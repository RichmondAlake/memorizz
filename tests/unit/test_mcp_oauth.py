"""OAuth sign-in for cloud apps (Grok Bot, ChatGPT dots) on the remote MCP server."""

import base64
import hashlib
import json
import secrets
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.testclient import TestClient

from memorizz.approval import SQLiteApprovalStore
from memorizz.mcp_server import (
    MemorizzMCPServerConfig,
    MemorizzRuntime,
    create_memorizz_http_app,
)
from memorizz.mcp_server import oauth as oauth_module
from memorizz.mcp_server.auth import RequestIdentity
from memorizz.mcp_server.config import (
    ALL_SCOPES,
    LOCAL_AGENTS_PRINCIPAL,
    PERSONAL_MEMORY_ID,
)
from memorizz.mcp_server.oauth_store import OAuthStore
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

BASE = "http://127.0.0.1:8766"
RESOURCE = BASE + "/mcp"
CHATGPT = "https://chatgpt.example/oauth/client.json"
CHATGPT_REDIRECT = "https://chatgpt.example/connector_platform_oauth_redirect"
GROK = "https://grok.example/oauth/client.json"
GROK_REDIRECT = "https://grok.example/oauth/callback"
# Grok Bot signs in with Cursor's shared client document (as published).
CURSOR = "https://cursor.com/oauth/mcp-client.json"
CURSOR_DOCUMENT = {
    "client_id": CURSOR,
    "client_name": "Cursor",
    "client_uri": "https://cursor.com",
    "redirect_uris": [
        "https://www.cursor.com/agents/mcp/oauth/callback",
        "https://www.cursor.com/bot/mcp/oauth/callback",
        "https://grok.com/connectors-oauth-exchange-code/",
        "http://localhost:8787/callback",
        "http://127.0.0.1:8787/callback",
    ],
    "grant_types": ["authorization_code", "refresh_token"],
    "response_types": ["code"],
    "token_endpoint_auth_method": "none",
}
# As published: no port, because Claude Code listens on whichever one is free.
CLAUDE_CODE = "https://claude.ai/oauth/claude-code-client-metadata"
CLAUDE_CODE_DOCUMENT = {
    "client_id": CLAUDE_CODE,
    "client_name": "Claude Code",
    "client_uri": "https://claude.ai",
    "redirect_uris": ["http://localhost/callback", "http://127.0.0.1/callback"],
    "grant_types": ["authorization_code", "refresh_token"],
    "response_types": ["code"],
    "token_endpoint_auth_method": "none",
}
# What a local agent (the plugins' stdio server) acts as.
LOCAL_AGENT = RequestIdentity(None, frozenset(ALL_SCOPES), False)


@pytest.fixture
def key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def documents(key):
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    public.update({"kid": "k1", "alg": "RS256", "use": "sig"})
    return {
        CHATGPT: {
            "client_id": CHATGPT,
            "client_name": "ChatGPT",
            "redirect_uris": [CHATGPT_REDIRECT],
            "token_endpoint_auth_method": "private_key_jwt",
            "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "jwks_uri": "https://chatgpt.example/oauth/jwks.json",
        },
        "https://chatgpt.example/oauth/jwks.json": {"keys": [public]},
        GROK: {
            "client_id": GROK,
            "client_name": "Grok",
            "redirect_uris": [GROK_REDIRECT],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
        },
        CURSOR: CURSOR_DOCUMENT,
        CLAUDE_CODE: CLAUDE_CODE_DOCUMENT,
    }


@pytest.fixture
def server(tmp_path, monkeypatch, documents):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    fetched = []

    async def fetch(url):
        fetched.append(url)
        return documents.get(url)

    monkeypatch.setattr(oauth_module, "fetch_json_document", fetch)
    config = MemorizzMCPServerConfig(
        transport="streamable-http",
        oauth=True,
        allow_writes=True,
        allow_agent_execution=False,
        tool_set="memory",
    )
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=tmp_path / "memory",
            embedding_provider=None,
            lazy_vector_indexes=True,
        )
    )
    runtime = MemorizzRuntime(
        config,
        provider=provider,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    store = OAuthStore(tmp_path / "apps.sqlite3")
    app = create_memorizz_http_app(config, runtime=runtime, oauth_store=store)
    with TestClient(app, base_url=BASE, follow_redirects=False) as client:
        client.store = store
        client.fetched = fetched
        client.runtime = runtime
        client.provider = provider
        yield client


def _pkce():
    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    return verifier, challenge


def _assertion(key, client_id, audience=BASE + "/token", **claims):
    now = int(time.time())
    payload = {
        "iss": client_id,
        "sub": client_id,
        "aud": audience,
        "iat": now,
        "exp": now + 120,
        "jti": secrets.token_hex(8),
        **claims,
    }
    return jwt.encode(payload, key, algorithm="RS256", headers={"kid": "k1"})


def _sign_in(client, client_id, redirect_uri, *, code=None, principal="richmond"):
    """Authorize, approve with a pairing code, and return (auth code, verifier)."""
    verifier, challenge = _pkce()
    response = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "st-1",
            "resource": RESOURCE,
        },
    )
    assert response.status_code == 302, response.text
    approve = urlparse(response.headers["location"])
    assert approve.path == "/oauth/approve"
    request_id = parse_qs(approve.query)["request"][0]
    page = client.get("/oauth/approve", params={"request": request_id})
    assert page.status_code == 200
    assert page.headers["x-frame-options"] == "DENY"
    if code is None:
        code, _ = client.store.create_pairing_code(
            principal, ["memorizz:read", "memorizz:write"]
        )
    approved = client.post(
        "/oauth/approve",
        data={"request": request_id, "action": "allow", "code": code.lower()},
    )
    assert approved.status_code == 302, approved.text
    back = urlparse(approved.headers["location"])
    assert f"{back.scheme}://{back.netloc}{back.path}" == redirect_uri
    query = parse_qs(back.query)
    assert query["state"] == ["st-1"]
    return query["code"][0], verifier


def _mcp(client, token, method, params=None, request_id=1):
    response = client.post(
        "/mcp",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": "2025-06-18",
        },
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        },
    )
    if response.status_code != 200:
        return response
    body = response.text
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        body = next(
            line[5:].strip() for line in body.splitlines() if line.startswith("data:")
        )
    return json.loads(body)


def _tool(client, token, name, arguments):
    result = _mcp(client, token, "tools/call", {"name": name, "arguments": arguments})
    assert "result" in result, result
    assert not result["result"].get("isError"), result
    content = result["result"]["structuredContent"]
    return content.get("result", content)


def _connect(client, client_id, redirect_uri, *, principal="richmond", code=None):
    """Sign an app in and return its access token."""
    auth_code, verifier = _sign_in(
        client, client_id, redirect_uri, principal=principal, code=code
    )
    tokens = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": auth_code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
            "resource": RESOURCE,
        },
    )
    assert tokens.status_code == 200, tokens.text
    return tokens.json()["access_token"]


def _authorize_status(client, client_id, redirect_uri):
    _, challenge = _pkce()
    return client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "st-1",
            "resource": RESOURCE,
        },
    )


def test_metadata_advertises_metadata_documents_signed_clients_and_both_scopes(server):
    meta = server.get("/.well-known/oauth-authorization-server").json()
    assert meta["issuer"] == BASE
    assert meta["client_id_metadata_document_supported"] is True
    assert {"none", "private_key_jwt"} <= set(
        meta["token_endpoint_auth_methods_supported"]
    )
    assert meta["code_challenge_methods_supported"] == ["S256"]
    assert meta["registration_endpoint"] == BASE + "/register"
    assert meta["scopes_supported"] == ["memorizz:read", "memorizz:write"]
    assert server.get("/.well-known/openid-configuration").json() == meta
    for path in (
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-protected-resource",
    ):
        resource = server.get(path).json()
        assert resource["resource"] == RESOURCE
        assert resource["authorization_servers"] == [BASE]
        assert resource["scopes_supported"] == ["memorizz:read", "memorizz:write"]
    challenge = server.post("/mcp", json={})
    assert challenge.status_code == 401
    assert "resource_metadata=" in challenge.headers["www-authenticate"]


def test_two_apps_share_one_memory_and_each_save_names_its_app(server, key):
    # ChatGPT: a client metadata document and signed token requests.
    code, verifier = _sign_in(server, CHATGPT, CHATGPT_REDIRECT)
    tokens = server.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": CHATGPT_REDIRECT,
            "client_id": CHATGPT,
            "code_verifier": verifier,
            "resource": RESOURCE,
            "client_assertion_type": oauth_module.JWT_ASSERTION_TYPE,
            "client_assertion": _assertion(key, CHATGPT),
        },
    )
    assert tokens.status_code == 200, tokens.text
    chatgpt = tokens.json()
    assert chatgpt["scope"] == "memorizz:read memorizz:write"
    saved = _tool(
        server,
        chatgpt["access_token"],
        "memorizz_store_memory",
        {"content": "Richmond's launch is on 12 November"},
    )
    assert saved["ok"] is True

    # Grok: dynamic client registration with no client secret.
    registered = server.post(
        "/register",
        json={
            "client_name": "Grok",
            "redirect_uris": [GROK_REDIRECT],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
        },
    ).json()
    code, verifier = _sign_in(server, registered["client_id"], GROK_REDIRECT)
    grok = server.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": GROK_REDIRECT,
            "client_id": registered["client_id"],
            "code_verifier": verifier,
        },
    ).json()
    found = _tool(server, grok["access_token"], "search", {"query": "launch"})
    assert [r["title"] for r in found["results"]] == [
        "Richmond's launch is on 12 November"
    ]
    record = _tool(
        server, grok["access_token"], "fetch", {"id": found["results"][0]["id"]}
    )
    assert record["text"] == "Richmond's launch is on 12 November"
    assert record["metadata"]["saved_by"] == "ChatGPT"

    apps = {g["app_name"]: g for g in server.store.list_grants()}
    assert set(apps) == {"ChatGPT", "Grok"}
    assert {g["principal"] for g in apps.values()} == {"richmond"}


@pytest.mark.parametrize(
    "redirect_uri, app",
    [
        ("https://www.cursor.com/bot/mcp/oauth/callback", "Grok Bot"),
        ("http://localhost:8787/callback", "Grok Bot"),
        ("https://grok.com/connectors-oauth-exchange-code/", "Grok"),
        ("https://www.cursor.com/agents/mcp/oauth/callback", "Cursor"),
    ],
)
def test_grok_bot_signs_in_as_cursor_but_is_named_by_its_callback(
    server, redirect_uri, app
):
    verifier, challenge = _pkce()
    response = server.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": CURSOR,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": "st-1",
            "resource": RESOURCE,
        },
    )
    page = server.get(response.headers["location"])
    assert f"Connect {app} to your MemoRizz memory?" in page.text

    code, verifier = _sign_in(server, CURSOR, redirect_uri)
    tokens = server.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CURSOR,
            "code_verifier": verifier,
            "resource": RESOURCE,
        },
    )
    assert tokens.status_code == 200, tokens.text
    assert [g["app_name"] for g in server.store.list_grants()] == [app]


def test_grok_bot_registration_with_its_app_link_callback_is_named_grok_bot(server):
    registered = server.post(
        "/register",
        json={
            "client_name": "Cursor",
            "redirect_uris": [
                "grokbot://mcp/oauth/callback",
                "cursor://anysphere.cursor-mcp/oauth/return",
                "http://localhost:8787/callback",
            ],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
        },
    )
    assert registered.status_code == 201, registered.text
    client_id = registered.json()["client_id"]
    code, _ = _sign_in(server, client_id, "grokbot://mcp/oauth/callback")
    assert code
    assert [g["app_name"] for g in server.store.list_grants()] == ["Grok Bot"]


def test_memory_tool_set_lists_only_memory_tools(server):
    code, verifier = _sign_in(server, GROK, GROK_REDIRECT)
    token = server.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": GROK_REDIRECT,
            "client_id": GROK,
            "code_verifier": verifier,
        },
    ).json()["access_token"]
    names = {t["name"] for t in _mcp(server, token, "tools/list")["result"]["tools"]}
    assert {
        "search",
        "fetch",
        "memorizz_store_memory",
        "memorizz_search_memories",
    } <= names
    assert not any("harness" in name or "agent" in name for name in names)


def test_refresh_rotates_and_revoking_the_app_cuts_it_off(server):
    code, verifier = _sign_in(server, GROK, GROK_REDIRECT)
    first = server.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": GROK_REDIRECT,
            "client_id": GROK,
            "code_verifier": verifier,
        },
    ).json()
    reused = server.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": GROK_REDIRECT,
            "client_id": GROK,
            "code_verifier": verifier,
        },
    )
    assert reused.status_code == 400
    refreshed = server.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": first["refresh_token"],
            "client_id": GROK,
        },
    )
    assert refreshed.status_code == 200, refreshed.text
    again = server.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": first["refresh_token"],
            "client_id": GROK,
        },
    )
    assert again.status_code == 400
    token = refreshed.json()["access_token"]
    assert _tool(server, token, "memorizz_server_info", {})["principal"] == "richmond"

    (grant,) = server.store.list_grants()
    assert server.store.revoke_grant(grant["grant_id"]) is True
    assert _mcp(server, token, "tools/list").status_code == 401
    assert server.store.list_grants() == []


def test_wrong_used_or_expired_pairing_codes_do_not_connect(server):
    response = server.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": GROK,
            "redirect_uri": GROK_REDIRECT,
            "code_challenge": _pkce()[1],
            "code_challenge_method": "S256",
            "state": "s",
        },
    )
    request_id = parse_qs(urlparse(response.headers["location"]).query)["request"][0]
    wrong = server.post(
        "/oauth/approve",
        data={"request": request_id, "action": "allow", "code": "AAAA-AAAA"},
    )
    assert wrong.status_code == 400
    assert "wrong, used or expired" in wrong.text
    for _ in range(4):
        server.post(
            "/oauth/approve",
            data={"request": request_id, "action": "allow", "code": "BBBB-BBBB"},
        )
    good, _ = server.store.create_pairing_code("richmond", ["memorizz:read"])
    locked = server.post(
        "/oauth/approve", data={"request": request_id, "action": "allow", "code": good}
    )
    assert locked.status_code == 302
    assert parse_qs(urlparse(locked.headers["location"]).query)["error"] == [
        "access_denied"
    ]
    # The good code was never spent, but a used one can't be used twice.
    assert server.store.redeem_pairing_code(good)["principal"] == "richmond"
    assert server.store.redeem_pairing_code(good) is None
    expired, _ = server.store.create_pairing_code(
        "richmond", ["memorizz:read"], ttl_seconds=60
    )
    with server.store._connect() as db:
        db.execute("UPDATE pairing_codes SET expires_at = 0")
    assert server.store.redeem_pairing_code(expired) is None
    assert server.store.list_grants() == []


def test_denying_redirects_back_with_access_denied(server):
    response = server.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": GROK,
            "redirect_uri": GROK_REDIRECT,
            "code_challenge": _pkce()[1],
            "code_challenge_method": "S256",
            "state": "s",
        },
    )
    request_id = parse_qs(urlparse(response.headers["location"]).query)["request"][0]
    denied = server.post(
        "/oauth/approve", data={"request": request_id, "action": "deny"}
    )
    query = parse_qs(urlparse(denied.headers["location"]).query)
    assert query["error"] == ["access_denied"] and query["state"] == ["s"]


def test_tokens_for_another_server_are_refused(server):
    response = server.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": GROK,
            "redirect_uri": GROK_REDIRECT,
            "code_challenge": _pkce()[1],
            "code_challenge_method": "S256",
            "resource": "https://elsewhere.example/mcp",
        },
    )
    assert parse_qs(urlparse(response.headers["location"]).query)["error"] == [
        "invalid_target"
    ]


def test_signed_requests_must_be_valid_fresh_and_unused(server, key):
    def exchange(assertion):
        code, verifier = _sign_in(server, CHATGPT, CHATGPT_REDIRECT)
        return server.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": CHATGPT_REDIRECT,
                "client_id": CHATGPT,
                "code_verifier": verifier,
                "client_assertion_type": oauth_module.JWT_ASSERTION_TYPE,
                "client_assertion": assertion,
            },
        )

    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert exchange(_assertion(other, CHATGPT)).status_code == 401
    assert (
        exchange(
            _assertion(key, CHATGPT, audience="https://elsewhere.example/token")
        ).status_code
        == 401
    )
    assert (
        exchange(_assertion(key, CHATGPT, exp=int(time.time()) + 3600)).status_code
        == 401
    )
    replayed = _assertion(key, CHATGPT, jti="once")
    assert exchange(replayed).status_code == 200
    assert exchange(replayed).status_code == 401
    code, verifier = _sign_in(server, CHATGPT, CHATGPT_REDIRECT)
    unsigned = server.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": CHATGPT_REDIRECT,
            "client_id": CHATGPT,
            "code_verifier": verifier,
        },
    )
    assert unsigned.status_code == 401


def test_client_documents_must_name_themselves_and_keep_keys_on_their_site(
    server, documents
):
    documents["https://bad.example/a.json"] = {
        "client_id": "https://bad.example/other.json",
        "redirect_uris": ["https://bad.example/cb"],
    }
    documents["https://bad.example/b.json"] = {
        "client_id": "https://bad.example/b.json",
        "redirect_uris": ["https://bad.example/cb"],
        "token_endpoint_auth_method": "private_key_jwt",
        "token_endpoint_auth_methods_supported": ["private_key_jwt"],
        "jwks_uri": "https://attacker.example/jwks.json",
    }
    for client_id in ("https://bad.example/a.json", "https://bad.example/b.json"):
        response = server.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "https://bad.example/cb",
                "code_challenge": _pkce()[1],
                "code_challenge_method": "S256",
            },
        )
        assert response.status_code == 400


def test_public_host_reaches_the_mcp_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    config = MemorizzMCPServerConfig(
        transport="streamable-http",
        oauth=True,
        public_url="https://memory.example.com",
    )
    app = create_memorizz_http_app(
        config, oauth_store=OAuthStore(tmp_path / "a.sqlite3")
    )
    with TestClient(app, base_url="https://memory.example.com") as client:
        response = client.post("/mcp", json={})
        assert response.status_code == 401  # reached auth, not "Invalid Host header"
        meta = client.get("/.well-known/oauth-protected-resource/mcp").json()
        assert meta["resource"] == "https://memory.example.com/mcp"


def test_config_rules_for_oauth():
    with pytest.raises(ValueError, match="streamable-http"):
        MemorizzMCPServerConfig(transport="stdio", oauth=True)
    with pytest.raises(ValueError, match="bare origin"):
        MemorizzMCPServerConfig(
            transport="streamable-http",
            oauth=True,
            public_url="https://x.example/memory",
        )
    config = MemorizzMCPServerConfig(transport="streamable-http", oauth=True)
    assert config.auth_required is True
    assert config.public_url == "http://127.0.0.1:8766"
    assert config.public_dict()["oauth"] is True
    with pytest.raises(ValueError, match="tool set"):
        MemorizzMCPServerConfig(tool_set="everything")


def test_stdio_server_has_no_search_or_fetch_tools(tmp_path, monkeypatch):
    import asyncio

    from memorizz.mcp_server import create_memorizz_mcp_server

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    server = create_memorizz_mcp_server(MemorizzMCPServerConfig())
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert "memorizz_store_memory" in names
    assert not {"search", "fetch"} & names


def test_store_keeps_only_hashes(tmp_path):
    store = OAuthStore(tmp_path / "s.sqlite3")
    code, _ = store.create_pairing_code("richmond", ["memorizz:read"])
    raw = Path(store.path).read_bytes()
    assert code.replace("-", "").encode() not in raw


def test_apps_cli_pairs_lists_and_revokes(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from memorizz.cli.mcp_apps import apps_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    runner = CliRunner()
    paired = runner.invoke(
        apps_app, ["pair", "--principal", "richmond", "--read-only", "--json"]
    )
    assert paired.exit_code == 0, paired.output
    code = json.loads(paired.output)["code"]
    store = OAuthStore()
    assert store.redeem_pairing_code(code) == {
        "principal": "richmond",
        "scopes": ["memorizz:read"],
    }
    assert runner.invoke(apps_app, ["pair", "--principal", "bad name"]).exit_code != 0

    grant_id = store.create_grant(
        "https://grok.example/c.json", "Grok", "richmond", ["memorizz:read"]
    )
    listed = json.loads(runner.invoke(apps_app, ["list", "--json"]).output)
    assert [g["app_name"] for g in listed] == ["Grok"]
    assert grant_id in runner.invoke(apps_app, ["list"], env={"COLUMNS": "60"}).output
    revoked = runner.invoke(apps_app, ["revoke", grant_id])
    assert revoked.exit_code == 0 and "Disconnected Grok" in revoked.output
    assert runner.invoke(apps_app, ["revoke", grant_id]).exit_code == 1
    assert json.loads(runner.invoke(apps_app, ["list", "--json"]).output) == []


def test_apps_serve_needs_a_public_address(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from memorizz.cli.mcp_apps import apps_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("MEMORIZZ_CLOUDFLARED", raising=False)
    runner = CliRunner()
    assert "public HTTPS" in runner.invoke(apps_app, ["serve"]).output
    assert "cloudflared" in runner.invoke(apps_app, ["serve", "--tunnel"]).output
    assert (
        "bare origin"
        in runner.invoke(
            apps_app, ["serve", "--public-url", "https://x.example/memory"]
        ).output
    )


def test_many_wrong_codes_from_any_sign_in_pause_code_checks(tmp_path):
    from memorizz.mcp_server import oauth_store

    store = OAuthStore(tmp_path / "s.sqlite3")
    good, _ = store.create_pairing_code("richmond", ["memorizz:read"])
    for _ in range(oauth_store.MAX_PAIRING_FAILURES):
        assert store.redeem_pairing_code("ZZZZ-ZZZZ") is None
    assert store.pairing_locked() is True
    with store._connect() as db:
        db.execute("UPDATE pairing_failures SET at = 0")
    assert store.pairing_locked() is False
    assert store.redeem_pairing_code(good) is not None


@pytest.mark.parametrize(
    "redirect_uri",
    ["http://localhost:61475/callback", "http://127.0.0.1:52011/callback"],
)
def test_claude_code_signs_in_on_the_local_port_it_picked(server, redirect_uri):
    token = _connect(server, CLAUDE_CODE, redirect_uri)
    saved = _tool(server, token, "memorizz_store_memory", {"content": "Hello"})
    assert saved["ok"] is True
    assert server.store.list_grants()[0]["app_name"] == "Claude Code"


@pytest.mark.parametrize(
    "redirect_uri",
    [
        "http://localhost.evil.example:61475/callback",
        "http://evil.example/callback",
        "http://localhost:61475/other",
        "http://localhost:61475/callback/extra",
        "http://localhost:61475/callback?next=evil",
        "https://localhost:61475/callback",
        "http://user@localhost:61475/callback",
    ],
)
def test_only_the_port_of_a_local_redirect_may_differ(server, redirect_uri):
    response = _authorize_status(server, CLAUDE_CODE, redirect_uri)
    assert response.status_code == 400
    assert "not registered" in response.text


def test_a_web_redirect_still_has_to_match_exactly(server):
    response = _authorize_status(
        server, GROK, "https://grok.example:8443/oauth/callback"
    )
    assert response.status_code == 400


def test_a_local_redirect_keeps_the_host_it_registered(server):
    registered = server.post(
        "/register",
        json={
            "client_name": "Only localhost",
            "redirect_uris": ["http://localhost/cb"],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        },
    ).json()
    client_id = registered["client_id"]
    assert (
        _authorize_status(server, client_id, "http://localhost:9/cb").status_code == 302
    )
    assert (
        _authorize_status(server, client_id, "http://127.0.0.1:9/cb").status_code == 400
    )


def test_a_shared_app_and_the_local_agents_read_each_others_memories(server):
    from memorizz.cli import plugin_commands

    project = "project-acme-abc123"
    server.runtime.remember(
        "The acme deploy needs the staging VPN",
        "knowledge_base",
        LOCAL_AGENT,
        memory_id=project,
    )
    token = _connect(server, GROK, GROK_REDIRECT, principal=LOCAL_AGENTS_PRINCIPAL)

    # The app finds what a local agent saved, in any project.
    found = _tool(server, token, "search", {"query": "acme deploy VPN"})["results"]
    assert len(found) == 1
    record = _tool(server, token, "fetch", {"id": found[0]["id"]})
    assert "staging VPN" in record["text"]
    assert record["metadata"]["memory_id"] == project

    # A save with no memory ID lands where the local agents read.
    saved = _tool(
        server,
        token,
        "memorizz_store_memory",
        {"content": "Richmond prefers short answers"},
    )
    assert saved["memory_id"] == PERSONAL_MEMORY_ID
    local = server.runtime.search_memories(
        "short answers", "knowledge_base", LOCAL_AGENT, memory_id=PERSONAL_MEMORY_ID
    )["memories"]
    assert [row["content"] for row in local] == ["Richmond prefers short answers"]
    assert local[0]["user_id"] is None
    assert local[0]["created_via"] == "Grok"

    context = plugin_commands.session_context("/tmp", provider=server.provider)
    assert "Richmond prefers short answers" in context
    assert f"`{PERSONAL_MEMORY_ID}`" in context

    # A fact it corrects stays in the shared memory, and names no other owner.
    updated = _tool(
        server,
        token,
        "memorizz_update_memory",
        {"record_id": saved["record_id"], "content": "Richmond prefers brief answers"},
    )
    assert updated["ok"] is True
    row = server.runtime.search_memories(
        "brief answers", "knowledge_base", LOCAL_AGENT, memory_id=PERSONAL_MEMORY_ID
    )["memories"][0]
    assert row["content"] == "Richmond prefers brief answers"
    assert row["user_id"] is None


def test_a_shared_apps_entities_land_where_local_agents_look(server):
    token = _connect(server, GROK, GROK_REDIRECT, principal=LOCAL_AGENTS_PRINCIPAL)
    saved = _tool(
        server,
        token,
        "memorizz_upsert_entity",
        {"name": "Harbor", "entity_type": "service", "attributes": {"owner": "Ada"}},
    )
    assert saved["ok"] is True
    found = server.runtime.lookup_entities(
        LOCAL_AGENT, query="Harbor", memory_id=PERSONAL_MEMORY_ID
    )
    assert [row["name"] for row in found["entities"]] == ["Harbor"]


@pytest.mark.parametrize(
    "requested, registered, same",
    [
        ("http://[::1]:5000/cb", "http://[::1]/cb", True),
        ("http://127.0.0.1:5000/cb", "http://127.0.0.1:80/cb", True),
        ("http://[::1]:5000/cb", "http://127.0.0.1/cb", False),
        ("https://127.0.0.1:5000/cb", "https://127.0.0.1/cb", False),
        ("http://example.com:5000/cb", "http://example.com/cb", False),
    ],
)
def test_loopback_redirects_compare_all_but_the_port(requested, registered, same):
    assert oauth_module._same_loopback_redirect(requested, registered) is same


def test_an_app_without_sharing_sees_nothing_of_the_local_agents(server):
    server.runtime.remember(
        "The acme deploy needs the staging VPN",
        "knowledge_base",
        LOCAL_AGENT,
        memory_id="project-acme-abc123",
    )
    token = _connect(server, GROK, GROK_REDIRECT)
    assert _tool(server, token, "search", {"query": "acme VPN"})["results"] == []
    saved = _tool(server, token, "memorizz_store_memory", {"content": "My own note"})
    assert saved["memory_id"] == PERSONAL_MEMORY_ID
    local = server.runtime.search_memories(
        "own note", "knowledge_base", LOCAL_AGENT, memory_id=PERSONAL_MEMORY_ID
    )["memories"]
    assert local == []


def test_a_shared_app_follows_the_local_principal_of_a_multi_user_store(
    server, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_MCP_SERVER_LOCAL_PRINCIPAL", "alice")
    alice = RequestIdentity("alice", frozenset(ALL_SCOPES), True)
    server.runtime.remember(
        "Alice's note", "knowledge_base", alice, memory_id="project-a-1"
    )
    server.runtime.remember(
        "Nobody's note", "knowledge_base", LOCAL_AGENT, memory_id="project-a-1"
    )
    token = _connect(server, GROK, GROK_REDIRECT, principal=LOCAL_AGENTS_PRINCIPAL)
    rows = _tool(server, token, "memorizz_list_memories", {"limit": 10})["memories"]
    assert [row["content"] for row in rows] == ["Alice's note"]


@pytest.mark.parametrize(
    "memory_type", ["tool_log", "semantic_cache", "workflow_memory"]
)
def test_a_shared_app_stays_out_of_the_agents_working_records(server, memory_type):
    token = _connect(server, GROK, GROK_REDIRECT, principal=LOCAL_AGENTS_PRINCIPAL)
    result = _mcp(
        server,
        token,
        "tools/call",
        {
            "name": "memorizz_list_memories",
            "arguments": {"memory_type": memory_type},
        },
    )
    assert result["result"].get("isError"), result
    assert "memory_type_not_tenant_scoped" in json.dumps(result)


def test_a_shared_app_can_read_past_sessions(server):
    token = _connect(server, GROK, GROK_REDIRECT, principal=LOCAL_AGENTS_PRINCIPAL)
    for memory_type in ("conversation_memory", "summaries", "entity_memory"):
        listed = _tool(
            server, token, "memorizz_list_memories", {"memory_type": memory_type}
        )
        assert listed["ok"] is True


def test_sharing_needs_a_server_that_lists_only_the_memory_tools(
    tmp_path, monkeypatch, documents
):
    """With every tool listed, acting as the local owner would reach the rest."""

    async def fetch(url):
        return documents.get(url)

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setattr(oauth_module, "fetch_json_document", fetch)
    config = MemorizzMCPServerConfig(
        transport="streamable-http",
        oauth=True,
        allow_writes=True,
        allow_agent_execution=False,
    )
    runtime = MemorizzRuntime(
        config,
        provider=FileSystemProvider(
            FileSystemConfig(
                root_path=tmp_path / "memory",
                embedding_provider=None,
                lazy_vector_indexes=True,
            )
        ),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    store = OAuthStore(tmp_path / "apps.sqlite3")
    app = create_memorizz_http_app(config, runtime=runtime, oauth_store=store)
    with TestClient(app, base_url=BASE, follow_redirects=False) as client:
        client.store = store
        _, challenge = _pkce()
        location = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": GROK,
                "redirect_uri": GROK_REDIRECT,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "st-1",
                "resource": RESOURCE,
            },
        ).headers["location"]
        request_id = parse_qs(urlparse(location).query)["request"][0]
        code, _ = store.create_pairing_code(
            LOCAL_AGENTS_PRINCIPAL, ["memorizz:read", "memorizz:write"]
        )
        refused = client.post(
            "/oauth/approve",
            data={"request": request_id, "action": "allow", "code": code},
        )
        assert refused.status_code == 400
        assert "memorizz mcp apps serve" in refused.text
        assert store.list_grants() == []

        # And a grant made some other way doesn't get a usable token.
        grant = store.create_grant(
            GROK, "Grok", LOCAL_AGENTS_PRINCIPAL, ["memorizz:read"]
        )
        store.save_token(
            "mza_shared", "access", grant, GROK, ["memorizz:read"], RESOURCE, None
        )
        response = client.post(
            "/mcp",
            headers={
                "Authorization": "Bearer mza_shared",
                "Accept": "application/json, text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert response.status_code == 401


def test_apps_cli_shares_the_local_agents_memory_only_when_asked(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from memorizz.cli.mcp_apps import apps_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()
    shared = runner.invoke(apps_app, ["pair", "--shared", "--json"])
    assert shared.exit_code == 0, shared.output
    assert json.loads(shared.output)["principal"] == LOCAL_AGENTS_PRINCIPAL
    plain = runner.invoke(apps_app, ["pair", "--json"])
    assert json.loads(plain.output)["principal"] != LOCAL_AGENTS_PRINCIPAL

    both = runner.invoke(apps_app, ["pair", "--shared", "--principal", "alice"])
    assert both.exit_code != 0
    forged = runner.invoke(apps_app, ["pair", "--principal", LOCAL_AGENTS_PRINCIPAL])
    assert forged.exit_code != 0

    store = OAuthStore()
    store.create_grant("c", "Grok Bot", LOCAL_AGENTS_PRINCIPAL, ["memorizz:read"])
    listed = runner.invoke(apps_app, ["list"])
    assert "local agents" in listed.output


def test_apps_serve_refuses_the_reserved_principal(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from memorizz.cli.mcp_apps import apps_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    result = CliRunner().invoke(
        apps_app,
        [
            "serve",
            "--public-url",
            "https://memory.example.com",
            "--principal",
            LOCAL_AGENTS_PRINCIPAL,
        ],
    )
    assert result.exit_code != 0
