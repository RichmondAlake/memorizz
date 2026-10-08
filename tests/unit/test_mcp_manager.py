"""Protocol, credential, and policy tests for first-class MCP support."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz.mcp import MCPClientManager, MCPConfigurationError
from memorizz.mcp.errors import MCPAuthorizationRequired
from memorizz.mcp.manager import (
    _find_mcp_error,
    _reject_auth_response,
    _same_origin_issuer,
    _scope_pinned_metadata_class,
    _sign_in_probe,
    _tolerate_origin_issuer_slash,
)
from memorizz.mcp.oauth import OAuthFlowRegistry, PendingOAuthFlow

SERVER = Path(__file__).parents[1] / "fixtures" / "mcp_stdio_server.py"


def _manager(tmp_path, monkeypatch, **overrides):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    config = {
        "name": "test",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER)],
        "timeout": 10,
        # The fixture's tools carry no annotations and "echo" is not a read
        # verb, so the host allowlist is what makes it callable without approval.
        "read_only_tools": ["echo"],
        **overrides,
    }
    return MCPClientManager(owner_id="agent-1", servers=[config])


@pytest.mark.unit
def test_stdio_protocol_handles_real_json_resources_and_prompts(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)

    tools = manager.list_tools("test")
    assert tools["ok"] is True
    assert {item["name"] for item in tools["tools"]} == {"echo", "create_item"}

    nested = {"enabled": True, "missing": None, "items": [1, False, None]}
    called = manager.call_tool("test", "echo", {"payload": nested})
    assert called["ok"] is True
    assert nested == called["result"]["structuredContent"]

    resources = manager.list_resources("test")
    assert resources["ok"] is True
    assert resources["resources"][0]["uri"] == "memo://welcome"
    read = manager.read_resource("test", "memo://welcome")
    assert read["ok"] is True
    assert "hello from MCP" in json.dumps(read)

    prompts = manager.list_prompts("test")
    assert prompts["ok"] is True
    prompt = manager.get_prompt("test", "greeting", {"name": "Ada"})
    assert prompt["ok"] is True
    assert "Hello Ada" in json.dumps(prompt)


@pytest.mark.unit
def test_mutations_require_durable_approval_and_are_not_retried(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch, max_retries=5)
    denied = manager.call_tool("test", "create_item", {"name": "draft"})
    assert denied["ok"] is False
    assert denied["error_code"] == "approval_required"

    proposal_id = denied["proposal"]["proposal_id"]
    approved = manager.approve_tool_call(
        proposal_id, approver_id="operator@example.com"
    )
    assert approved["status"] == "approved"
    resumed = manager.resume_tool_call(proposal_id)
    assert resumed["ok"] is True
    assert (
        manager.resume_tool_call(proposal_id)["error_code"] == "invalid_approval_state"
    )
    assert manager.diagnostics()["metrics"]["test"]["requests"] == 1


@pytest.mark.unit
def test_mcp_annotations_override_name_heuristics_for_approval(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)
    manager._tool_metadata["test"] = {
        "delete_preview": {
            "name": "delete_preview",
            "annotations": {"readOnlyHint": True},
        },
        "calculate": {
            "name": "calculate",
            "annotations": {"destructiveHint": True},
        },
    }
    monkeypatch.setattr(
        manager,
        "_call_tool_authorized",
        lambda server, tool, arguments: {
            "ok": True,
            "server_name": server,
            "tool_name": tool,
            "arguments": arguments,
        },
    )

    read = manager.call_tool("test", "delete_preview", {"record_id": "r-1"})
    destructive = manager.call_tool("test", "calculate", {"value": 1})

    assert read["ok"] is True
    assert destructive["error_code"] == "approval_required"


@pytest.mark.unit
def test_inline_secrets_are_encrypted_and_never_exported(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    manager = MCPClientManager(
        owner_id="agent-secret",
        servers=[
            {
                "name": "remote",
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
                "headers": {"X-API-Key": "header-secret"},
                "auth": {"type": "bearer", "token": "bearer-secret"},
            }
        ],
    )
    public = json.dumps(manager.server_dicts())
    assert "header-secret" not in public
    assert "bearer-secret" not in public
    assert manager.server_dicts()[0]["header_names"] == ["X-API-Key"]

    encrypted = (tmp_path / "mcp_credentials.enc").read_bytes()
    assert b"header-secret" not in encrypted
    assert b"bearer-secret" not in encrypted
    assert (tmp_path / "mcp_credentials.key").stat().st_mode & 0o077 == 0


@pytest.mark.unit
def test_remote_security_defaults_to_https_and_public_network(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    manager = MCPClientManager(
        owner_id="agent-1",
        servers=[
            {
                "name": "unsafe",
                "transport": "http",
                "url": "http://127.0.0.1:8765/mcp",
            }
        ],
    )
    result = manager.test_connection("unsafe")
    assert result["ok"] is False
    assert result["error_code"] == "configuration_error"


@pytest.mark.unit
def test_notion_oauth_required_path_stops_before_network_use(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    manager = MCPClientManager(
        owner_id="agent-oauth",
        servers=[
            {
                "name": "notion",
                "transport": "streamable_http",
                "url": "https://unresolvable.invalid/mcp",
                "auth": {
                    "type": "oauth",
                    "redirect_uri": "http://127.0.0.1:8765/api/mcp/oauth/callback",
                },
            }
        ],
    )
    result = manager.test_connection("notion")
    assert result["ok"] is False
    assert result["error_code"] == "authorization_required"


@pytest.mark.unit
def test_notion_unauthenticated_path_reports_missing_bearer_token(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    manager = MCPClientManager(
        owner_id="agent-notion",
        servers=[
            {
                "name": "notion",
                "transport": "streamable_http",
                "url": "https://unresolvable.invalid/mcp",
                "auth": {"type": "bearer"},
            }
        ],
    )

    result = manager.test_connection("notion")

    assert result["ok"] is False
    assert result["error_code"] == "authorization_required"
    assert "token" in result["error"].lower()


@pytest.mark.unit
def test_http_auth_status_survives_transport_exception_group():
    with pytest.raises(MCPAuthorizationRequired):
        asyncio.run(_reject_auth_response(SimpleNamespace(status_code=401), "notion"))
    authorization = MCPAuthorizationRequired("authorization required", "notion")
    grouped = RuntimeError("task group failed")
    grouped.exceptions = [RuntimeError("transport"), authorization]
    assert _find_mcp_error(grouped) is authorization


@pytest.mark.unit
def test_notion_mutation_path_returns_approval_before_network_use(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    manager = MCPClientManager(
        owner_id="agent-notion",
        servers=[
            {
                "name": "notion",
                "transport": "streamable_http",
                "url": "https://unresolvable.invalid/mcp",
                "mutation_tools": ["create_page"],
            }
        ],
    )

    result = manager.call_tool(
        "notion", "create_page", {"parent": "page-1", "title": "Roadmap"}
    )

    assert result["ok"] is False
    assert result["error_code"] == "approval_required"
    assert result["proposal"]["tool_name"] == "mcp:notion:create_page"
    assert result["proposal"]["arguments"]["title"] == "Roadmap"


@pytest.mark.unit
def test_notion_streamable_http_success_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("The execution sandbox does not allow loopback sockets")
        port = listener.getsockname()[1]
    environment = dict(os.environ)
    environment.update(
        {"MCP_TEST_TRANSPORT": "streamable-http", "MCP_TEST_PORT": str(port)}
    )
    process = subprocess.Popen(
        [sys.executable, str(SERVER)],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if process.poll() is not None:
                    pytest.fail("Streamable HTTP test server exited during startup")
                time.sleep(0.05)
        else:
            pytest.fail("Streamable HTTP test server did not start")

        manager = MCPClientManager(
            owner_id="agent-http",
            servers=[
                {
                    "name": "notion",
                    "transport": "streamable_http",
                    "url": f"http://127.0.0.1:{port}/mcp",
                    "allow_private_network": True,
                    "timeout": 10,
                }
            ],
        )
        result = manager.test_connection("notion")
        assert result["ok"] is True, result
        assert result["tool_count"] == 2
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.unit
def test_invalid_duplicate_names_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    config = {
        "name": "same",
        "transport": "stdio",
        "command": sys.executable,
    }
    with pytest.raises(MCPConfigurationError):
        MCPClientManager(owner_id="agent", servers=[config, config])


@pytest.mark.unit
def test_oauth_callback_state_is_single_use():
    registry = OAuthFlowRegistry()
    flow = PendingOAuthFlow(owner_id="agent", server_name="notion", state="state-1")
    registry.add(flow)

    completed = registry.complete_callback(state="state-1", code="code-1")

    assert completed is flow
    assert completed.callback_ready.is_set()
    assert registry.complete_callback(state="state-1", code="replay") is None


@pytest.mark.unit
def test_sign_in_probe_picks_an_argument_free_read_only_tool():
    tools = [
        {"name": "create_draft", "annotations": {"readOnlyHint": False}},
        {
            "name": "get_thread",
            "annotations": {"readOnlyHint": True},
            "inputSchema": {"required": ["threadId"]},
        },
        {"name": "search_threads", "annotations": {"readOnlyHint": True}},
        {"name": "list_labels", "annotations": {"readOnlyHint": True}},
        {"name": "list_everything"},
    ]
    assert _sign_in_probe(tools) == "list_labels"
    assert _sign_in_probe(tools[:2]) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("auth", "raises_on_401"),
    [
        ({"type": "oauth", "redirect_uri": "http://127.0.0.1:8765/cb"}, False),
        ({"type": "none"}, True),
    ],
)
def test_oauth_clients_let_the_provider_see_401s(
    tmp_path, monkeypatch, auth, raises_on_401
):
    """httpx runs response hooks before the auth flow, so an OAuth client
    must not turn a 401 into an error before sign-in can start."""
    import httpx2

    captured = {}

    class Stop(Exception):
        pass

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            raise Stop()

    monkeypatch.setattr(httpx2, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        "memorizz.mcp.manager.validate_remote_url", lambda *args, **kwargs: None
    )
    manager = _manager(
        tmp_path,
        monkeypatch,
        name="remote",
        transport="streamable_http",
        url="https://mcp.example.com/mcp",
        command=None,
        args=[],
        auth=auth,
    )

    async def enter():
        async def noop(*args):
            return None

        async with manager._client(
            manager.get_server("remote"),
            redirect_handler=noop,
            callback_handler=noop,
        ):
            pass

    with pytest.raises(Stop):
        asyncio.run(enter())
    assert bool(captured["event_hooks"]["response"]) is raises_on_401


@pytest.mark.unit
def test_configured_oauth_scopes_cannot_be_widened_by_the_sdk():
    metadata_class = _scope_pinned_metadata_class()
    pinned = metadata_class(
        redirect_uris=["http://127.0.0.1:8765/cb"],
        scope="https://www.googleapis.com/auth/gmail.readonly",
    )
    pinned.scope = (
        "https://mail.google.com/ https://www.googleapis.com/auth/gmail.modify"
    )
    assert pinned.scope == "https://www.googleapis.com/auth/gmail.readonly"

    open_ended = metadata_class(redirect_uris=["http://127.0.0.1:8765/cb"])
    open_ended.scope = "default"
    assert open_ended.scope == "default"


@pytest.mark.unit
def test_issuer_check_tolerates_only_a_bare_origin_slash():
    from mcp.client.auth import oauth2
    from mcp.client.auth.exceptions import OAuthFlowError

    assert _same_origin_issuer(
        "https://accounts.google.com", "https://accounts.google.com/"
    )
    assert not _same_origin_issuer(
        "https://accounts.google.com", "https://evil.example/"
    )
    assert not _same_origin_issuer(
        "https://as.example/tenant", "https://as.example/tenant/"
    )

    _tolerate_origin_issuer_slash()
    _tolerate_origin_issuer_slash()  # idempotent
    metadata = SimpleNamespace(issuer="https://accounts.google.com")
    oauth2.validate_metadata_issuer(metadata, "https://accounts.google.com/")
    with pytest.raises(OAuthFlowError):
        oauth2.validate_metadata_issuer(metadata, "https://other.example/")
