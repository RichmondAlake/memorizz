"""Settings → Connected apps: list, approve (pairing code) and disconnect the
cloud apps that use this memory through the MCP server."""

import re
import time
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.cli import mcp_apps  # noqa: E402
from memorizz.mcp_server.config import LOCAL_AGENTS_PRINCIPAL  # noqa: E402
from memorizz.mcp_server.config import READ_SCOPE, WRITE_SCOPE
from memorizz.mcp_server.oauth_store import OAuthStore  # noqa: E402
from memorizz.memory_provider import FileSystemConfig  # noqa: E402
from memorizz.memory_provider import FileSystemProvider
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "false")
    for key in ("MEMORIZZ_UI_AUTH_TOKEN", "MEMORIZZ_UI_AUTH_ACCOUNTS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(mcp_apps, "default_principal", lambda: "ada")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def provider(home):
    return FileSystemProvider(
        FileSystemConfig(root_path=home / "store", lazy_vector_indexes=True)
    )


def _connected(provider, **extra):
    return patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"path": "/srv/store"},
            "provider_secrets": {},
            "embedding_check": None,
            "read_only": False,
            **extra,
        },
    )


def _client():
    return TestClient(create_app(), follow_redirects=False)


def test_list_is_empty_then_shows_each_app(home):
    client = _client()
    response = client.get("/api/connected-apps")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "apps": []}

    store = OAuthStore()
    grok = store.create_grant("client-grok", "Grok", "ada", [READ_SCOPE, WRITE_SCOPE])
    store.create_grant("client-dot", "ChatGPT", "ada", [READ_SCOPE])
    store.touch_grant(grok)

    response = client.get("/api/connected-apps")
    assert response.headers["cache-control"] == "no-store"
    apps = {app["app_name"]: app for app in response.json()["apps"]}
    assert apps["Grok"]["grant_id"] == grok
    assert apps["Grok"]["principal"] == "ada"
    assert apps["Grok"]["can"] == "Read and save"
    assert apps["Grok"]["read_only"] is False
    assert apps["Grok"]["last_used"] == "just now"
    assert apps["ChatGPT"]["can"] == "Read only"
    assert apps["ChatGPT"]["last_used"] == "Never"
    assert apps["ChatGPT"]["last_used_at"] is None
    assert "hash" not in response.text


def test_pairing_code_connects_one_app_to_read_and_save(home):
    response = _client().post("/api/connected-apps/pairing-code", json={})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert re.fullmatch(r"[A-Z2-9]{4}-[A-Z2-9]{4}", body["code"])
    assert body["principal"] == "ada"
    assert body["read_only"] is False
    assert 590 < body["expires_at"] - time.time() <= 600
    assert set(body) == {
        "ok",
        "code",
        "principal",
        "expires_at",
        "read_only",
        "shared",
    }

    store = OAuthStore()
    assert store.redeem_pairing_code(body["code"]) == {
        "principal": "ada",
        "scopes": [READ_SCOPE, WRITE_SCOPE],
    }
    # Single use.
    assert store.redeem_pairing_code(body["code"]) is None


def test_read_only_pairing_code_lets_the_app_read_only(home):
    client = _client()
    body = client.post(
        "/api/connected-apps/pairing-code", json={"read_only": True}
    ).json()
    assert body["read_only"] is True
    assert OAuthStore().redeem_pairing_code(body["code"])["scopes"] == [READ_SCOPE]
    # No body at all is the default code; a non-boolean is refused.
    assert client.post("/api/connected-apps/pairing-code").status_code == 200
    refused = client.post("/api/connected-apps/pairing-code", json={"read_only": "yes"})
    assert refused.status_code == 400


def test_disconnect_removes_the_app_and_its_access(home):
    store = OAuthStore()
    grant_id = store.create_grant("client-grok", "Grok", "ada", [READ_SCOPE])
    store.save_token(
        "token-1", "access", grant_id, "client-grok", [READ_SCOPE], None, None
    )
    client = _client()

    response = client.post(f"/api/connected-apps/{grant_id}/disconnect")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "app_name": "Grok"}
    assert client.get("/api/connected-apps").json()["apps"] == []
    assert store.grant(grant_id)["revoked_at"] is not None
    assert store.token("token-1", "access") is None

    # Already disconnected, or never connected.
    assert client.post(f"/api/connected-apps/{grant_id}/disconnect").status_code == 404
    assert client.post("/api/connected-apps/app_missing/disconnect").status_code == 404


def test_read_only_ui_refuses_both_changes(home, provider, monkeypatch):
    store = OAuthStore()
    grant_id = store.create_grant("client-grok", "Grok", "ada", [READ_SCOPE])
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    client = _client()
    with _connected(provider, read_only=True):
        assert (
            client.post("/api/connected-apps/pairing-code", json={}).status_code == 403
        )
        assert (
            client.post(f"/api/connected-apps/{grant_id}/disconnect").status_code == 403
        )
        assert len(client.get("/api/connected-apps").json()["apps"]) == 1
        page = client.get("/settings").text
    assert store.grant(grant_id)["revoked_at"] is None
    section = page[page.index('id="connected-apps"') :]
    assert "Grok" in section
    assert 'id="apps-pair"' not in section
    assert "data-disconnect" not in section
    assert "This UI is read-only, so apps can't be approved" in section


def test_read_only_connection_also_refuses_changes(home, provider):
    # The process is writable, but the page shows a read-only connection.
    client = _client()
    with _connected(provider, read_only=True):
        response = client.post("/api/connected-apps/pairing-code", json={})
    assert response.status_code == 403
    assert "read-only" in response.json()["detail"]


def test_cross_site_changes_are_refused(home):
    grant_id = OAuthStore().create_grant("client-grok", "Grok", "ada", [READ_SCOPE])
    client = _client()
    evil = {"Origin": "https://evil.example"}
    assert (
        client.post(
            "/api/connected-apps/pairing-code", json={}, headers=evil
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/connected-apps/{grant_id}/disconnect", headers=evil
        ).status_code
        == 403
    )
    assert OAuthStore().grant(grant_id)["revoked_at"] is None


def test_settings_page_lists_apps_with_their_names_escaped(home, provider):
    client = _client()
    with _connected(provider):
        empty = client.get("/settings").text
    assert 'href="#connected-apps"' in empty
    assert 'id="connected-apps"' in empty
    assert "No apps are connected yet" in empty
    assert 'id="apps-empty" hidden' not in empty
    assert "memorizz mcp apps serve --tunnel" in empty
    assert "grok.com/connectors" in empty
    assert "Add custom MCP server (Developer mode)" in empty

    OAuthStore().create_grant(
        "client-x", "<img src=x onerror=alert(1)>", "ada", [READ_SCOPE, WRITE_SCOPE]
    )
    with _connected(provider):
        page = client.get("/settings").text
    assert "<img src=x onerror=alert(1)>" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "Read and save" in page
    assert 'id="apps-empty" hidden' in page
    assert 'id="apps-pair"' in page and "data-mutates" in page
    assert "1 connected" in page


def test_a_code_can_share_the_local_agents_memory_when_asked(home):
    client = _client()
    plain = client.post("/api/connected-apps/pairing-code", json={}).json()
    assert plain["shared"] is False
    assert OAuthStore().redeem_pairing_code(plain["code"])["principal"] == "ada"

    shared = client.post(
        "/api/connected-apps/pairing-code", json={"shared": True}
    ).json()
    assert shared["shared"] is True
    redeemed = OAuthStore().redeem_pairing_code(shared["code"])
    assert redeemed["principal"] == LOCAL_AGENTS_PRINCIPAL

    refused = client.post("/api/connected-apps/pairing-code", json={"shared": "yes"})
    assert refused.status_code == 400


def test_the_list_says_when_an_app_shares_the_local_agents_memory(home, provider):
    OAuthStore().create_grant(
        "client-grok", "Grok Bot", LOCAL_AGENTS_PRINCIPAL, [READ_SCOPE, WRITE_SCOPE]
    )
    client = _client()
    app = client.get("/api/connected-apps").json()["apps"][0]
    assert app["shared"] is True
    assert app["memory_of"] == "Your local agents"
    with _connected(provider):
        page = client.get("/settings").text
    assert "Your local agents" in page
    assert LOCAL_AGENTS_PRINCIPAL not in page
    assert 'id="apps-shared"' in page
