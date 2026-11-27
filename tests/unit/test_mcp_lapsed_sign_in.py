"""A saved sign-in the server refuses is reported as one that needs renewing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from memorizz.mcp import MCPClientManager
from memorizz.mcp.credentials import MCPOAuthTokenStorage

NOTION = {
    "name": "notion",
    "transport": "streamable_http",
    "url": "https://mcp.notion.com/mcp",
    "auth": {
        "type": "oauth",
        "redirect_uri": "http://127.0.0.1:8765/api/mcp/oauth/callback",
        "tokens": {"access_token": "old-access", "token_type": "Bearer"},
    },
}


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    return MCPClientManager(owner_id="agent-1", servers=[dict(NOTION)])


def _status(manager):
    return manager.connection_status("notion")["servers"][0]


@pytest.mark.unit
def test_a_refused_sign_in_is_no_longer_reported_as_signed_in(manager):
    assert _status(manager)["authenticated"] is True
    server = manager.get_server("notion")

    error = manager._safe_error(RuntimeError("401 Unauthorized"), server)

    assert error.code == "authorization_required"
    # The model is told to stop, not left to try the server's other tools.
    assert "stop calling them" in error.message
    assert "sign in to 'notion'" in error.message
    status = _status(manager)
    assert status["authenticated"] is False
    assert status["state"] == "reauth_required"
    # The tokens are kept: signing in again replaces them.
    assert manager._credential_record(server)["oauth_tokens"]


@pytest.mark.unit
def test_new_tokens_or_an_accepted_request_end_a_lapsed_sign_in(manager):
    server = manager.get_server("notion")
    manager._safe_error(RuntimeError("invalid_grant"), server)
    assert _status(manager)["authenticated"] is False

    manager._record_metric("notion", "tools/call", True)
    assert _status(manager)["authenticated"] is True

    manager._safe_error(RuntimeError("invalid_grant"), server)
    assert _status(manager)["authenticated"] is False
    storage = MCPOAuthTokenStorage(manager.credential_store, server.auth.credential_ref)
    tokens = SimpleNamespace(
        model_dump=lambda **_: {"access_token": "new-access", "token_type": "Bearer"}
    )
    asyncio.run(storage.set_tokens(tokens))
    assert _status(manager)["authenticated"] is True
    assert manager._credential_record(server)["oauth_tokens"]["access_token"] == (
        "new-access"
    )


@pytest.mark.unit
def test_other_failures_and_other_sign_ins_are_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    never = {
        **NOTION,
        "auth": {k: v for k, v in NOTION["auth"].items() if k != "tokens"},
    }
    manager = MCPClientManager(owner_id="agent-2", servers=[dict(NOTION)])
    server = manager.get_server("notion")

    assert manager._safe_error(RuntimeError("connection reset"), server).code != (
        "authorization_required"
    )
    assert _status(manager)["authenticated"] is True

    unsigned = MCPClientManager(owner_id="agent-3", servers=[never])
    unsigned._safe_error(RuntimeError("401"), unsigned.get_server("notion"))
    assert "sign_in_lapsed_at" not in unsigned._credential_record(
        unsigned.get_server("notion")
    )


@pytest.mark.unit
def test_the_agent_offers_the_sign_in_when_a_call_is_refused(monkeypatch):
    from memorizz import MemAgent
    from memorizz.memagent import core

    agent = MemAgent(memory_provider=False, auto_register=False)
    try:
        emitted = []
        session = SimpleNamespace(
            emit=lambda kind, **payload: emitted.append((kind, payload))
        )
        monkeypatch.setattr(core, "session_for", lambda _agent: session)
        refused = {
            "ok": False,
            "error_code": "authorization_required",
            "error": "Authorization is required for MCP server 'notion'",
        }
        agent.mcp_manager = SimpleNamespace(call_tool=lambda **_: refused)
        monkeypatch.setattr(
            agent,
            "_capability_state",
            lambda: [
                {
                    "id": "notes",
                    "title": "Notes and docs",
                    "status": "needs_sign_in",
                    "via": "notion",
                },
                {
                    "id": "email",
                    "title": "Email",
                    "status": "needs_sign_in",
                    "via": "gmail",
                },
            ],
        )

        result = agent._invoke_mcp_tool("notion", "notion-fetch", {"id": "x"})

        assert result is refused
        assert emitted == [
            (
                "capability.requested",
                {
                    "capability": "notes",
                    "title": "Notes and docs",
                    "status": "needs_sign_in",
                    "reason": "notion needs signing in again",
                },
            )
        ]
        # A call that worked offers nothing.
        agent.mcp_manager = SimpleNamespace(call_tool=lambda **_: {"ok": True})
        agent._invoke_mcp_tool("notion", "notion-fetch", {"id": "x"})
        assert len(emitted) == 1
    finally:
        agent.close()
