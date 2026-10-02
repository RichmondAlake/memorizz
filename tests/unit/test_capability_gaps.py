"""Detecting what an agent can't do yet, and the card that offers to enable it."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memorizz import capability_gaps
from memorizz.mcp import catalog

pytestmark = pytest.mark.unit

GMAIL = {
    "name": "gmail",
    "transport": "streamable_http",
    "url": catalog.GMAIL_URL,
    "auth": {"type": "oauth"},
}


def _row(state, capability_id):
    return next(row for row in state if row["id"] == capability_id)


def test_state_reflects_configuration_and_environment():
    bare = capability_gaps.capability_state(environ={"TAVILY_API_KEY": "tvly-x"})
    web = _row(bare, "web_search")
    assert web["status"] == "missing"
    providers = {
        offer["provider"]: offer
        for offer in web["offers"]
        if offer["kind"] == "internet_provider"
    }
    assert providers["tavily"]["key_set"] is True
    assert providers["firecrawl"]["key_set"] is False
    assert any(offer["kind"] == "mcp_search" for offer in web["offers"])
    assert _row(bare, "email")["offers"][0] == {
        "kind": "mcp_preset",
        "preset": "gmail",
        "title": "Connect Gmail",
        "needs_client": True,
        "docs": catalog.PRESETS["gmail"]["docs"],
    }

    configured = capability_gaps.capability_state(
        internet_provider="tavily",
        mcp_servers=[GMAIL, {"name": "notion", "url": catalog.NOTION_URL}],
        signed_in={"notion": True},
        sandbox_provider="e2b",
        environ={},
    )
    assert _row(configured, "web_search")["status"] == "enabled"
    assert _row(configured, "notes")["status"] == "enabled"
    assert _row(configured, "code_execution")["via"] == "e2b"
    email = _row(configured, "email")
    assert email["status"] == "needs_sign_in"
    assert email["offers"] == [
        {
            "kind": "mcp_sign_in",
            "server": "gmail",
            "title": "Sign in to gmail",
            "needs_client": True,
        }
    ]
    # An offline provider does not count as web search.
    offline = capability_gaps.capability_state(internet_provider="offline", environ={})
    assert _row(offline, "web_search")["status"] == "missing"


def test_suggestions_only_name_missing_capabilities_the_request_needs():
    state = capability_gaps.capability_state(environ={})
    assert capability_gaps.suggest("can you search the internet", state) == [
        "web_search"
    ]
    assert capability_gaps.suggest("What's on my calendar tomorrow?", state) == [
        "calendar"
    ]
    assert capability_gaps.suggest("summarise https://example.com/a", state) == [
        "web_search"
    ]
    assert capability_gaps.suggest("hi there", state) == []

    searching = capability_gaps.capability_state(internet_provider="tavily", environ={})
    assert capability_gaps.suggest("search the web for news", searching) == []


def test_manifest_lists_only_what_is_missing():
    state = capability_gaps.capability_state(
        internet_provider="tavily", mcp_servers=[GMAIL], environ={}
    )
    note = capability_gaps.manifest(state)
    assert "Web search" not in note
    assert "Email: Read and draft email (connected but not signed in)" in note
    assert "request_capability" in note
    everything = [dict(row, status="enabled") for row in state]
    assert capability_gaps.manifest(everything) == ""


def test_agent_asks_for_a_capability_instead_of_improvising(monkeypatch):
    from memorizz import MemAgent
    from memorizz.memagent import core

    agent = MemAgent(memory_provider=False, auto_register=False)
    try:
        assert "request_capability" in agent.tool_manager.list_tools()
        assert "request_capability" in agent.semantic_tool_router.always_visible
        prompt = agent._build_system_prompt()
        assert "Capabilities not enabled for this agent" in prompt
        assert "- Web search: Search the web and read pages" in prompt

        emitted = []
        session = SimpleNamespace(
            emit=lambda kind, **payload: emitted.append((kind, payload))
        )
        monkeypatch.setattr(core, "session_for", lambda _agent: session)
        result, _ = agent.tool_manager.execute_tool(
            "request_capability", {"capability": "web search", "reason": "news"}
        )
        assert result["ok"] is True and result["capability"] == "web_search"
        assert "enable it from the app" in result["message"]
        assert emitted == [
            (
                "capability.requested",
                {
                    "capability": "web_search",
                    "title": "Web search",
                    "status": "missing",
                    "reason": "news",
                },
            )
        ]
        unknown, _ = agent.tool_manager.execute_tool(
            "request_capability", {"capability": "teleport"}
        )
        assert unknown["ok"] is False and "web_search" in unknown["error"]

        agent._suggest_capabilities(session, "please search the internet for it")
        assert emitted[-1] == ("capability.suggested", {"capabilities": ["web_search"]})
    finally:
        agent.close()


class _FakeModel:
    model = "fake"

    def get_config(self):
        return {"provider": "openai", "model": "fake"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None

    def generate_stream(self, messages, tools=None, **kwargs):
        yield {"type": "content", "content": "No."}
        yield {"type": "done", "content": "No."}


def test_stream_reports_a_needed_capability_before_the_model_answers():
    from memorizz import MemAgent

    agent = MemAgent(model=_FakeModel(), memory_provider=False, auto_register=False)
    try:
        with agent.run_stream_events("can you search the internet for me") as events:
            kinds = [(event["type"], event.get("capabilities")) for event in events]
        suggested = [caps for kind, caps in kinds if kind == "capability.suggested"]
        assert suggested == [["web_search"]]
        first_model = (
            next(
                index for index, (kind, _) in enumerate(kinds) if kind == "answer.delta"
            )
            if any(kind == "answer.delta" for kind, _ in kinds)
            else len(kinds)
        )
        assert kinds.index(("capability.suggested", ["web_search"])) < first_model
    finally:
        agent.close()


# --------------------------------------------------------------------- UI


pytest.importorskip("fastapi")


class _Provider:
    def __init__(self):
        self.agent = SimpleNamespace(
            agent_id="agent-1",
            name="Assistant",
            persona=None,
            mcp_servers=[],
            internet_access_provider=None,
            internet_access_config=None,
            sandbox_provider=None,
            browser_control=None,
        )

    def list_memagents(self):
        return [self.agent]

    def retrieve_memagent(self, agent_id):
        return self.agent if agent_id == "agent-1" else None

    def store_memagent(self, agent):
        self.agent = agent
        return agent.agent_id


@pytest.fixture
def ui(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _Provider()
    client = TestClient(create_app(), follow_redirects=False)
    # create_app loads layered .env files, which may define a real key.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with monkeypatch.context() as patcher:
        patcher.setitem(state._state, "provider", provider)
        patcher.setitem(state._state, "provider_type", "filesystem")
        patcher.setitem(state._state, "connection_info", {})
        yield client, provider


def test_capability_api_reads_live_state_and_enables_offers(ui, monkeypatch):
    client, provider = ui
    listed = client.get("/api/agents/agent-1/capabilities?ids=web_search,email").json()
    assert [row["id"] for row in listed["capabilities"]] == ["web_search", "email"]
    tavily = listed["capabilities"][0]["offers"][0]
    assert tavily["provider"] == "tavily" and tavily["key_set"] is False

    saved = {}

    def fake_apply(updates):
        saved.update(updates)
        monkeypatch.setenv("TAVILY_API_KEY", updates["TAVILY_API_KEY"])

    monkeypatch.setattr("memorizz._env_io.apply_env_updates", fake_apply)
    monkeypatch.setattr(
        "memorizz.ui.routers.capabilities._validate_internet_provider_choice",
        lambda provider_name, config: None,
    )
    enabled = client.post(
        "/api/agents/agent-1/capabilities/web_search/enable",
        json={
            "kind": "internet_provider",
            "provider": "tavily",
            "api_key": "tvly-secret",
        },
    )
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["capability"]["status"] == "enabled"
    assert provider.agent.internet_access_provider == "tavily"
    assert saved == {"TAVILY_API_KEY": "tvly-secret"}
    assert "tvly-secret" not in json.dumps(provider.agent.internet_access_config or {})

    gmail = client.post(
        "/api/agents/agent-1/capabilities/email/enable", json={"kind": "mcp_preset"}
    )
    assert gmail.status_code == 200, gmail.text
    assert gmail.json()["capability"]["status"] == "needs_sign_in"
    assert provider.agent.mcp_servers[0]["name"] == "gmail"

    bad = client.post(
        "/api/agents/agent-1/capabilities/web_search/enable", json={"kind": "setting"}
    )
    assert bad.status_code == 400


def test_sign_in_returns_only_to_same_site_paths():
    from memorizz.ui.routers.mcp import _local_path

    assert _local_path("/agents/a/playground") == "/agents/a/playground"
    for unsafe in (
        "//evil.example/x",
        "https://evil.example",
        "/\\evil",
        "agents",
        "/a\r\nb",
    ):
        assert _local_path(unsafe) is None


def test_oauth_callback_redirects_to_the_card_that_started_it():
    from fastapi.testclient import TestClient

    from memorizz.mcp.oauth import PendingOAuthFlow, oauth_flows
    from memorizz.ui.app import create_app

    flow = PendingOAuthFlow(
        owner_id="agent-1",
        server_name="gmail",
        state="card-state-1",
        return_to="/agents/agent-1/playground",
    )
    flow.result = {"ok": True}
    flow.completed.set()
    oauth_flows.add(flow)
    response = TestClient(create_app(), follow_redirects=False).get(
        "/api/mcp/oauth/callback?state=card-state-1&code=abc"
    )
    assert response.status_code == 302
    assert response.headers["location"] == (
        "/agents/agent-1/playground?oauth=success&server=gmail"
    )


def test_context_panel_uses_the_agents_real_window(monkeypatch):
    from memorizz.memagent.models import MemAgentModel
    from memorizz.ui.routers.playground import _agent_context_window

    pinned = MemAgentModel(
        instruction="x",
        llm_config={"provider": "ollama", "model": "m", "context_window_tokens": 12288},
    )
    assert _agent_context_window(pinned) == 12288
    live = SimpleNamespace(_context_window_tokens=16384)
    assert _agent_context_window(live) == 16384
    assert _agent_context_window(SimpleNamespace()) == 128000
