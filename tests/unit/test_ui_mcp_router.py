"""Local UI coverage for MCP connection management routes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.approval import SQLiteApprovalStore  # noqa: E402
from memorizz.mcp.oauth import PendingOAuthFlow, oauth_flows  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402


class _Provider:
    def __init__(self):
        self.agent = SimpleNamespace(
            agent_id="agent-1",
            name="MCP Agent",
            mcp_servers=[],
        )

    def list_memagents(self):
        return [self.agent]

    def retrieve_memagent(self, agent_id):
        return self.agent if agent_id == self.agent.agent_id else None

    def store_memagent(self, agent):
        self.agent = agent
        return agent.agent_id


@pytest.mark.unit
def test_mcp_page_and_server_crud(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _Provider()
    client = TestClient(create_app(), follow_redirects=False)
    with monkeypatch.context() as patcher:
        patcher.setitem(state._state, "provider", provider)
        patcher.setitem(state._state, "provider_type", "filesystem")
        patcher.setitem(state._state, "connection_info", {})

        page = client.get("/mcp?agent_id=agent-1")
        assert page.status_code == 200
        assert "Connect Notion" in page.text
        assert "Google Calendar" in page.text
        assert "Expose MemoRizz as an MCP server" in page.text
        assert '"args": ["mcp", "serve"]' in page.text

        added = client.post(
            "/api/mcp/agents/agent-1/servers",
            json={
                "name": "notion",
                "transport": "streamable_http",
                "url": "https://mcp.notion.com/mcp",
                "auth": {
                    "type": "bearer",
                    "token": "notion-secret",
                },
            },
        )
        assert added.status_code == 200, added.text
        assert "notion-secret" not in str(provider.agent.mcp_servers)
        assert provider.agent.mcp_servers[0]["name"] == "notion"

        listed = client.get("/api/mcp/agents/agent-1/servers")
        assert listed.status_code == 200
        assert listed.json()["status"][0]["authenticated"] is True

        removed = client.delete("/api/mcp/agents/agent-1/servers/notion")
        assert removed.status_code == 200
        assert provider.agent.mcp_servers == []


@pytest.mark.unit
def test_oauth_callback_reports_exchange_result_and_rejects_replay():
    client = TestClient(create_app(), follow_redirects=False)
    flow = PendingOAuthFlow(
        owner_id="agent-1", server_name="notion", state="ui-state-1"
    )
    flow.result = {"ok": True}
    flow.completed.set()
    oauth_flows.add(flow)

    response = client.get(
        "/api/mcp/oauth/callback?state=ui-state-1&code=authorization-code"
    )

    assert response.status_code == 302
    assert "oauth=success" in response.headers["location"]
    replay = client.get("/api/mcp/oauth/callback?state=ui-state-1&code=replayed-code")
    assert replay.status_code == 400


@pytest.mark.unit
def test_generic_approval_ui_can_decide_and_resume_exact_checkpoint(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _Provider()
    store = SQLiteApprovalStore(tmp_path / "ui-approvals.sqlite3")
    proposal = store.propose(
        owner_id="agent-1",
        tool_name="calendar_create_event",
        arguments={"title": "Launch", "hour": 9},
        policy_reason="Creates an external calendar event",
        checkpoint={"kind": "agent_tool_call", "thread_id": "thread-1"},
    )

    class RuntimeAgent:
        def list_approval_proposals(self, status=None, limit=100):
            return [
                item.to_dict(include_arguments=True)
                for item in store.list(owner_id="agent-1", status=status, limit=limit)
            ]

        def approve(self, proposal_id, *, approver_id, reason=None):
            return store.approve(
                proposal_id,
                approver_id=approver_id,
                decision_reason=reason,
            ).to_dict(include_arguments=True)

        def reject(self, proposal_id, *, approver_id, reason=None):
            return store.reject(
                proposal_id,
                approver_id=approver_id,
                decision_reason=reason,
            ).to_dict(include_arguments=True)

        def cancel_approval(self, proposal_id, *, approver_id, reason=None):
            return self.reject(
                proposal_id,
                approver_id=approver_id,
                reason=reason or "Cancelled by host",
            )

        def resume_approval(self, proposal_id):
            stored = store.get(proposal_id)
            store.consume(
                proposal_id,
                expected_tool_name=stored.tool_name,
                expected_arguments=stored.arguments,
            )
            return "calendar event created"

    from memorizz.memagent import MemAgent

    monkeypatch.setattr(
        MemAgent,
        "load",
        classmethod(
            lambda cls, agent_id, memory_provider=None, **kwargs: RuntimeAgent()
        ),
    )
    client = TestClient(create_app())
    with monkeypatch.context() as patcher:
        patcher.setitem(state._state, "provider", provider)
        patcher.setitem(state._state, "provider_type", "filesystem")
        patcher.setitem(state._state, "connection_info", {})

        pending = client.get("/api/agents/agent-1/approvals?status=pending")
        assert pending.status_code == 200
        assert pending.json()["approvals"][0]["argument_hash"] == proposal.argument_hash
        assert pending.json()["approvals"][0]["thread_id"] == "thread-1"

        approved = client.post(
            f"/api/agents/agent-1/approvals/{proposal.proposal_id}/approve",
            json={"approver_id": "operator@example.com"},
        )
        assert approved.status_code == 200
        assert approved.json()["proposal"]["approver_id"] == "operator@example.com"

        resumed = client.post(
            f"/api/agents/agent-1/approvals/{proposal.proposal_id}/resume"
        )
        assert resumed.status_code == 200
        assert resumed.json()["result"] == "calendar event created"
        assert store.get(proposal.proposal_id).status.value == "consumed"
