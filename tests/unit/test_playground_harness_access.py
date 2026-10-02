"""Harness delegates in the playground: the access and folder they may use."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.approval import SQLiteApprovalStore  # noqa: E402
from memorizz.memagent.models import MemAgentModel  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.metaharness import MetaHarness, SQLiteHarnessRunStore  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture()
def playground(tmp_path: Path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
        scratch_root=tmp_path / "scratch",
    )
    llm = {"provider": "ollama", "model": "qwen2.5:7b"}
    provider.store_memagent(
        MemAgentModel(
            agent_id="codex-researcher",
            name="Codex researcher",
            llm_config=llm,
            meta_harness=True,
            meta_harness_mode="runtime",
            default_harness="codex",
        )
    )
    provider.store_memagent(
        MemAgentModel(agent_id="note-taker", name="Note taker", llm_config=llm)
    )
    provider.store_memagent(
        MemAgentModel(
            agent_id="desk",
            name="Research desk",
            llm_config=llm,
            delegates=["codex-researcher", "note-taker"],
            delegation_config={"enabled": True, "mode": "auto"},
        )
    )
    provider.store_memagent(
        MemAgentModel(
            agent_id="plain", name="Plain", llm_config=llm, delegates=["note-taker"]
        )
    )
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "meta_harness": meta,
        "meta_harness_provider": provider,
        "read_only": False,
    }
    workspace = tmp_path / "project"
    workspace.mkdir()
    try:
        with patch.dict(state._state, values):
            yield SimpleNamespace(
                client=TestClient(create_app(), follow_redirects=False),
                meta=meta,
                workspace=workspace,
                tmp_path=tmp_path,
            )
    finally:
        meta.close()
        provider.close()


def test_a_grant_needs_a_name_for_web_access_or_edits(playground):
    client = playground.client
    url = "/api/agents/desk/harness-access"

    for body in ({"network": "full"}, {"write": True}):
        refused = client.post(url, json=body)
        assert refused.status_code == 400
        assert "Give your name" in refused.json()["detail"]
    assert client.post(url, json={"network": "sometimes"}).status_code == 400
    assert client.post(url, json={"write": "yes"}).status_code == 400

    granted = client.post(
        url,
        json={
            "workspace": str(playground.workspace),
            "network": "full",
            "approver_id": "Ada",
        },
    )
    assert granted.status_code == 200, granted.text
    grant = granted.json()["grant"]
    assert grant["workspace"] == str(playground.workspace.resolve())
    assert grant["permissions"]["network"] == "full"
    assert grant["permissions"]["workspace_mode"] == "read_only"
    assert grant["approver_id"] == "Ada"

    # Nothing risky: no name needed, and a blank folder is a fresh one.
    quiet = client.post(url, json={}).json()["grant"]
    assert quiet["approver_id"] == "playground"
    assert playground.meta.is_scratch_workspace(quiet["workspace"])

    outside = client.post(url, json={"workspace": str(Path.home())})
    assert outside.status_code == 400
    assert "outside the configured allowed roots" in outside.json()["detail"]
    assert client.post("/api/agents/plain/harness-access", json={}).status_code == 400
    assert client.post("/api/agents/missing/harness-access", json={}).status_code == 404


def test_a_grant_can_be_checked_until_it_expires(playground):
    client = playground.client
    grant = client.post(
        "/api/agents/desk/harness-access",
        json={"network": "full", "approver_id": "Ada"},
    ).json()["grant"]
    url = f"/api/agents/desk/harness-access/{grant['grant_id']}"

    assert client.get(url).json()["grant"]["grant_id"] == grant["grant_id"]
    # Another agent can't use it.
    assert (
        client.get(f"/api/agents/plain/harness-access/{grant['grant_id']}").status_code
        == 404
    )
    later = datetime.now(timezone.utc) + timedelta(days=2)
    with patch("memorizz.approval._utcnow", return_value=later):
        assert client.get(url).status_code == 404


def test_each_conversation_can_get_its_own_folder(playground):
    first = playground.client.post("/api/agents/desk/harness-workspace").json()[
        "workspace"
    ]
    second = playground.client.post("/api/agents/desk/harness-workspace").json()[
        "workspace"
    ]
    assert first != second
    assert playground.meta.is_scratch_workspace(first)
    assert Path(first).is_dir()


def _fake_agent(delegates):
    return SimpleNamespace(
        delegates=delegates,
        delegation_config={"enabled": True, "mode": "auto"},
        has_sandbox=lambda: True,
        has_browser_control=lambda: True,
        has_internet_access=lambda: True,
        tool_manager=None,
        memory_ids=[],
    )


def _prepared_run(client, agent, form):
    """Run the stream route's preparation and return its run kwargs and the
    status events it emitted."""
    captured = {}

    class _Stream:
        def poll(self, timeout=None):
            raise StopIteration

        def close(self):
            pass

    def fake_stream(_agent, _query, *, _prepare=None, **kwargs):
        events = []
        with ExitStack() as stack:
            session = SimpleNamespace(
                stack=stack,
                emit=lambda kind, **payload: events.append({"type": kind, **payload}),
                persistence={},
            )
            _, run_kwargs = _prepare(session)
        captured.update(kwargs=run_kwargs, events=events)
        return _Stream()

    with patch("memorizz.streaming.agent_event_stream", fake_stream), patch(
        "memorizz.memagent.MemAgent.load", return_value=agent
    ):
        response = client.post(
            "/agents/desk/playground/stream", data={"query": "Compare", **form}
        )
    assert response.status_code == 200, response.text
    return captured["kwargs"], captured["events"]


def test_a_message_gives_harness_delegates_the_grant_or_the_folder(playground):
    client = playground.client
    delegate = SimpleNamespace(
        agent_id="codex-researcher",
        name="Codex researcher",
        meta_harness_mode="runtime",
        default_harness="codex",
    )
    agent = _fake_agent([delegate])
    grant = client.post(
        "/api/agents/desk/harness-access",
        json={
            "workspace": str(playground.workspace),
            "network": "full",
            "approver_id": "Ada",
        },
    ).json()["grant"]

    kwargs, events = _prepared_run(client, agent, {"harness_grant": grant["grant_id"]})
    parent = kwargs["tool_context"]["harness_parent"]
    assert parent["grant_id"] == grant["grant_id"]
    assert parent["permissions"]["network"] == "full"
    assert not [e for e in events if e.get("scope") == "harness_access"]

    folder = client.post("/api/agents/desk/harness-workspace").json()["workspace"]
    kwargs, _ = _prepared_run(client, agent, {"harness_workspace": folder})
    assert kwargs["tool_context"]["harness_parent"] == {
        "workspace": str(Path(folder).resolve())
    }

    # An expired grant is reported and the conversation's folder used instead.
    later = datetime.now(timezone.utc) + timedelta(days=2)
    with patch("memorizz.approval._utcnow", return_value=later):
        kwargs, events = _prepared_run(
            client,
            agent,
            {"harness_grant": grant["grant_id"], "harness_workspace": folder},
        )
    assert kwargs["tool_context"]["harness_parent"] == {
        "workspace": str(Path(folder).resolve())
    }
    warning = [e for e in events if e.get("scope") == "harness_access"][0]
    assert warning["grant_expired"] is True and "expired" in warning["message"]

    # A folder outside the allowed roots isn't passed on.
    kwargs, events = _prepared_run(
        client, agent, {"harness_workspace": str(Path.home())}
    )
    assert "tool_context" not in kwargs
    assert [e for e in events if e.get("scope") == "harness_access"]

    # Agents without harness delegates get nothing extra.
    kwargs, _ = _prepared_run(
        client, _fake_agent([]), {"harness_grant": grant["grant_id"]}
    )
    assert "tool_context" not in kwargs


def test_the_page_offers_the_control_only_with_harness_delegates(playground):
    page = playground.client.get("/agents/desk/playground")
    assert page.status_code == 200, page.text[:500]
    assert 'id="pg-hx"' in page.text and 'id="pg-hx-chip"' in page.text
    assert "Codex researcher (codex)" in page.text
    assert "Note taker (" not in page.text  # runs in process, not on a harness
    plain = playground.client.get("/agents/plain/playground").text
    assert 'id="pg-hx"' not in plain and 'id="pg-hx-chip"' not in plain
