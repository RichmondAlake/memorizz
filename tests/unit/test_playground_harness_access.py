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
    # Without them the panel is there but hidden: it shows when a
    # conversation is answered by a harness.
    plain = playground.client.get("/agents/plain/playground").text
    assert (
        'id="pg-hx" aria-labelledby="pg-hx-title" data-delegates="false" hidden'
        in plain
    )
    assert 'id="pg-hx-chip"' not in plain
    assert 'data-delegates="true"' in page.text


# --- a conversation answered by a harness ---------------------------------


def _harness(name, ready=True, models=("gpt-test",)):
    """Stands in for an adapter: only what the page and the route ask of it."""
    state_of = {
        "name": name,
        "ready": ready,
        "models": list(models),
        "error": None if ready else "not signed in",
        "remediation": None if ready else f"Sign in to {name}.",
    }
    return SimpleNamespace(
        probe=lambda: SimpleNamespace(to_dict=lambda: dict(state_of))
    )


@pytest.fixture()
def answering(playground):
    from memorizz.ui.routers import playground as module

    module._ANSWER_HARNESSES.update(at=0.0, rows=None)
    playground.meta.adapters.update(
        codex=_harness("codex"),
        deepseek=_harness("deepseek", ready=False),
        memagent=_harness("memagent"),
    )
    yield playground
    module._ANSWER_HARNESSES.update(at=0.0, rows=None)


def test_the_page_lists_the_harnesses_that_can_answer(answering):
    listed = answering.client.get("/api/agents/plain/answer-harnesses").json()

    # Ready ones first; the agent's own runtime cannot answer for itself.
    assert [(row["name"], row["ready"]) for row in listed["harnesses"]] == [
        ("codex", True),
        ("deepseek", False),
    ]
    assert listed["harnesses"][0]["models"] == ["gpt-test"]
    assert listed["harnesses"][1]["why"] == "Sign in to deepseek."
    # This agent answers with its own model, so the page keeps that label.
    assert listed["runtime"] is None
    page = answering.client.get("/agents/plain/playground").text
    assert 'id="pg-answer"' in page and "playground-harness.js" in page
    assert (
        answering.client.get("/api/agents/nobody/answer-harnesses").status_code == 404
    )


def test_a_turn_can_be_answered_by_a_harness_without_saving_it(answering):
    agent = _fake_agent([])
    switched = {}
    agent.harness_config = {"model": "the-agents-own-model", "budget": {"max_steps": 5}}
    agent.with_meta_harness = lambda service, **options: switched.update(
        service=service, **options
    )
    kept = []
    with patch(
        "memorizz.memagent.persistence.keep_stored_fields",
        lambda _agent, *fields: kept.extend(fields),
    ):
        kwargs, events = _prepared_run(
            answering.client,
            agent,
            {"answer_harness": "Codex", "harness_workspace": str(answering.workspace)},
        )

    assert switched["service"] is answering.meta
    assert (switched["mode"], switched["default_harness"]) == ("runtime", "codex")
    # The agent's own model name means nothing to a harness; its limits stay.
    assert switched["config"] == {"budget": {"max_steps": 5}}
    # The choice is for this turn: a save during it keeps the stored agent.
    assert {"meta_harness_mode", "default_harness", "harness_config"} <= set(kept)
    assert {"stage": "answered_by", "harness": "codex"}.items() <= next(
        event for event in events if event.get("stage") == "answered_by"
    ).items()
    # The folder reaches the turn although the agent has no delegates.
    assert kwargs["tool_context"]["harness_parent"]["workspace"] == str(
        answering.workspace.resolve()
    )

    agent.with_meta_harness = lambda service, **options: switched.update(options)
    _prepared_run(
        answering.client,
        agent,
        {"answer_harness": "codex", "answer_model": "gpt-other"},
    )
    assert switched["config"]["model"] == "gpt-other"


def test_a_harness_that_is_unknown_or_not_ready_is_refused_before_the_turn(answering):
    client = answering.client

    def ask(name):
        return client.post(
            "/agents/plain/playground/stream",
            data={"query": "Hello", "answer_harness": name},
        )

    assert ask("nope").status_code == 400
    assert ask("memagent").status_code == 400
    refused = ask("deepseek")
    assert (
        refused.status_code == 409
        and "Sign in to deepseek." in refused.json()["detail"]
    )
    with patch.dict(state._state, {"read_only": True}):
        assert ask("codex").status_code == 403


def test_access_can_be_allowed_for_a_conversation_a_harness_answers(answering):
    client = answering.client
    body = {"workspace": str(answering.workspace), "write": True, "approver_id": "ada"}

    # An agent with no harness delegates has nothing to allow access for...
    assert client.post("/api/agents/plain/harness-access", json=body).status_code == 400
    # ...unless a harness answers the conversation.
    allowed = client.post(
        "/api/agents/plain/harness-access", json={**body, "answer_harness": "codex"}
    )
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["grant"]["permissions"]["workspace_mode"] == "direct"
    assert (
        client.post(
            "/api/agents/plain/harness-access",
            json={**body, "answer_harness": "deepseek"},
        ).status_code
        == 409
    )


def test_a_stored_answer_a_harness_gave_is_linked_to_its_run(answering):
    from memorizz.ui.routers.playground import _mark_harness_turns

    def run(run_id, question, memory="chat-1", agent="plain", mode="runtime"):
        return {
            "run_id": run_id,
            "harness": "codex",
            "status": "succeeded",
            "task": {
                "task": question,
                "memory_id": memory,
                "agent_id": agent,
                "mode": mode,
            },
            "result": {"latency_ms": 1200, "cost_usd": 0.02},
        }

    runs = [
        run("r3", "Second question"),
        run("r2", "First question", memory="another-chat"),
        run("r1", "First question"),
        run("r0", "Its own model answered", mode="delegate"),
    ]
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": "Its own model answered"},
        {"role": "assistant", "content": "Two."},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Three."},
    ]
    with patch.object(answering.meta, "list_runs", lambda limit=300: runs):
        _mark_harness_turns(messages, "plain", "chat-1")

    assert messages[1]["harness_run"] == {
        "run_id": "r1", "harness": "codex", "status": "succeeded",
        "latency_ms": 1200, "cost_usd": 0.02,
    }  # fmt: skip
    assert "harness_run" not in messages[3]
    assert messages[5]["harness_run"]["run_id"] == "r3"


def test_a_harness_turn_carries_on_after_its_page_leaves(
    answering, tmp_path, monkeypatch
):
    import time

    from memorizz import live_turns
    from memorizz.streaming import StreamEvent
    from memorizz.ui.routers import playground as module

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))

    def event(kind, **fields):
        return StreamEvent({"type": kind, "run_id": "turn-1", "seq": 0, **fields})

    class Stream:
        """A turn that still has its ending to come when the page leaves."""

        def __init__(self):
            self.left = [
                event(
                    "status",
                    message="Codex: writing",
                    harness_run_id="run-9",
                    harness="codex",
                ),
                event("answer.delta", delta="Done."),
                event("run.done", status="completed"),
            ]
            self.closed = False

        def poll(self, timeout=None):
            if not self.left:
                raise StopIteration
            return self.left.pop(0)

        def close(self):
            self.closed = True

    stream = Stream()
    # A page already listening; one opened later gets a finished turn from
    # the stored conversation instead.
    tail = live_turns.LiveTail("plain", "chat-1")
    assert tail.poll() == []
    module._follow_detached(
        stream,
        "plain",
        "Write the note",
        event("run.started", memory_id="chat-1", thread_id="t-1"),
        event(
            "status",
            message="Codex is working",
            harness_run_id="run-9",
            harness="codex",
        ),
        set(),
    )
    for _ in range(100):
        if stream.closed:
            break
        time.sleep(0.05)
    assert stream.closed  # the turn was followed to its end, then released

    seen = tail.poll()
    assert [item["type"] for item in seen] == [
        "turn.started", "status", "status", "answer.delta", "turn.done",
    ]  # fmt: skip
    # The page that finds it knows it is its own, and which run to show.
    assert seen[0]["source"] == "playground" and seen[0]["query"] == "Write the note"
    assert seen[1]["harness_run_id"] == "run-9" and seen[1]["harness"] == "codex"
    assert seen[-1]["status"] == "completed"
