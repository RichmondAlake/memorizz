"""CLI turns written as they stream and followed by the UI's playground."""

import json
import os
import time

import pytest

from memorizz import live_turns
from memorizz.live_turns import LiveTail, LiveTurn, agent_dir

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    # Every delta is written at once.
    monkeypatch.setattr(live_turns, "DELTA_SECONDS", 0.0)
    return tmp_path


def _run(turn, memory_id="m1", deltas=("Hel", "lo"), status="completed"):
    turn.event({"type": "run.started", "memory_id": memory_id, "thread_id": "t1"})
    turn.event({"type": "status", "stage": "retrieving_memory"})
    turn.event({"type": "tool.started", "tool_name": "lookup", "span_id": "s1"})
    turn.event(
        {
            "type": "tool.completed",
            "tool_name": "lookup",
            "span_id": "s1",
            "status": "ok",
        }
    )
    for delta in deltas:
        turn.event({"type": "answer.delta", "delta": delta})
    if status:
        turn.event({"type": "run.done", "status": status})


def test_a_turn_is_written_privately_and_read_back_in_order(home):
    tail = LiveTail("agent-1", "m1")
    assert tail.poll() == []  # nothing yet

    turn = LiveTurn("agent-1", "how can you help me?")
    _run(turn)
    events = tail.poll()
    assert [e["type"] for e in events] == [
        "turn.started",
        "status",
        "tool.started",
        "tool.completed",
        "answer.delta",
        "answer.delta",
        "turn.done",
    ]
    assert events[0]["query"] == "how can you help me?"
    assert events[0]["source"] == "cli"
    assert {e["turn_id"] for e in events} == {turn.turn_id}
    assert events[1]["message"] == "retrieving memory"
    assert "".join(e["delta"] for e in events if e["type"] == "answer.delta") == "Hello"
    assert events[-1]["status"] == "completed"
    assert tail.poll() == []  # only new lines afterwards

    journal = next(agent_dir("agent-1").glob("*.jsonl"))
    assert journal.stat().st_mode & 0o777 == 0o600
    assert agent_dir("agent-1").stat().st_mode & 0o777 == 0o700


def test_a_page_opened_mid_turn_catches_up_and_skips_finished_turns():
    _run(LiveTurn("agent-1", "finished earlier"))
    running = LiveTurn("agent-1", "still going")
    _run(running, deltas=("par",), status=None)

    tail = LiveTail("agent-1")
    events = tail.poll()
    assert {e["turn_id"] for e in events} == {running.turn_id}
    assert events[0]["query"] == "still going"

    running.event({"type": "answer.delta", "delta": "tial"})
    running.close()
    later = tail.poll()
    assert [e["type"] for e in later] == ["answer.delta", "turn.done"]
    assert later[-1]["status"] == "interrupted"


def test_the_tail_keeps_one_conversation_and_held_text_is_flushed(monkeypatch):
    monkeypatch.setattr(live_turns, "DELTA_SECONDS", 3600.0)
    other = LiveTurn("agent-1", "elsewhere")
    _run(other, memory_id="m2")
    turn = LiveTurn("agent-1", "here")
    turn.event({"type": "run.started", "memory_id": "m1"})
    turn.event({"type": "answer.delta", "delta": "a"})  # first delta: written
    turn.event({"type": "answer.delta", "delta": "b"})  # held back
    turn.event({"type": "answer.delta", "delta": "c"})
    tail = LiveTail("agent-1", "m1")
    assert [e.get("delta") for e in tail.poll()] == [None, "a"]
    turn.event({"type": "run.done", "status": "completed"})
    assert [(e["type"], e.get("delta")) for e in tail.poll()] == [
        ("answer.delta", "bc"),
        ("turn.done", None),
    ]


def test_nothing_is_written_without_a_conversation_or_a_safe_agent_id(home):
    LiveTurn("agent-1", "q").event({"type": "run.started"})
    for unsafe in ("../escape", "a/b", "", None):
        turn = LiveTurn(unsafe, "q")
        turn.event({"type": "run.started", "memory_id": "m1"})
        turn.close()
    assert not (home / "live").exists() or not any((home / "live").rglob("*.jsonl"))
    assert LiveTail("../escape").poll() == []


def test_old_journals_are_pruned_when_a_turn_starts():
    old = LiveTurn("agent-1", "old")
    _run(old)
    path = next(agent_dir("agent-1").glob("*.jsonl"))
    stale = time.time() - live_turns.KEEP_SECONDS - 10
    os.utime(path, (stale, stale))
    _run(LiveTurn("agent-1", "new"))
    remaining = list(agent_dir("agent-1").glob("*.jsonl"))
    assert len(remaining) == 1 and remaining[0] != path


def test_the_playground_streams_running_cli_turns(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from unittest.mock import Mock

    from fastapi.testclient import TestClient

    from memorizz import MemAgent
    from memorizz.llms.openai import OpenAI
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.ui import state
    from memorizz.ui.app import create_app

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", Mock())
    memory = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    agent = MemAgent(
        model=OpenAI(model="gpt-test"), memory_provider=memory, auto_register=False
    )
    agent.save()
    agent_id = agent.agent_id
    agent.close()
    running = LiveTurn(agent_id, "hi there")
    _run(running, deltas=("Hi",), status=None)

    client = TestClient(create_app())
    monkeypatch.setitem(state._state, "provider", memory)
    monkeypatch.setitem(state._state, "provider_type", "filesystem")
    monkeypatch.setitem(state._state, "connection_info", {})
    response = client.get(f"/api/agents/{agent_id}/live?memory_id=m1&wait=0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0]["type"] == "turn.started" and events[0]["query"] == "hi there"
    assert events[-1] == {**events[-1], "type": "answer.delta", "delta": "Hi"}
    assert client.get(f"/api/agents/{agent_id}/live?memory_id=m2&wait=0").text == (
        "retry: 2000\n\n"
    )
    assert client.get("/api/agents/missing/live?wait=0").status_code == 404
    page = client.get(f"/agents/{agent_id}/playground").text
    assert "pgFollowLiveTurns" in page


def test_the_cli_publishes_its_turns(monkeypatch):
    from io import StringIO
    from types import SimpleNamespace

    from memorizz.cli import streaming

    def run_stream_events(query, **kwargs):
        yield {"type": "run.started", "memory_id": "m1", "thread_id": "t1"}
        yield {"type": "answer.delta", "delta": "Hi "}
        yield {"type": "answer.delta", "delta": "Richmond"}
        yield {"type": "answer.done"}
        yield {"type": "run.done", "status": "completed"}

    agent = SimpleNamespace(agent_id="agent-1", run_stream_events=run_stream_events)
    session = SimpleNamespace(agent=agent, memory_id=None, thread_id=None, user_id=None)
    monkeypatch.setattr(streaming, "save_stream_session", lambda session: "written")
    tail = LiveTail("agent-1", "m1")
    tail.poll()
    code = streaming.consume_stream(
        session, "hi there", stdout=StringIO(), stderr=StringIO()
    )
    assert code == 0
    events = tail.poll()
    assert events[0]["query"] == "hi there"
    assert "".join(e.get("delta", "") for e in events) == "Hi Richmond"
    assert events[-1]["type"] == "turn.done"
