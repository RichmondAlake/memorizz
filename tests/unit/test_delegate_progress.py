"""Seeing a coordinator's delegates work: in the stream, the terminal and the
playground."""

from __future__ import annotations

import time
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.metaharness import (
    AgentHarness,
    HarnessCapabilities,
    HarnessEvent,
    HarnessEventType,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness.base import AdapterOutcome

pytestmark = pytest.mark.unit


class _Harness(AgentHarness):
    """Runs a command, then answers, or fails with ``tests_failed``."""

    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail

    def probe(self):
        return HarnessCapabilities(name=self.name, available=True, command=self.name)

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        emit(
            HarnessEvent(
                task.run_id,
                HarnessEventType.COMMAND,
                {"id": "c1", "command": "pytest -q", "status": "in_progress"},
            )
        )
        time.sleep(0.4)
        if self.fail:
            return AdapterOutcome(
                error_code="tests_failed", error="1 failed", exit_code=1
            )
        return AdapterOutcome(final_response=f"{self.name} found 1 bug", exit_code=0)


class _Model:
    model = "fake"

    def get_config(self):
        return {"provider": "openai", "model": "fake"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def generate_stream(self, messages, tools=None, **kwargs):
        yield {"type": "content", "content": "ok"}
        yield {"type": "done", "content": "ok"}


@pytest.fixture
def service(tmp_path: Path):
    harnesses = MetaHarness(
        adapters=[_Harness("busy"), _Harness("broken", fail=True)],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    yield harnesses
    harnesses.close()


def _delegate(memory, service, workspace, name, harness):
    from memorizz import MemAgent

    agent = MemAgent(model=_Model(), memory_provider=memory, auto_register=False)
    agent.name = name
    agent.with_meta_harness(
        service,
        mode="runtime",
        default_harness=harness,
        config={"workspace": str(workspace), "permissions": {"mcp_access": "none"}},
    )
    return agent


def test_a_delegates_harness_run_is_named_and_tied_to_its_step(tmp_path, service):
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.streaming import current_stream, harness_event_listener

    memory = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", embedding_provider=None)
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = _delegate(memory, service, workspace, "Bug hunter", "busy")
    session = SimpleNamespace(events=[], agent=None)
    session.emit = lambda kind, **payload: session.events.append((kind, payload))
    seen = []
    tokens = (
        current_stream.set(session),
        harness_event_listener.set(lambda run, harness, e: seen.append(e)),
    )
    try:
        agent.run_on_harness(
            "Check", workspace=str(workspace), context={"delegation": {"task_id": "t1"}}
        )
    finally:
        harness_event_listener.reset(tokens[1])
        current_stream.reset(tokens[0])
        agent.close()
    assert session.events[0][1]["message"] == "Bug hunter (busy) is working"
    assert {payload["span_id"] for _, payload in session.events} == {"delegate:t1"}
    assert seen[0] == {"type": "run.started", "data": {"delegate": "Bug hunter"}}


def test_a_crew_streams_each_delegates_start_progress_and_outcome(tmp_path, service):
    from memorizz import MemAgent
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    memory = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", embedding_provider=None)
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    hunter = _delegate(memory, service, workspace, "Bug hunter", "busy")
    checker = _delegate(memory, service, workspace, "Doc checker", "broken")
    # A plain MemAgent delegate, on its own model in this process.
    notes = MemAgent(model=_Model(), memory_provider=memory, auto_register=False)
    notes.name = "Note taker"
    crew = MemAgent(
        model=_Model(),
        memory_provider=memory,
        auto_register=False,
        delegates=[hunter, checker, notes],
        delegation={
            "enabled": True,
            "mode": "deterministic",
            "consolidation_strategy": "deterministic",
            "plan": [
                {
                    "task_id": "hunt",
                    "description": "Hunt",
                    "assigned_agent_id": hunter.agent_id,
                },
                {
                    "task_id": "docs",
                    "description": "Docs",
                    "assigned_agent_id": checker.agent_id,
                },
                {
                    "task_id": "notes",
                    "description": "Notes",
                    "assigned_agent_id": notes.agent_id,
                },
            ],
        },
    )
    try:
        with crew.run_stream_events("review") as stream:
            events = [
                event
                for event in stream
                if str(event.get("span_id") or "").startswith("delegate:")
            ]
    finally:
        crew.close()
    by_kind = {}
    for event in events:
        by_kind.setdefault((event["type"], event["span_id"]), []).append(event)
    assert by_kind[("tool.started", "delegate:hunt")][0]["tool_name"] == (
        "Delegate · Bug hunter (busy)"
    )
    progress = by_kind[("status", "delegate:hunt")]
    assert progress[0]["message"] == "Bug hunter (busy) is working"
    assert progress[0]["harness_run_id"]
    assert "Bug hunter (busy): running a command" in [e["message"] for e in progress]
    assert "pytest" not in repr(events)  # the stream names the kind of work only
    assert by_kind[("tool.completed", "delegate:hunt")][0]["status"] == "success"
    failed = by_kind[("tool.completed", "delegate:docs")][0]
    assert failed["status"] == "error" and failed["reason_code"]
    notes_progress = [e["message"] for e in by_kind[("status", "delegate:notes")]]
    assert notes_progress[:2] == ["Note taker is working", "Note taker: thinking"]
    assert by_kind[("tool.completed", "delegate:notes")][0]["status"] == "success"


def test_the_terminal_names_each_delegates_lines_and_says_how_it_ended():
    from memorizz.cli.harness_feed import DelegateLines, HarnessFeed

    feed = HarnessFeed()
    feed.listener(
        "r1", "codex", {"type": "run.started", "data": {"delegate": "Bug hunter"}}
    )
    feed.listener(
        "r1",
        "codex",
        {
            "type": "command",
            "data": {"id": "c", "command": "ls", "status": "in_progress"},
        },
    )
    feed.listener("r1", "codex", {"type": "message", "data": {"text": "1 bug"}})
    feed.listener("r1", "codex", {"type": "run.finished"})
    # No harness header (the delegate has its own line); every line is named
    # and its result prints when its run ends, not with the reply's answer.
    assert [line.plain for line in feed.drain()] == [
        "  Bug hunter · $ ls",
        "  Bug hunter · 1 bug",
    ]

    lines = DelegateLines()
    start = {
        "type": "tool.started",
        "span_id": "delegate:a",
        "tool_name": "Delegate · Bug hunter (codex)",
    }
    assert [t.plain for t in lines.event(start)] == ["● Bug hunter (codex) is working"]
    assert lines.event(start) == []  # once
    done = {**start, "type": "tool.completed", "status": "success"}
    assert [t.plain for t in lines.event(done)] == ["✓ Bug hunter (codex) done · 0s"]
    lines.event({**start, "span_id": "delegate:b"})
    failed = {
        **start,
        "span_id": "delegate:b",
        "type": "tool.completed",
        "status": "error",
        "reason_code": "tests_failed",
    }
    assert [t.plain for t in lines.event(failed)] == [
        "✗ Bug hunter (codex) failed · tests failed · 0s"
    ]
    assert lines.event({"type": "tool.started", "span_id": "s", "tool_name": "x"}) == []


def test_the_repl_prints_delegates_outside_the_tools_line(monkeypatch, tmp_path):
    from rich.console import Console

    from memorizz.cli import repl, streaming

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))

    class Stream:
        def __init__(self, events):
            self.events = list(events)

        def poll(self, timeout=None):
            if not self.events:
                raise StopIteration
            return self.events.pop(0)

        def close(self):
            pass

    name = "Delegate · Bug hunter (codex)"
    events = [
        {"type": "run.started", "memory_id": "m1", "thread_id": "t1"},
        {"type": "tool.started", "tool_name": name, "span_id": "delegate:a"},
        {"type": "tool.started", "tool_name": "lookup", "span_id": "s1"},
        {
            "type": "tool.completed",
            "tool_name": name,
            "span_id": "delegate:a",
            "status": "error",
            "reason_code": "HarnessSecurityError",
        },
        {"type": "run.done", "status": "completed"},
    ]
    out = StringIO()
    session = SimpleNamespace(
        agent=SimpleNamespace(
            agent_id="a1", run_stream_events=lambda q, **k: Stream(events)
        ),
        memory_id=None,
        thread_id=None,
        user_id=None,
        console=Console(file=out, force_terminal=True, color_system=None, width=120),
    )
    monkeypatch.setattr(streaming, "save_stream_session", lambda s: "written")
    repl._stream_turn(session, "review")
    text = out.getvalue()
    assert "● Bug hunter (codex) is working" in text
    assert "✗ Bug hunter (codex) failed · HarnessSecurityError" in text
    assert "Tools: lookup" in text and "Tools: Delegate" not in text


def test_the_playground_follows_a_cli_turns_delegates(tmp_path, monkeypatch):
    from memorizz.live_turns import LiveTail, LiveTurn

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    turn = LiveTurn("agent-1", "review")
    turn.event({"type": "run.started", "memory_id": "m1"})
    turn.event(
        {
            "type": "status",
            "message": "Bug hunter (Codex) is working",
            "span_id": "delegate:a",
            "harness_run_id": "run-9",
        }
    )
    tail = LiveTail("agent-1")
    lines = [e for e in tail.poll() if e["type"] == "status"]
    assert lines[0]["span_id"] == "delegate:a" and lines[0]["harness_run_id"] == "run-9"
