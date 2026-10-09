"""A turn on a harness reports its progress while it runs, not only at the end."""

from __future__ import annotations

import os
import time
from pathlib import Path

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
from memorizz.streaming import HarnessProgress, _harness_activity


class _BusyHarness(AgentHarness):
    """Calls a tool, runs a command and thinks before answering."""

    name = "busy"

    def probe(self) -> HarnessCapabilities:
        return HarnessCapabilities(name=self.name, available=True, command="busy")

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        steps = [
            (HarnessEventType.STATUS, {"thread_id": "t-1", "model": "busy-model"}),
            (
                HarnessEventType.TOOL_CALL,
                {"name": "memorizz_search_memories", "status": "in_progress"},
            ),
            (
                HarnessEventType.COMMAND,
                {"command": "cat secret-plan.txt", "status": "in_progress"},
            ),
            (HarnessEventType.REASONING, {"text": "private chain of thought"}),
        ]
        for kind, data in steps:
            emit(HarnessEvent(task.run_id, kind, data))
            time.sleep(0.35)
        return AdapterOutcome(final_response="all done", exit_code=0)


def test_activity_names_the_kind_of_work_never_its_content():
    assert _harness_activity(
        {"type": "tool_call", "data": {"name": "web_search", "status": "in_progress"}}
    ) == {"activity": "calling web_search", "tool_name": "web_search"}
    assert (
        _harness_activity(
            {"type": "tool_call", "data": {"name": "web_search", "status": "completed"}}
        )
        is None
    )
    command = _harness_activity(
        {
            "type": "command",
            "data": {"command": "rm -rf build", "status": "in_progress"},
        }
    )
    assert command == {"activity": "running a command"}
    assert _harness_activity({"type": "reasoning", "data": {"text": "secret"}}) == {
        "activity": "thinking"
    }
    assert _harness_activity({"type": "message", "data": {"text": "draft"}}) == {
        "activity": "writing"
    }
    for quiet in ("log", "usage", "complete", "error"):
        assert _harness_activity({"type": quiet, "data": {}}) is None


class _Session:
    def __init__(self):
        self.events = []

    def emit(self, kind, **payload):
        self.events.append((kind, payload))


class _Service:
    def __init__(self, events):
        self._events = events

    def events(self, run_id, *, after=0, limit=1000):
        return [e for e in self._events if e["sequence"] > after][:limit]


def test_progress_relays_model_and_activity_then_stops():
    service = _Service(
        [
            {"sequence": 1, "type": "status", "data": {"status": "running"}},
            {"sequence": 2, "type": "status", "data": {"model": "gpt-x"}},
            {"sequence": 3, "type": "reasoning", "data": {"text": "a"}},
            {"sequence": 4, "type": "reasoning", "data": {"text": "b"}},
            {"sequence": 5, "type": "tool_call", "data": {"name": "web_search"}},
        ]
    )
    session = _Session()
    progress = HarnessProgress(session, service, "codex", poll_seconds=0.01)
    progress.start("run-1")
    deadline = time.monotonic() + 2
    while len(session.events) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    progress.stop()
    messages = [payload["message"] for _, payload in session.events]
    assert messages == [
        "Codex is working",
        "Codex is working (gpt-x)",
        "Codex: thinking",  # once, not once per reasoning event
        "Codex: calling web_search",
    ]
    assert all(kind == "status" for kind, _ in session.events)
    assert {payload["harness_run_id"] for _, payload in session.events} == {"run-1"}
    count = len(session.events)
    time.sleep(0.05)
    assert len(session.events) == count  # stopped


def test_progress_stops_quietly_when_the_ledger_fails():
    class Broken:
        def events(self, *args, **kwargs):
            raise RuntimeError("ledger closed")

    session = _Session()
    progress = HarnessProgress(session, Broken(), "pi", poll_seconds=0.01)
    progress.start("run-2")
    progress.stop()
    assert [payload["message"] for _, payload in session.events] == ["pi is working"]


def test_streamed_runtime_turn_reports_harness_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from memorizz.memagent.builders import MemAgentBuilder

    monkeypatch.setenv("MEMORIZZ_MEMORY_ROOT", str(tmp_path / "memory"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = MetaHarness(
        adapters=[_BusyHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    agent = (
        MemAgentBuilder()
        .with_instruction("Answer on the harness")
        .with_execution_harness(
            "busy",
            meta_harness=service,
            config={"workspace": str(workspace), "permissions": {"mcp_access": "none"}},
        )
        .build()
    )
    try:
        stream = agent.run_stream_events(
            "What is in the plan?", memory_id="memory-1", thread_id="thread-1"
        )
        events = list(stream)
    finally:
        agent.close(close_memory_provider=True)
        service.close()

    statuses = [e for e in events if e["type"] == "status"]
    messages = [e.get("message") for e in statuses if e.get("harness") == "busy"]
    assert messages[0] == "busy is working"
    assert "busy is working (busy-model)" in messages
    assert "busy: calling memorizz_search_memories" in messages
    assert "busy: running a command" in messages
    assert "busy: thinking" in messages
    answer = "".join(e["delta"] for e in events if e["type"] == "answer.delta")
    assert answer == "all done"
    # Progress comes before the answer and never carries tool input or reasoning.
    first_delta = next(e["seq"] for e in events if e["type"] == "answer.delta")
    assert all(e["seq"] < first_delta for e in statuses if e.get("harness"))
    text = repr(events)
    assert "secret-plan" not in text and "chain of thought" not in text


def test_quiet_keys_turns_echo_off_inside_and_restores_it():
    termios = pytest.importorskip("termios")
    from memorizz.cli import ui

    master, slave = os.openpty()
    try:
        stream = os.fdopen(slave, "r", closefd=False)
        before = termios.tcgetattr(slave)
        assert before[3] & termios.ECHO
        with ui.quiet_keys(stream):
            assert not termios.tcgetattr(slave)[3] & termios.ECHO
        assert termios.tcgetattr(slave) == before
    finally:
        os.close(master)
        os.close(slave)


def test_quiet_keys_is_a_no_op_without_a_terminal():
    from io import StringIO

    from memorizz.cli import ui

    with ui.quiet_keys(StringIO()):
        pass


def test_elapsed_suffix_counts_up_and_offers_the_way_out():
    from memorizz.cli import ui

    assert ui.elapsed_suffix(0.4) == ""
    assert ui.elapsed_suffix(3.2) == " · 3s"
    assert ui.elapsed_suffix(12) == " · 12s · ctrl-c to stop"
    assert ui.elapsed_suffix(185) == " · 3m 05s · ctrl-c to stop"
