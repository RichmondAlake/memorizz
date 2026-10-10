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
    agent = None  # a stream owned by some other agent, e.g. the coordinator

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


def test_listener_gets_full_events_while_the_stream_stays_content_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from memorizz.memagent.builders import MemAgentBuilder
    from memorizz.streaming import harness_event_listener

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
    seen = []
    token = harness_event_listener.set(lambda run, harness, e: seen.append(e))
    try:
        events = list(
            agent.run_stream_events("Plan?", memory_id="m-1", thread_id="t-1")
        )
    finally:
        harness_event_listener.reset(token)
        agent.close(close_memory_provider=True)
        service.close()
    kinds = [e["type"] for e in seen]
    assert kinds[0] == "run.started" and kinds[-1] == "run.finished"
    commands = [e["data"]["command"] for e in seen if e["type"] == "command"]
    assert commands == ["cat secret-plan.txt"]  # the terminal sees the command
    assert "secret-plan" not in repr(events)  # the stream does not


def test_any_harness_run_inside_a_stream_reports_progress(tmp_path: Path):
    """Delegates and the run_harness_task tool go through run_on_harness too."""
    from memorizz.memagent.builders import MemAgentBuilder
    from memorizz.streaming import current_stream

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
        .with_instruction("Delegate")
        .with_meta_harness(
            service,
            default_harness="busy",
            mode="delegate",
            config={"workspace": str(workspace), "permissions": {"mcp_access": "none"}},
        )
        .build()
    )
    session = _Session()
    token = current_stream.set(session)
    try:
        result = agent.run_on_harness("Check the plan", workspace=str(workspace))
    finally:
        current_stream.reset(token)
        agent.close(close_memory_provider=True)
        service.close()
    assert result.final_response == "all done"
    messages = [payload.get("message") for _, payload in session.events]
    assert messages[0] == "busy is working"
    assert "busy: running a command" in messages


def _feed_lines(feed, answer=None):
    lines = feed.drain() if answer is None else feed.finish(answer)
    return [line.plain for line in lines]


def test_feed_prints_commands_tools_edits_and_holds_back_the_answer():
    from memorizz.cli.harness_feed import HarnessFeed

    feed = HarnessFeed()
    put = lambda kind, data: feed.listener("r1", "codex", {"type": kind, "data": data})
    feed.listener("r1", "codex", {"type": "run.started"})
    put("message", {"role": "assistant", "text": "I'll look at the folder first."})
    put(
        "command",
        {"id": "c1", "command": '/bin/zsh -lc "ls -la"', "status": "in_progress"},
    )
    put(
        "command",
        {
            "id": "c1",
            "command": '/bin/zsh -lc "ls -la"',
            "status": "completed",
            "exit_code": 0,
            "aggregated_output": "README.md\nsrc\ntests\nsetup.py\n",
        },
    )
    put(
        "tool_call",
        {
            "id": "t1",
            "name": "mcp__memorizz__memorizz_search_memories",
            "arguments": {"query": "release steps", "limit": 5},
            "status": "in_progress",
        },
    )
    put("tool_call", {"id": "t1", "name": "x", "status": "completed"})
    put("tool_result", {"tool_use_id": "t1", "content": '{"ok": true, "count": 2}'})
    put("file_change", {"changes": [{"path": "src/app.py", "kind": "update"}]})
    put(
        "command",
        {"id": "c2", "command": "pytest -q", "exit_code": 1, "status": "failed"},
    )
    put("message", {"role": "assistant", "text": "The answer."})
    assert _feed_lines(feed) == [
        "● Codex",
        "  I'll look at the folder first.",
        "  $ ls -la",
        "    README.md",
        "    src",
        "    tests",
        "    … 1 more lines",
        "  ⚙ memorizz_search_memories  query=release steps, limit=5",
        '    ↳ {"ok": true, "count": 2}',
        "  ✎ update src/app.py",
        "  $ pytest -q",
        "    exit 1",
    ]
    feed.listener("r1", "codex", {"type": "run.finished"})
    # The last message is the harness's answer, which the chat prints itself.
    assert _feed_lines(feed, answer="The answer.") == []


def test_feed_keeps_a_last_message_that_is_not_the_answer_and_labels_parallel_runs():
    from memorizz.cli.harness_feed import HarnessFeed

    feed = HarnessFeed()
    feed.listener("a", "codex", {"type": "run.started"})
    feed.listener("b", "claude-code", {"type": "run.started"})
    feed.listener(
        "a",
        "codex",
        {"type": "command", "data": {"command": "ls", "status": "in_progress"}},
    )
    feed.listener(
        "b", "claude-code", {"type": "message", "data": {"text": "Delegate summary"}}
    )
    feed.listener("a", "codex", {"type": "run.finished"})
    feed.listener("b", "claude-code", {"type": "run.finished"})
    assert _feed_lines(feed) == ["● Codex", "● Claude Code", "  Codex · $ ls"]
    assert _feed_lines(feed, answer="Coordinator answer") == ["  Delegate summary"]


def test_feed_skips_hidden_reasoning_user_messages_and_delegate_markers():
    from memorizz.cli.harness_feed import HarnessFeed, command_text

    feed = HarnessFeed()
    for kind, data in [
        ("reasoning", {"text": "", "hidden": True}),
        ("message", {"role": "user", "text": "hi"}),
        (
            "tool_call",
            {"subagent": True, "harness_run_id": "x", "status": "in_progress"},
        ),
        ("usage", {"input_tokens": 3}),
        (
            "tool_call",
            {"id": "w", "type": "web_search", "query": "", "status": "in_progress"},
        ),
    ]:
        feed.listener("r", "codex", {"type": kind, "data": data})
    assert _feed_lines(feed) == ["● Codex", "  ⌕ web search "]
    assert command_text("bash -lc 'echo \"hi\"'") == 'echo "hi"'
    assert command_text(["git", "status"]) == "git status"
    assert command_text("/bin/zsh -lc ls") == "ls"
    assert command_text("/bin/zsh -lc 'cat README.md'") == "cat README.md"
    assert command_text("python -c 'print(1)'") == "python -c 'print(1)'"


def test_a_delegate_worker_reports_progress_though_its_answer_stream_is_cleared(
    tmp_path: Path,
):
    """Delegates run with current_stream cleared; progress_stream still reaches the user."""
    from memorizz.memagent.builders import MemAgentBuilder
    from memorizz.streaming import current_stream, progress_stream

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
        .with_instruction("Delegate")
        .with_meta_harness(
            service,
            default_harness="busy",
            mode="delegate",
            config={"workspace": str(workspace), "permissions": {"mcp_access": "none"}},
        )
        .build()
    )
    session = _Session()
    tokens = (current_stream.set(None), progress_stream.set(session))
    try:
        agent.run_on_harness("Check the plan", workspace=str(workspace))
    finally:
        progress_stream.reset(tokens[1])
        current_stream.reset(tokens[0])
        agent.close(close_memory_provider=True)
        service.close()
    assert [p.get("message") for _, p in session.events][0] == "busy is working"


def test_busy_spins_on_a_terminal_and_stays_quiet_elsewhere():
    from io import StringIO

    from rich.console import Console

    from memorizz.cli import ui

    plain = Console(file=StringIO(), force_terminal=False)
    with ui.busy(plain, "Checking which harnesses are ready…"):
        pass
    assert plain.file.getvalue() == ""
    term = Console(file=StringIO(), force_terminal=True, width=80)
    with ui.busy(term, "Checking which harnesses are ready…"):
        import time as _t

        _t.sleep(0.2)
    assert "Checking which harnesses are ready" in term.file.getvalue()
