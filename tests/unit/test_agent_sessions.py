"""Codex and Claude Code sessions recorded as harness runs (the plugin's trajectories)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from memorizz.metaharness import SQLiteHarnessRunStore
from memorizz.metaharness.agent_sessions import (
    detect_agent,
    find_session_log,
    read_session,
    record_session,
    session_run_id,
)

pytestmark = pytest.mark.unit

CONTEXT = (
    "MemoRizz memory is connected (the `memorizz` MCP tools).\n"
    "Last session summary (2026-10-02, claude-code): - Moved checkout to int()."
)


@pytest.fixture(autouse=True)
def _restore_environment():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _item(at: str, item: dict, started_ms: int = None) -> dict:
    payload = {"type": "item_completed", "item": item}
    if started_ms is not None:
        payload["started_at_ms"] = started_ms
    return {"timestamp": at, "type": "event_msg", "payload": payload}


def codex_rollout(path: Path, project: Path, *, second_turn: bool = False) -> Path:
    rows = [
        {
            "timestamp": "2026-10-02T11:57:14.910Z",
            "type": "session_meta",
            "payload": {
                "id": "codex-session-1",
                "cwd": str(project),
                "cli_version": "0.159.3",
            },
        },
        {
            "timestamp": "2026-10-02T11:57:15.000Z",
            "type": "turn_context",
            "payload": {"model": "gpt-5.1-codex", "cwd": str(project)},
        },
        {
            "timestamp": "2026-10-02T11:57:15.100Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": CONTEXT}],
            },
        },
        {
            "timestamp": "2026-10-02T11:57:15.200Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Sandbox rules …"}],
            },
        },
        _item(
            "2026-10-02T11:57:16.000Z",
            {
                "type": "UserMessage",
                "content": [{"type": "text", "text": "Fix the failing test"}],
            },
        ),
        _item(
            "2026-10-02T11:57:20.000Z",
            {
                "type": "CommandExecution",
                "id": "exec-1",
                "command": ["/bin/zsh", "-lc", "python -m pytest -q"],
                "aggregated_output": "1 failed\nOPENAI_API_KEY=sk-proj-abcdefghijklmnop1234",
                "exit_code": 1,
                "status": "completed",
            },
            started_ms=1_790_942_237_000,  # 11:57:17, three seconds earlier
        ),
        _item(
            "2026-10-02T11:57:25.000Z",
            {
                "type": "McpToolCall",
                "id": "exec-2",
                "server": "memorizz",
                "tool": "memorizz_search_memories",
                "arguments": {"query": "rewards rounding"},
                "status": "completed",
                "result": {
                    "content": [{"type": "text", "text": '{"count": 1}'}],
                    "isError": False,
                },
            },
        ),
        _item(
            "2026-10-02T11:57:30.000Z",
            {
                "type": "FileChange",
                "id": "exec-3",
                "changes": {
                    str(project / "src/rewards.py"): {
                        "type": "update",
                        "unified_diff": "-round\n+int\n",
                    }
                },
                "status": "completed",
            },
        ),
        _item(
            "2026-10-02T11:57:33.000Z",
            {"type": "Reasoning", "id": "rs-1", "summary_text": []},
        ),
        _item(
            "2026-10-02T11:57:35.000Z",
            {
                "type": "AgentMessage",
                "content": [{"type": "Text", "text": "Fixed: int() not round()."}],
            },
        ),
        {
            "timestamp": "2026-10-02T11:57:35.500Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 20_000,
                        "cached_input_tokens": 12_000,
                        "output_tokens": 500,
                        "reasoning_output_tokens": 50,
                    }
                },
            },
        },
        {
            "timestamp": "2026-10-02T11:57:36.000Z",
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "last_agent_message": "Fixed: int() not round().",
            },
        },
    ]
    if second_turn:
        rows += [
            _item(
                "2026-10-02T12:10:00.000Z",
                {
                    "type": "UserMessage",
                    "content": [{"type": "text", "text": "Now add a test for $0"}],
                },
            ),
            _item(
                "2026-10-02T12:10:04.000Z",
                {
                    "type": "AgentMessage",
                    "content": [{"type": "Text", "text": "Added the test."}],
                },
            ),
        ]
    return _write(path, rows)


def _claude_assistant(at: str, message_id: str, block: dict, **extra) -> dict:
    return {
        "type": "assistant",
        "timestamp": at,
        "sessionId": "claude-session-1",
        "message": {
            "id": message_id,
            "model": "claude-haiku-4-5-20251001",
            "role": "assistant",
            "content": [block],
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 1_000,
                "cache_creation_input_tokens": 2_000,
                "output_tokens": 100,
            },
        },
        **extra,
    }


def claude_transcript(path: Path, project: Path, *, cost_state: bool = True) -> Path:
    base = {"sessionId": "claude-session-1", "cwd": str(project), "version": "2.1.287"}
    rows = [
        {
            **base,
            "type": "attachment",
            "timestamp": "2026-10-02T11:58:17.000Z",
            "attachment": {
                "type": "hook_success",
                "hookEvent": "SessionStart",
                "content": CONTEXT,
            },
        },
        {
            **base,
            "type": "user",
            "isMeta": True,
            "timestamp": "2026-10-02T11:58:18.000Z",
            "message": {"role": "user", "content": "Caveat: local commands below"},
        },
        {
            **base,
            "type": "user",
            "timestamp": "2026-10-02T11:58:18.500Z",
            "message": {"role": "user", "content": "Why int() and not round()?"},
        },
        # One model call logged as two lines (thinking, then a tool call) with
        # the same usage: counted once.
        _claude_assistant(
            "2026-10-02T11:58:20.000Z",
            "msg_1",
            {"type": "thinking", "thinking": ""},
            cwd=str(project),
        ),
        _claude_assistant(
            "2026-10-02T11:58:21.000Z",
            "msg_1",
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "mcp__plugin_memorizz_memorizz__memorizz_search_memories",
                "input": {"query": "rewards rounding"},
            },
        ),
        {
            **base,
            "type": "user",
            "timestamp": "2026-10-02T11:58:22.000Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "one point per dollar"}],
                    }
                ],
            },
        },
        {
            **base,
            "type": "assistant",
            "isSidechain": True,
            "timestamp": "2026-10-02T11:58:22.500Z",
            "message": {"id": "side", "model": "claude-haiku-4-5", "content": []},
        },
        _claude_assistant(
            "2026-10-02T11:58:25.000Z",
            "msg_2",
            {"type": "text", "text": "Points are per whole dollar."},
        ),
    ]
    if cost_state:
        rows.append(
            {
                "type": "cost-state",
                "sessionId": "claude-session-1",
                "totalCostUSD": 0.0123,
            }
        )
    return _write(path, rows)


def test_a_codex_session_becomes_a_run_with_its_steps(tmp_path: Path):
    project = tmp_path / "shop"
    log = codex_rollout(tmp_path / "rollout-x-codex-session-1.jsonl", project)
    assert detect_agent(log) == "codex"
    run, events = read_session(log, memory_id="project-shop-abc123")

    assert run.run_id == session_run_id("codex", "codex-session-1")
    assert run.harness == "codex" and run.status.value == "succeeded"
    assert run.task["task"] == "Fix the failing test"
    assert run.task["workspace"] == str(project)
    assert run.task["memory_id"] == "project-shop-abc123"
    assert run.task["metadata"]["source"] == "plugin"
    assert run.task["permissions"]["workspace_mode"] == "direct"  # it edited a file
    assert run.result["final_response"] == "Fixed: int() not round()."
    assert run.result["usage"]["input_tokens"] == 20_000
    assert run.result["usage"]["model"] == "gpt-5.1-codex"
    assert run.result["latency_ms"] == 20_000  # prompt at :16, done at :36

    kinds = [event.type.value for event in events]
    assert kinds[0] == "status" and kinds[-2:] == ["usage", "complete"]
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    contexts = [e.data["memory_context"] for e in events if "memory_context" in e.data]
    assert len(contexts) == 1 and contexts[0]["hook"] == "SessionStart"
    began, command = [e for e in events if e.type.value == "command"]
    assert began.data["status"] == "in_progress"
    assert began.timestamp.startswith("2026-10-02T11:57:17")  # the command took 3s
    assert command.timestamp.startswith("2026-10-02T11:57:20")
    assert command.data["command"] == "python -m pytest -q"
    assert "sk-proj" not in command.data["aggregated_output"]  # secrets removed
    tool = next(e for e in events if e.type.value == "tool_call")
    assert tool.data["name"] == "mcp__memorizz__memorizz_search_memories"
    assert tool.data["arguments"] == {"query": "rewards rounding"}
    result = next(e for e in events if e.type.value == "tool_result")
    assert result.data["tool_use_id"] == "exec-2" and not result.data["is_error"]
    change = next(e for e in events if e.type.value == "file_change")
    assert change.data["changes"][0]["path"] == "src/rewards.py"
    assert next(e for e in events if e.type.value == "reasoning").data["hidden"]


def test_a_claude_code_session_becomes_a_run_with_its_steps(tmp_path: Path):
    project = tmp_path / "shop"
    log = claude_transcript(tmp_path / "claude-session-1.jsonl", project)
    assert detect_agent(log) == "claude-code"
    run, events = read_session(log)

    assert run.harness == "claude-code"
    assert run.task["task"] == "Why int() and not round()?"  # not the meta line
    assert run.task["metadata"]["turns"] == 1
    assert run.task["permissions"]["workspace_mode"] == "read_only"
    assert run.result["final_response"] == "Points are per whole dollar."
    assert run.result["cost_usd"] == pytest.approx(0.0123)  # Claude Code's own total
    usage = run.result["usage"]
    # msg_1 appears on two lines but is one model call.
    assert usage["input_tokens"] == 20 and usage["output_tokens"] == 200
    assert usage["cache_read_input_tokens"] == 2_000
    tool = next(e for e in events if e.type.value == "tool_call")
    assert tool.data["name"] == "mcp__memorizz__memorizz_search_memories"
    result = next(e for e in events if e.type.value == "tool_result")
    block = result.data["message"]["content"][0]
    assert block == {
        "tool_use_id": "toolu_1",
        "content": "one point per dollar",
        "is_error": False,
    }
    assert not any("side" in json.dumps(e.data) for e in events)  # sidechain skipped
    context = next(e for e in events if "memory_context" in e.data)
    assert "Last session summary" in context.data["memory_context"]["text"]


def test_without_claude_codes_total_the_cost_is_priced_from_usage(tmp_path: Path):
    log = claude_transcript(tmp_path / "s.jsonl", tmp_path, cost_state=False)
    run, _events = read_session(log)
    assert run.result["cost_usd"] and run.result["cost_usd"] > 0
    assert run.result["usage"]["cost_basis"] == "list_rate_estimate"


def test_recording_again_replaces_the_run_and_follows_new_turns(tmp_path: Path):
    project = tmp_path / "shop"
    log = codex_rollout(tmp_path / "rollout.jsonl", project)
    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    try:
        first = record_session(log, store=store)
        count = len(store.events(first.run_id))
        record_session(log, store=store)
        assert len(store.list()) == 1 and len(store.events(first.run_id)) == count

        codex_rollout(log, project, second_turn=True)
        again = record_session(log, store=store)
        assert again.run_id == first.run_id
        assert again.task["metadata"]["turns"] == 2
        assert again.result["final_response"] == "Added the test."
        assert len(store.events(first.run_id)) == count + 2
        # Two turns' working time, not the twelve minutes between them.
        assert again.result["latency_ms"] == 24_000
    finally:
        store.close()


def test_a_log_without_a_prompt_records_nothing(tmp_path: Path):
    log = _write(
        tmp_path / "rollout.jsonl",
        [{"type": "session_meta", "payload": {"id": "s", "cwd": str(tmp_path)}}],
    )
    assert read_session(log) is None
    assert read_session(_write(tmp_path / "other.jsonl", [{"hello": 1}])) is None


def test_session_logs_are_found_by_session_id(tmp_path: Path):
    codex_home = tmp_path / "codex"
    log = codex_rollout(
        codex_home / "sessions/2026/10/02/rollout-2026-10-02T11-57-14-abc.jsonl",
        tmp_path,
    )
    assert find_session_log("codex", "abc", home=codex_home) == log
    claude_home = tmp_path / "claude"
    transcript = claude_transcript(
        claude_home / "projects/-shop/claude-session-1.jsonl", tmp_path
    )
    assert (
        find_session_log("claude-code", "claude-session-1", home=claude_home)
        == transcript
    )
    assert find_session_log("codex", "", home=codex_home) is None


def test_the_plugin_records_the_session_after_each_turn(tmp_path: Path, monkeypatch):
    from memorizz.cli import plugin_commands

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("MEMORIZZ_HOME", str(home))
    monkeypatch.setenv("PLUGIN_ROOT", "/plugins/memorizz")  # Codex runs the hook
    monkeypatch.delenv(plugin_commands.REMOTE_URL_ENV, raising=False)
    project = tmp_path / "shop"
    project.mkdir()
    log = codex_rollout(tmp_path / "rollout.jsonl", project)
    payload = {
        "session_id": "codex-session-1",
        "cwd": str(project),
        "transcript_path": str(log),
    }
    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    try:
        run_id = plugin_commands.record_session_run(payload, store=store)
        run = store.get(run_id)
        assert run.harness == "codex"
        assert run.task["memory_id"] == plugin_commands.project_memory_id(project)

        monkeypatch.setenv(plugin_commands.RUNS_ENV, "false")
        assert plugin_commands.record_session_run(payload, store=store) is None
    finally:
        store.close()


def test_a_hosted_plugin_does_not_record_runs(tmp_path: Path, monkeypatch):
    from memorizz.cli import plugin_commands

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv(plugin_commands.REMOTE_URL_ENV, "https://memory.example.com/mcp")
    log = codex_rollout(tmp_path / "rollout.jsonl", tmp_path)
    with patch("memorizz.metaharness.agent_sessions.record_session") as record:
        assert plugin_commands.record_session_run({"transcript_path": str(log)}) is None
    record.assert_not_called()


def test_import_session_shows_past_sessions_in_the_ui(tmp_path: Path, monkeypatch):
    from typer.testing import CliRunner

    from memorizz.cli.plugin_commands import plugin_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    codex = codex_rollout(tmp_path / "rollout.jsonl", tmp_path / "shop")
    claude = claude_transcript(tmp_path / "claude.jsonl", tmp_path / "shop")
    junk = _write(tmp_path / "junk.jsonl", [{"hello": 1}])
    result = CliRunner().invoke(
        plugin_app, ["import-session", str(codex), str(claude), str(junk), "--json"]
    )
    assert result.exit_code == 0, result.output
    runs = json.loads(result.output)["runs"]
    assert [item["agent"] for item in runs[:2]] == ["codex", "claude-code"]
    assert runs[2]["run_id"] is None
    store = SQLiteHarnessRunStore(tmp_path / "home" / "harness-runs.sqlite3")
    try:
        assert {run.harness for run in store.list()} == {"codex", "claude-code"}
        assert all(
            run.task["memory_id"].startswith("project-shop-") for run in store.list()
        )
    finally:
        store.close()


def test_the_harnesses_page_shows_a_plugin_session(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from memorizz.approval import SQLiteApprovalStore
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.metaharness import MetaHarness
    from memorizz.ui import state
    from memorizz.ui.app import create_app

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[],
        run_store=store,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    run = record_session(
        claude_transcript(tmp_path / "claude.jsonl", tmp_path / "shop"), store=store
    )
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "meta_harness": meta,
        "meta_harness_provider": provider,
        "read_only": False,
    }
    try:
        with patch.dict(state._state, values):
            client = TestClient(create_app(), follow_redirects=False)
            page = client.get("/harnesses")
            assert page.status_code == 200
            assert "plugin session" in page.text and "First prompt" in page.text
            assert "Recorded by" in page.text
            detail = client.get(f"/api/harness-runs/{run.run_id}?include_events=true")
            assert detail.status_code == 200
            events = detail.json()["run"]["events"]
            assert any("memory_context" in event["data"] for event in events)
    finally:
        meta.close()
        provider.close()
