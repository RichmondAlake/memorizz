"""Harness page: conversations, step counts, MemAgent defaults, Codex costs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.approval import SQLiteApprovalStore  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.metaharness import (  # noqa: E402
    AgentHarness,
    HarnessCapabilities,
    HarnessEvent,
    HarnessEventType,
    HarnessStatus,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness.base import AdapterOutcome, SubprocessHarness  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

pytestmark = pytest.mark.unit


class _Recorder(AgentHarness):
    """Answers every task and keeps what it was given."""

    name = "recorder"

    def __init__(self):
        self.tasks = []

    def probe(self):
        return HarnessCapabilities(name=self.name, available=True, models=["m-1"])

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        self.tasks.append(task)
        answer = f"Answer {len(self.tasks)}"
        emit(
            HarnessEvent(
                task.run_id,
                HarnessEventType.MESSAGE,
                {"role": "assistant", "text": answer},
            )
        )
        emit(
            HarnessEvent(
                task.run_id,
                HarnessEventType.COMMAND,
                {"id": "c1", "command": "ls", "status": "in_progress"},
            )
        )
        emit(
            HarnessEvent(
                task.run_id,
                HarnessEventType.COMMAND,
                {"id": "c1", "command": "ls", "exit_code": 0},
            )
        )
        return AdapterOutcome(final_response=answer, exit_code=0)


def _app(tmp_path: Path, adapter):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[adapter],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "meta_harness": meta,
        "meta_harness_provider": provider,
        "read_only": False,
    }
    return provider, meta, values


def _wait(client, run_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        run = client.get(f"/api/harness-runs/{run_id}").json()["run"]
        if run["status"] in {HarnessStatus.SUCCEEDED.value, HarnessStatus.FAILED.value}:
            return run
        assert time.monotonic() < deadline, run
        time.sleep(0.02)


def test_a_run_continues_as_a_conversation_with_its_setup(tmp_path: Path):
    recorder = _Recorder()
    provider, meta, values = _app(tmp_path, recorder)
    workspace = tmp_path / "project"
    workspace.mkdir()
    try:
        with patch.dict(state._state, values):
            client = TestClient(create_app(), follow_redirects=False)
            first = client.post(
                "/api/harness-runs",
                json={
                    "task": "Find the failing test",
                    "harness": "recorder",
                    "model": "m-1",
                    "workspace": str(workspace),
                    "mcp_access": "none",
                    "memory_id": "repo",
                },
            )
            assert first.status_code == 200, first.text
            root = first.json()["run"]["run_id"]
            _wait(client, root)

            # The ledger shows the steps (the command counted once) and the way on.
            page = client.get("/harnesses").text
            assert f"/harnesses/chat?from={root}" in page
            assert f'data-trajectory-for="{root}"' in page
            assert 'data-steps="2"' in page  # one message, one command

            chat = client.get(f"/harnesses/chat?from={root}")
            assert chat.status_code == 200, chat.text
            assert "Harness chat" in chat.text
            assert f'value="{workspace}"' in chat.text
            assert 'value="m-1"' in chat.text

            conversation = f"hxc-{root}"
            second = client.post(
                f"/api/harness-conversations/{conversation}/turns",
                json={
                    "task": "Now fix it",
                    "harness": "recorder",
                    "model": "m-1",
                    "workspace": str(workspace),
                    "mcp_access": "none",
                    "memory_id": "repo",
                },
            )
            assert second.status_code == 200, second.text
            _wait(client, second.json()["run"]["run_id"])

            task = recorder.tasks[-1]
            assert task.task == "Now fix it"
            assert task.thread_id == conversation
            assert task.metadata["conversation_id"] == conversation
            assert task.metadata["conversation_turn"] == 2
            earlier = task.context["conversation"]
            assert earlier[0]["request"] == "Find the failing test"
            assert earlier[0]["response"] == "Answer 1"
            prompt = SubprocessHarness._prompt(task, SimpleNamespace(rendered=""))
            assert "--- Conversation so far ---" in prompt
            assert "User: Find the failing test" in prompt
            assert prompt.index("Conversation so far") < prompt.index("--- Task ---")

            listed = client.get(f"/api/harness-conversations/{conversation}").json()
            assert [turn["message"] for turn in listed["turns"]] == [
                "Find the failing test",
                "Now fix it",
            ]
            assert listed["turns"][1]["answer"] == "Answer 2"
            assert listed["turns"][1]["steps"]["command"] == 1
            assert (
                client.get("/api/harness-conversations/hxc-missing").status_code == 404
            )
    finally:
        meta.close()
        provider.close()


def test_memagent_uses_a_saved_agent_or_says_how_to_get_one():
    from memorizz.ui.routers import harnesses as routes

    service = SimpleNamespace(
        list_runs=lambda limit=50: [{"task": {"agent_id": "a2"}}, {"task": {}}]
    )
    agents = [
        {"agent_id": "a1", "name": "First", "created_at": "2026-09-01"},
        {"agent_id": "a2", "name": "Harness helper", "created_at": "2026-08-01"},
    ]
    # The agent last used with a harness wins; otherwise the newest agent.
    assert routes.default_memagent(service, agents) == {
        "id": "a2",
        "name": "Harness helper",
    }
    fresh = SimpleNamespace(list_runs=lambda limit=50: [])
    assert routes.default_memagent(fresh, agents)["id"] == "a1"

    with patch.object(
        routes, "default_memagent", return_value={"id": "a1", "name": "First"}
    ):
        payload, chosen = routes._with_memagent(
            {"task": "x"}, fresh, ["codex", "memagent"]
        )
        assert payload["agent_id"] == "a1" and chosen["name"] == "First"
        unchanged, none = routes._with_memagent({"agent_id": "a9"}, fresh, ["memagent"])
        assert unchanged["agent_id"] == "a9" and none is None
        other, none = routes._with_memagent({}, fresh, ["codex"])
        assert "agent_id" not in other and none is None
    with patch.object(routes, "default_memagent", return_value=None):
        with pytest.raises(ValueError, match="Create one on the Agents page"):
            routes._with_memagent({}, fresh, ["memagent"])


def test_codex_names_its_default_model_and_prices_usage(monkeypatch):
    from memorizz.metaharness import adapters
    from memorizz.metaharness.models import HarnessTask

    catalog = {
        "models": [
            {"slug": "gpt-5.5", "visibility": "list", "priority": 13},
            {"slug": "gpt-reserve", "visibility": "hide", "priority": 1},
            {"slug": "gpt-6-astra", "visibility": "list", "priority": 2},
        ]
    }
    monkeypatch.setattr(adapters.CodexHarness, "_catalogs", {})
    monkeypatch.setattr(
        adapters.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps(catalog), returncode=0),
    )
    codex = adapters.CodexHarness(command="codex-for-test")
    assert codex.model_catalog() == ["gpt-6-astra", "gpt-5.5"]

    task = HarnessTask(task="hi", workspace="/tmp")
    assert codex._resolve_model(task) == "gpt-6-astra"
    command = codex.build_command(task, workspace=Path("/tmp"), prompt="hi")
    assert command[command.index("--model") + 1] == "gpt-6-astra"
    pinned = HarnessTask(task="hi", workspace="/tmp", model="gpt-5.5")
    assert codex._resolve_model(pinned) == "gpt-5.5"

    codex._run_models[task.run_id] = "gpt-6-astra"
    _, updates = codex.parse_event(
        task.run_id,
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 60708,
                "cached_input_tokens": 47744,
                "output_tokens": 258,
                "reasoning_output_tokens": 12,
            },
        },
    )
    assert updates["cost_usd"] == pytest.approx(0.190284)
    assert updates["usage"]["model"] == "gpt-6-astra"
    assert updates["usage"]["cost_basis"] == "list_rate_estimate"
    assert adapters.codex_list_cost("not-a-model", {"input_tokens": 1}) is None


def test_the_ledger_marks_estimates_and_the_model_used():
    from memorizz.ui.execution_view import shape_harness_run

    row = shape_harness_run(
        {
            "run_id": "r1",
            "status": "succeeded",
            "task": {"task": "x", "harness": "codex"},
            "result": {
                "cost_usd": 0.19,
                "usage": {"model": "gpt-6-astra", "cost_basis": "list_rate_estimate"},
            },
        }
    )
    assert row["model"] == "gpt-6-astra" and row["model_is_default"] is True
    assert row["cost_estimated"] is True


def test_step_counts_count_each_harness_item_once(tmp_path: Path):
    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    from memorizz.metaharness.models import HarnessRun, HarnessTask

    task = HarnessTask(task="x", workspace=str(tmp_path))
    run = store.create(
        HarnessRun(run_id=task.run_id, task=task.to_dict(), status="running")
    )
    for kind, data in (
        (HarnessEventType.MESSAGE, {"text": "a"}),
        (HarnessEventType.MESSAGE, {"text": "b"}),
        (HarnessEventType.COMMAND, {"id": "item_1", "status": "in_progress"}),
        (HarnessEventType.COMMAND, {"id": "item_1", "exit_code": 0}),
        (HarnessEventType.TOOL_CALL, {"id": "t1", "name": "Read"}),
    ):
        store.append_event(HarnessEvent(run.run_id, kind, data))
    counts = store.event_counts([run.run_id])[run.run_id]
    assert (
        counts["message"] == 2 and counts["command"] == 1 and counts["tool_call"] == 1
    )
    assert store.event_counts([]) == {}
    store.close()


def test_model_output_renders_only_through_the_safe_markdown_renderer():
    ui = Path(__file__).parents[2] / "src" / "memorizz" / "ui"
    shared = (ui / "static" / "js" / "safe-markdown.js").read_text()
    assert "html(token) { return esc(" in shared
    for name in ("playground.html", "harness_chat.html"):
        page = (ui / "templates" / name).read_text()
        assert "/static/js/safe-markdown.js" in page, name
        # Raw marked output would run HTML a model or harness wrote.
        assert "marked.parse(" not in page and "new marked.Marked" not in page, name


class _Other(_Recorder):
    name = "other"


def _wait_workflow(client, workflow_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        value = client.get(f"/api/harness-orchestrations/{workflow_id}").json()
        if value["orchestration"]["status"] not in {"queued", "running"}:
            return value["orchestration"]
        assert time.monotonic() < deadline, value
        time.sleep(0.02)


def test_a_finished_workflow_runs_again_with_its_settings(tmp_path: Path):
    first, second = _Recorder(), _Other()
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[first, second],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
        scratch_root=tmp_path / "scratch",
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
            started = client.post(
                "/api/harness-orchestrations",
                json={
                    "kind": "compare",
                    "task": "Price check",
                    "harnesses": ["recorder", "other"],
                    "mcp_access": "none",
                    "max_steps": 12,
                },
            )
            assert started.status_code == 200, started.text
            original = started.json()["orchestration"]
            _wait_workflow(client, original["orchestration_id"])
            scratch = original["task"]["workspace"]
            assert meta.is_scratch_workspace(scratch)

            # The card offers both ways back, with the form's settings.
            page = client.get("/harnesses").text
            assert f'data-rerun-workflow="{original["orchestration_id"]}"' in page
            assert f'data-workflow-setup="{original["orchestration_id"]}"' in page

            again = client.post(
                f"/api/harness-orchestrations/{original['orchestration_id']}/rerun"
            )
            assert again.status_code == 200, again.text
            repeat = again.json()["orchestration"]
            assert repeat["orchestration_id"] != original["orchestration_id"]
            assert [step["harness"] for step in repeat["steps"]] == [
                "recorder",
                "other",
            ]
            assert repeat["task"]["task"] == "Price check"
            assert repeat["task"]["budget"]["max_steps"] == 12
            assert (
                repeat["task"]["metadata"]["rerun_of"] == original["orchestration_id"]
            )
            # A fresh scratch folder, as a blank workspace had the first time.
            assert repeat["task"]["workspace"] != scratch
            assert meta.is_scratch_workspace(repeat["task"]["workspace"])
            _wait_workflow(client, repeat["orchestration_id"])
            assert len(first.tasks) == 2 and first.tasks[-1].task == "Price check"
            assert "Run again from" in client.get("/harnesses").text

            assert (
                client.post("/api/harness-orchestrations/missing/rerun").status_code
                == 404
            )
    finally:
        meta.close()
        provider.close()


def test_rerun_fills_in_an_agent_and_refuses_masked_settings(tmp_path: Path):
    from memorizz.metaharness.models import HarnessOrchestration

    first, second = _Recorder(), _Other()
    meta = MetaHarness(
        adapters=[first, second],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    try:
        store = meta.run_store
        for workflow_id, text in (("wf-ok", "Plan it"), ("wf-masked", "[REDACTED]")):
            store.create_orchestration(
                HarnessOrchestration(
                    orchestration_id=workflow_id,
                    kind="plan",
                    status="failed",
                    task={
                        "task": "Goal",
                        "workspace": str(tmp_path),
                        "run_id": "old",
                        "permissions": {"mcp_access": "none"},
                    },
                    steps=[
                        {
                            "name": "Plan",
                            "harness": "recorder",
                            "instruction": f"{text}\n\nGoal:\nGoal",
                            "run_id": "old-step",
                            "handoff": {"files_changed": []},
                        }
                    ],
                )
            )
        repeat = meta.rerun_orchestration("wf-ok", agent_id="agent-7")
        assert repeat["task"]["agent_id"] == "agent-7"
        assert repeat["task"]["run_id"] != "old"
        assert repeat["task"]["workspace"] == str(tmp_path)
        assert repeat["steps"][0]["instruction"] == "Plan it\n\nGoal:\nGoal"
        assert repeat["steps"][0]["run_id"] is None
        with pytest.raises(ValueError, match="Edit and run"):
            meta.rerun_orchestration("wf-masked")
        with pytest.raises(KeyError):
            meta.rerun_orchestration("nope")
    finally:
        meta.close()


def test_edit_and_run_gets_the_launch_form_settings():
    from memorizz.ui.execution_view import workflow_setup

    setup = workflow_setup(
        "plan",
        {
            "task": "Fix the bug",
            "workspace": "/repo",
            "permissions": {"network": "full", "allowed_tools": ["Read", "Grep"]},
            "budget": {"max_wall_time_seconds": 600.0, "max_steps": 40},
            "metadata": {"execution_backend": "docker"},
        },
        [
            {
                "name": "Implement",
                "harness": "codex",
                "instruction": "Make the change\n\nGoal:\nFix the bug",
                "workspace_mode": "direct",
                "verification": {"command": "pytest -q"},
                "run_id": "r1",
            }
        ],
    )
    assert setup["mode"] == "plan" and setup["network"] == "full"
    assert setup["allowed_tools"] == "Read, Grep"
    assert setup["timeout_seconds"] == 600.0 and setup["execution_backend"] == "docker"
    stage = setup["stages"][0]
    # The goal the launch appended comes off; the form adds it again.
    assert stage["instruction"] == "Make the change"
    assert stage["write"] is True and stage["verification_command"] == "pytest -q"
    compare = workflow_setup(
        "compare", {"task": "x"}, [{"harness": "a"}, {"harness": "b"}]
    )
    assert compare["harnesses"] == ["a", "b"] and "stages" not in compare


def test_harnesses_report_reasoning_subagents_and_their_model(monkeypatch):
    from memorizz.metaharness import adapters

    codex = adapters.CodexHarness(command="codex-for-test")
    codex._run_models["r1"] = "gpt-6-astra"
    events, _ = codex.parse_event("r1", {"type": "thread.started", "thread_id": "t1"})
    assert events[0].data == {"thread_id": "t1", "model": "gpt-6-astra"}
    events, _ = codex.parse_event(
        "r1",
        {
            "type": "item.completed",
            "item": {"id": "i1", "type": "reasoning", "text": "Check the quote"},
        },
    )
    assert events[0].type == HarnessEventType.REASONING
    assert events[0].data["text"] == "Check the quote"
    events, _ = codex.parse_event(
        "r1",
        {
            "type": "item.started",
            "item": {
                "id": "i2",
                "type": "collab_tool_call",
                "tool": "spawn_agent",
                "prompt": "Find the close",
                "status": "in_progress",
            },
        },
    )
    call = events[0].data
    assert events[0].type == HarnessEventType.TOOL_CALL and call["subagent"] is True
    # A hosted web search says when it finished, though not what it searched.
    started, _ = codex.parse_event(
        "r1",
        {
            "type": "item.started",
            "item": {"id": "w1", "type": "web_search", "query": ""},
        },
    )
    finished, _ = codex.parse_event(
        "r1",
        {
            "type": "item.completed",
            "item": {"id": "w1", "type": "web_search", "query": "", "results": []},
        },
    )
    assert started[0].data["status"] == "in_progress"
    assert finished[0].data["status"] == "completed"
    assert call["name"] == "spawn_agent" and call["input"] == {
        "prompt": "Find the close"
    }

    claude = adapters.ClaudeCodeHarness(command="claude-for-test")
    claude.parse_event(
        "r2",
        {
            "type": "system",
            "subtype": "init",
            "session_id": "s",
            "model": "claude-opus-5-5",
        },
    )
    events, _ = claude.parse_event(
        "r2",
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "Search first"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Task",
                        "input": {"description": "Look up"},
                    },
                ]
            },
        },
    )
    assert [e.type for e in events] == [
        HarnessEventType.REASONING,
        HarnessEventType.TOOL_CALL,
    ]
    assert events[0].data == {"text": "Search first", "hidden": False}
    assert events[1].data["subagent"] is True
    # Claude Code often leaves the thinking text out; the step still shows.
    events, _ = claude.parse_event(
        "r2",
        {
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "thinking": ""}]},
        },
    )
    assert (
        events[0].type == HarnessEventType.REASONING
        and events[0].data["hidden"] is True
    )
    events, _ = claude.parse_event(
        "r2",
        {
            "type": "assistant",
            "parent_tool_use_id": "t1",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "WebSearch", "input": {}}
                ]
            },
        },
    )
    assert events[0].data["parent_id"] == "t1" and "subagent" not in events[0].data
    _, updates = claude.parse_event(
        "r2",
        {
            "type": "result",
            "result": "done",
            "usage": {"input_tokens": 4, "output_tokens": 491},
            "modelUsage": {
                "claude-opus-5-5": {
                    "inputTokens": 28270,
                    "outputTokens": 5520,
                    "cacheReadInputTokens": 66367,
                    "cacheCreationInputTokens": 23262,
                    "costUSD": 0.373,
                },
                "claude-haiku-4-5-20251001": {
                    "inputTokens": 43459,
                    "outputTokens": 332,
                    "costUSD": 0.045,
                },
            },
        },
    )
    usage = updates["usage"]
    # The whole run, not the last model call.
    assert usage["model"] == "claude-opus-5-5"
    assert usage["input_tokens"] == 71729 and usage["output_tokens"] == 5852
    assert usage["cache_read_input_tokens"] == 66367
    assert set(usage["models"]) == {"claude-opus-5-5", "claude-haiku-4-5-20251001"}


def test_a_memagent_run_reports_its_steps_as_they_happen():
    from memorizz.metaharness import adapters
    from memorizz.metaharness.models import HarnessTask

    class Agent:
        llm_config = {"provider": "openai", "model": "gpt-5"}
        internet_access_manager = None
        delegates = [
            SimpleNamespace(
                agent_id="researcher", name="Market researcher", persona=None
            )
        ]
        _stream_event_callback = None
        _delegation_event_callback = None

        def set_stream_event_callback(self, callback):
            self._stream_event_callback = callback

        def set_delegation_event_callback(self, callback):
            self._delegation_event_callback = callback

        def run(self, query, **kwargs):
            trace = self._stream_event_callback
            trace(
                {
                    "type": "trace",
                    "trace_kind": "model_call",
                    "span_id": "m1",
                    "iteration": 0,
                }
            )
            trace(
                {
                    "type": "trace",
                    "trace_kind": "model_result",
                    "span_id": "m1",
                    "success": True,
                }
            )
            trace(
                {
                    "type": "trace",
                    "trace_kind": "tool_call",
                    "tool_call_id": "c1",
                    "tool_name": "internet_search",
                    "content": '{"query": "ORCL"}',
                }
            )
            # A long result arrives in pieces; it is recorded once, whole.
            trace(
                {
                    "type": "trace",
                    "trace_kind": "tool_result",
                    "tool_call_id": "c1",
                    "success": True,
                    "content": "3 res",
                    "append": False,
                    "complete": False,
                }
            )
            trace(
                {
                    "type": "trace",
                    "trace_kind": "tool_result",
                    "tool_call_id": "c1",
                    "success": True,
                    "content": "ults",
                    "append": True,
                    "complete": True,
                }
            )
            trace(
                {
                    "type": "trace",
                    "trace_kind": "memory_retrieval",
                    "content": "ignored",
                }
            )
            self._delegation_event_callback(
                {"type": "task_started", "task_id": "d1", "agent_id": "researcher"}
            )
            self._delegation_event_callback(
                {
                    "type": "task_finished",
                    "task_id": "d1",
                    "status": "succeeded",
                    "result": "found it",
                }
            )
            return "ORCL is up"

    agent = Agent()
    seen = []
    outcome = adapters.NativeMemAgentHarness(agent).run(
        HarnessTask(task="Price", workspace="/tmp"),
        workspace=Path("/tmp"),
        context_pack=SimpleNamespace(rendered="", source_ids=[]),
        emit=seen.append,
        cancel_event=SimpleNamespace(is_set=lambda: False),
    )
    kinds = [
        (
            e.type.value,
            e.data.get("phase")
            or e.data.get("name")
            or e.data.get("tool_use_id")
            or e.data.get("status"),
        )
        for e in seen
    ]
    assert kinds == [
        ("status", "running"),
        ("status", "model_call"),
        ("status", "model_result"),
        ("tool_call", "internet_search"),
        ("tool_result", "c1"),
        ("tool_call", "Market researcher"),
        ("tool_result", "d1"),
        ("message", None),
    ]
    assert seen[0].data["model"] == "gpt-5"
    assert seen[3].data["input"] == {"query": "ORCL"}
    assert seen[4].data["content"] == "3 results"
    assert seen[5].data["subagent"] is True and seen[6].data["is_error"] is False
    assert outcome.usage["model"] == "gpt-5"
    # The run's callbacks are put back afterwards.
    assert (
        agent._stream_event_callback is None
        and agent._delegation_event_callback is None
    )


def test_the_page_names_each_runs_model_from_its_events(tmp_path: Path):
    from memorizz.metaharness.models import HarnessRun, HarnessTask
    from memorizz.ui.routers.harnesses import fill_reported_models

    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    task = HarnessTask(task="x", workspace=str(tmp_path))
    run = store.create(
        HarnessRun(run_id=task.run_id, task=task.to_dict(), status="running")
    )
    store.append_event(
        HarnessEvent(
            run.run_id,
            HarnessEventType.STATUS,
            {"session_id": "s", "model": "claude-opus-5-5"},
        )
    )
    store.append_event(
        HarnessEvent(run.run_id, HarnessEventType.MESSAGE, {"text": "model"})
    )
    assert store.event_models([run.run_id]) == {run.run_id: "claude-opus-5-5"}
    rows = [{"run_id": run.run_id, "model": ""}, {"run_id": "other", "model": "pinned"}]
    fill_reported_models(SimpleNamespace(run_store=store), rows)
    assert rows[0]["model"] == "claude-opus-5-5" and rows[0]["model_is_default"] is True
    assert rows[1]["model"] == "pinned"
    store.close()


def test_the_subagents_switch_reaches_claude_code_and_codex(tmp_path: Path):
    from memorizz.metaharness import adapters
    from memorizz.metaharness.models import HarnessTask
    from memorizz.metaharness.requests import harness_task
    from memorizz.ui.execution_view import workflow_setup

    def claude_tools(task):
        harness = adapters.ClaudeCodeHarness(command="claude-for-test")
        harness._cli_flags = frozenset(
            {"--restricted", "--tools", "--strict-mcp-config"}
        )
        command = harness.build_command(task, workspace=tmp_path, prompt="P")
        return command[command.index("--allowedTools") + 1].split(","), command

    def codex_configs(task):
        command = adapters.CodexHarness(command="codex-for-test").build_command(
            task, workspace=tmp_path, prompt="P"
        )
        return [command[i + 1] for i, v in enumerate(command[:-1]) if v == "--config"]

    off = harness_task("Research", str(tmp_path), mcp_access="none")
    on = harness_task(
        "Research", str(tmp_path), mcp_access="none", allow_subagents=True
    )
    assert off.permissions.allow_subagents is False
    # Subagents need no approval of their own.
    assert on.permissions.require_approval is False

    allowed, _ = claude_tools(off)
    assert "Task" not in allowed
    allowed, command = claude_tools(on)
    # Task joins the defaults rather than replacing them.
    assert allowed[:3] == ["Read", "Glob", "Grep"] and "Task" in allowed
    assert "Task" in command[command.index("--tools") + 1].split(",")

    assert "features.multi_agent=false" in codex_configs(off)
    assert "features.multi_agent=true" in codex_configs(on)
    # Codex sub-agents need a saved parent session, so that run isn't ephemeral.
    codex = adapters.CodexHarness(command="codex-for-test")
    assert "--ephemeral" in codex.build_command(off, workspace=tmp_path, prompt="P")
    assert "--ephemeral" not in codex.build_command(on, workspace=tmp_path, prompt="P")

    # Stored runs without the field still load, and Edit and run carries it.
    legacy = HarnessTask.from_dict(
        {"task": "x", "workspace": "/w", "permissions": {"network": "none"}}
    )
    assert legacy.permissions.allow_subagents is False
    setup = workflow_setup(
        "compare", {"task": "x", "permissions": {"allow_subagents": True}}, []
    )
    assert setup["allow_subagents"] is True


def test_codex_sub_agents_are_followed_from_their_session_files(tmp_path: Path):
    from datetime import datetime

    from memorizz.metaharness import adapters

    sessions = tmp_path / "sessions"
    folder = sessions / datetime.now().strftime("%Y/%m/%d")
    folder.mkdir(parents=True)
    parent, child = "parent-thread", "child-thread"

    def line(kind, payload):
        return json.dumps({"type": kind, "payload": payload}) + "\n"

    rows = [
        line(
            "session_meta",
            {
                "id": child,
                "subagent_history_start_ordinal": 2,
                "source": {
                    "subagent": {
                        "thread_spawn": {
                            "parent_thread_id": parent,
                            "agent_path": "/root/oracle_quote",
                            "agent_nickname": "Avicenna",
                        }
                    }
                },
            },
        ),
        # The parent's history the sub-agent was forked with: skipped.
        line("event_msg", {"type": "task_complete", "last_agent_message": "parent"}),
        line("event_msg", {"type": "thread_settings_applied", "thread_id": child}),
        line(
            "event_msg",
            {
                "type": "item_completed",
                "thread_id": child,
                "item": {
                    "type": "Extension",
                    "kind": "web.search",
                    "id": "s1",
                    "query": "",
                },
            },
        ),
        line(
            "event_msg",
            {
                "type": "item_completed",
                "thread_id": child,
                "item": {"type": "Reasoning", "summary_text": []},
            },
        ),
        line(
            "event_msg",
            {
                "type": "item_completed",
                "thread_id": child,
                "item": {
                    "type": "AgentMessage",
                    "content": [{"type": "Text", "text": "ORCL $137.12"}],
                },
            },
        ),
    ]
    rollout = folder / "rollout-2026-09-30T12-39-34-child-thread.jsonl"
    rollout.write_text("".join(rows))
    (folder / "rollout-2026-09-30T12-00-00-other.jsonl").write_text(
        line("session_meta", {"id": "other", "source": "exec"})
    )

    seen = []
    watcher = adapters._CodexSubagentWatcher(
        "r1", seen.append, sessions, time.time() - 60
    )
    watcher.threads.add(parent)
    watcher.poll()
    kinds = [
        (
            e.type.value,
            e.data.get("name")
            or e.data.get("role")
            or ("hidden" if e.data.get("hidden") else ""),
        )
        for e in seen
    ]
    assert kinds == [
        ("tool_call", "spawn_agent"),
        ("tool_call", "web_search"),
        ("reasoning", "hidden"),
        ("message", "assistant"),
    ]
    assert seen[0].data["subagent"] is True
    assert seen[0].data["input"]["description"] == "Avicenna · oracle_quote"
    assert all(e.data.get("parent_id") == child for e in seen[1:])

    # Later lines are read once, as they arrive; the answer closes the sub-agent.
    with rollout.open("a") as handle:
        handle.write(
            line(
                "event_msg",
                {"type": "task_complete", "last_agent_message": "ORCL $137.12"},
            )
        )
    watcher.poll()
    assert seen[-1].type == HarnessEventType.TOOL_RESULT
    assert seen[-1].data == {
        "tool_use_id": child,
        "content": "ORCL $137.12",
        "is_error": False,
    }
    watcher.poll()
    assert len(seen) == 5


def test_codex_sub_agents_survive_format_changes_and_say_when_unreadable(
    tmp_path: Path,
):
    from memorizz.metaharness import adapters

    def line(kind, payload):
        return (
            json.dumps(
                {"timestamp": "2026-10-01T12:00:00Z", "type": kind, "payload": payload}
            )
            + "\n"
        )

    sessions = tmp_path / "sessions"
    # Another folder layout, other casing, and the parent named elsewhere.
    folder = sessions / "by-thread" / "x"
    folder.mkdir(parents=True)
    (folder / "rollout-2026-10-01T12-00-00-kid.jsonl").write_text(
        line(
            "SessionMeta",
            {"id": "kid", "source": {"subagent": {"parent_thread_id": "lead"}}},
        )
        + line(
            "EventMsg",
            {
                "type": "ItemCompleted",
                "thread_id": "kid",
                "item": {"type": "agent_message", "content": [{"text": "Found it"}]},
            },
        )
        + line("EventMsg", {"type": "TaskComplete", "last_agent_message": "Found it"})
    )
    seen = []
    watcher = adapters._CodexSubagentWatcher(
        "r1", seen.append, sessions, time.time() - 60
    )
    watcher.threads.add("lead")
    watcher.saw_spawn("call-1")
    watcher.poll()
    assert [e.type.value for e in seen] == ["tool_call", "message", "tool_result"]
    assert seen[1].data["text"] == "Found it" and seen[1].data["parent_id"] == "kid"
    watcher.stop()
    assert not any(e.data.get("status") == "subagent_details_unavailable" for e in seen)

    # Spawns reported, but no session file to read: the trace says why.
    lost = []
    blind = adapters._CodexSubagentWatcher(
        "r2", lost.append, tmp_path / "missing", time.time()
    )
    blind.saw_spawn("call-1")
    blind.saw_spawn("call-1")  # the same spawn reported twice counts once
    blind.stop()
    assert len(lost) == 1 and lost[0].data["status"] == "subagent_details_unavailable"
    assert "Codex started 1 sub-agent," in lost[0].data["notice"]


def test_each_compared_harness_runs_its_own_model(tmp_path: Path):
    first, second = _Recorder(), _Other()
    meta = MetaHarness(
        adapters=[first, second],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    try:
        started = meta.start_compare(
            {
                "task": "x",
                "workspace": str(tmp_path),
                "model": "shared",
                "permissions": {"mcp_access": "none"},
            },
            ["recorder", "other"],
            models={"other": "model-b", "recorder": " "},
        )
        assert [step["model"] for step in started["steps"]] == [None, "model-b"]
        deadline = time.monotonic() + 5
        while len(first.tasks) + len(second.tasks) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        # A harness without its own pick keeps the shared model.
        assert first.tasks[0].model == "shared" and second.tasks[0].model == "model-b"
        while meta.get_orchestration(started["orchestration_id"])["status"] in {
            "queued",
            "running",
        }:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        again = meta.rerun_orchestration(started["orchestration_id"])
        assert [step["model"] for step in again["steps"]] == [None, "model-b"]
    finally:
        meta.close()


def test_a_memagent_run_can_use_another_model(tmp_path: Path):
    from memorizz.memagent import MemAgentModel
    from memorizz.metaharness import adapters
    from memorizz.ui.execution_view import shape_harness_run, workflow_setup

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    saved = {"provider": "openai", "model": "gpt-5", "context_window_tokens": 128000}
    provider.store_memagent(
        MemAgentModel(agent_id="a1", instruction="x", llm_config=saved)
    )
    # The same provider: only the model changes, and the old window size goes.
    assert adapters._memagent_model_config(provider, "a1", "gpt-5.5") == {
        "provider": "openai",
        "model": "gpt-5.5",
    }
    # "provider/model" switches provider.
    assert adapters._memagent_model_config(provider, "a1", "ollama/qwen2.5:7b") == {
        "provider": "ollama",
        "model": "qwen2.5:7b",
    }
    loaded = {}

    class _Loader:
        @classmethod
        def load(cls, agent_id, **kwargs):
            loaded.update(kwargs)
            raise RuntimeError("stop after load")

    with patch("memorizz.memagent.MemAgent", _Loader):
        adapters.PersistedMemAgentHarness(provider).run(
            HarnessTaskForTest(model="ollama/qwen2.5:7b"),
            workspace=tmp_path,
            context_pack=SimpleNamespace(rendered="", source_ids=[]),
            emit=lambda event: None,
            cancel_event=SimpleNamespace(is_set=lambda: False),
        )
    assert loaded["llm_config"] == {"provider": "ollama", "model": "qwen2.5:7b"}
    # The saved agent is untouched.
    assert provider.retrieve_memagent("a1").llm_config == saved
    provider.close()

    running = shape_harness_run(
        {
            "run_id": "r",
            "status": "running",
            "task": {"task": "x", "harness": "memagent", "model": "gpt-5.5"},
            "result": {},
        }
    )
    assert running["model"] == "gpt-5.5"
    done = shape_harness_run(
        {
            "run_id": "r",
            "status": "succeeded",
            "task": {"task": "x", "harness": "codex", "model": "gpt-6-luna"},
            "result": {"usage": {"model": "gpt-6-astra"}},
        }
    )
    # What the harness reported running wins over what was asked for.
    assert done["model"] == "gpt-6-astra"
    setup = workflow_setup(
        "compare",
        {"task": "x"},
        [{"harness": "codex", "model": "gpt-6-luna"}, {"harness": "pi"}],
    )
    assert setup["harness_models"] == {"codex": "gpt-6-luna"}


def HarnessTaskForTest(**values):
    from memorizz.metaharness.models import HarnessTask

    return HarnessTask(
        task="x", workspace="/tmp", harness="memagent", agent_id="a1", **values
    )


def test_pi_defaults_to_a_provider_you_have_a_key_for(monkeypatch):
    from memorizz.metaharness import adapters
    from memorizz.metaharness.models import HarnessTask

    for name in [key for keys in adapters.PI_PROVIDER_KEYS.values() for key in keys]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    pi = adapters.PiHarness(command="pi-for-test")
    pi._catalog = []
    task = HarnessTask(task="x", workspace="/tmp", permissions={"mcp_access": "none"})
    command = pi.build_command(task, workspace=Path("/tmp"), prompt="P")
    assert command[command.index("--model") + 1] == "openai/gpt-5.5"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    command = pi.build_command(task, workspace=Path("/tmp"), prompt="P")
    assert command[command.index("--model") + 1] == "anthropic/claude-sonnet-5-5"
    # A configured model or provider is never overridden.
    pinned = adapters.PiHarness(command="pi-for-test", default_model="openai/gpt-6-sol")
    command = pinned.build_command(task, workspace=Path("/tmp"), prompt="P")
    assert command[command.index("--model") + 1] == "openai/gpt-6-sol"


def test_openhands_reads_current_events_and_gets_the_run_settings(tmp_path: Path):
    from memorizz.metaharness import adapters
    from memorizz.metaharness.models import HarnessTask

    hands = adapters.OpenHandsHarness(
        command="openhands-for-test", external_isolation=True
    )
    assert hands.probe().metadata["network_modes"] == ["full"]
    task = HarnessTask(
        task="x",
        workspace=str(tmp_path),
        harness="openhands",
        model="anthropic/claude-haiku-4-5",
        permissions={"network": "full", "mcp_access": "none"},
    )
    env = hands._child_environment(task)
    assert env["LLM_MODEL"] == "anthropic/claude-haiku-4-5"
    # Read-only runs get a read-only workspace mount from the wrapper.
    assert env["MEMORIZZ_WORKSPACE_WRITABLE"] == "0"

    events, _ = hands.parse_event(
        "r",
        {
            "kind": "ActionEvent",
            "tool_name": "terminal",
            "tool_call_id": "t1",
            "action": {"kind": "x", "command": "cat README.md"},
            "reasoning_content": "Look first",
        },
    )
    assert [e.type for e in events] == [
        HarnessEventType.REASONING,
        HarnessEventType.COMMAND,
    ]
    assert events[1].data["command"] == "cat README.md"
    events, _ = hands.parse_event(
        "r",
        {
            "kind": "ObservationEvent",
            "tool_call_id": "t1",
            "observation": {"content": [{"type": "text", "text": "# Tiny demo"}]},
        },
    )
    assert (
        events[0].type == HarnessEventType.TOOL_RESULT
        and events[0].data["content"] == "# Tiny demo"
    )
    events, updates = hands.parse_event(
        "r",
        {
            "kind": "MessageEvent",
            "source": "agent",
            "llm_message": {
                "content": [{"type": "text", "text": "A small demo note."}]
            },
        },
    )
    assert (
        updates["final_response"] == "A small demo note."
        and events[0].type == HarnessEventType.MESSAGE
    )
    _, updates = hands.parse_event(
        "r", {"kind": "ConversationErrorEvent", "detail": "model not found"}
    )
    assert updates["error"] == "model not found"
    # The older "type" events still parse.
    _, updates = hands.parse_event("r", {"type": "final", "message": "done"})
    assert updates["final_response"] == "done"


def test_hermes_run_homes_live_under_its_profiles(tmp_path: Path, monkeypatch):
    from memorizz.metaharness import adapters

    root = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    assert adapters._hermes_profiles_root() is None  # not installed there
    (root / "hermes-agent").mkdir(parents=True)
    profiles = adapters._hermes_profiles_root()
    assert profiles == root / "profiles" and profiles.is_dir()


def test_a_harness_delegate_shares_its_parent_runs_workspace_and_approval(
    tmp_path: Path,
):
    from memorizz import MemAgent
    from memorizz.metaharness.models import HarnessStatus

    ran = []

    class _Service:
        def run(self, task):
            ran.append(task)
            return SimpleNamespace(
                status=HarnessStatus.SUCCEEDED, final_response="ok", run_id=task.run_id
            )

    delegate = SimpleNamespace(
        meta_harness=_Service(),
        harness_config={"model": "gpt-6-luna"},
        default_harness="codex",
        agent_id="delegate-1",
        meta_harness_mode="runtime",
        _resolve_execution_state=lambda memory_id, thread_id: (memory_id, thread_id),
        _record_interaction=lambda *a, **k: None,
    )
    started = []
    parent = {
        "run_id": "parent-run",
        "workspace": str(tmp_path),
        "permissions": {
            "network": "full",
            "mcp_access": "none",
            "allow_subagents": True,
            "workspace_mode": "direct",
        },
    }
    MemAgent.run_on_harness(
        delegate, "Review mathlib.py", parent_run=parent, on_start=started.append
    )
    task = ran[0]
    assert (
        task.workspace == str(tmp_path)
        and task.harness == "codex"
        and task.model == "gpt-6-luna"
    )
    # What the parent was approved for, edits included, without another approval.
    assert (
        task.permissions.network == "full" and task.permissions.allow_subagents is True
    )
    assert task.permissions.workspace_mode == "direct"
    assert task.permissions.require_approval is False
    assert task.metadata["parent_run_id"] == "parent-run"
    assert started == [task.run_id]

    # A delegate set to read-only stays read-only; one set to use the web
    # can't when the parent wasn't approved for it.
    ran.clear()
    delegate.harness_config = {
        "permissions": {"workspace_mode": "read_only", "network": "full"}
    }
    calm = {**parent, "permissions": {**parent["permissions"], "network": "none"}}
    MemAgent.run_on_harness(delegate, "Look only", parent_run=calm)
    assert ran[0].permissions.workspace_mode == "read_only"
    assert ran[0].permissions.network == "none"

    # A parent approved read-only keeps its delegates read-only.
    ran.clear()
    delegate.harness_config = {}
    MemAgent.run_on_harness(
        delegate,
        "x",
        parent_run={
            **parent,
            "permissions": {"network": "none", "workspace_mode": "read_only"},
        },
    )
    assert ran[0].permissions.workspace_mode == "read_only"

    # A playground grant works like a parent run, without a parent run ID.
    ran.clear()
    grant = {
        "grant_id": "grant-1",
        "workspace": str(tmp_path),
        "permissions": {"network": "full", "workspace_mode": "read_only"},
    }
    MemAgent.run_on_harness(delegate, "Research", parent_run=grant)
    assert (
        ran[0].permissions.network == "full"
        and ran[0].permissions.require_approval is False
    )
    assert (
        ran[0].metadata["access_grant_id"] == "grant-1"
        and "parent_run_id" not in ran[0].metadata
    )

    # Without a parent run or grant nothing is granted.
    ran.clear()
    MemAgent.run_on_harness(delegate, "x", workspace=str(tmp_path))
    assert (
        ran[0].permissions.network == "none" and "parent_run_id" not in ran[0].metadata
    )


def test_canceling_a_turn_cancels_its_delegates_harness_run(tmp_path: Path):
    import threading

    from memorizz import MemAgent
    from memorizz.metaharness.models import HarnessStatus
    from memorizz.streaming import (
        CancellationToken,
        StreamCancelled,
        current_cancellation,
    )

    canceled, started = [], threading.Event()

    class _Service:
        def run(self, task):
            started.set()
            for _ in range(200):
                if task.run_id in canceled:
                    return SimpleNamespace(
                        status=HarnessStatus.CANCELED,
                        final_response="",
                        run_id=task.run_id,
                    )
                threading.Event().wait(0.01)
            return SimpleNamespace(
                status=HarnessStatus.SUCCEEDED, final_response="ok", run_id=task.run_id
            )

        def cancel(self, run_id, before_start=False):
            canceled.append(run_id)

    delegate = SimpleNamespace(
        meta_harness=_Service(),
        harness_config={},
        default_harness="codex",
        agent_id="delegate-1",
        meta_harness_mode="runtime",
        _resolve_execution_state=lambda memory_id, thread_id: (memory_id, thread_id),
        _record_interaction=lambda *a, **k: None,
    )
    token = CancellationToken()
    results = []

    def work():
        scope = current_cancellation.set(token)
        try:
            results.append(
                MemAgent.run_on_harness(delegate, "Long job", workspace=str(tmp_path))
            )
        finally:
            current_cancellation.reset(scope)

    worker = threading.Thread(target=work)
    worker.start()
    assert started.wait(2)
    token.cancel()
    worker.join(2)
    assert len(canceled) == 1 and results[0].status == HarnessStatus.CANCELED

    # A turn already canceled starts no harness run at all.
    scope = current_cancellation.set(token)
    try:
        with pytest.raises(StreamCancelled):
            MemAgent.run_on_harness(delegate, "Too late", workspace=str(tmp_path))
    finally:
        current_cancellation.reset(scope)


def test_a_memagent_harness_run_stops_when_canceled(tmp_path: Path):
    import threading

    from memorizz.metaharness import adapters
    from memorizz.metaharness.models import HarnessContextPack, HarnessTask
    from memorizz.streaming import check_cancelled

    steps = []

    class _Agent:
        agent_id = "coordinator"
        model = None

        def run(self, query, **kwargs):
            for step in range(200):
                check_cancelled()  # MemAgent checks between model and tool calls
                steps.append(step)
                threading.Event().wait(0.01)
            return "finished"

        def get_last_run_usage(self):
            return {
                "calls": 1,
                "input_tokens": 100,
                "output_tokens": 10,
                "cost_usd": 0.01,
            }

    cancel_event = threading.Event()
    threading.Timer(0.1, cancel_event.set).start()
    outcome = adapters.NativeMemAgentHarness(_Agent()).run(
        HarnessTask(task="Coordinate", workspace=str(tmp_path), harness="memagent"),
        workspace=tmp_path,
        context_pack=HarnessContextPack(query="Coordinate"),
        emit=lambda event: None,
        cancel_event=cancel_event,
    )
    assert outcome.error_code == "canceled"
    assert 0 < len(steps) < 200
    # What it spent before stopping still counts.
    assert outcome.cost_usd == 0.01 and outcome.usage["calls"] == 1


def test_the_trace_names_harness_delegates_and_links_their_runs():
    from memorizz.metaharness import adapters

    agent = SimpleNamespace(
        delegates=[
            SimpleNamespace(
                agent_id="d1",
                name="Codex reviewer",
                persona=None,
                meta_harness_mode="runtime",
                default_harness="codex",
            )
        ],
        _stream_event_callback=None,
        _delegation_event_callback=None,
    )
    agent.set_stream_event_callback = lambda cb: setattr(
        agent, "_stream_event_callback", cb
    )
    agent.set_delegation_event_callback = lambda cb: setattr(
        agent, "_delegation_event_callback", cb
    )
    seen = []
    with adapters._live_trace(agent, "run-1", seen.append):
        agent._delegation_event_callback(
            {"type": "task_started", "task_id": "t1", "agent_id": "d1"}
        )
        agent._delegation_event_callback(
            {
                "type": "task_progress",
                "task_id": "t1",
                "harness": "codex",
                "harness_run_id": "child-run",
            }
        )
        agent._delegation_event_callback(
            {
                "type": "task_finished",
                "task_id": "t1",
                "status": "completed",
                "result": "2 bugs",
            }
        )
    assert seen[0].data["name"] == "Codex reviewer (codex)"
    assert seen[1].data["id"] == "t1" and seen[1].data["harness_run_id"] == "child-run"
    assert (
        seen[2].type == HarnessEventType.TOOL_RESULT
        and seen[2].data["is_error"] is False
    )


def test_the_agent_picker_names_the_harnesses_a_coordinator_hands_work_to():
    from memorizz.ui.routers import harnesses as routes

    agents = [
        {
            "agent_id": "desk",
            "name": "Research desk",
            "delegates": ["cx", "cc", "helper", "gone"],
        },
        {
            "agent_id": "cx",
            "name": "Codex researcher",
            "meta_harness_mode": "runtime",
            "default_harness": "codex",
        },
        {
            "agent_id": "cc",
            "name": "Claude Code researcher",
            "meta_harness_mode": "runtime",
            "default_harness": "claude-code",
        },
        {"agent_id": "helper", "name": "Note taker"},
        {
            "agent_id": "off",
            "name": "Paused",
            "delegates": ["cx"],
            "delegation_config": {"enabled": False},
        },
        # Saved agents can come back as models rather than dicts.
        SimpleNamespace(
            agent_id="crew", name="Crew", delegates=["cx"], delegation_config=None
        ),
    ]

    from memorizz.metaharness import catalog

    assert catalog.agent_teams(agents) == {
        "desk": [
            "Codex researcher (codex)",
            "Claude Code researcher (claude-code)",
            "Note taker",
        ],
        "crew": ["Codex researcher (codex)"],
    }
    page = (
        Path(routes.__file__).parents[1] / "templates" / "harnesses.html"
    ).read_text()
    assert 'data-team="{{ team|tojson|forceescape }}"' in page
    assert "which hands parts of the task to" in page


def test_finished_runs_and_workflows_can_be_deleted(tmp_path: Path, monkeypatch):
    from memorizz.metaharness.models import HarnessRun

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[_Recorder(), _Other()],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
        scratch_root=tmp_path / "scratch",
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
            started = []
            for text in ("Coordinate the review", "Summarise the notes"):
                response = client.post(
                    "/api/harness-runs",
                    json={"task": text, "harness": "recorder", "mcp_access": "none"},
                )
                assert response.status_code == 200, response.text
                started.append(
                    _wait(client, response.json()["run"]["run_id"])["run_id"]
                )
            parent, other = started
            # A delegate's run made for the first, and a run still working.
            meta.run_store.create(
                HarnessRun(
                    run_id="delegate-run",
                    task={"task": "part", "metadata": {"parent_run_id": parent}},
                    status="succeeded",
                    harness="other",
                )
            )
            meta.run_store.create(
                HarnessRun(
                    run_id="live-run",
                    task={"task": "busy"},
                    status="running",
                    harness="other",
                )
            )
            workflow = client.post(
                "/api/harness-orchestrations",
                json={
                    "kind": "compare",
                    "task": "Price check",
                    "harnesses": ["recorder", "other"],
                    "mcp_access": "none",
                },
            ).json()["orchestration"]
            workflow_id = workflow["orchestration_id"]
            steps = [
                step["run_id"] for step in _wait_workflow(client, workflow_id)["steps"]
            ]
            assert meta.events(parent)

            page = client.get("/harnesses").text
            assert 'id="hx-delete-picked"' in page and 'id="hx-pick-all"' in page
            assert f'data-delete-workflow="{workflow_id}"' in page
            assert 'data-delegate-runs="1"' in page  # the parent's detail pane

            # One run: its events and its delegate's run go with it.
            deleted = client.delete(f"/api/harness-runs/{parent}")
            assert deleted.status_code == 200, deleted.text
            assert deleted.json()["deleted"] == [parent]
            assert deleted.json()["delegates_deleted"] == ["delegate-run"]
            assert client.get(f"/api/harness-runs/{parent}").status_code == 404
            assert meta.get_run("delegate-run") is None
            assert meta.events(parent) == []

            busy = client.delete("/api/harness-runs/live-run")
            assert busy.status_code == 400 and "Cancel it" in busy.json()["detail"]
            stage = client.delete(f"/api/harness-runs/{steps[0]}")
            assert (
                stage.status_code == 400
                and "Delete the workflow" in stage.json()["detail"]
            )
            assert client.delete("/api/harness-runs/missing").status_code == 404

            # Several at once: what can't go is kept, with the reason.
            bulk = client.post(
                "/api/harness-runs/delete",
                json={"run_ids": [other, "live-run", steps[1], "missing"]},
            )
            assert bulk.status_code == 200, bulk.text
            assert bulk.json()["deleted"] == [other]
            assert {item["run_id"]: item["reason"] for item in bulk.json()["kept"]} == {
                "live-run": "active",
                steps[1]: "workflow",
                "missing": "not_found",
            }
            assert client.post("/api/harness-runs/delete", json={}).status_code == 400

            # A workflow goes with all its runs, and the scratch folder made
            # for its blank workspace.
            scratch = workflow["task"]["workspace"]
            assert meta.is_scratch_workspace(scratch) and Path(scratch).is_dir()
            gone = client.delete(f"/api/harness-orchestrations/{workflow_id}")
            assert gone.status_code == 200, gone.text
            assert sorted(gone.json()["deleted"]) == sorted(steps)
            assert gone.json()["scratch_removed"] == [str(Path(scratch).resolve())]
            assert not Path(scratch).exists()
            assert (
                client.get(f"/api/harness-orchestrations/{workflow_id}").status_code
                == 404
            )
            assert all(meta.get_run(run_id) is None for run_id in steps)
            assert [run["run_id"] for run in meta.list_runs()] == ["live-run"]
            assert (
                client.delete(f"/api/harness-orchestrations/{workflow_id}").status_code
                == 404
            )

            # A read-only UI offers no delete and refuses one.
            monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "1")
            locked = TestClient(create_app(), follow_redirects=False)
            locked_page = locked.get("/harnesses").text
            assert 'id="hx-delete-picked"' not in locked_page
            assert "data-delete-workflow=" not in locked_page
            assert locked.delete("/api/harness-runs/live-run").status_code == 403
    finally:
        meta.close()
        provider.close()


def test_a_running_workflow_is_not_deleted(tmp_path: Path):
    from memorizz.metaharness.models import HarnessOrchestration

    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    meta = MetaHarness(
        adapters=[_Recorder()],
        run_store=store,
        approval_store=SQLiteApprovalStore(tmp_path / "a.sqlite3"),
    )
    try:
        store.create_orchestration(
            HarnessOrchestration(
                orchestration_id="wf",
                kind="plan",
                status="running",
                task={"task": "x"},
                steps=[{"name": "one", "harness": "recorder"}],
                heartbeat_at="2999-01-01T00:00:00+00:00",
            )
        )
        with pytest.raises(ValueError, match="still working"):
            meta.delete_orchestration("wf")
        assert store.get_orchestration("wf") is not None
    finally:
        meta.close()


def test_delegates_that_edit_share_the_parent_lease_and_take_turns(tmp_path: Path):
    import threading

    from memorizz.metaharness.models import HarnessRun, HarnessTask

    spans, lock = [], threading.Lock()

    class _Editor(_Recorder):
        name = "editor"

        def run(self, task, *, workspace, context_pack, emit, cancel_event):
            began = time.monotonic()
            time.sleep(0.15)
            with lock:
                spans.append((began, time.monotonic()))
            return AdapterOutcome(final_response="edited", exit_code=0)

    workspace = tmp_path / "project"
    workspace.mkdir()
    provider, meta, _ = _app(tmp_path, _Editor())
    try:
        folder = str(workspace.resolve())
        for run_id, mode in (
            ("parent-run", "direct"),
            ("other-parent", "direct"),
            ("calm-parent", "read_only"),
        ):
            meta.run_store.create(
                HarnessRun(
                    run_id=run_id,
                    task={
                        "task": "lead",
                        "workspace": folder,
                        "permissions": {"workspace_mode": mode},
                    },
                    status="running",
                    harness="memagent",
                )
            )
        assert meta.run_store.acquire_workspace(folder, "parent-run")

        def edit(metadata):
            return meta.run(
                HarnessTask(
                    task="Fix the bug",
                    workspace=str(workspace),
                    harness="editor",
                    permissions={
                        "workspace_mode": "direct",
                        "require_approval": False,
                        "mcp_access": "none",
                    },
                    metadata=metadata,
                )
            )

        # Under its parent's lease a delegate may edit without asking again.
        assert edit({"parent_run_id": "parent-run"}).status == HarnessStatus.SUCCEEDED
        # Not covered: no parent, or a parent approved only to read.
        assert edit({}).status == HarnessStatus.PENDING_APPROVAL
        assert (
            edit({"parent_run_id": "calm-parent"}).status
            == HarnessStatus.PENDING_APPROVAL
        )
        # Covered, but another run holds the workspace.
        busy = edit({"parent_run_id": "other-parent"})
        assert (
            busy.status == HarnessStatus.FAILED and busy.error_code == "workspace_busy"
        )

        # Two delegates editing at once take turns instead of failing.
        spans.clear()
        results = []
        workers = [
            threading.Thread(
                target=lambda: results.append(edit({"parent_run_id": "parent-run"}))
            )
            for _ in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(5)
        assert [r.status for r in results] == [HarnessStatus.SUCCEEDED] * 2
        first, second = sorted(spans)
        assert second[0] >= first[1]
        # The parent's lease is still the parent's.
        assert meta.run_store.workspace_holder(str(workspace.resolve())) == "parent-run"
    finally:
        meta.close()
        provider.close()


def test_a_playground_grant_lets_harness_delegates_use_approved_access(tmp_path: Path):
    from datetime import datetime, timedelta, timezone

    from memorizz.metaharness.models import HarnessTask

    workspace = tmp_path / "project"
    workspace.mkdir()
    provider, meta, _ = _app(tmp_path, _Recorder())
    try:
        grant = meta.grant_delegate_access(
            agent_id="desk", approver_id="Ada", workspace=str(workspace), network="full"
        )
        assert grant["workspace"] == str(workspace.resolve())
        assert grant["permissions"]["network"] == "full"
        assert grant["permissions"]["workspace_mode"] == "read_only"
        assert grant["approver_id"] == "Ada"
        # Recorded as a decided approval.
        record = meta.approval_store.get(grant["grant_id"])
        assert (
            record.tool_name == "metaharness.delegate_access"
            and record.status.value == "approved"
        )
        assert (
            meta.delegate_access(grant["grant_id"], agent_id="desk")["workspace"]
            == grant["workspace"]
        )
        assert meta.delegate_access(grant["grant_id"], agent_id="someone-else") is None
        assert meta.delegate_access("nope", agent_id="desk") is None

        def research(metadata, network="full"):
            return meta.run(
                HarnessTask(
                    task="Look it up",
                    workspace=str(workspace),
                    harness="recorder",
                    permissions={
                        "network": network,
                        "require_approval": False,
                        "mcp_access": "none",
                    },
                    metadata=metadata,
                )
            )

        assert (
            research({"access_grant_id": grant["grant_id"]}).status
            == HarnessStatus.SUCCEEDED
        )

        # A blank workspace becomes a fresh scratch folder.
        scratch = meta.grant_delegate_access(agent_id="desk", approver_id="Ada")
        assert meta.is_scratch_workspace(scratch["workspace"])
        # Folders outside the allowed roots are refused.
        with pytest.raises(Exception, match="outside the configured allowed roots"):
            meta.grant_delegate_access(
                agent_id="desk", approver_id="Ada", workspace=str(Path.home())
            )

        # An expired grant no longer counts.
        with patch(
            "memorizz.approval._utcnow",
            return_value=datetime.now(timezone.utc) + timedelta(days=2),
        ):
            assert meta.delegate_access(grant["grant_id"], agent_id="desk") is None
    finally:
        meta.close()
        provider.close()


def test_a_coordinators_cost_includes_its_delegates_runs():
    from memorizz.ui.execution_view import build_harness_monitor

    def run(run_id, cost, parent=None, basis=None, harness="codex"):
        usage = {"cost_basis": basis} if basis else {}
        return {
            "run_id": run_id,
            "status": "succeeded",
            "harness": harness,
            "task": {
                "task": run_id,
                "metadata": {"parent_run_id": parent} if parent else {},
            },
            "result": {"status": "succeeded", "cost_usd": cost, "usage": usage},
        }

    runs = [
        run("lead", 0.05, harness="memagent", basis="list_rate"),
        run("codex-part", 1.40, parent="lead", basis="list_rate_estimate"),
        run("claude-part", 0.15, parent="lead"),
        run("nested", 0.10, parent="claude-part"),
        run("local-part", None, parent="lead"),
        run("alone", 0.02),
    ]
    monitor = build_harness_monitor([], runs, [])
    rows = {row["run_id"]: row for row in monitor["runs"]}
    lead = rows["lead"]
    assert lead["delegate_runs"] == 4 and lead["delegate_unpriced"] == 1
    assert round(lead["delegate_cost_usd"], 2) == 1.65
    assert round(lead["total_cost_usd"], 2) == 1.70
    assert lead["total_cost_estimated"] is True  # Codex's share is an estimate
    assert rows["claude-part"]["delegate_runs"] == 1
    assert (
        rows["alone"]["total_cost_usd"] == 0.02 and rows["alone"]["delegate_runs"] == 0
    )
    # The tape's spend counts each run once.
    assert round(monitor["spend"], 2) == 1.72


def test_a_coordinators_planning_and_combining_calls_are_counted():
    from memorizz.memagent import MemAgent
    from memorizz.multi_agent_orchestrator import MultiAgentOrchestrator
    from tests.mocks.mock_providers import MockLLMProvider, MockMemoryProvider

    model = MockLLMProvider(
        responses=["Combined"], provider="anthropic", model="claude-sonnet-5-5"
    )
    model.last_usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 1200,
    }
    root = MemAgent(model=model, memory_provider=MockMemoryProvider(), agent_id="root")
    orchestrator = MultiAgentOrchestrator(
        root, [MemAgent(memory_provider=MockMemoryProvider(), agent_id="d1")]
    )
    orchestrator._coordinator_memory_context = lambda *a, **k: ("", {})
    root._run_usage = {}
    orchestrator._consolidate_results(
        "q",
        [
            {
                "description": "part",
                "assigned_agent_id": "d1",
                "result": "r",
                "status": "completed",
            }
        ],
    )
    usage = root.get_last_run_usage()
    assert (
        usage["calls"] == 1
        and usage["input_tokens"] == 1000
        and usage["output_tokens"] == 200
    )
    # Priced at Sonnet 5.5's list rates: 1000 x $2/M + 200 x $10/M.
    assert usage["cost_usd"] == 0.004 and not usage.get("unpriced_calls")


def test_a_memagent_run_reports_its_priced_cost(tmp_path: Path):
    import threading

    from memorizz.metaharness import adapters
    from memorizz.metaharness.models import HarnessContextPack, HarnessTask

    class _Agent:
        agent_id = "coordinator"
        model = None
        usage = {}

        def run(self, query, **kwargs):
            return "done"

        def get_last_run_usage(self):
            return dict(self.usage)

    def outcome(usage):
        agent = _Agent()
        agent.usage = usage
        return adapters.NativeMemAgentHarness(agent).run(
            HarnessTask(task="Coordinate", workspace=str(tmp_path), harness="memagent"),
            workspace=tmp_path,
            context_pack=HarnessContextPack(query="Coordinate"),
            emit=lambda event: None,
            cancel_event=threading.Event(),
        )

    priced = outcome(
        {"calls": 2, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.12}
    )
    assert priced.cost_usd == 0.12 and priced.usage["cost_basis"] == "list_rate"
    # One call that couldn't be priced (a local model, say) leaves it unpriced.
    partial = outcome({"calls": 2, "cost_usd": 0.12, "unpriced_calls": 1})
    assert partial.cost_usd is None and "cost_basis" not in partial.usage


def test_scratch_folders_still_in_use_and_your_own_folders_are_kept(tmp_path: Path):
    recorder = _Recorder()
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[recorder],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
        scratch_root=tmp_path / "scratch",
    )
    own = tmp_path / "project"
    own.mkdir()
    try:
        shared = meta.scratch_workspace()

        def run(workspace):
            return meta.run(
                {
                    "task": "x",
                    "workspace": workspace,
                    "harness": "recorder",
                    "permissions": {"mcp_access": "none"},
                }
            ).run_id

        first, second, mine = run(shared), run(shared), run(str(own))
        # Another run still uses the scratch folder: it stays.
        assert meta.delete_runs([first])["scratch_removed"] == []
        assert Path(shared).is_dir()
        # The last run using it goes: so does the folder. Your own folder stays.
        assert meta.delete_runs([second, mine])["scratch_removed"] == [
            str(Path(shared).resolve())
        ]
        assert not Path(shared).exists() and own.is_dir()
        # Keeping it is a choice.
        kept = meta.scratch_workspace()
        assert (
            meta.delete_runs([run(kept)], remove_scratch=False)["scratch_removed"] == []
        )
        assert Path(kept).is_dir()
        # A delegate grant still using a folder keeps it.
        granted = meta.grant_delegate_access(agent_id="desk", approver_id="Ada")
        assert meta.delete_runs([run(granted["workspace"])])["scratch_removed"] == []
        assert Path(granted["workspace"]).is_dir()
        # The scratch root itself is never removed.
        assert meta._remove_unused_scratch([str(tmp_path / "scratch")]) == []
    finally:
        meta.close()
        provider.close()


def test_a_conversation_can_be_deleted_from_its_page(tmp_path: Path):
    recorder = _Recorder()
    provider, meta, values = _app(tmp_path, recorder)
    workspace = tmp_path / "project"
    workspace.mkdir()
    try:
        with patch.dict(state._state, values):
            client = TestClient(create_app(), follow_redirects=False)
            first = client.post(
                "/api/harness-runs",
                json={
                    "task": "Start",
                    "harness": "recorder",
                    "workspace": str(workspace),
                    "mcp_access": "none",
                },
            ).json()["run"]["run_id"]
            _wait(client, first)
            conversation = f"hxc-{first}"
            turn = client.post(
                f"/api/harness-conversations/{conversation}/turns",
                json={
                    "task": "Next",
                    "harness": "recorder",
                    "workspace": str(workspace),
                    "mcp_access": "none",
                },
            )
            assert turn.status_code == 200, turn.text
            _wait(client, turn.json()["run"]["run_id"])
            page = client.get(f"/harnesses/chat?from={first}").text
            assert 'id="hxc-delete"' in page and "page-dialogs.js" in page

            gone = client.delete(f"/api/harness-conversations/{conversation}")
            assert gone.status_code == 200, gone.text
            assert len(gone.json()["deleted"]) == 2
            assert (
                client.get(f"/api/harness-conversations/{conversation}").status_code
                == 404
            )
            assert (
                client.delete(f"/api/harness-conversations/{conversation}").status_code
                == 404
            )
    finally:
        meta.close()
        provider.close()
