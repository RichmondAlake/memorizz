"""Staged plans and harness comparisons driven in the background."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.metaharness import (
    AgentHarness,
    HarnessCapabilities,
    HarnessEvent,
    HarnessEventType,
    HarnessStatus,
    HarnessTask,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness.base import AdapterOutcome


class _Recording(AgentHarness):
    """Records every task it runs; optional failure or blocking."""

    def __init__(self, name, *, fail=False, block=False, cost=0.01):
        self.name = name
        self.fail = fail
        self.block = block
        self.cost = cost
        self.tasks = []
        self.context_packs = []

    def probe(self):
        return HarnessCapabilities(
            name=self.name, available=True, mcp=True, usage_reporting=True
        )

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        self.tasks.append(task)
        self.context_packs.append(context_pack)
        emit(HarnessEvent(task.run_id, HarnessEventType.MESSAGE, {"text": "working"}))
        if self.block:
            deadline = time.monotonic() + 5
            while not cancel_event.wait(0.02) and time.monotonic() < deadline:
                pass
            if cancel_event.is_set():
                return AdapterOutcome(error_code="canceled", error="canceled")
        if self.fail:
            return AdapterOutcome(error_code="harness_failed", error="it broke")
        if task.writes_workspace:
            (workspace / f"{self.name}.txt").write_text("done\n", encoding="utf-8")
        return AdapterOutcome(
            final_response=f"{self.name} finished {task.context.get('harness_stage')}",
            usage={"input_tokens": 10, "output_tokens": 2},
            cost_usd=self.cost,
            exit_code=0,
        )


def _service(tmp_path: Path, *adapters) -> MetaHarness:
    return MetaHarness(
        adapters=list(adapters),
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )


def _workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


def _wait(service, orchestration_id, until, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        value = service.get_orchestration(orchestration_id)
        if until(value):
            return value
        assert time.monotonic() < deadline, value
        time.sleep(0.02)


def _finished(value):
    return HarnessStatus(value["status"]).terminal


@pytest.mark.unit
def test_plan_runs_stages_in_order_with_prior_results(tmp_path: Path) -> None:
    planner, reviewer = _Recording("planner"), _Recording("reviewer")
    service = _service(tmp_path, planner, reviewer)
    try:
        started = service.start_plan(
            {"task": "Fix the parser", "workspace": str(_workspace(tmp_path))},
            [
                {"name": "Plan", "harness": "planner", "instruction": "Write a plan"},
                {"name": "Review", "harness": "reviewer"},
            ],
        )
        assert started["kind"] == "plan"
        done = _wait(service, started["orchestration_id"], _finished)
        assert done["status"] == "succeeded", done
        assert [step["run_id"] is not None for step in done["steps"]] == [True, True]
        assert done["summary"]["succeeded"] == 2
        assert done["summary"]["cost_usd"] == pytest.approx(0.02)

        plan_task, review_task = planner.tasks[0], reviewer.tasks[0]
        assert plan_task.task == "Write a plan"
        assert review_task.task == "Fix the parser"  # no instruction: the goal
        (handoff,) = review_task.context["prior_stage_results"]
        assert handoff["stage"] == "Plan" and handoff["harness"] == "planner"
        assert handoff["status"] == "succeeded"
        assert handoff["response"] == "planner finished Plan"
        assert handoff["run_id"] == done["steps"][0]["run_id"]
        assert plan_task.context["prior_stage_results"] == []
        # Both stages retrieve memory for the goal and share one snapshot.
        assert plan_task.context["memory_query"] == "Fix the parser"
        assert reviewer.context_packs[0].metadata.get("shared_context_reused")
        tags = review_task.metadata["orchestration"]
        assert tags["id"] == started["orchestration_id"]
        assert (tags["kind"], tags["step"], tags["steps"]) == ("plan", 1, 2)
    finally:
        service.close()


@pytest.mark.unit
def test_plan_edit_stage_waits_for_approval_then_continues(tmp_path: Path) -> None:
    worker = _Recording("worker")
    service = _service(tmp_path, worker)
    workspace = _workspace(tmp_path)
    try:
        started = service.start_plan(
            {"task": "Implement it", "workspace": str(workspace)},
            [
                {"name": "Plan", "harness": "worker"},
                {"name": "Implement", "harness": "worker", "workspace_mode": "direct"},
                {"name": "Review", "harness": "worker"},
            ],
        )
        oid = started["orchestration_id"]
        waiting = _wait(service, oid, lambda v: v["status"] == "pending_approval")
        assert waiting["current_step"] == 1
        # Read-only stages never asked for approval; only the edit stage did.
        approvals = service.list_approvals(status="pending")
        assert len(approvals) == 1
        assert approvals[0]["arguments"]["run_id"] == waiting["steps"][1]["run_id"]

        proposal_id = approvals[0]["proposal_id"]
        service.approve(proposal_id, approver_id="operator@example.com")
        service.resume_approval_start(proposal_id)
        done = _wait(service, oid, _finished)
        assert done["status"] == "succeeded", done
        assert (workspace / "worker.txt").exists()
        assert [task.permissions.workspace_mode for task in worker.tasks] == [
            "read_only",
            "direct",
            "read_only",
        ]
    finally:
        service.close()


@pytest.mark.unit
def test_plan_stops_at_the_first_failed_stage(tmp_path: Path) -> None:
    broken, never = _Recording("broken", fail=True), _Recording("never")
    service = _service(tmp_path, broken, never)
    try:
        started = service.start_plan(
            {"task": "Goal", "workspace": str(_workspace(tmp_path))},
            [
                {"name": "Investigate", "harness": "broken"},
                {"name": "Review", "harness": "never"},
            ],
        )
        done = _wait(service, started["orchestration_id"], _finished)
        assert done["status"] == "failed"
        assert done["error_code"] == "harness_failed"
        assert "Stage 1 of 2 (Investigate) on broken: it broke." in done["error"]
        assert "Later stages did not run." in done["error"]
        assert done["steps"][1]["run_id"] is None
        assert never.tasks == []
    finally:
        service.close()


@pytest.mark.unit
def test_cancel_stops_the_running_stage_and_starts_no_more(tmp_path: Path) -> None:
    slow, never = _Recording("slow", block=True), _Recording("never")
    service = _service(tmp_path, slow, never)
    try:
        started = service.start_plan(
            {"task": "Goal", "workspace": str(_workspace(tmp_path))},
            [{"name": "One", "harness": "slow"}, {"name": "Two", "harness": "never"}],
        )
        oid = started["orchestration_id"]
        _wait(
            service, oid, lambda v: v["steps"][0]["run_id"] and v["status"] == "running"
        )
        assert service.cancel_orchestration(oid)["status"] == "canceling"
        done = _wait(service, oid, _finished)
        assert done["status"] == "canceled"
        assert service.get_run(done["steps"][0]["run_id"])["status"] == "canceled"
        assert done["steps"][1]["run_id"] is None and never.tasks == []
        assert service.cancel_orchestration(oid)["reason"] == "already_terminal"
    finally:
        service.close()


@pytest.mark.unit
def test_expired_stage_approval_cancels_the_plan(tmp_path: Path) -> None:
    worker = _Recording("worker")
    service = _service(tmp_path, worker)
    try:
        started = service.start_plan(
            {
                "task": "Goal",
                "workspace": str(_workspace(tmp_path)),
                "metadata": {"approval_ttl_seconds": 1},
            },
            [{"name": "Edit", "harness": "worker", "workspace_mode": "direct"}],
        )
        oid = started["orchestration_id"]
        _wait(service, oid, lambda v: v["status"] == "pending_approval")
        done = _wait(service, oid, _finished, timeout=5.0)
        assert done["status"] == "canceled"
        assert done["error_code"] == "approval_expired"
        assert worker.tasks == []
    finally:
        service.close()


@pytest.mark.unit
def test_compare_runs_every_harness_on_one_snapshot(tmp_path: Path) -> None:
    first, second = _Recording("first"), _Recording("second", cost=0.03)
    service = _service(tmp_path, first, second)
    try:
        started = service.start_compare(
            {"task": "Find the bug", "workspace": str(_workspace(tmp_path))},
            ["first", "second", "first"],
        )
        assert [step["harness"] for step in started["steps"]] == ["first", "second"]
        done = _wait(service, started["orchestration_id"], _finished)
        assert done["status"] == "succeeded"
        assert done["summary"] == {
            "runs": 2,
            "succeeded": 2,
            "verified": 0,
            "cost_usd": pytest.approx(0.04),
        }
        assert first.tasks[0].task == second.tasks[0].task == "Find the bug"
        fingerprints = {
            first.context_packs[0].fingerprint,
            second.context_packs[0].fingerprint,
        }
        assert len(fingerprints) == 1
    finally:
        service.close()


@pytest.mark.unit
def test_compare_reports_partial_failure(tmp_path: Path) -> None:
    service = _service(tmp_path, _Recording("good"), _Recording("bad", fail=True))
    try:
        started = service.start_compare(
            {"task": "Goal", "workspace": str(_workspace(tmp_path))}, ["good", "bad"]
        )
        done = _wait(service, started["orchestration_id"], _finished)
        assert done["status"] == "failed"
        assert done["error"] == "1 of 2 harnesses succeeded."
    finally:
        service.close()


@pytest.mark.unit
def test_workflow_requests_are_validated(tmp_path: Path) -> None:
    service = _service(tmp_path, _Recording("one"), _Recording("two"))
    task = {"task": "Goal", "workspace": str(_workspace(tmp_path))}
    try:
        with pytest.raises(ValueError, match="at least two"):
            service.start_compare(task, ["one", "one"])
        with pytest.raises(ValueError, match="auto is not allowed"):
            service.start_compare(task, ["one", "auto"])
        with pytest.raises(ValueError, match="Unknown harness"):
            service.start_compare(task, ["one", "missing"])
        with pytest.raises(ValueError, match="read-only"):
            service.start_compare(
                {**task, "permissions": {"workspace_mode": "direct"}}, ["one", "two"]
            )
        with pytest.raises(ValueError, match="at most one write-capable"):
            service.start_plan(
                task,
                [
                    {"name": "A", "harness": "one", "workspace_mode": "direct"},
                    {"name": "B", "harness": "two", "workspace_mode": "direct"},
                ],
            )
        with pytest.raises(ValueError, match="Unknown harness"):
            service.start_plan(task, [{"name": "A", "harness": "missing"}])
        with pytest.raises(ValueError, match="at most 8 stages"):
            service.start_plan(
                task, [{"name": str(i), "harness": "one"} for i in range(9)]
            )
        assert service.list_orchestrations() == []
    finally:
        service.close()


@pytest.mark.unit
def test_a_workflow_without_a_live_driver_reads_as_interrupted(tmp_path: Path) -> None:
    service = _service(tmp_path, _Recording("slow", block=True))
    try:
        started = service.start_plan(
            {"task": "Goal", "workspace": str(_workspace(tmp_path))},
            [{"name": "One", "harness": "slow"}],
        )
        oid = started["orchestration_id"]
        _wait(service, oid, lambda v: v["status"] == "running")
    finally:
        service.close()
    # close() stops the driver, which records the interruption itself.
    reopened = _service(tmp_path, _Recording("slow"))
    try:
        assert reopened.get_orchestration(oid)["status"] == "interrupted"
        # A record left running by a host that died is reported interrupted
        # once its heartbeat is stale, without writing on the read.
        stale = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        reopened.run_store.update_orchestration(
            oid, status="running", heartbeat_at=stale, error=None, error_code=None
        )
        view = reopened.get_orchestration(oid)
        assert view["status"] == "interrupted" and view["stale"] is True
        assert reopened.run_store.get_orchestration(oid).status.value == "running"
        reopened.recover_interrupted_runs()
        assert reopened.run_store.get_orchestration(oid).status.value == "interrupted"
    finally:
        reopened.close()


@pytest.mark.unit
def test_run_plan_read_only_stage_does_not_inherit_edit_approval(
    tmp_path: Path,
) -> None:
    worker = _Recording("worker")
    service = _service(tmp_path, worker)
    try:
        base = HarnessTask(
            task="Goal",
            workspace=str(_workspace(tmp_path)),
            permissions={"workspace_mode": "direct"},
        )
        results = service.run_plan(base, [{"name": "Read", "harness": "worker"}])
        assert [result.status for result in results] == [HarnessStatus.SUCCEEDED]
        assert worker.tasks[0].permissions.require_approval is False
    finally:
        service.close()


@pytest.mark.unit
def test_close_stops_drivers_before_runs(tmp_path: Path) -> None:
    service = _service(tmp_path, _Recording("slow", block=True))
    started = service.start_plan(
        {"task": "Goal", "workspace": str(_workspace(tmp_path))},
        [{"name": "One", "harness": "slow"}, {"name": "Two", "harness": "slow"}],
    )
    _wait(service, started["orchestration_id"], lambda v: v["status"] == "running")
    closer = threading.Thread(target=service.close)
    closer.start()
    closer.join(timeout=10)
    assert not closer.is_alive()
    store = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    try:
        record = store.get_orchestration(started["orchestration_id"])
        assert record.status == HarnessStatus.INTERRUPTED
        assert record.steps[1]["run_id"] is None
    finally:
        store.close()


@pytest.mark.unit
def test_handoffs_keep_every_stage_within_the_prompt_budget() -> None:
    from memorizz.metaharness.handoff import fit_handoffs

    long_plan = "PLAN-START " + "step " * 6_000 + " PLAN-END"
    handoffs = [
        {"stage": "Plan", "status": "succeeded", "run_id": "r1", "response": long_plan},
        {
            "stage": "Build",
            "status": "succeeded",
            "run_id": "r2",
            "response": "x" * 9_000,
        },
    ]
    fitted = fit_handoffs(handoffs, budget=6_000)
    rendered = __import__("json").dumps(fitted)
    assert len(rendered) <= 6_000
    # Both stages survive, and the first plan keeps its beginning and end.
    assert [item["stage"] for item in fitted] == ["Plan", "Build"]
    assert "PLAN-START" in fitted[0]["response"] and "PLAN-END" in fitted[0]["response"]
    assert fitted[0]["response_truncated"] and fitted[1]["response_truncated"]
    # The most recent stage gets the larger share.
    assert len(fitted[1]["response"]) > len(fitted[0]["response"])
    small = [{"stage": "Plan", "response": "short"}]
    assert fit_handoffs(small, budget=6_000) == small


@pytest.mark.unit
def test_handoff_collects_files_commands_and_verification() -> None:
    from memorizz.metaharness.handoff import render_handoff, stage_handoff

    run = {
        "run_id": "run-1",
        "harness": "codex",
        "status": "succeeded",
        "task": {"workspace": "/repo", "context": {"harness_stage": "Implement"}},
        "result": {
            "final_response": "Changed the parser.",
            "verified": True,
            "verification": {
                "required": True,
                "command": "pytest -q",
                "return_code": 0,
            },
            "workspace_diff": "# git status --porcelain\n M src/parser.py\n?? tests/new.py\n"
            "?? __pycache__/parser.cpython-312.pyc\n"
            "\n# git diff --binary HEAD\ndiff --git ...",
        },
    }
    events = [
        {"type": "command", "data": {"command": "pytest -q tests/test_parser.py"}},
        {"type": "file_change", "data": {"changes": [{"path": "/repo/src/parser.py"}]}},
        {
            "type": "tool_call",
            "data": {"name": "Edit", "input": {"file_path": "README.md"}},
        },
        {"type": "message", "data": {"text": "token=sk-test-this-must-not-survive"}},
    ]
    handoff = stage_handoff(run, events)
    assert handoff["stage"] == "Implement"
    assert handoff["files_changed"] == ["src/parser.py", "README.md", "tests/new.py"]
    assert handoff["commands"] == ["pytest -q tests/test_parser.py"]
    assert handoff["verification"] == {
        "command": "pytest -q",
        "verified": True,
        "return_code": 0,
    }
    text = render_handoff(handoff, workflow={"kind": "plan", "step": 1, "steps": 3})
    assert text.startswith("[Harness plan stage 2/3: Implement · codex · succeeded]")
    assert "Files changed: src/parser.py" in text and "Harness run: run-1" in text


@pytest.mark.unit
def test_workflow_handoffs_are_saved_to_conversation_memory(tmp_path: Path) -> None:
    from memorizz.enums.memory_type import MemoryType
    from memorizz.memory_provider.filesystem.provider import (
        FileSystemConfig,
        FileSystemProvider,
    )

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    service = MetaHarness(
        memory_provider=provider,
        adapters=[_Recording("planner"), _Recording("reviewer")],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    try:
        started = service.start_plan(
            {
                "task": "Fix the parser",
                "workspace": str(_workspace(tmp_path)),
                "memory_id": "repo-memory",
            },
            [
                {"name": "Plan", "harness": "planner"},
                {"name": "Review", "harness": "reviewer"},
            ],
        )
        oid = started["orchestration_id"]
        done = _wait(service, oid, _finished)
        assert done["status"] == "succeeded"
        assert all(step["handoff"]["memory_record_id"] for step in done["steps"])
        rows = provider.retrieve_conversation_history_ordered_by_timestamp(
            "repo-memory", thread_id=f"harness-workflow-{oid}"
        )
        texts = [row.get("content") for row in rows if row.get("role") == "assistant"]
        assert any("planner finished Plan" in text for text in texts)
        assert any(
            text.startswith("[Harness plan stage 2/2: Review · reviewer")
            for text in texts
        )
        # Later runs in the same memory scope can retrieve what earlier ones did.
        hits = provider.retrieve_by_query(
            "planner finished",
            memory_store_type=MemoryType.CONVERSATION_MEMORY,
            memory_id="repo-memory",
            limit=5,
        )
        assert hits
    finally:
        service.close()


@pytest.mark.unit
def test_workspace_errors_name_the_allowed_roots(tmp_path: Path) -> None:
    service = _service(tmp_path, _Recording("one"))
    try:
        missing = service.run({"task": "Goal", "workspace": "/memorizz"})
        assert missing.status == HarnessStatus.FAILED
        assert "Workspace /memorizz does not exist on this machine" in missing.error
        assert f"Allowed workspace roots: {tmp_path.resolve()}" in missing.error
        outside = service.run({"task": "Goal", "workspace": "/"})
        assert "outside the configured allowed roots" in outside.error
    finally:
        service.close()
