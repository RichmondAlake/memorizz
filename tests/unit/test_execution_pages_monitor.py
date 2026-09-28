# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Execution pages (harnesses, automations, playground) read as monitors.

The shaping is pure: tape figures, one row per record, and the detail each
pane shows. The page tests render real stores so the templates stay wired to
the same controls (approval, pause, run now, delete) they always had.
"""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from memorizz.ui.execution_view import (
    build_automation_monitor,
    build_harness_monitor,
    build_playground_launcher,
    describe_schedule,
    first_line,
    parse_time,
    shape_harness_run,
)

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def _run(run_id, status, **extra):
    task = {
        "task": extra.pop("task", "Fix the flaky test\nand explain why."),
        "workspace": "/work/memorizz",
        "harness": extra.pop("requested", "codex"),
        "permissions": {"workspace_mode": extra.pop("mode", "read_only")},
        "budget": {"max_wall_time_seconds": 900, "max_steps": 80},
        "verification": {"command": extra.pop("verify", None)},
        "metadata": {"execution_backend": "docker", "source": "ui"},
    }
    value = {
        "run_id": run_id,
        "status": status,
        "harness": extra.pop("harness", "codex"),
        "task": task,
        "created_at": "2026-09-28T10:00:00+00:00",
        "updated_at": extra.pop("updated_at", "2026-09-28T11:00:00+00:00"),
    }
    value.update(extra)
    return value


@pytest.mark.unit
def test_parse_time_accepts_iso_z_naive_and_datetimes():
    assert parse_time("2026-09-28T10:00:00Z") == datetime(
        2026, 9, 28, 10, tzinfo=timezone.utc
    )
    naive = parse_time("2026-09-28T10:00:00")
    assert naive.tzinfo is timezone.utc, "naive times are read as UTC"
    assert parse_time(NOW) is NOW
    assert parse_time("") is None and parse_time("soon") is None
    assert first_line("\n  Fix   the\ttest \nsecond") == "Fix the test"
    assert first_line("x" * 300, 10) == "x" * 9 + "…"


@pytest.mark.unit
def test_harness_tape_counts_groups_spend_and_availability():
    harnesses = [
        {"name": "codex", "ready": True, "version": "0.157"},
        {
            "name": "claude-code",
            "ready": False,
            "error_code": "authentication_required",
            "error": "No key",
            "remediation": "Set ANTHROPIC_API_KEY.",
        },
        {"name": "openhands", "ready": False, "error": "not installed"},
    ]
    runs = [
        _run(
            "r1",
            "succeeded",
            result={
                "verified": True,
                "verification": {"required": True},
                "cost_usd": 0.5,
            },
        ),
        _run(
            "r2",
            "verification_failed",
            result={"verification": {"required": True}, "cost_usd": 1.25},
        ),
        _run("r3", "running", harness=None, requested="auto"),
        _run("r4", "pending_approval", harness=None, requested="claude-code"),
        _run("r5", "canceled"),
        _run("r6", "budget_exceeded", result={"error_code": "budget_exceeded"}),
    ]
    approvals = [
        {
            "proposal_id": "p1",
            "arguments": {"run_id": "r4"},
            "expires_at": "2026-09-28T13:00:00+00:00",
        },
        {"proposal_id": "p2", "arguments": {"run_id": "gone"}},
    ]
    m = build_harness_monitor(harnesses, runs, approvals, now=NOW, run_limit=6)

    assert (m["ready"], m["total_harnesses"]) == (1, 3)
    states = {item["name"]: item["label"] for item in m["availability"]}
    assert states == {
        "codex": "Ready",
        "claude-code": "Setup required",
        "openhands": "Unavailable",
    }
    assert m["counts"] == {
        "active": 1,
        "approval": 1,
        "succeeded": 1,
        "failed": 2,
        "canceled": 1,
    }
    assert m["awaiting_approval"] == 2
    assert [p["proposal_id"] for p in m["unlisted_approvals"]] == ["p2"]
    assert m["spend"] == pytest.approx(1.75)
    assert (m["verified"], m["verification_checked"]) == (1, 2)
    assert m["has_more"] is True, "a full page means older runs exist"
    assert m["harness_names"] == ["claude-code", "codex"]
    pending = next(row for row in m["runs"] if row["run_id"] == "r4")
    assert pending["approval"]["proposal_id"] == "p1"
    assert pending["approval"]["expires"] == datetime(
        2026, 9, 28, 13, tzinfo=timezone.utc
    )
    assert pending["harness_label"] == "claude-code", "requested harness before routing"
    routed = next(row for row in m["runs"] if row["run_id"] == "r3")
    assert routed["harness_label"] == "Auto route"


@pytest.mark.unit
def test_harness_run_row_carries_limits_verification_and_outcome():
    row = shape_harness_run(
        _run(
            "r1",
            "verification_failed",
            mode="direct",
            verify="pytest -q",
            started_at="2026-09-28T10:00:00+00:00",
            finished_at="2026-09-28T10:02:30+00:00",
            result={
                "verified": False,
                "verification": {
                    "required": True,
                    "return_code": 1,
                    "stderr": "1 failed",
                },
                "usage": {"input_tokens": 1000, "output_tokens": 200},
            },
        ),
        now=NOW,
    )
    assert row["status_label"] == "Verification failed"
    assert (row["group"], row["signal"]) == ("failed", "bad")
    assert row["task_line"] == "Fix the flaky test"
    assert row["workspace_name"] == "memorizz"
    assert row["writes"] and row["mode_label"] == "Direct edits"
    assert row["backend"] == "Docker"
    assert row["limits"]["wall_seconds"] == 900 and row["limits"]["cost_usd"] is None
    assert (row["verify_state"], row["verify_return_code"]) == ("failed", 1)
    assert row["verify_command"] == "pytest -q"
    assert row["verify_output"] == "1 failed"
    assert row["duration_ms"] == 150_000, "from start and finish when no latency"
    assert row["tokens"] == 1200 and row["cost_usd"] is None
    assert row["outcome"] == "Verification failed"
    assert row["cancellable"] is False

    running = shape_harness_run(
        _run(
            "r2", "running", started_at="2026-09-28T11:59:00+00:00", verify="make test"
        ),
        now=NOW,
    )
    assert running["duration_ms"] == 60_000, "live elapsed time while running"
    assert running["verify_state"] == "pending" and running["cancellable"]
    succeeded = shape_harness_run(
        _run("r3", "succeeded", result={"final_response": "ok"})
    )
    assert succeeded["outcome"] == "Complete" and succeeded["verify_state"] == "none"


@pytest.mark.unit
def test_empty_harness_monitor_is_safe():
    m = build_harness_monitor([], [], [], now=NOW)
    assert m["runs"] == [] and m["spend"] is None and m["last_activity"] is None
    assert m["ready"] == 0 and m["has_more"] is False


def _job(name, *, enabled=True, next_in_hours=2, runs=(), **extra):
    job_id = extra.pop("job_id", str(uuid.uuid4()))
    recent = [
        {
            "status": status,
            "started_at": (NOW - timedelta(hours=i + 1)).isoformat(),
            "finished_at": (
                NOW - timedelta(hours=i + 1) + timedelta(seconds=30)
            ).isoformat(),
            "error": "rate limited\nretry later" if status == "failed" else None,
            "result_summary": f"{name} ok" if status == "succeeded" else None,
            "result_payload": {"memory_id": "mem-1"} if status == "succeeded" else {},
        }
        for i, status in enumerate(runs)
    ]
    value = {
        "job_id": job_id,
        "agent_id": "agent-1",
        "name": name,
        "enabled": enabled,
        "schedule_type": "cron",
        "cron_expr": "0 8 * * 1-5",
        "timezone": "Europe/London",
        "next_run_at": NOW + timedelta(hours=next_in_hours),
        "action_type": "agent_query",
        "action_config": {"query_template": "Summarise the inbox"},
        "latest_run": recent[0] if recent else None,
        "recent_runs": recent,
    }
    value.update(extra)
    return value


@pytest.mark.unit
def test_describe_schedule_reads_like_a_sentence():
    assert (
        describe_schedule({"schedule_type": "cron", "cron_expr": "0 8 * * *"})
        == "Cron 0 8 * * *"
    )
    assert (
        describe_schedule({"schedule_type": "interval", "interval_seconds": 3600})
        == "Every 1h"
    )
    assert (
        describe_schedule({"schedule_type": "interval", "interval_seconds": 90})
        == "Every 90s"
    )
    assert (
        describe_schedule({"schedule_type": "interval", "interval_seconds": 172800})
        == "Every 2d"
    )
    assert describe_schedule({"schedule_type": "one_shot"}) == "One-shot"


@pytest.mark.unit
def test_automation_tape_next_run_failures_and_overdue():
    jobs = [
        _job("Digest", runs=["succeeded", "failed", "succeeded"], next_in_hours=5),
        _job("Health", runs=["failed", "succeeded"], next_in_hours=1),
        _job("Paused", enabled=False, runs=["succeeded"], next_in_hours=-3),
        _job("Stuck", runs=[], next_in_hours=-1),
    ]
    m = build_automation_monitor(
        jobs, agent_names={"agent-1": "Support bot"}, now=NOW, runs_per_job=3
    )
    assert (m["total"], m["active"], m["paused"]) == (4, 3, 1)
    assert m["failed_last"] == 1 and m["never_run"] == 1
    assert m["overdue"] == 1, "paused jobs are never overdue"
    assert m["runs_recorded"] == 6 and m["runs_capped"] is True
    assert m["next"]["name"] == "Health"
    assert [row["name"] for row in m["latest_results"]][0] in {
        "Digest",
        "Health",
        "Paused",
    }

    health = next(row for row in m["jobs"] if row["name"] == "Health")
    assert health["agent_name"] == "Support bot"
    assert health["tags"] == ["active", "failed"]
    assert health["last_label"] == "Failed" and health["last_signal"] == "bad"
    assert health["error"].startswith("rate limited")
    assert [run["status"] for run in health["history"]] == ["succeeded", "failed"]
    assert health["recent"][0]["duration_ms"] == 30_000
    assert health["recent"][0]["error"] == "rate limited"
    digest = next(row for row in m["jobs"] if row["name"] == "Digest")
    assert digest["output"] == "Digest ok" and digest["trace_memory_id"] == "mem-1"
    assert digest["failures"] == 1 and digest["query"] == "Summarise the inbox"
    stuck = next(row for row in m["jobs"] if row["name"] == "Stuck")
    assert stuck["tags"] == ["active", "never"] and stuck["last_label"] == "Never run"


@pytest.mark.unit
def test_automation_monitor_falls_back_to_latest_run_only():
    job = _job("Solo", runs=["succeeded"])
    job.pop("recent_runs")
    m = build_automation_monitor([job], now=NOW)
    assert m["runs_recorded"] == 1 and m["runs_capped"] is False
    assert m["jobs"][0]["last_status"] == "succeeded"


@pytest.mark.unit
def test_playground_launcher_orders_by_last_use_without_extra_reads():
    agents = [
        SimpleNamespace(
            agent_id="a-old",
            name="Old",
            llm_config={"provider": "openai", "model": "gpt-4.1-mini"},
            application_mode="assistant",
            memory_ids=["m1", "m2"],
            persona=None,
            instruction="You are old.\nMore.",
        ),
        {
            "agent_id": "a-never",
            "name": "",
            "persona": {"name": "Persona name", "role": "Guide"},
            "llm_config": {},
        },
        SimpleNamespace(
            agent_id="a-new",
            name="New",
            llm_config={"provider": "anthropic", "model": "claude-sonnet-5"},
            application_mode="workflow",
            memory_ids=[],
            persona=None,
            instruction="",
        ),
    ]
    last = {
        "a-old": (NOW - timedelta(days=3)).timestamp(),
        "a-new": (NOW - timedelta(hours=2)).timestamp(),
    }
    launcher = build_playground_launcher(
        agents,
        last_run_by_agent=last,
        now=NOW,
        default_model=lambda provider: "default-model",
    )
    assert [row["agent_id"] for row in launcher["agents"]] == [
        "a-new",
        "a-old",
        "a-never",
    ]
    never = launcher["agents"][2]
    assert never["name"] == "Persona name" and never["role"] == "Guide"
    assert never["model"] == "default-model" and never["last_run"] is None
    old = launcher["agents"][1]
    assert old["threads"] == 2 and old["instruction"] == "You are old."
    assert launcher["agents"][0]["mode"] == "Workflow"
    assert (launcher["used_day"], launcher["used_week"]) == (1, 2)
    assert launcher["threads"] == 2 and launcher["models"] == 3
    assert launcher["last_activity"] == datetime.fromtimestamp(
        last["a-new"], tz=timezone.utc
    )


# ---------------------------------------------------------------- pages

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402


@pytest.fixture()
def provider(tmp_path):
    fs = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "ui", lazy_vector_indexes=True)
    )
    with patch.dict(
        state._state,
        {
            "provider": fs,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(fs.root_path)},
        },
    ):
        yield fs


@pytest.mark.unit
def test_automations_page_renders_grid_pane_and_unchanged_controls(provider):
    from memorizz.automation.models import AutomationJob
    from memorizz.automation.store.factory import get_automation_store

    store = get_automation_store(provider)
    job = store.create_job(
        AutomationJob(
            job_id="job-1",
            agent_id="agent-1",
            name="Morning digest",
            schedule_type="interval",
            interval_seconds=3600,
            timezone="UTC",
            next_run_at=datetime.now(timezone.utc) + timedelta(hours=1),
            action_type="agent_query",
            action_config={"query_template": "Summarise the inbox"},
        )
    )
    run = store.start_run(job, datetime.now(timezone.utc), "test-worker")
    store.finish_run(run.run_id, "failed", "Rate limit reached", None, None)

    client = TestClient(create_app(), follow_redirects=False)
    html = client.get("/automations").text
    assert 'id="ax-table"' in html and 'data-key="job-1"' in html
    assert "Every 1h" in html and "Rate limit reached" in html
    for action in ("run-now", "pause", "delete"):
        assert f'action="/automations/job-1/{action}"' in html
    assert 'href="/automations/job-1/edit"' in html
    assert 'name="confirm"' in html, "delete still needs the confirmation box"
    assert 'id="automation-agent-filter"' in html and "Create automation" in html
    assert "Failed last run" in html


@pytest.mark.unit
def test_harness_page_renders_ledger_with_pane_approval_and_launch_form(
    provider, tmp_path
):
    from memorizz.approval import SQLiteApprovalStore
    from memorizz.metaharness import (
        HarnessPermissions,
        HarnessRun,
        HarnessTask,
        MetaHarness,
        SQLiteHarnessRunStore,
    )

    runs = SQLiteHarnessRunStore(tmp_path / "runs.sqlite3")
    approvals = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    task = HarnessTask(
        task="Upgrade the schema",
        workspace=str(tmp_path),
        harness="codex",
        permissions=HarnessPermissions(workspace_mode="direct"),
    )
    proposal = approvals.propose(
        owner_id="ui:operator",
        tool_name="metaharness.run",
        arguments=task.approval_arguments(workspace_fingerprint="x"),
        policy_reason="Direct edits need approval.",
    )
    runs.create(
        HarnessRun(
            run_id=task.run_id,
            task=task.to_dict(),
            status="pending_approval",
            approval_proposal_id=proposal.proposal_id,
        )
    )
    meta = MetaHarness(
        memory_provider=provider, adapters=[], run_store=runs, approval_store=approvals
    )
    try:
        with patch.dict(
            state._state,
            {
                "meta_harness": meta,
                "meta_harness_provider": provider,
                "read_only": False,
            },
        ):
            html = (
                TestClient(create_app(), follow_redirects=False).get("/harnesses").text
            )
    finally:
        meta.close()
    assert 'id="hx-table"' in html and f'data-key="{task.run_id}"' in html
    assert "Pending approval" in html and "Upgrade the schema" in html
    # One approval card in the queue and one in the run's pane, both actionable.
    assert html.count(f'data-proposal-id="{proposal.proposal_id}"') == 2
    assert 'data-decision="approve"' in html and 'data-decision="reject"' in html
    # The launch form keeps its fields and stays collapsed while runs exist.
    assert 'id="harness-run-form"' in html and 'name="verification_command"' in html
    assert (
        '<details class="panel harness-launch-panel hx-launch" id="harness-launch">'
        in html
    )


@pytest.mark.unit
def test_playground_launcher_page_lists_agents_as_rows(provider):
    from memorizz.memagent.models import MemAgentModel

    provider.store_memagent(
        MemAgentModel(
            agent_id="agent-launch", name="Launcher agent", instruction="Help."
        )
    )
    html = TestClient(create_app(), follow_redirects=False).get("/playground").text
    assert 'id="pg-table"' in html and 'data-key="agent-launch"' in html
    assert 'href="/agents/agent-launch/playground"' in html
    assert "Launcher agent" in html and "Create agent" in html
