"""Plans, comparisons, retries and scratch folders on every surface."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from memorizz.approval import SQLiteApprovalStore
from memorizz.cli.app import app
from memorizz.memory_provider.filesystem.provider import (
    FileSystemConfig,
    FileSystemProvider,
)
from memorizz.metaharness import (
    AgentHarness,
    HarnessCapabilities,
    HarnessEvent,
    HarnessEventType,
    HarnessStatus,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness.base import AdapterOutcome
from memorizz.metaharness.requests import harness_task, parse_stage, plan_stages


class _Worker(AgentHarness):
    def __init__(self, name: str) -> None:
        self.name = name
        self.tasks = []

    def probe(self):
        return HarnessCapabilities(
            name=self.name,
            available=True,
            mcp=True,
            usage_reporting=True,
            metadata={"task_tool_policy": True},
        )

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        self.tasks.append(task)
        emit(HarnessEvent(task.run_id, HarnessEventType.MESSAGE, {"text": "ok"}))
        if task.writes_workspace:
            (workspace / f"{self.name}.txt").write_text("done\n", encoding="utf-8")
        return AdapterOutcome(
            final_response=f"{self.name} answered", cost_usd=0.01, exit_code=0
        )


def _provider(tmp_path: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )


def _meta(tmp_path: Path, *names: str, provider=None, **options) -> MetaHarness:
    return MetaHarness(
        memory_provider=provider,
        adapters=[_Worker(name) for name in names],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
        scratch_root=tmp_path / "scratch",
        **options,
    )


def _wait_run(meta, run_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        run = meta.get_run(run_id)
        if run and HarnessStatus(run["status"]).terminal:
            return run
        assert time.monotonic() < deadline, run
        time.sleep(0.02)


def _workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir(exist_ok=True)
    return path


# ------------------------------------------------------------------- SDK


@pytest.mark.unit
def test_shared_request_helpers() -> None:
    task = harness_task(
        "Goal",
        "/w",
        write=True,
        network="full",
        allowed_env="A, B,A",
        allowed_tools=["Read", " ", "Grep"],
        verification_command="  ",
        output_schema={"type": "object"},
    )
    assert task.permissions.workspace_mode == "direct"
    assert task.permissions.allowed_env == ["A", "B"]
    assert task.permissions.allowed_tools == ["Read", "Grep"]
    assert task.verification.command is None
    assert task.output_schema == {"type": "object"}

    assert parse_stage("implement:codex:edit") == {
        "name": "implement",
        "harness": "codex",
        "write": True,
    }
    for bad in ("implement", "implement:codex:fast", ":codex"):
        with pytest.raises(ValueError):
            parse_stage(bad)
    stages = plan_stages(
        [{"name": "Plan", "harness": "pi", "instruction": "Plan it"}, {}], "Goal"
    )
    assert stages[0]["instruction"] == "Plan it\n\nGoal:\nGoal"
    assert stages[1]["name"] == "Stage 2" and stages[1]["instruction"] is None
    with pytest.raises(ValueError, match="at least one stage"):
        plan_stages([], "Goal")


@pytest.mark.unit
def test_retry_start_and_single_run_answers_in_memory(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    meta = _meta(tmp_path, "alpha", provider=provider)
    try:
        first = meta.run(
            {
                "task": "Summarize",
                "workspace": str(_workspace(tmp_path)),
                "memory_id": "repo",
                "permissions": {"mcp_access": "none"},
            }
        )
        assert first.status == HarnessStatus.SUCCEEDED
        rows = provider.retrieve_conversation_history_ordered_by_timestamp(
            "repo", thread_id=f"harness-run-{first.run_id}"
        )
        assert rows and rows[0]["content"].startswith(
            "[Harness run: alpha · succeeded]"
        )

        again = meta.retry_start(first.run_id)
        assert again.run_id != first.run_id
        assert _wait_run(meta, again.run_id)["status"] == "succeeded"
        with pytest.raises(ValueError, match="terminal"):
            queued = meta.start(
                {
                    "task": "Edit",
                    "workspace": str(_workspace(tmp_path)),
                    "permissions": {"workspace_mode": "direct"},
                }
            )
            meta.retry_start(queued.run_id)

        # A MemAgent records harness answers itself; no second copy.
        meta.run(
            {
                "task": "Delegated",
                "workspace": str(_workspace(tmp_path)),
                "memory_id": "agent-memory",
                "permissions": {"mcp_access": "none"},
                "metadata": {"origin_agent_id": "agent-1"},
            }
        )
        assert not provider.retrieve_conversation_history_ordered_by_timestamp(
            "agent-memory"
        )
    finally:
        meta.close()
        provider.close()


# ------------------------------------------------------------------- CLI


def _cli(make_meta, monkeypatch):
    """Each command opens and closes its own service, like the real CLI."""
    monkeypatch.setattr(
        "memorizz.cli.harness_commands._service",
        lambda **_: (make_meta(), None, []),
    )
    monkeypatch.setattr("memorizz.cli.harness_commands.FOLLOW_POLL_SECONDS", 0.02)
    return CliRunner()


@pytest.mark.unit
def test_cli_plan_compare_and_workflow_commands(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    runner = _cli(lambda: _meta(tmp_path, "alpha", "beta"), monkeypatch)
    planned = runner.invoke(
        app,
        [
            "harness",
            "plan",
            "Ship it",
            "--stage",
            "Plan:alpha",
            "--stage",
            "Review:beta",
            "--instruction",
            "Plan=Write a plan",
            "--mcp-access",
            "none",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert planned.exit_code == 0, planned.output
    body = json.loads(planned.stdout.strip().splitlines()[-1])
    assert body["workflow"]["status"] == "succeeded"
    assert body["runs"][0]["task"]["task"] == "Write a plan\n\nGoal:\nShip it"
    assert "[1/2] Plan (alpha): succeeded" in planned.stderr

    compared = runner.invoke(
        app,
        [
            "harness",
            "compare",
            "Where is the bug?",
            "--harness",
            "alpha",
            "--harness",
            "beta",
            "--scratch",
            "--mcp-access",
            "none",
            "--json",
        ],
    )
    assert compared.exit_code == 0, compared.output
    body = json.loads(compared.stdout.strip().splitlines()[-1])
    folder = Path(body["runs"][0]["task"]["workspace"])
    assert folder.parent == (tmp_path / "scratch").resolve()
    workflow_id = body["workflow"]["orchestration_id"]

    listed = runner.invoke(app, ["harness", "workflows", "--json"])
    assert json.loads(listed.stdout)["count"] == 2
    shown = runner.invoke(app, ["harness", "show-workflow", workflow_id, "--json"])
    assert len(json.loads(shown.stdout)["runs"]) == 2
    canceled = runner.invoke(app, ["harness", "cancel-workflow", workflow_id, "--json"])
    assert json.loads(canceled.stdout)["reason"] == "already_terminal"
    missing = runner.invoke(app, ["harness", "show-workflow", "nope"])
    assert missing.exit_code == 1

    bad = runner.invoke(
        app,
        ["harness", "plan", "Goal", "--stage", "nope", "--workspace", str(workspace)],
    )
    assert bad.exit_code != 0 and "NAME:HARNESS" in bad.output


@pytest.mark.unit
def test_cli_run_passes_tool_lists_and_output_schema(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _cli(lambda: _meta(tmp_path, "alpha"), monkeypatch)
    schema = tmp_path / "schema.json"
    schema.write_text('{"type": "object"}', encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "harness",
            "run",
            "Inspect",
            "--scratch",
            "--harness",
            "alpha",
            "--mcp-access",
            "none",
            "--allow-tool",
            "Read",
            "--deny-tool",
            "WebFetch",
            "--output-schema",
            str(schema),
            "--json",
        ],
    )
    # The fake adapter has no output-schema support, so routing refuses it,
    # after recording exactly what was asked for.
    body = json.loads(result.stdout)
    shown = runner.invoke(app, ["harness", "show", body["run_id"], "--json"])
    run = json.loads(shown.stdout)["run"]
    assert run["task"]["permissions"]["allowed_tools"] == ["Read"]
    assert run["task"]["permissions"]["denied_tools"] == ["WebFetch"]
    assert run["task"]["output_schema"] == {"type": "object"}
    assert "output_schema_unsupported" in json.dumps(body["routing"])


# ------------------------------------------------------------------- MCP


@pytest.mark.unit
def test_mcp_workflows_are_tenant_scoped(tmp_path: Path) -> None:
    from memorizz.mcp_server.auth import RequestIdentity
    from memorizz.mcp_server.config import (
        ALL_SCOPES,
        READ_SCOPE,
        MemorizzMCPServerConfig,
    )
    from memorizz.mcp_server.runtime import MemorizzRuntime, MemorizzServerError

    def identity(principal, *scopes):
        return RequestIdentity(
            principal=principal,
            scopes=frozenset(scopes or ALL_SCOPES),
            authenticated=True,
        )

    workspace = _workspace(tmp_path)
    approvals = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    meta = _meta(tmp_path, "alpha", "beta")
    meta.approval_store = approvals
    provider = _provider(tmp_path)
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(
            allow_writes=True,
            allow_harness_execution=True,
            harness_workspace_roots={str(workspace)},
        ),
        provider=provider,
        approval_store=approvals,
        meta_harness=meta,
    )
    alice, bob = identity("alice"), identity("bob")
    try:
        # No workspace: a scratch folder outside the MCP roots is allowed.
        started = runtime.start_harness_run(
            "What time is it?", None, alice, harness="alpha", mcp_access="none"
        )
        run_id = started["run"]["run_id"]
        assert _wait_run(meta, run_id)["status"] == "succeeded"
        retried = runtime.retry_harness_run(run_id, alice)
        assert _wait_run(meta, retried["run"]["run_id"])["status"] == "succeeded"
        with pytest.raises(MemorizzServerError, match="not found"):
            runtime.retry_harness_run(run_id, bob)

        compared = runtime.start_harness_comparison(
            "Compare", ["alpha", "beta"], str(workspace), alice, mcp_access="none"
        )
        workflow_id = compared["workflow"]["orchestration_id"]
        deadline = time.monotonic() + 5
        while runtime.get_harness_workflow(workflow_id, alice)["workflow"][
            "status"
        ] not in {"succeeded", "failed"}:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert len(runtime.get_harness_workflow(workflow_id, alice)["runs"]) == 2
        assert runtime.list_harness_workflows(alice)["count"] == 1
        assert runtime.list_harness_workflows(bob)["count"] == 0
        with pytest.raises(MemorizzServerError, match="not found"):
            runtime.get_harness_workflow(workflow_id, bob)
        with pytest.raises(MemorizzServerError, match="not found"):
            runtime.cancel_harness_workflow(workflow_id, bob)

        # An edit stage needs the write scope and then host approval.
        stages = [
            {"name": "Plan", "harness": "alpha"},
            {"name": "Build", "harness": "beta", "write": True},
        ]
        with pytest.raises(MemorizzServerError):
            runtime.start_harness_plan(
                "Build it", stages, str(workspace), identity("carol", READ_SCOPE)
            )
        plan = runtime.start_harness_plan(
            "Build it", stages, str(workspace), alice, mcp_access="none"
        )
        plan_id = plan["workflow"]["orchestration_id"]
        deadline = time.monotonic() + 5
        while (
            runtime.get_harness_workflow(plan_id, alice)["workflow"]["status"]
            != "pending_approval"
        ):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        plan_runs = runtime.get_harness_workflow(plan_id, alice)["runs"]
        # The read-only stage ran without approval; only the edit stage waits.
        assert [run["status"] for run in plan_runs] == [
            "succeeded",
            "pending_approval",
        ]
        assert runtime.cancel_harness_workflow(plan_id, alice)["ok"] is True
    finally:
        meta.close()
        provider.close()


# -------------------------------------------------------------------- UI


@pytest.mark.unit
def test_ui_retry_and_advanced_launch_fields(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    provider = _provider(tmp_path)
    meta = _meta(tmp_path, "alpha")
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
                "/api/harness-runs",
                json={
                    "task": "Inspect",
                    "harness": "alpha",
                    "mcp_access": "none",
                    "thread_id": "thread-7",
                },
            )
            assert started.status_code == 200, started.text
            run = started.json()["run"]
            assert run["task"]["thread_id"] == "thread-7"
            _wait_run(meta, run["run_id"])

            page = client.get("/harnesses")
            assert "harness-retry" in page.text
            assert 'name="output_schema"' in page.text
            retried = client.post(f"/api/harness-runs/{run['run_id']}/retry", json={})
            assert retried.status_code == 200, retried.text
            assert retried.json()["run"]["run_id"] != run["run_id"]

            # Tool lists change what the harness may do, so they need approval.
            narrowed = client.post(
                "/api/harness-runs",
                json={
                    "task": "Inspect",
                    "harness": "alpha",
                    "mcp_access": "none",
                    "allowed_tools": "Read, Grep",
                    "denied_tools": "WebFetch",
                },
            )
            assert narrowed.json()["status"] == "approval_required"
            permissions = narrowed.json()["run"]["task"]["permissions"]
            assert permissions["allowed_tools"] == ["Read", "Grep"]
            assert permissions["denied_tools"] == ["WebFetch"]

            bad_schema = client.post(
                "/api/harness-runs",
                json={"task": "Inspect", "output_schema": "{not json"},
            )
            assert bad_schema.status_code == 400
            assert "valid JSON" in bad_schema.json()["detail"]
    finally:
        meta.close()
        provider.close()
