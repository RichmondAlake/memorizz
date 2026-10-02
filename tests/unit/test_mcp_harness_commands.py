"""The MCP server's harness tools match the UI and CLI, tenant-scoped."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from memorizz.approval import SQLiteApprovalStore
from tests.unit.test_harness_parity import _meta, _provider, _wait_run, _workspace

pytestmark = pytest.mark.unit


def _wait_workflow(runtime, workflow_id, who):
    deadline = time.monotonic() + 5
    while True:
        value = runtime.get_harness_workflow(workflow_id, who)["workflow"]
        if value["status"] not in {"queued", "running"}:
            return value
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_mcp_reruns_continues_and_deletes_only_the_callers_work(tmp_path: Path):
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
        compared = runtime.start_harness_comparison(
            "Compare",
            ["alpha", "beta"],
            None,
            alice,
            mcp_access="none",
            harness_models={"beta": "model-b"},
            allow_subagents=True,
        )
        workflow = _wait_workflow(
            runtime, compared["workflow"]["orchestration_id"], alice
        )
        runs = runtime.get_harness_workflow(workflow["orchestration_id"], alice)["runs"]
        assert {run["harness"]: run["task"]["model"] for run in runs} == {
            "alpha": None,
            "beta": "model-b",
        }
        assert all(run["task"]["permissions"]["allow_subagents"] for run in runs)

        again = runtime.rerun_harness_workflow(workflow["orchestration_id"], alice)
        assert (
            again["workflow"]["task"]["metadata"]["rerun_of"]
            == workflow["orchestration_id"]
        )
        _wait_workflow(runtime, again["workflow"]["orchestration_id"], alice)
        with pytest.raises(MemorizzServerError, match="not found"):
            runtime.rerun_harness_workflow(workflow["orchestration_id"], bob)

        # A conversation: read it and add a turn; bob sees nothing.
        first = runtime.start_harness_run(
            "Start", str(workspace), alice, harness="alpha", mcp_access="none"
        )
        run_id = first["run"]["run_id"]
        _wait_run(meta, run_id)
        shown = runtime.get_harness_conversation(run_id, alice)
        assert shown["conversation_id"] == f"hxc-{run_id}" and len(shown["turns"]) == 1
        turn = runtime.continue_harness_conversation(
            shown["conversation_id"], "Next", alice, harness="beta"
        )
        assert _wait_run(meta, turn["run"]["run_id"])["harness"] == "beta"
        assert len(runtime.get_harness_conversation(run_id, alice)["turns"]) == 2
        with pytest.raises(MemorizzServerError, match="not found"):
            runtime.get_harness_conversation(run_id, bob)

        # Deleting needs the write scope and touches only the caller's runs.
        with pytest.raises(MemorizzServerError):
            runtime.delete_harness_runs([run_id], identity("alice", READ_SCOPE))
        refused = runtime.delete_harness_runs([run_id], bob)
        assert refused["deleted"] == [] and refused["kept"][0]["reason"] == "not_found"
        deleted = runtime.delete_harness_runs([run_id, runs[0]["run_id"]], alice)
        assert deleted["deleted"] == [run_id]
        assert deleted["kept"][0]["reason"] == "workflow"
        with pytest.raises(MemorizzServerError, match="not found"):
            runtime.delete_harness_workflow(workflow["orchestration_id"], bob)
        gone = runtime.delete_harness_workflow(workflow["orchestration_id"], alice)
        assert len(gone["deleted"]) == 2 and gone["scratch_removed"]
    finally:
        meta.close()
        provider.close()
