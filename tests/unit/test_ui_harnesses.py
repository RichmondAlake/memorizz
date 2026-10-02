"""Functional coverage for the local meta-harness operator console."""

from __future__ import annotations

import time
from pathlib import Path
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
from memorizz.metaharness.base import AdapterOutcome  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402


class _UIHarness(AgentHarness):
    name = "ui-fake"

    def probe(self):
        return HarnessCapabilities(name=self.name, available=True, mcp=True)

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        emit(
            HarnessEvent(
                task.run_id,
                HarnessEventType.MESSAGE,
                {"role": "assistant", "text": "UI harness complete"},
            )
        )
        if task.writes_workspace:
            (workspace / "ui-result.txt").write_text("complete\n", encoding="utf-8")
        return AdapterOutcome(final_response="UI harness complete", exit_code=0)


class _UIAuthRequiredHarness(AgentHarness):
    name = "ui-auth-required"

    def probe(self):
        return HarnessCapabilities(
            name=self.name,
            available=True,
            command="ui-auth-required",
            error_code="authentication_required",
            error="The harness API key is not configured.",
            remediation="Set TEST_HARNESS_API_KEY before starting this harness.",
            metadata={"authentication_configured": False},
        )

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        raise AssertionError("An unauthenticated harness must not start")


@pytest.mark.unit
def test_harness_ui_launch_approval_resume_and_evidence(tmp_path: Path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    approvals = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[_UIHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=approvals,
        allowed_workspace_roots=[str(tmp_path)],
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
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
            assert "Agent harnesses" in page.text
            assert "ui-fake" in page.text

            started = client.post(
                "/api/harness-runs",
                json={
                    "task": "Create verified output",
                    "workspace": str(workspace),
                    "harness": "ui-fake",
                    "write": True,
                    "mcp_access": "none",
                    "verification_command": "test -f ui-result.txt",
                },
            )
            assert started.status_code == 200, started.text
            payload = started.json()
            assert payload["status"] == "approval_required"
            proposal_id = payload["proposal"]["proposal_id"]
            run_id = payload["run"]["run_id"]

            approved = client.post(
                f"/api/harness-approvals/{proposal_id}/approve",
                json={"approver_id": "ui-operator@example.com"},
            )
            assert approved.status_code == 200
            resumed = client.post(
                f"/api/harness-approvals/{proposal_id}/resume", json={}
            )
            assert resumed.status_code == 200, resumed.text

            deadline = time.monotonic() + 3
            while True:
                run = client.get(
                    f"/api/harness-runs/{run_id}?include_events=true"
                ).json()["run"]
                if run["status"] in {
                    HarnessStatus.SUCCEEDED.value,
                    HarnessStatus.FAILED.value,
                    HarnessStatus.VERIFICATION_FAILED.value,
                }:
                    break
                assert time.monotonic() < deadline
                time.sleep(0.02)
            assert run["status"] == HarnessStatus.SUCCEEDED.value
            assert run["result"]["verified"] is True
            assert any(event["type"] == "verification" for event in run["events"])
    finally:
        meta.close()
        provider.close()


@pytest.mark.unit
def test_harness_ui_renders_and_returns_missing_authentication_guidance(
    tmp_path: Path,
):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[_UIAuthRequiredHarness()],
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
    try:
        with patch.dict(state._state, values):
            client = TestClient(create_app(), follow_redirects=False)
            page = client.get("/harnesses")
            assert page.status_code == 200
            assert "Setup required" in page.text
            assert "TEST_HARNESS_API_KEY" in page.text

            started = client.post(
                "/api/harness-runs",
                json={
                    "task": "Inspect the workspace",
                    "workspace": str(tmp_path),
                    "harness": "ui-auth-required",
                    "mcp_access": "none",
                },
            )
            assert started.status_code == 409
            payload = started.json()
            assert payload["ok"] is False
            assert payload["error"]["code"] == "authentication_required"
            assert "TEST_HARNESS_API_KEY" in payload["error"]["remediation"]
            assert payload["run"]["status"] == "failed"
    finally:
        meta.close()
        provider.close()


class _UINamedHarness(_UIHarness):
    def __init__(self, name):
        self.name = name


def _workflow_app(tmp_path: Path, *names):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    meta = MetaHarness(
        memory_provider=provider,
        adapters=[_UINamedHarness(name) for name in names],
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


def _wait_workflow(client, workflow_id, until, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        body = client.get(f"/api/harness-orchestrations/{workflow_id}").json()
        if until(body["orchestration"]):
            return body
        assert time.monotonic() < deadline, body
        time.sleep(0.02)


@pytest.mark.unit
def test_harness_ui_runs_a_staged_plan_through_approval(tmp_path: Path):
    provider, meta, values = _workflow_app(tmp_path, "alpha", "beta")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        with patch.dict(state._state, values):
            client = TestClient(create_app(), follow_redirects=False)
            page = client.get("/harnesses")
            assert "Staged plan" in page.text and "Compare" in page.text
            assert 'id="hx-launch-data"' in page.text

            started = client.post(
                "/api/harness-orchestrations",
                json={
                    "kind": "plan",
                    "task": "Add a result file",
                    "workspace": str(workspace),
                    "mcp_access": "none",
                    "memory_id": "ui-workflows",
                    "stages": [
                        {"name": "Plan", "harness": "alpha", "instruction": "Plan it"},
                        {"name": "Build", "harness": "beta", "write": True},
                    ],
                },
            )
            assert started.status_code == 200, started.text
            workflow_id = started.json()["orchestration"]["orchestration_id"]
            waiting = _wait_workflow(
                client, workflow_id, lambda w: w["status"] == "pending_approval"
            )
            first = waiting["runs"][0]
            # The stage's role comes first and the goal follows it.
            assert first["task"]["task"] == "Plan it\n\nGoal:\nAdd a result file"
            assert first["task"]["metadata"]["orchestration"]["id"] == workflow_id

            page = client.get("/harnesses")
            assert "Workflows" in page.text
            assert f'id="hx-workflow-{workflow_id}"' in page.text
            assert "Waiting for approval in the queue above." in page.text
            assert "Stage 2/2" in page.text  # ledger tag on the stage run

            proposal = client.get("/api/harness-runs").json()["runs"][0][
                "approval_proposal_id"
            ]
            assert (
                client.post(
                    f"/api/harness-approvals/{proposal}/approve",
                    json={"approver_id": "ui-operator@example.com"},
                ).status_code
                == 200
            )
            assert (
                client.post(
                    f"/api/harness-approvals/{proposal}/resume", json={}
                ).status_code
                == 200
            )
            done = _wait_workflow(
                client, workflow_id, lambda w: w["status"] == "succeeded"
            )
            assert (workspace / "ui-result.txt").exists()
            assert all(
                step["handoff"]["memory_record_id"]
                for step in done["orchestration"]["steps"]
            )

            listed = client.get("/api/harness-orchestrations").json()
            assert listed["count"] == 1
            activity = client.get("/api/harness-activity").json()
            assert activity["active"] is False and len(activity["fingerprint"]) == 16
            page = client.get("/harnesses")
            assert "Answers saved to memory" in page.text and "(2 of 2)" in page.text
    finally:
        meta.close()
        provider.close()


@pytest.mark.unit
def test_harness_ui_compares_harnesses_and_validates_requests(tmp_path: Path):
    provider, meta, values = _workflow_app(tmp_path, "alpha", "beta")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    base = {"task": "Inspect", "workspace": str(workspace), "mcp_access": "none"}
    try:
        with patch.dict(state._state, values):
            client = TestClient(create_app(), follow_redirects=False)
            started = client.post(
                "/api/harness-orchestrations",
                json={**base, "kind": "compare", "harnesses": ["alpha", "beta"]},
            )
            assert started.status_code == 200, started.text
            workflow_id = started.json()["orchestration"]["orchestration_id"]
            done = _wait_workflow(
                client, workflow_id, lambda w: w["status"] == "succeeded"
            )
            assert {run["harness"] for run in done["runs"]} == {"alpha", "beta"}
            assert "Comparison" in client.get("/harnesses").text

            for body, message in [
                ({**base, "kind": "compare", "harnesses": ["alpha"]}, "at least two"),
                ({**base, "kind": "plan", "stages": []}, "at least one stage"),
                ({**base, "kind": "other"}, "kind must be plan or compare"),
                (
                    {
                        **base,
                        "kind": "plan",
                        "stages": [
                            {"harness": "alpha", "write": True},
                            {"harness": "beta", "write": True},
                        ],
                    },
                    "at most one write-capable",
                ),
            ]:
                response = client.post("/api/harness-orchestrations", json=body)
                assert response.status_code == 400, response.text
                assert message in response.json()["detail"]
            assert (
                client.post(
                    f"/api/harness-orchestrations/{workflow_id}/cancel", json={}
                ).json()["reason"]
                == "already_terminal"
            )
            assert client.get("/api/harness-orchestrations/missing").status_code == 404
    finally:
        meta.close()
        provider.close()


@pytest.mark.unit
def test_read_only_harness_ui_hides_launch_and_blocks_workflows(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    provider, meta, values = _workflow_app(tmp_path, "alpha", "beta")
    try:
        with patch.dict(state._state, values):
            client = TestClient(create_app(), follow_redirects=False)
            page = client.get("/harnesses")
            assert page.status_code == 200
            assert 'id="harness-run-form"' not in page.text
            assert "launching is turned off here" in page.text
            blocked = client.post(
                "/api/harness-orchestrations",
                json={"kind": "compare", "harnesses": ["alpha", "beta"]},
            )
            assert blocked.status_code == 403
    finally:
        meta.close()
        provider.close()


@pytest.mark.unit
def test_quick_launch_runs_a_task_without_a_project_folder(
    tmp_path: Path, monkeypatch, tmp_path_factory
):
    # The managed scratch folder lives outside the operator's workspace roots.
    home = tmp_path_factory.mktemp("memorizz-home")
    monkeypatch.setenv("MEMORIZZ_HOME", str(home))
    provider, meta, values = _workflow_app(tmp_path, "alpha", "beta")
    try:
        with patch.dict(state._state, values):
            client = TestClient(create_app(), follow_redirects=False)
            page = client.get("/harnesses")
            assert 'id="hx-quick"' in page.text
            assert "None: a fresh scratch folder" in page.text

            started = client.post(
                "/api/harness-runs",
                json={
                    "task": "Answer a question",
                    "harness": "alpha",
                    "mcp_access": "none",
                },
            )
            assert started.status_code == 200, started.text
            run_id = started.json()["run"]["run_id"]
            deadline = time.monotonic() + 3
            while True:
                run = client.get(f"/api/harness-runs/{run_id}").json()["run"]
                if run["status"] == HarnessStatus.SUCCEEDED.value:
                    break
                assert time.monotonic() < deadline, run
                time.sleep(0.02)
            workspace = Path(run["task"]["workspace"])
            assert workspace.is_dir() and not any(workspace.iterdir())
            assert workspace.parent == (home / "harness-workspaces").resolve()
            # Scratch folders are not suggested as project folders.
            assert (
                str(workspace)
                not in client.get("/harnesses").text.split("hx-workspaces")[1]
            )

            empty = client.post("/api/harness-runs", json={"task": "  "})
            assert (
                empty.status_code == 400
                and empty.json()["detail"] == "Describe the task"
            )
            bad = client.post(
                "/api/harness-orchestrations",
                json={"kind": "plan", "task": "Goal", "stages": []},
            )
            assert bad.status_code == 400
            # Validation failures create no scratch folder.
            assert len(list((home / "harness-workspaces").iterdir())) == 1
    finally:
        meta.close()
        provider.close()
