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
            assert "Agent Harnesses" in page.text
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
