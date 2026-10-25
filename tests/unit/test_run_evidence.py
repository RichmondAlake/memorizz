"""Run evidence pages and complete exports of long, delegated activity."""

from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from memorizz.approval import SQLiteApprovalStore
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.metaharness import (
    HarnessEvent,
    HarnessEventType,
    HarnessRun,
    HarnessStatus,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.ui import state
from memorizz.ui.app import create_app

pytestmark = pytest.mark.unit


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    provider = FileSystemProvider(FileSystemConfig(root_path=tmp_path / "memory"))
    service = MetaHarness(
        memory_provider=provider,
        adapters=[],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    run = HarnessRun(
        run_id="evidence-root",
        harness="codex",
        status=HarnessStatus.SUCCEEDED,
        task={"task": "Keep every recorded event"},
    )
    events = [
        HarnessEvent(run.run_id, HarnessEventType.LOG, {"text": f"event {i}"})
        for i in range(10_005)
    ]
    events.append(
        HarnessEvent(
            run.run_id,
            HarnessEventType.TOOL_RESULT,
            {"id": "tool-1", "content": "λ" * 9000 + "OUTPUT-END"},
        )
    )
    service.run_store.replace(run, events)
    for child_id, parent in [("delegate", run.run_id), ("nested", "delegate")]:
        child = HarnessRun(
            run_id=child_id,
            harness="claude-code",
            status=HarnessStatus.SUCCEEDED,
            task={"task": child_id, "metadata": {"parent_run_id": parent}},
        )
        service.run_store.replace(
            child,
            [HarnessEvent(child_id, HarnessEventType.MESSAGE, {"text": child_id})],
        )
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "meta_harness": service,
        "meta_harness_provider": provider,
        "read_only": True,
    }
    with patch.dict(state._state, values):
        with TestClient(create_app()) as client:
            yield client, service, events
    service.close()
    provider.close()


def test_evidence_is_a_linkable_page_and_keeps_all_views(evidence):
    client, _, _ = evidence
    response = client.get("/harnesses/runs/evidence-root")
    assert response.status_code == 200
    page = response.text
    assert "<h1>Run evidence</h1>" in page
    assert 'aria-label="Copy full activity"' in page
    assert 'id="evidence-trajectory"' in page
    assert 'id="evidence-record"' in page
    assert 'id="evidence-memory"' in page
    assert 'id="harness-run-dialog"' not in page
    assert client.get("/harnesses/runs/missing").status_code == 404


def test_activity_has_every_event_full_outputs_and_nested_delegates(evidence):
    client, _, events = evidence
    response = client.get("/api/harness-runs/evidence-root/activity")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    activity = response.json()
    assert activity["run"]["events"] == [event.to_dict() for event in events]
    assert activity["run"]["events"][-1]["data"]["content"] == "λ" * 9000 + "OUTPUT-END"
    assert [run["run_id"] for run in activity["delegated_runs"]] == [
        "delegate",
        "nested",
    ]
    assert all(run["events"] for run in activity["delegated_runs"])
    assert client.get("/api/harness-runs/missing/activity").status_code == 404


def test_raw_record_does_not_stop_at_the_first_thousand_events(evidence):
    client, _, events = evidence
    run = client.get("/api/harness-runs/evidence-root?include_events=true").json()[
        "run"
    ]
    assert run["events"] == [event.to_dict() for event in events]


def test_linked_runs_are_included_once_even_with_cyclic_references(evidence):
    _, service, _ = evidence
    child = HarnessRun(
        run_id="linked",
        harness="codex",
        status=HarnessStatus.SUCCEEDED,
        task={"task": "Linked from an event, without a parent field"},
    )
    service.run_store.replace(
        child,
        [
            HarnessEvent(
                "linked",
                HarnessEventType.TOOL_CALL,
                {"harness_run_id": "evidence-root"},
            )
        ],
    )
    service.run_store.append_event(
        HarnessEvent(
            "evidence-root", HarnessEventType.TOOL_CALL, {"harness_run_id": "linked"}
        )
    )
    activity = service.run_activity("evidence-root")
    assert {run["run_id"] for run in activity["delegated_runs"]} == {
        "delegate",
        "nested",
        "linked",
    }
    assert len(activity["delegated_runs"]) == 3
