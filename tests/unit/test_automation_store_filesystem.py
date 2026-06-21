# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tests for the filesystem-backed automation store (jobs/runs/deliveries)."""

from datetime import datetime, timedelta, timezone

import pytest

from memorizz.automation.models import AutomationDelivery, AutomationJob
from memorizz.automation.store.factory import get_automation_store
from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider


def _store(tmp_path):
    provider = FileSystemProvider(FileSystemConfig(root_path=str(tmp_path)))
    return get_automation_store(provider)


def _job(job_id="j1", agent_id="a1", name="Daily", enabled=True, next_run_at=None):
    return AutomationJob(
        job_id=job_id,
        agent_id=agent_id,
        name=name,
        enabled=enabled,
        schedule_type="interval",
        interval_seconds=3600,
        timezone="UTC",
        next_run_at=next_run_at or datetime(2030, 1, 1, tzinfo=timezone.utc),
        action_type="agent_query",
        action_config={"query_template": "hi"},
    )


@pytest.mark.unit
def test_factory_returns_filesystem_store(tmp_path):
    assert type(_store(tmp_path)).__name__ == "FileSystemAutomationStore"


@pytest.mark.unit
def test_crud_and_list_filters(tmp_path):
    store = _store(tmp_path)
    store.create_job(_job("j1", "a1", "One"))
    store.create_job(_job("j2", "a1", "Two", enabled=False))
    store.create_job(_job("j3", "other", "Theirs"))
    assert store.get_job("j1").name == "One"
    assert {j.job_id for j in store.list_jobs(agent_id="a1")} == {"j1", "j2"}
    assert {j.job_id for j in store.list_jobs(agent_id="a1", enabled=True)} == {"j1"}
    assert {j.job_id for j in store.list_jobs(agent_id="a1", enabled=False)} == {"j2"}


@pytest.mark.unit
def test_update_pause_resume(tmp_path):
    store = _store(tmp_path)
    store.create_job(_job("j1"))
    store.update_job("j1", {"name": "Renamed", "interval_seconds": 60})
    assert store.get_job("j1").name == "Renamed"
    assert store.get_job("j1").interval_seconds == 60
    store.pause_job("j1")
    assert store.get_job("j1").enabled is False
    store.resume_job("j1")
    assert store.get_job("j1").enabled is True


@pytest.mark.unit
def test_claim_due_jobs_leases_once(tmp_path):
    store = _store(tmp_path)
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    store.create_job(_job("due", next_run_at=past))
    store.create_job(_job("not_due", next_run_at=future))
    now = datetime.now(timezone.utc)
    claimed = store.claim_due_jobs("worker1", now, limit=10, lease_seconds=120)
    assert [j.job_id for j in claimed] == ["due"]
    assert store.get_job("due").locked_by == "worker1"
    # already leased -> a second worker gets nothing
    assert store.claim_due_jobs("worker2", now, limit=10, lease_seconds=120) == []


@pytest.mark.unit
def test_claim_job_run_now(tmp_path):
    store = _store(tmp_path)
    store.create_job(_job("j1", enabled=False))
    now = datetime.now(timezone.utc)
    job = store.claim_job(
        "j1", worker_id="w1", now_utc=now, lease_seconds=60, force_enable=True
    )
    assert job is not None and job.enabled is True
    # held lease blocks a second claim
    assert store.claim_job("j1", worker_id="w2", now_utc=now, lease_seconds=60) is None


@pytest.mark.unit
def test_runs_and_deliveries(tmp_path):
    store = _store(tmp_path)
    job = _job("j1")
    store.create_job(job)
    run = store.start_run(job, job.next_run_at, "w1")
    assert run.status == "running"
    store.record_delivery(
        run.run_id,
        AutomationDelivery(
            delivery_id="d1", run_id=run.run_id, recipient="+100", status="sent"
        ),
    )
    store.finish_run(run.run_id, "succeeded", None, "done", {"response": "hi there"})
    runs = store.list_runs("j1")
    assert len(runs) == 1 and runs[0].status == "succeeded"
    assert runs[0].result_payload["response"] == "hi there"


@pytest.mark.unit
def test_delete_job_cascades(tmp_path):
    store = _store(tmp_path)
    job = _job("j1")
    store.create_job(job)
    run = store.start_run(job, job.next_run_at, "w1")
    store.record_delivery(
        run.run_id,
        AutomationDelivery(
            delivery_id="d1", run_id=run.run_id, recipient="+1", status="sent"
        ),
    )
    assert store.delete_job("j1") is True
    assert store.get_job("j1") is None
    assert store.list_runs("j1") == []
