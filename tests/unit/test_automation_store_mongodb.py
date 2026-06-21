# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tests for the MongoDB-backed automation store.

Run against an in-memory ``mongomock`` instance (skipped if not installed), so no
real MongoDB server is required.
"""

import types
from datetime import datetime, timedelta, timezone

import pytest

mongomock = pytest.importorskip("mongomock")

from memorizz.automation.models import AutomationDelivery, AutomationJob  # noqa: E402
from memorizz.automation.store.mongodb import MongoDBAutomationStore  # noqa: E402


def _store():
    db = mongomock.MongoClient()["testdb"]
    return MongoDBAutomationStore(types.SimpleNamespace(db=db))


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
def test_crud_and_list_filters():
    store = _store()
    store.create_job(_job("j1", "a1", "One"))
    store.create_job(_job("j2", "a1", "Two", enabled=False))
    store.create_job(_job("j3", "other", "Theirs"))
    assert store.get_job("j1").name == "One"
    assert {j.job_id for j in store.list_jobs(agent_id="a1")} == {"j1", "j2"}
    assert {j.job_id for j in store.list_jobs(agent_id="a1", enabled=True)} == {"j1"}


@pytest.mark.unit
def test_update_pause_resume():
    store = _store()
    store.create_job(_job("j1"))
    store.update_job("j1", {"name": "Renamed", "interval_seconds": 60})
    assert store.get_job("j1").name == "Renamed"
    assert store.get_job("j1").interval_seconds == 60
    store.pause_job("j1")
    assert store.get_job("j1").enabled is False
    store.resume_job("j1")
    assert store.get_job("j1").enabled is True


@pytest.mark.unit
def test_claim_due_jobs_leases_once():
    store = _store()
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    store.create_job(_job("due", next_run_at=past))
    store.create_job(_job("not_due", next_run_at=future))
    now = datetime.now(timezone.utc)
    claimed = store.claim_due_jobs("worker1", now, limit=10, lease_seconds=120)
    assert [j.job_id for j in claimed] == ["due"]
    assert store.claim_due_jobs("worker2", now, limit=10, lease_seconds=120) == []


@pytest.mark.unit
def test_claim_job_run_now():
    store = _store()
    store.create_job(_job("j1", enabled=False))
    now = datetime.now(timezone.utc)
    job = store.claim_job(
        "j1", worker_id="w1", now_utc=now, lease_seconds=60, force_enable=True
    )
    assert job is not None and job.enabled is True
    assert store.claim_job("j1", worker_id="w2", now_utc=now, lease_seconds=60) is None


@pytest.mark.unit
def test_runs_and_deliveries():
    store = _store()
    job = _job("j1")
    store.create_job(job)
    run = store.start_run(job, job.next_run_at, "w1")
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
def test_delete_job_cascades():
    store = _store()
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
