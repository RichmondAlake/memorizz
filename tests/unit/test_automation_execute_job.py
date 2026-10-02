"""The job runner the worker and the UI's Run now share."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from memorizz.automation import worker


class _Store:
    def __init__(self):
        self.patches, self.finished = [], []

    def update_job(self, job_id, patch):
        self.patches.append(patch)

    def finish_run(self, run_id, **values):
        self.finished.append(values)


def _job(**values):
    base = dict(
        job_id="job-1",
        max_run_seconds=30,
        retry_max_attempts=1,
        retry_backoff_seconds=1,
        schedule_type="interval",
        cron_expr=None,
        interval_seconds=60,
        timezone="UTC",
    )
    return SimpleNamespace(**{**base, **values})


def _run(job, store):
    worker.execute_claimed_job(
        job,
        SimpleNamespace(run_id="run-1"),
        scheduled_for=None,
        memory_provider=None,
        store=store,
    )
    return store.finished[-1], store.patches[-1]


@pytest.mark.unit
def test_success_records_the_answer_and_reschedules(monkeypatch) -> None:
    monkeypatch.setattr(worker, "run_job_once", lambda *a, **k: {"response": "Done."})
    finished, patch = _run(_job(), _Store())
    assert finished["status"] == "succeeded" and finished["result_summary"] == "Done."
    assert patch["next_run_at"] > patch["last_run_at"] and patch["locked_by"] is None


@pytest.mark.unit
def test_all_failed_deliveries_count_as_a_failure(monkeypatch) -> None:
    payload = {"delivery_summary": {"total": 2, "sent": 0, "failed": 2}}
    monkeypatch.setattr(worker, "run_job_once", lambda *a, **k: payload)
    finished, _ = _run(_job(), _Store())
    assert finished["status"] == "failed"
    assert finished["error"] == "All deliveries failed."


@pytest.mark.unit
def test_a_hung_attempt_times_out_instead_of_blocking(monkeypatch) -> None:
    release = threading.Event()
    monkeypatch.setattr(worker, "run_job_once", lambda *a, **k: release.wait(10))
    # Shrink the runner's 30-second floor so the test stays fast.
    monkeypatch.setattr(worker, "max", lambda *values: 0.2, raising=False)
    started = time.monotonic()
    try:
        finished, _ = _run(_job(), _Store())
    finally:
        release.set()
    assert time.monotonic() - started < 5
    assert finished["status"] == "failed"


@pytest.mark.unit
def test_an_invalid_schedule_disables_the_job(monkeypatch) -> None:
    monkeypatch.setattr(worker, "run_job_once", lambda *a, **k: {"response": "ok"})

    def broken(**_):
        raise ValueError("bad cron")

    monkeypatch.setattr(worker, "compute_next_run_at", broken)
    finished, patch = _run(_job(schedule_type="cron", cron_expr="nope"), _Store())
    assert patch["enabled"] is False
    assert finished["status"] == "failed"
    assert finished["error"] == "Reschedule failed: bad cron"
