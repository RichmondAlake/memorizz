import threading
from datetime import datetime, timezone


def test_worker_claims_runs_and_reschedules(monkeypatch):
    from memorizz.automation.models import AutomationJob, AutomationRun
    from memorizz.automation.worker import run_worker

    scheduled = datetime(2026, 2, 13, 9, 0, tzinfo=timezone.utc)
    job = AutomationJob(
        job_id="job-1",
        agent_id="agent-1",
        name="Test job",
        enabled=True,
        schedule_type="interval",
        cron_expr=None,
        interval_seconds=60,
        timezone="UTC",
        start_at=None,
        next_run_at=scheduled,
        last_run_at=None,
        action_type="agent_query",
        action_config={"query_template": "hi"},
        delivery_type=None,
        delivery_config={},
    )

    stop_event = threading.Event()

    # Patch runner to avoid invoking real agents/tools.
    def _fake_run_job_once(*args, **kwargs):
        return {"ok": True}

    monkeypatch.setattr("memorizz.automation.worker.run_job_once", _fake_run_job_once)

    class FakeStore:
        def __init__(self):
            self.claimed_once = False
            self.started = False
            self.finished = False
            self.updated_patch = None

        def claim_due_jobs(self, worker_id, now_utc, limit, lease_seconds):
            if self.claimed_once:
                return []
            self.claimed_once = True
            return [job]

        def start_run(self, job_obj, scheduled_for, worker_id):
            self.started = True
            return AutomationRun(
                run_id="run-1",
                job_id=job_obj.job_id,
                scheduled_for=scheduled_for,
                started_at=now_utc,
                status="running",
                attempt=1,
            )

        def update_job(self, job_id, patch):
            self.updated_patch = dict(patch)
            stop_event.set()
            return job

        def finish_run(
            self,
            run_id,
            status,
            error,
            result_summary,
            result_payload,
            attempt=1,
        ):
            self.finished = True
            return AutomationRun(
                run_id=run_id,
                job_id=job.job_id,
                scheduled_for=scheduled,
                started_at=now_utc,
                finished_at=now_utc,
                status=status,
                attempt=attempt,
                error=error,
                result_summary=result_summary,
                result_payload=result_payload or {},
            )

    now_utc = datetime(2026, 2, 13, 9, 0, tzinfo=timezone.utc)
    store = FakeStore()

    t = threading.Thread(
        target=run_worker,
        kwargs={
            "store": store,
            "memory_provider": object(),
            "poll_interval_s": 1,
            "lease_seconds": 30,
            "max_concurrency": 1,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    t.start()
    t.join(timeout=5)

    assert store.started is True
    assert store.finished is True
    assert isinstance(store.updated_patch, dict)
    assert "next_run_at" in store.updated_patch
