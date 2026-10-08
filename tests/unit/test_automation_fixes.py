# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Regression tests for the automation worker, runner and stores.

Pins two defects: "Run now" re-enabling a paused automation, and a timed-out
attempt being retried (and delivering) while the abandoned attempt was still
running. Agents are never executed; Twilio is replaced by an in-memory sender.
"""

import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memorizz.automation import runner, worker
from memorizz.automation.models import AutomationDelivery, AutomationJob
from memorizz.automation.store.factory import get_automation_store
from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider


def _store(tmp_path):
    provider = FileSystemProvider(FileSystemConfig(root_path=str(tmp_path)))
    return get_automation_store(provider)


def _mongo_store():
    mongomock = pytest.importorskip("mongomock")
    from memorizz.automation.store.mongodb import MongoDBAutomationStore

    return MongoDBAutomationStore(SimpleNamespace(db=mongomock.MongoClient()["t"]))


def _job(**values):
    base = dict(
        job_id="j1",
        agent_id="a1",
        name="Nightly digest",
        enabled=True,
        schedule_type="cron",
        cron_expr="0 9 * * *",
        timezone="UTC",
        next_run_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        action_type="agent_query",
        action_config={"query_template": "hi"},
        max_run_seconds=30,
        retry_max_attempts=1,
        retry_backoff_seconds=1,
    )
    return AutomationJob(**{**base, **values})


class _FakeSender:
    def __init__(self):
        self.sent = []

    def send(self, *, to, body):
        self.sent.append(to)
        return {"provider_message_id": f"SM{len(self.sent)}"}


@pytest.fixture
def sender(monkeypatch):
    fake = _FakeSender()
    monkeypatch.setattr(
        runner.TwilioWhatsAppSender, "from_env", staticmethod(lambda: fake)
    )
    return fake


class _PatchStore:
    """Records the worker's final job patch; never runs anything."""

    def __init__(self):
        self.patches, self.finished = [], []

    def update_job(self, job_id, patch):
        self.patches.append(dict(patch))

    def finish_run(self, run_id, **values):
        self.finished.append(values)


# ---------------------------------------------------------------------------
# Finding 7: Run now must not re-enable a paused job
# ---------------------------------------------------------------------------


class TestRunNowKeepsPausedJobsPaused:
    @pytest.mark.unit
    def test_filesystem_claim_job_does_not_enable_by_default(self, tmp_path):
        store = _store(tmp_path)
        store.create_job(_job(enabled=False))
        now = datetime.now(timezone.utc)
        claimed = store.claim_job("j1", worker_id="w1", now_utc=now, lease_seconds=60)
        assert claimed is not None
        assert claimed.enabled is False
        stored = store.get_job("j1")
        assert stored.enabled is False
        assert stored.locked_by == "w1"

    @pytest.mark.unit
    def test_mongodb_claim_job_does_not_enable_by_default(self):
        store = _mongo_store()
        store.create_job(_job(enabled=False))
        claimed = store.claim_job(
            "j1", worker_id="w1", now_utc=datetime.now(timezone.utc), lease_seconds=60
        )
        assert claimed.enabled is False
        assert store.get_job("j1").enabled is False

    @pytest.mark.unit
    def test_manual_run_of_a_paused_cron_job_leaves_it_paused(
        self, tmp_path, monkeypatch
    ):
        store = _store(tmp_path)
        store.create_job(_job(enabled=False))
        monkeypatch.setattr(worker, "run_job_once", lambda *a, **k: {"response": "ok"})
        now = datetime.now(timezone.utc)
        claimed = store.claim_job("j1", worker_id="w1", now_utc=now, lease_seconds=60)
        run = store.start_run(claimed, now, "w1")

        worker.execute_claimed_job(
            claimed, run, scheduled_for=now, memory_provider=None, store=store
        )

        after = store.get_job("j1")
        assert after.enabled is False
        assert after.locked_by is None and after.lock_expires_at is None
        assert after.next_run_at > now
        assert after.last_run_at is not None
        assert store.list_runs("j1")[0].status == "succeeded"

    @pytest.mark.unit
    def test_worker_patch_keeps_a_paused_job_paused(self, monkeypatch):
        monkeypatch.setattr(worker, "run_job_once", lambda *a, **k: {"response": "ok"})
        store = _PatchStore()
        worker.execute_claimed_job(
            _job(enabled=False),
            SimpleNamespace(run_id="run-1"),
            scheduled_for=None,
            memory_provider=None,
            store=store,
        )
        assert store.patches[-1]["enabled"] is False
        assert "next_run_at" in store.patches[-1]

    @pytest.mark.unit
    def test_worker_patch_does_not_touch_an_enabled_job(self, monkeypatch):
        # A pause made while the job runs must not be undone when it finishes.
        monkeypatch.setattr(worker, "run_job_once", lambda *a, **k: {"response": "ok"})
        store = _PatchStore()
        worker.execute_claimed_job(
            _job(enabled=True),
            SimpleNamespace(run_id="run-1"),
            scheduled_for=None,
            memory_provider=None,
            store=store,
        )
        assert "enabled" not in store.patches[-1]


# ---------------------------------------------------------------------------
# Finding 8: a timed-out attempt is neither retried nor delivered twice
# ---------------------------------------------------------------------------


class TestTimeoutDoesNotDoubleRun:
    @pytest.mark.unit
    def test_timed_out_attempt_is_neither_retried_nor_delivered(
        self, tmp_path, monkeypatch, sender
    ):
        store = _store(tmp_path)
        job = _job(
            delivery_type="whatsapp_twilio",
            delivery_config={"whatsapp_to": ["+15550001111"]},
            retry_max_attempts=3,
        )
        store.create_job(job)
        release = threading.Event()
        attempts = []

        def slow_action(job, *, scheduled_for_utc, memory_provider):
            attempts.append(time.monotonic())
            release.wait(10)
            return {"response": "late answer"}

        monkeypatch.setattr(runner, "execute_job_action", slow_action)
        # Shrink the runner's 30-second floor so the test stays fast.
        monkeypatch.setattr(worker, "max", lambda *values: 0.2, raising=False)
        now = datetime.now(timezone.utc)
        run = store.start_run(job, now, "w1")

        started = time.monotonic()
        try:
            worker.execute_claimed_job(
                job, run, scheduled_for=now, memory_provider=None, store=store
            )
            assert time.monotonic() - started < 3
            finished = store.list_runs("j1")[0]
            assert finished.status == "failed"
            assert finished.attempt == 1
            assert "timed out" in (finished.error or "").lower()
            assert len(attempts) == 1, "a retry started while attempt 1 was alive"
        finally:
            release.set()
        time.sleep(0.5)  # let the abandoned attempt finish on its own
        assert sender.sent == []
        assert store.list_deliveries(run.run_id) == []
        assert len(attempts) == 1

    @pytest.mark.unit
    def test_run_job_once_does_not_deliver_after_its_deadline(
        self, tmp_path, monkeypatch, sender
    ):
        store = _store(tmp_path)
        job = _job(
            delivery_type="whatsapp_twilio",
            delivery_config={"whatsapp_to": ["+15550001111"]},
        )
        monkeypatch.setattr(
            runner, "execute_job_action", lambda job, **_: {"response": "answer"}
        )
        stop = threading.Event()
        stop.set()
        with pytest.raises(runner.AttemptAbandoned):
            runner.run_job_once(
                job,
                run_id="run-1",
                scheduled_for_utc=datetime.now(timezone.utc),
                memory_provider=None,
                store=store,
                stop_event=stop,
            )
        assert sender.sent == []
        assert store.list_deliveries("run-1") == []

    @pytest.mark.unit
    def test_delivery_is_idempotent_per_run_and_recipient(self, tmp_path, sender):
        store = _store(tmp_path)
        job = _job(
            delivery_type="whatsapp_twilio",
            delivery_config={"whatsapp_to": ["+15550001111", "+15550002222"]},
        )
        first = runner.deliver_job_output(
            job, run_id="run-1", output_text="hello", store=store
        )
        second = runner.deliver_job_output(
            job, run_id="run-1", output_text="hello", store=store
        )
        assert sender.sent == ["whatsapp:+15550001111", "whatsapp:+15550002222"]
        assert [d.status for d in first] == ["sent", "sent"]
        assert [d.status for d in second] == ["sent", "sent"]
        assert {d.delivery_id for d in first} == {d.delivery_id for d in second}
        assert len(store.list_deliveries("run-1")) == 2
        # A different run to the same recipient is a new delivery.
        runner.deliver_job_output(job, run_id="run-2", output_text="hi", store=store)
        assert len(sender.sent) == 4

    @pytest.mark.unit
    def test_delivery_stops_once_the_attempt_is_abandoned(self, tmp_path, sender):
        store = _store(tmp_path)
        stop = threading.Event()
        job = _job(
            delivery_type="whatsapp_twilio",
            delivery_config={"whatsapp_to": ["+15550001111", "+15550002222"]},
        )
        original = sender.send

        def send_then_stop(*, to, body):
            result = original(to=to, body=body)
            stop.set()
            return result

        sender.send = send_then_stop
        runner.deliver_job_output(
            job, run_id="run-2", output_text="hello", store=store, stop_event=stop
        )
        assert sender.sent == ["whatsapp:+15550001111"]
        recorded = store.list_deliveries("run-2")
        assert [d.recipient for d in recorded] == ["whatsapp:+15550001111"]

    @pytest.mark.unit
    def test_mongodb_record_delivery_is_idempotent(self):
        store = _mongo_store()
        delivery = AutomationDelivery(
            delivery_id="d-1", run_id="run-1", recipient="whatsapp:+1", status="sent"
        )
        store.record_delivery("run-1", delivery)
        store.record_delivery("run-1", delivery)
        assert len(store.list_deliveries("run-1")) == 1
        assert store.list_deliveries("other") == []
