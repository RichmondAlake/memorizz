# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Automation worker loop: claim due jobs, run them, and reschedule."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from .runner import run_job_once
from .schedule import compute_next_run_at, utcnow

logger = logging.getLogger(__name__)


def _worker_id() -> str:
    host = socket.gethostname()
    pid = os.getpid()
    suffix = uuid.uuid4().hex[:8]
    return f"{host}:{pid}:{suffix}"


def _delivery_outcome(result_payload: Any) -> tuple[str, Optional[str]]:
    """A run whose every delivery failed counts as failed, so retries kick in."""
    summary = (
        result_payload.get("delivery_summary")
        if isinstance(result_payload, dict)
        else None
    )
    if (
        isinstance(summary, dict)
        and int(summary.get("total") or 0) > 0
        and int(summary.get("sent") or 0) == 0
        and int(summary.get("failed") or 0) > 0
    ):
        return "failed", str(
            result_payload.get("delivery_error") or "All deliveries failed."
        )
    return "succeeded", None


def execute_claimed_job(
    job: Any,
    run: Any,
    *,
    scheduled_for: Any,
    memory_provider: Any,
    store: Any,
) -> None:
    """Run a claimed job with retries, then reschedule or disable it, unlock it
    and finish its run record. Used by the worker and the UI's Run now."""
    attempt = 1
    last_error: Optional[str] = None
    result_payload: Any = None
    status = "failed"
    max_seconds = max(30, int(job.max_run_seconds or 900))
    attempts = int(job.retry_max_attempts or 1)
    try:
        for attempt in range(1, attempts + 1):
            # A separate thread bounds each attempt; shutdown(wait=False) keeps
            # a hung model call from blocking past the timeout.
            inner = ThreadPoolExecutor(max_workers=1)
            try:
                future = inner.submit(
                    run_job_once,
                    job,
                    run_id=run.run_id,
                    scheduled_for_utc=scheduled_for,
                    memory_provider=memory_provider,
                    store=store,
                )
                result_payload = future.result(timeout=max_seconds)
                status, last_error = _delivery_outcome(result_payload)
                break
            except Exception as exc:
                last_error = str(exc) or type(exc).__name__
                status = "failed"
                if attempt < attempts:
                    time.sleep(max(1, int(job.retry_backoff_seconds or 60)))
            finally:
                inner.shutdown(wait=False)

        now_utc = utcnow()
        patch = {"last_run_at": now_utc, "locked_by": None, "lock_expires_at": None}
        try:
            if str(job.schedule_type) == "one_shot":
                patch.update(enabled=False, next_run_at=now_utc)
            else:
                patch["next_run_at"] = compute_next_run_at(
                    schedule_type=job.schedule_type,
                    cron_expr=job.cron_expr,
                    interval_seconds=job.interval_seconds,
                    tz_name=job.timezone,
                    after_utc=now_utc,
                )
        except Exception as exc:
            # An invalid schedule would otherwise retry in a tight loop.
            patch.update(enabled=False, next_run_at=now_utc)
            last_error = last_error or f"Reschedule failed: {exc}"
            status = "failed"
        try:
            store.update_job(job.job_id, patch)
        except Exception:
            logger.exception("Failed to reschedule automation %s", job.job_id)
            try:
                store.update_job(
                    job.job_id, {"locked_by": None, "lock_expires_at": None}
                )
            except Exception:
                logger.exception("Failed to unlock automation %s", job.job_id)
    finally:
        payload = result_payload if isinstance(result_payload, dict) else {}
        try:
            store.finish_run(
                run.run_id,
                status=status,
                error=last_error,
                result_summary=str(payload.get("response") or "")[:2000] or None,
                result_payload=payload,
                attempt=attempt,
            )
        except Exception:
            logger.exception("Failed to finish automation run %s", run.run_id)


def run_worker(
    *,
    store: Any,
    memory_provider: Any,
    poll_interval_s: int = 5,
    lease_seconds: int = 120,
    max_concurrency: int = 2,
    stop_event: Optional[threading.Event] = None,
) -> None:
    worker_id = _worker_id()
    poll_interval = max(1, int(poll_interval_s or 5))
    lease_seconds = max(5, int(lease_seconds or 120))
    max_concurrency = max(1, int(max_concurrency or 1))

    def _process_job(job):
        run = store.start_run(job, job.next_run_at, worker_id)
        execute_claimed_job(
            job,
            run,
            scheduled_for=job.next_run_at,
            memory_provider=memory_provider,
            store=store,
        )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        inflight = set()
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            # Clear completed futures and log any errors.
            done = {f for f in inflight if f.done()}
            for f in done:
                exc = f.exception()
                if exc is not None:
                    logger.error("Automation job execution failed: %s", exc)
            inflight -= done

            capacity = max_concurrency - len(inflight)
            now_utc = utcnow()

            jobs = []
            if capacity > 0:
                try:
                    jobs = store.claim_due_jobs(
                        worker_id, now_utc, limit=capacity, lease_seconds=lease_seconds
                    )
                except Exception:
                    logger.exception("Failed to claim due automation jobs")
                    jobs = []

            if jobs:
                for job in jobs:
                    inflight.add(executor.submit(_process_job, job))
                continue

            time.sleep(poll_interval)
