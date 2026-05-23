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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Optional

from .runner import run_job_once
from .schedule import compute_next_run_at, utcnow

logger = logging.getLogger(__name__)


def _worker_id() -> str:
    host = socket.gethostname()
    pid = os.getpid()
    suffix = uuid.uuid4().hex[:8]
    return f"{host}:{pid}:{suffix}"


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
        scheduled_for = job.next_run_at
        run = store.start_run(job, scheduled_for, worker_id)
        attempt = 1
        last_error = None
        result_payload = None
        status = "failed"
        max_seconds = max(30, int(job.max_run_seconds or 900))

        try:
            for attempt in range(1, int(job.retry_max_attempts or 1) + 1):
                try:
                    # Run with timeout to prevent hung LLM calls from blocking the thread pool.
                    with ThreadPoolExecutor(max_workers=1) as inner:
                        future = inner.submit(
                            run_job_once,
                            job,
                            run_id=run.run_id,
                            scheduled_for_utc=scheduled_for,
                            memory_provider=memory_provider,
                            store=store,
                        )
                        result_payload = future.result(timeout=max_seconds)
                    # Treat "all deliveries failed" as a failed run so retries kick in.
                    delivery_summary = (
                        result_payload.get("delivery_summary")
                        if isinstance(result_payload, dict)
                        else None
                    )
                    if (
                        isinstance(delivery_summary, dict)
                        and int(delivery_summary.get("total") or 0) > 0
                        and int(delivery_summary.get("sent") or 0) == 0
                        and int(delivery_summary.get("failed") or 0) > 0
                    ):
                        status = "failed"
                        last_error = str(
                            result_payload.get("delivery_error")
                            or "All deliveries failed."
                        )
                    else:
                        status = "succeeded"
                        last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)
                    status = "failed"
                    if attempt < int(job.retry_max_attempts or 1):
                        time.sleep(max(1, int(job.retry_backoff_seconds or 60)))

            # Reschedule / disable
            now_utc = utcnow()
            patch = {
                "last_run_at": now_utc,
                "locked_by": None,
                "lock_expires_at": None,
            }

            if str(job.schedule_type) == "one_shot":
                patch["enabled"] = False
                patch["next_run_at"] = now_utc
            else:
                patch["next_run_at"] = compute_next_run_at(
                    schedule_type=job.schedule_type,
                    cron_expr=job.cron_expr,
                    interval_seconds=job.interval_seconds,
                    tz_name=job.timezone,
                    after_utc=now_utc,
                )

            store.update_job(job.job_id, patch)

        finally:
            try:
                store.finish_run(
                    run.run_id,
                    status=status,
                    error=last_error,
                    result_summary=None,
                    result_payload=result_payload
                    if isinstance(result_payload, dict)
                    else {},
                    attempt=attempt,
                )
            except Exception:
                logger.exception("Failed to finish automation run %s", run.run_id)

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
