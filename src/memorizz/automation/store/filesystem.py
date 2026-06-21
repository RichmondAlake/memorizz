# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Filesystem-backed automation store.

Jobs, runs, and deliveries are stored as one JSON document per record under
``<root_path>/automations/{jobs,runs,deliveries}/``. Writes are atomic
(tmp file + ``os.replace``) and serialized with an in-process ``RLock`` — which
covers the common single-worker case (``memorizz automations run`` or the CLI's
``/automation run``). Cross-process leasing is best-effort via the job's
``lock_expires_at``.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import AutomationDelivery, AutomationJob, AutomationRun

_UPDATABLE_FIELDS = {
    "name",
    "enabled",
    "schedule_type",
    "cron_expr",
    "interval_seconds",
    "timezone",
    "start_at",
    "next_run_at",
    "last_run_at",
    "misfire_policy",
    "max_run_seconds",
    "retry_max_attempts",
    "retry_backoff_seconds",
    "action_type",
    "action_config",
    "delivery_type",
    "delivery_config",
    "locked_by",
    "lock_expires_at",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class FileSystemAutomationStore:
    def __init__(self, provider: Any):
        self.provider = provider
        root = Path(getattr(provider, "root_path", ".")) / "automations"
        self._jobs_dir = root / "jobs"
        self._runs_dir = root / "runs"
        self._deliveries_dir = root / "deliveries"
        for d in (self._jobs_dir, self._runs_dir, self._deliveries_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ io
    @staticmethod
    def _write(path: Path, model: Any) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(model.model_dump(mode="json")))
        tmp.replace(path)

    @staticmethod
    def _load(path: Path, model_cls: Any) -> Optional[Any]:
        try:
            return model_cls.model_validate(json.loads(path.read_text()))
        except Exception:
            return None

    # ----------------------------------------------------------------- jobs
    def create_job(self, job: AutomationJob) -> AutomationJob:
        now = _utcnow()
        if job.created_at is None:
            job.created_at = now
        job.updated_at = now
        with self._lock:
            self._write(self._jobs_dir / f"{job.job_id}.json", job)
        return job

    def get_job(self, job_id: str) -> Optional[AutomationJob]:
        path = self._jobs_dir / f"{job_id}.json"
        return self._load(path, AutomationJob) if path.exists() else None

    def update_job(self, job_id: str, patch: Dict[str, Any]) -> AutomationJob:
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                raise ValueError("Job not found")
            if patch:
                data = job.model_dump()
                for key, value in patch.items():
                    if key in _UPDATABLE_FIELDS:
                        data[key] = value
                data["updated_at"] = _utcnow()
                job = AutomationJob.model_validate(data)
                self._write(self._jobs_dir / f"{job_id}.json", job)
        return job

    def list_jobs(
        self, agent_id: Optional[str] = None, enabled: Optional[bool] = None
    ) -> List[AutomationJob]:
        with self._lock:
            jobs = [self._load(p, AutomationJob) for p in self._jobs_dir.glob("*.json")]
        jobs = [j for j in jobs if j is not None]
        if agent_id is not None:
            jobs = [j for j in jobs if j.agent_id == agent_id]
        if enabled is not None:
            jobs = [j for j in jobs if j.enabled == enabled]
        jobs.sort(key=lambda j: _as_aware(j.created_at) or _utcnow(), reverse=True)
        return jobs

    def pause_job(self, job_id: str) -> AutomationJob:
        return self.update_job(job_id, {"enabled": False})

    def resume_job(self, job_id: str) -> AutomationJob:
        return self.update_job(job_id, {"enabled": True})

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            job_path = self._jobs_dir / f"{job_id}.json"
            existed = job_path.exists()
            if existed:
                job_path.unlink()
            for run_path in list(self._runs_dir.glob("*.json")):
                run = self._load(run_path, AutomationRun)
                if run is None or run.job_id != job_id:
                    continue
                for d_path in list(self._deliveries_dir.glob("*.json")):
                    delivery = self._load(d_path, AutomationDelivery)
                    if delivery is not None and delivery.run_id == run.run_id:
                        d_path.unlink()
                run_path.unlink()
        return existed

    # -------------------------------------------------------------- claiming
    def claim_due_jobs(
        self,
        worker_id: str,
        now_utc: datetime,
        limit: int,
        lease_seconds: int,
    ) -> List[AutomationJob]:
        now = _as_aware(now_utc) or _utcnow()
        lease_expiry = now + timedelta(seconds=int(lease_seconds or 0))
        claimed: List[AutomationJob] = []
        with self._lock:
            due = []
            for job in self.list_jobs(enabled=True):
                nxt = _as_aware(job.next_run_at)
                lock = _as_aware(job.lock_expires_at)
                if nxt and nxt <= now and (lock is None or lock <= now):
                    due.append(job)
            due.sort(key=lambda j: _as_aware(j.next_run_at) or now)
            for job in due[: max(1, int(limit or 1))]:
                job.locked_by = worker_id
                job.lock_expires_at = lease_expiry
                job.updated_at = now
                self._write(self._jobs_dir / f"{job.job_id}.json", job)
                claimed.append(job)
        return claimed

    def claim_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        now_utc: datetime,
        lease_seconds: int,
        force_enable: bool = True,
    ) -> Optional[AutomationJob]:
        now = _as_aware(now_utc) or _utcnow()
        lease_expiry = now + timedelta(seconds=int(lease_seconds or 0))
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                return None
            lock = _as_aware(job.lock_expires_at)
            if lock is not None and lock > now:
                return None  # locked by another worker
            job.locked_by = worker_id
            job.lock_expires_at = lease_expiry
            if force_enable:
                job.enabled = True
            job.updated_at = now
            self._write(self._jobs_dir / f"{job_id}.json", job)
        return job

    # ----------------------------------------------------------------- runs
    def start_run(
        self, job: AutomationJob, scheduled_for: datetime, worker_id: str
    ) -> AutomationRun:
        run = AutomationRun(
            run_id=str(uuid.uuid4()),
            job_id=job.job_id,
            scheduled_for=_as_aware(scheduled_for) or _utcnow(),
            started_at=_utcnow(),
            status="running",
            attempt=1,
            created_at=_utcnow(),
        )
        with self._lock:
            self._write(self._runs_dir / f"{run.run_id}.json", run)
        return run

    def finish_run(
        self,
        run_id: str,
        status: str,
        error: Optional[str],
        result_summary: Optional[str],
        result_payload: Optional[Dict[str, Any]],
        attempt: int = 1,
    ) -> AutomationRun:
        with self._lock:
            path = self._runs_dir / f"{run_id}.json"
            run = self._load(path, AutomationRun)
            if run is None:
                raise ValueError("Run not found")
            run.finished_at = _utcnow()
            run.status = status  # type: ignore[assignment]
            run.error = error
            run.result_summary = result_summary
            run.result_payload = result_payload or {}
            run.attempt = int(attempt or 1)
            self._write(path, run)
        return run

    def list_runs(self, job_id: str, limit: int = 50) -> List[AutomationRun]:
        with self._lock:
            runs = [self._load(p, AutomationRun) for p in self._runs_dir.glob("*.json")]
        runs = [r for r in runs if r is not None and r.job_id == job_id]
        runs.sort(key=lambda r: _as_aware(r.created_at) or _utcnow(), reverse=True)
        return runs[: max(1, int(limit or 1))]

    # ----------------------------------------------------------- deliveries
    def record_delivery(
        self, run_id: str, delivery: AutomationDelivery
    ) -> AutomationDelivery:
        if delivery.created_at is None:
            delivery.created_at = _utcnow()
        with self._lock:
            self._write(self._deliveries_dir / f"{delivery.delivery_id}.json", delivery)
        return delivery
