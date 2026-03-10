# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Oracle-backed automation store (jobs, runs, deliveries)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..models import AutomationDelivery, AutomationJob, AutomationRun


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return json.dumps(value)


def _json_loads(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


class OracleAutomationStore:
    def __init__(self, oracle_provider: Any):
        self.provider = oracle_provider
        self.schema = getattr(getattr(oracle_provider, "config", None), "schema", None)

    def _table(self, name: str) -> str:
        schema = str(self.schema or "").strip()
        if schema:
            return f"{schema}.{name}"
        return name

    def _row_to_job(self, row: Any) -> AutomationJob:
        (
            job_id,
            agent_id,
            name,
            enabled,
            schedule_type,
            cron_expr,
            interval_seconds,
            timezone_name,
            start_at,
            next_run_at,
            last_run_at,
            misfire_policy,
            max_run_seconds,
            retry_max_attempts,
            retry_backoff_seconds,
            action_type,
            action_config,
            delivery_type,
            delivery_config,
            locked_by,
            lock_expires_at,
            created_at,
            updated_at,
        ) = row

        return AutomationJob(
            job_id=str(job_id),
            agent_id=str(agent_id),
            name=str(name),
            enabled=bool(int(enabled) if enabled is not None else 0),
            schedule_type=str(schedule_type),
            cron_expr=str(cron_expr) if cron_expr is not None else None,
            interval_seconds=int(interval_seconds)
            if interval_seconds is not None
            else None,
            timezone=str(timezone_name),
            start_at=_as_aware(start_at),
            next_run_at=_as_aware(next_run_at) or _utcnow(),
            last_run_at=_as_aware(last_run_at),
            misfire_policy=str(misfire_policy or "skip"),
            max_run_seconds=int(max_run_seconds or 900),
            retry_max_attempts=int(retry_max_attempts or 1),
            retry_backoff_seconds=int(retry_backoff_seconds or 60),
            action_type=str(action_type),
            action_config=_json_loads(action_config),
            delivery_type=str(delivery_type) if delivery_type is not None else None,
            delivery_config=_json_loads(delivery_config),
            locked_by=str(locked_by) if locked_by is not None else None,
            lock_expires_at=_as_aware(lock_expires_at),
            created_at=_as_aware(created_at),
            updated_at=_as_aware(updated_at),
        )

    def _select_job_columns(self) -> str:
        return (
            "job_id, agent_id, name, enabled, schedule_type, cron_expr, interval_seconds, "
            "timezone, start_at, next_run_at, last_run_at, misfire_policy, max_run_seconds, "
            "retry_max_attempts, retry_backoff_seconds, action_type, action_config, "
            "delivery_type, delivery_config, locked_by, lock_expires_at, created_at, updated_at"
        )

    def create_job(self, job: AutomationJob) -> AutomationJob:
        jobs_table = self._table("automation_jobs")
        payload = job.model_dump()

        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {jobs_table} (
                    job_id, agent_id, name, enabled,
                    schedule_type, cron_expr, interval_seconds, timezone,
                    start_at, next_run_at, last_run_at,
                    misfire_policy, max_run_seconds, retry_max_attempts, retry_backoff_seconds,
                    action_type, action_config,
                    delivery_type, delivery_config,
                    locked_by, lock_expires_at
                ) VALUES (
                    :job_id, :agent_id, :name, :enabled,
                    :schedule_type, :cron_expr, :interval_seconds, :timezone,
                    :start_at, :next_run_at, :last_run_at,
                    :misfire_policy, :max_run_seconds, :retry_max_attempts, :retry_backoff_seconds,
                    :action_type, :action_config,
                    :delivery_type, :delivery_config,
                    :locked_by, :lock_expires_at
                )
                """,
                {
                    "job_id": payload["job_id"],
                    "agent_id": payload["agent_id"],
                    "name": payload["name"],
                    "enabled": 1 if payload.get("enabled") else 0,
                    "schedule_type": payload["schedule_type"],
                    "cron_expr": payload.get("cron_expr"),
                    "interval_seconds": payload.get("interval_seconds"),
                    "timezone": payload["timezone"],
                    "start_at": payload.get("start_at"),
                    "next_run_at": payload["next_run_at"],
                    "last_run_at": payload.get("last_run_at"),
                    "misfire_policy": payload.get("misfire_policy") or "skip",
                    "max_run_seconds": int(payload.get("max_run_seconds") or 900),
                    "retry_max_attempts": int(payload.get("retry_max_attempts") or 1),
                    "retry_backoff_seconds": int(
                        payload.get("retry_backoff_seconds") or 60
                    ),
                    "action_type": payload["action_type"],
                    "action_config": _json_dumps(payload.get("action_config") or {}),
                    "delivery_type": payload.get("delivery_type"),
                    "delivery_config": _json_dumps(
                        payload.get("delivery_config") or {}
                    ),
                    "locked_by": payload.get("locked_by"),
                    "lock_expires_at": payload.get("lock_expires_at"),
                },
            )
            conn.commit()

        return self.get_job(job.job_id) or job

    def update_job(self, job_id: str, patch: Dict[str, Any]) -> AutomationJob:
        if not patch:
            existing = self.get_job(job_id)
            if not existing:
                raise ValueError("Job not found")
            return existing

        jobs_table = self._table("automation_jobs")
        allowed = {
            "name": "name",
            "enabled": "enabled",
            "schedule_type": "schedule_type",
            "cron_expr": "cron_expr",
            "interval_seconds": "interval_seconds",
            "timezone": "timezone",
            "start_at": "start_at",
            "next_run_at": "next_run_at",
            "last_run_at": "last_run_at",
            "misfire_policy": "misfire_policy",
            "max_run_seconds": "max_run_seconds",
            "retry_max_attempts": "retry_max_attempts",
            "retry_backoff_seconds": "retry_backoff_seconds",
            "action_type": "action_type",
            "action_config": "action_config",
            "delivery_type": "delivery_type",
            "delivery_config": "delivery_config",
            "locked_by": "locked_by",
            "lock_expires_at": "lock_expires_at",
        }

        set_parts: List[str] = []
        params: Dict[str, Any] = {"job_id": job_id}
        for key, value in patch.items():
            if key not in allowed:
                continue
            col = allowed[key]
            bind = f"v_{key}"
            if key in {"enabled"}:
                params[bind] = 1 if bool(value) else 0
            elif key in {"action_config", "delivery_config"}:
                params[bind] = _json_dumps(value or {})
            else:
                params[bind] = value
            set_parts.append(f"{col} = :{bind}")

        if not set_parts:
            existing = self.get_job(job_id)
            if not existing:
                raise ValueError("Job not found")
            return existing

        set_sql = ", ".join(set_parts) + ", updated_at = CURRENT_TIMESTAMP"
        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {jobs_table} SET {set_sql} WHERE job_id = :job_id",
                params,
            )
            conn.commit()

        updated = self.get_job(job_id)
        if not updated:
            raise ValueError("Job not found after update")
        return updated

    def get_job(self, job_id: str) -> Optional[AutomationJob]:
        jobs_table = self._table("automation_jobs")
        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self._select_job_columns()} FROM {jobs_table} WHERE job_id = :job_id",
                {"job_id": job_id},
            )
            row = cursor.fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(
        self, agent_id: Optional[str] = None, enabled: Optional[bool] = None
    ) -> List[AutomationJob]:
        jobs_table = self._table("automation_jobs")
        where = []
        params: Dict[str, Any] = {}
        if agent_id:
            where.append("agent_id = :agent_id")
            params["agent_id"] = agent_id
        if enabled is not None:
            where.append("enabled = :enabled")
            params["enabled"] = 1 if enabled else 0
        where_sql = " WHERE " + " AND ".join(where) if where else ""

        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self._select_job_columns()} FROM {jobs_table}{where_sql} ORDER BY created_at DESC",
                params,
            )
            rows = cursor.fetchall() or []
        return [self._row_to_job(row) for row in rows]

    def pause_job(self, job_id: str) -> AutomationJob:
        return self.update_job(job_id, {"enabled": False})

    def resume_job(self, job_id: str) -> AutomationJob:
        return self.update_job(job_id, {"enabled": True})

    def delete_job(self, job_id: str) -> bool:
        jobs_table = self._table("automation_jobs")
        runs_table = self._table("automation_runs")
        deliveries_table = self._table("automation_deliveries")

        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT run_id FROM {runs_table} WHERE job_id = :job_id",
                {"job_id": job_id},
            )
            run_rows = cursor.fetchall() or []
            run_ids = [str(row[0]) for row in run_rows if row and row[0] is not None]
            if run_ids:
                # Delete deliveries first
                in_list = ", ".join([f":rid{i}" for i in range(len(run_ids))])
                params = {f"rid{i}": run_id for i, run_id in enumerate(run_ids)}
                cursor.execute(
                    f"DELETE FROM {deliveries_table} WHERE run_id IN ({in_list})",
                    params,
                )
                cursor.execute(
                    f"DELETE FROM {runs_table} WHERE run_id IN ({in_list})",
                    params,
                )

            cursor.execute(
                f"DELETE FROM {jobs_table} WHERE job_id = :job_id",
                {"job_id": job_id},
            )
            deleted = cursor.rowcount or 0
            conn.commit()
        return deleted > 0

    def claim_due_jobs(
        self,
        worker_id: str,
        now_utc: datetime,
        limit: int,
        lease_seconds: int,
    ) -> List[AutomationJob]:
        jobs_table = self._table("automation_jobs")
        lease_expiry = now_utc + timedelta(seconds=int(lease_seconds or 0))
        safe_limit = max(1, int(limit or 1))

        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {self._select_job_columns()}
                FROM {jobs_table}
                WHERE enabled = 1
                  AND next_run_at <= :now_utc
                  AND (lock_expires_at IS NULL OR lock_expires_at <= :now_utc)
                ORDER BY next_run_at
                FETCH FIRST {safe_limit} ROWS ONLY
                FOR UPDATE SKIP LOCKED
                """,
                {"now_utc": now_utc},
            )
            rows = cursor.fetchall() or []
            jobs = [self._row_to_job(row) for row in rows]

            for job in jobs:
                cursor.execute(
                    f"""
                    UPDATE {jobs_table}
                    SET locked_by = :worker_id,
                        lock_expires_at = :lease_expiry,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = :job_id
                    """,
                    {
                        "worker_id": worker_id,
                        "lease_expiry": lease_expiry,
                        "job_id": job.job_id,
                    },
                )
                job.locked_by = worker_id
                job.lock_expires_at = lease_expiry

            conn.commit()
        return jobs

    def claim_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        now_utc: datetime,
        lease_seconds: int,
        force_enable: bool = True,
    ) -> Optional[AutomationJob]:
        """Claim a specific job by ID for immediate execution.

        Returns None if the job is locked by another worker or does not exist.
        """
        jobs_table = self._table("automation_jobs")
        lease_expiry = now_utc + timedelta(seconds=int(lease_seconds or 0))

        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {self._select_job_columns()}
                FROM {jobs_table}
                WHERE job_id = :job_id
                  AND (lock_expires_at IS NULL OR lock_expires_at <= :now_utc)
                FOR UPDATE SKIP LOCKED
                """,
                {"job_id": job_id, "now_utc": now_utc},
            )
            row = cursor.fetchone()
            if not row:
                conn.commit()
                return None

            job = self._row_to_job(row)

            cursor.execute(
                f"""
                UPDATE {jobs_table}
                SET locked_by = :worker_id,
                    lock_expires_at = :lease_expiry,
                    enabled = :enabled,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = :job_id
                """,
                {
                    "worker_id": worker_id,
                    "lease_expiry": lease_expiry,
                    "enabled": 1 if force_enable else (1 if job.enabled else 0),
                    "job_id": job.job_id,
                },
            )
            conn.commit()

        job.locked_by = worker_id
        job.lock_expires_at = lease_expiry
        if force_enable:
            job.enabled = True
        return job

    def start_run(
        self, job: AutomationJob, scheduled_for: datetime, worker_id: str
    ) -> AutomationRun:
        runs_table = self._table("automation_runs")
        run_id = str(uuid.uuid4())
        started_at = _utcnow()

        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {runs_table} (
                    run_id, job_id, scheduled_for, started_at, status, attempt,
                    error, result_summary, result_payload
                ) VALUES (
                    :run_id, :job_id, :scheduled_for, :started_at, :status, :attempt,
                    :error, :result_summary, :result_payload
                )
                """,
                {
                    "run_id": run_id,
                    "job_id": job.job_id,
                    "scheduled_for": scheduled_for,
                    "started_at": started_at,
                    "status": "running",
                    "attempt": 1,
                    "error": None,
                    "result_summary": None,
                    "result_payload": _json_dumps({}),
                },
            )
            conn.commit()

        return AutomationRun(
            run_id=run_id,
            job_id=job.job_id,
            scheduled_for=_as_aware(scheduled_for) or scheduled_for,
            started_at=started_at,
            status="running",
            attempt=1,
        )

    def finish_run(
        self,
        run_id: str,
        status: str,
        error: Optional[str],
        result_summary: Optional[str],
        result_payload: Optional[Dict[str, Any]],
        attempt: int = 1,
    ) -> AutomationRun:
        runs_table = self._table("automation_runs")
        finished_at = _utcnow()

        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE {runs_table}
                SET finished_at = :finished_at,
                    status = :status,
                    attempt = :attempt,
                    error = :error,
                    result_summary = :result_summary,
                    result_payload = :result_payload
                WHERE run_id = :run_id
                """,
                {
                    "finished_at": finished_at,
                    "status": status,
                    "attempt": int(attempt or 1),
                    "error": error,
                    "result_summary": result_summary,
                    "result_payload": _json_dumps(result_payload or {}),
                    "run_id": run_id,
                },
            )
            conn.commit()

            cursor.execute(
                f"""
                SELECT run_id, job_id, scheduled_for, started_at, finished_at,
                       status, attempt, error, result_summary, result_payload, created_at
                FROM {runs_table}
                WHERE run_id = :run_id
                """,
                {"run_id": run_id},
            )
            row = cursor.fetchone()

        if not row:
            raise ValueError("Run not found after update")

        (
            run_id_val,
            job_id,
            scheduled_for,
            started_at,
            finished_at_val,
            status_val,
            attempt_val,
            error_val,
            summary_val,
            payload_val,
            created_at,
        ) = row

        return AutomationRun(
            run_id=str(run_id_val),
            job_id=str(job_id),
            scheduled_for=_as_aware(scheduled_for) or scheduled_for,
            started_at=_as_aware(started_at),
            finished_at=_as_aware(finished_at_val),
            status=str(status_val),
            attempt=int(attempt_val or 1),
            error=str(error_val) if error_val is not None else None,
            result_summary=str(summary_val) if summary_val is not None else None,
            result_payload=_json_loads(payload_val),
            created_at=_as_aware(created_at),
        )

    def record_delivery(
        self, run_id: str, delivery: AutomationDelivery
    ) -> AutomationDelivery:
        deliveries_table = self._table("automation_deliveries")
        payload = delivery.model_dump()

        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {deliveries_table} (
                    delivery_id, run_id, channel, provider, recipient,
                    status, provider_message_id, error
                ) VALUES (
                    :delivery_id, :run_id, :channel, :provider, :recipient,
                    :status, :provider_message_id, :error
                )
                """,
                {
                    "delivery_id": payload["delivery_id"],
                    "run_id": run_id,
                    "channel": payload.get("channel") or "whatsapp",
                    "provider": payload.get("provider") or "twilio",
                    "recipient": payload["recipient"],
                    "status": payload["status"],
                    "provider_message_id": payload.get("provider_message_id"),
                    "error": payload.get("error"),
                },
            )
            conn.commit()

        return delivery

    def list_runs(self, job_id: str, limit: int = 50) -> List[AutomationRun]:
        runs_table = self._table("automation_runs")
        safe_limit = max(1, int(limit or 1))

        with self.provider.pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT run_id, job_id, scheduled_for, started_at, finished_at,
                       status, attempt, error, result_summary, result_payload, created_at
                FROM {runs_table}
                WHERE job_id = :job_id
                ORDER BY created_at DESC
                FETCH FIRST {safe_limit} ROWS ONLY
                """,
                {"job_id": job_id},
            )
            rows = cursor.fetchall() or []

        runs: List[AutomationRun] = []
        for row in rows:
            (
                run_id_val,
                job_id_val,
                scheduled_for,
                started_at,
                finished_at_val,
                status_val,
                attempt_val,
                error_val,
                summary_val,
                payload_val,
                created_at,
            ) = row
            runs.append(
                AutomationRun(
                    run_id=str(run_id_val),
                    job_id=str(job_id_val),
                    scheduled_for=_as_aware(scheduled_for) or scheduled_for,
                    started_at=_as_aware(started_at),
                    finished_at=_as_aware(finished_at_val),
                    status=str(status_val),
                    attempt=int(attempt_val or 1),
                    error=str(error_val) if error_val is not None else None,
                    result_summary=str(summary_val)
                    if summary_val is not None
                    else None,
                    result_payload=_json_loads(payload_val),
                    created_at=_as_aware(created_at),
                )
            )
        return runs
