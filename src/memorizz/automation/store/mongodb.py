# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""MongoDB-backed automation store.

Jobs, runs, and deliveries live in the ``automation_jobs`` / ``automation_runs``
/ ``automation_deliveries`` collections of the provider's database. Claiming
uses ``find_one_and_update`` so it is atomic across concurrent workers. Datetimes
are stored as native BSON dates so schedule queries (``next_run_at <= now``) work
server-side.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..models import AutomationDelivery, AutomationJob, AutomationRun
from .filesystem import _UPDATABLE_FIELDS


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


def _strip_id(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if doc is None:
        return None
    doc.pop("_id", None)
    return doc


class MongoDBAutomationStore:
    def __init__(self, provider: Any):
        self.provider = provider
        db = provider.db
        self.jobs = db["automation_jobs"]
        self.runs = db["automation_runs"]
        self.deliveries = db["automation_deliveries"]
        try:
            self.jobs.create_index("job_id", unique=True)
            self.jobs.create_index([("agent_id", 1)])
            self.jobs.create_index([("next_run_at", 1)])
            self.runs.create_index([("job_id", 1)])
            self.deliveries.create_index([("run_id", 1)])
        except Exception:
            pass

    # ----------------------------------------------------------------- jobs
    def create_job(self, job: AutomationJob) -> AutomationJob:
        now = _utcnow()
        if job.created_at is None:
            job.created_at = now
        job.updated_at = now
        self.jobs.insert_one(job.model_dump(mode="python"))
        return job

    def get_job(self, job_id: str) -> Optional[AutomationJob]:
        doc = self.jobs.find_one({"job_id": job_id})
        return AutomationJob.model_validate(_strip_id(doc)) if doc else None

    def update_job(self, job_id: str, patch: Dict[str, Any]) -> AutomationJob:
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
            self.jobs.replace_one({"job_id": job_id}, job.model_dump(mode="python"))
        return job

    def list_jobs(
        self, agent_id: Optional[str] = None, enabled: Optional[bool] = None
    ) -> List[AutomationJob]:
        query: Dict[str, Any] = {}
        if agent_id is not None:
            query["agent_id"] = agent_id
        if enabled is not None:
            query["enabled"] = enabled
        docs = self.jobs.find(query).sort("created_at", -1)
        return [AutomationJob.model_validate(_strip_id(d)) for d in docs]

    def pause_job(self, job_id: str) -> AutomationJob:
        return self.update_job(job_id, {"enabled": False})

    def resume_job(self, job_id: str) -> AutomationJob:
        return self.update_job(job_id, {"enabled": True})

    def delete_job(self, job_id: str) -> bool:
        run_ids = [
            d["run_id"] for d in self.runs.find({"job_id": job_id}, {"run_id": 1})
        ]
        if run_ids:
            self.deliveries.delete_many({"run_id": {"$in": run_ids}})
            self.runs.delete_many({"job_id": job_id})
        result = self.jobs.delete_one({"job_id": job_id})
        return (result.deleted_count or 0) > 0

    # -------------------------------------------------------------- claiming
    def claim_due_jobs(
        self,
        worker_id: str,
        now_utc: datetime,
        limit: int,
        lease_seconds: int,
    ) -> List[AutomationJob]:
        from pymongo import ReturnDocument

        now = _as_aware(now_utc) or _utcnow()
        lease_expiry = now + timedelta(seconds=int(lease_seconds or 0))
        claimed: List[AutomationJob] = []
        for _ in range(max(1, int(limit or 1))):
            doc = self.jobs.find_one_and_update(
                {
                    "enabled": True,
                    "next_run_at": {"$lte": now},
                    "$or": [
                        {"lock_expires_at": None},
                        {"lock_expires_at": {"$lte": now}},
                    ],
                },
                {
                    "$set": {
                        "locked_by": worker_id,
                        "lock_expires_at": lease_expiry,
                        "updated_at": now,
                    }
                },
                sort=[("next_run_at", 1)],
                return_document=ReturnDocument.AFTER,
            )
            if not doc:
                break
            claimed.append(AutomationJob.model_validate(_strip_id(doc)))
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
        from pymongo import ReturnDocument

        now = _as_aware(now_utc) or _utcnow()
        lease_expiry = now + timedelta(seconds=int(lease_seconds or 0))
        update: Dict[str, Any] = {
            "locked_by": worker_id,
            "lock_expires_at": lease_expiry,
            "updated_at": now,
        }
        if force_enable:
            update["enabled"] = True
        doc = self.jobs.find_one_and_update(
            {
                "job_id": job_id,
                "$or": [
                    {"lock_expires_at": None},
                    {"lock_expires_at": {"$lte": now}},
                ],
            },
            {"$set": update},
            return_document=ReturnDocument.AFTER,
        )
        return AutomationJob.model_validate(_strip_id(doc)) if doc else None

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
        self.runs.insert_one(run.model_dump(mode="python"))
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
        from pymongo import ReturnDocument

        doc = self.runs.find_one_and_update(
            {"run_id": run_id},
            {
                "$set": {
                    "finished_at": _utcnow(),
                    "status": status,
                    "error": error,
                    "result_summary": result_summary,
                    "result_payload": result_payload or {},
                    "attempt": int(attempt or 1),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not doc:
            raise ValueError("Run not found")
        return AutomationRun.model_validate(_strip_id(doc))

    def list_runs(self, job_id: str, limit: int = 50) -> List[AutomationRun]:
        docs = (
            self.runs.find({"job_id": job_id})
            .sort("created_at", -1)
            .limit(max(1, int(limit or 1)))
        )
        return [AutomationRun.model_validate(_strip_id(d)) for d in docs]

    # ----------------------------------------------------------- deliveries
    def record_delivery(
        self, run_id: str, delivery: AutomationDelivery
    ) -> AutomationDelivery:
        if delivery.created_at is None:
            delivery.created_at = _utcnow()
        self.deliveries.insert_one(delivery.model_dump(mode="python"))
        return delivery
