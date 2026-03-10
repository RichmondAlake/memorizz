# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Automation store interface (provider-agnostic)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from ..models import AutomationDelivery, AutomationJob, AutomationRun


class AutomationStore(Protocol):
    def create_job(self, job: AutomationJob) -> AutomationJob:
        ...

    def update_job(self, job_id: str, patch: Dict[str, Any]) -> AutomationJob:
        ...

    def get_job(self, job_id: str) -> Optional[AutomationJob]:
        ...

    def list_jobs(
        self, agent_id: Optional[str] = None, enabled: Optional[bool] = None
    ) -> List[AutomationJob]:
        ...

    def pause_job(self, job_id: str) -> AutomationJob:
        ...

    def resume_job(self, job_id: str) -> AutomationJob:
        ...

    def delete_job(self, job_id: str) -> bool:
        ...

    def claim_due_jobs(
        self,
        worker_id: str,
        now_utc: datetime,
        limit: int,
        lease_seconds: int,
    ) -> List[AutomationJob]:
        ...

    def start_run(
        self, job: AutomationJob, scheduled_for: datetime, worker_id: str
    ) -> AutomationRun:
        ...

    def finish_run(
        self,
        run_id: str,
        status: str,
        error: Optional[str],
        result_summary: Optional[str],
        result_payload: Optional[Dict[str, Any]],
        attempt: int = 1,
    ) -> AutomationRun:
        ...

    def record_delivery(
        self, run_id: str, delivery: AutomationDelivery
    ) -> AutomationDelivery:
        ...

    def list_runs(self, job_id: str, limit: int = 50) -> List[AutomationRun]:
        ...
