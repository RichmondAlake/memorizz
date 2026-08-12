# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Automation manager: exposes scheduling tools to MemAgent when supported."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from ...automation.models import AutomationJob
from ...automation.schedule import compute_next_run_at, utcnow, validate_timezone_name
from ...automation.store.factory import get_automation_store
from ...tooling import governed_tool

# Tool names registered by ``register_tools``. The automation runner strips these
# for scheduled runs (an executing agent should do its task, not manage other
# automations) — keep in sync with the add_tool calls in register_tools.
AUTOMATION_TOOL_NAMES = (
    "automation_create_job",
    "automation_list_jobs",
    "automation_pause_job",
    "automation_resume_job",
    "automation_delete_job",
    "automation_run_now",
)


class AutomationManager:
    def __init__(self, memory_provider: Any, enabled: bool = True):
        self._enabled = bool(enabled)
        self._store = get_automation_store(memory_provider) if self._enabled else None

    def is_enabled(self) -> bool:
        return self._enabled and self._store is not None

    @property
    def store(self):
        return self._store

    def register_tools(
        self,
        tool_manager: Any,
        *,
        agent_id: str,
        default_timezone: Optional[str] = None,
    ) -> None:
        if not self.is_enabled() or tool_manager is None:
            return
        store = self._store
        assert store is not None

        def _resolve_timezone(value: str) -> str:
            tz = str(value or "").strip()
            if tz:
                return tz
            tz = str(default_timezone or "").strip()
            if tz:
                return tz
            tz = str(os.environ.get("MEMORIZZ_DEFAULT_TIMEZONE", "")).strip()
            if tz:
                return tz
            raise ValueError("timezone is required (or set MEMORIZZ_DEFAULT_TIMEZONE)")

        def _normalize_whatsapp_to(values: Any) -> List[str]:
            if values is None:
                raw_list: List[str] = []
            elif isinstance(values, str):
                raw_list = [values]
            elif isinstance(values, list):
                raw_list = values
            else:
                raw_list = []

            normalized: List[str] = []
            seen = set()
            for item in raw_list:
                text = str(item or "").strip()
                if not text:
                    continue
                to_val = (
                    text if text.lower().startswith("whatsapp:") else f"whatsapp:{text}"
                )
                if to_val in seen:
                    continue
                seen.add(to_val)
                normalized.append(to_val)
            return normalized

        @governed_tool(
            deterministic=False,
            side_effects=True,
            requires_approval=True,
            approval_reason="Create a durable scheduled automation",
            domains=("automations",),
        )
        def automation_create_job(
            name: str,
            schedule_type: str,
            cron_expr: str = "",
            interval_seconds: int = 0,
            timezone: str = "",  # noqa: F811
            query_template: str = "",
            memory_id: str = "",
            whatsapp_to: List[str] = None,
            client_request_id: str = "",
        ) -> Dict[str, Any]:
            """Create a scheduled automation job for this agent."""
            if store is None:
                return {"ok": False, "error": "Automations are unavailable (no store)."}

            job_name = str(name or "").strip()
            if not job_name:
                return {"ok": False, "error": "name is required."}

            schedule = str(schedule_type or "").strip().lower()
            tz_name = _resolve_timezone(timezone)
            validate_timezone_name(tz_name)

            query = str(query_template or "").strip()
            if not query:
                return {"ok": False, "error": "query_template is required."}

            mem_id = str(memory_id or "").strip()
            if not mem_id:
                mem_id = str(uuid.uuid4())

            to_list = _normalize_whatsapp_to(whatsapp_to)

            now_utc = utcnow()
            next_run_at = compute_next_run_at(
                schedule_type=schedule,
                cron_expr=str(cron_expr or "").strip() or None,
                interval_seconds=int(interval_seconds or 0) or None,
                tz_name=tz_name,
                after_utc=now_utc,
            )

            resolved_job_id = (
                f"{agent_id}:{str(client_request_id).strip()}"
                if str(client_request_id or "").strip()
                else str(uuid.uuid4())
            )

            proposed = {
                "job_id": resolved_job_id,
                "agent_id": agent_id,
                "name": job_name,
                "enabled": True,
                "schedule_type": schedule,
                "cron_expr": str(cron_expr or "").strip() or None,
                "interval_seconds": int(interval_seconds or 0) or None,
                "timezone": tz_name,
                "start_at": now_utc,
                "next_run_at": next_run_at,
                "misfire_policy": "skip",
                "max_run_seconds": 900,
                "retry_max_attempts": 1,
                "retry_backoff_seconds": 60,
                "action_type": "agent_query",
                "action_config": {
                    "query_template": query,
                    "memory_id": mem_id,
                    "client_request_id": str(client_request_id or "").strip() or None,
                },
                "delivery_type": "whatsapp_twilio" if to_list else None,
                "delivery_config": {"whatsapp_to": to_list} if to_list else {},
            }

            job = AutomationJob(**proposed)
            try:
                created = store.create_job(job)
            except Exception as exc:
                # If idempotency hit, return existing.
                existing = store.get_job(job.job_id)
                if existing:
                    created = existing
                else:
                    return {"ok": False, "error": str(exc)}

            return {
                "ok": True,
                "created": True,
                "job_id": created.job_id,
                "next_run_at": created.next_run_at.isoformat(),
                "job": created.model_dump(),
            }

        def automation_list_jobs() -> Dict[str, Any]:
            """List automation jobs for this agent."""
            if store is None:
                return {"ok": False, "error": "Automations are unavailable (no store)."}
            jobs = store.list_jobs(agent_id=agent_id, enabled=None)
            return {
                "ok": True,
                "count": len(jobs),
                "jobs": [job.model_dump() for job in jobs],
            }

        @governed_tool(
            side_effects=True,
            requires_approval=True,
            approval_reason="Pause a durable automation",
            domains=("automations",),
        )
        def automation_pause_job(job_id: str) -> Dict[str, Any]:
            """Pause a job by setting enabled=false."""
            if store is None:
                return {"ok": False, "error": "Automations are unavailable (no store)."}
            updated = store.pause_job(str(job_id or "").strip())
            return {"ok": True, "job": updated.model_dump()}

        @governed_tool(
            side_effects=True,
            requires_approval=True,
            approval_reason="Resume a durable automation",
            domains=("automations",),
        )
        def automation_resume_job(job_id: str) -> Dict[str, Any]:
            """Resume a paused job by setting enabled=true."""
            if store is None:
                return {"ok": False, "error": "Automations are unavailable (no store)."}
            updated = store.resume_job(str(job_id or "").strip())
            return {"ok": True, "job": updated.model_dump()}

        @governed_tool(
            side_effects=True,
            requires_approval=True,
            approval_reason="Permanently delete a durable automation",
            domains=("automations",),
        )
        def automation_delete_job(job_id: str) -> Dict[str, Any]:
            """Delete a job."""
            if store is None:
                return {"ok": False, "error": "Automations are unavailable (no store)."}
            normalized = str(job_id or "").strip()
            if not normalized:
                return {"ok": False, "error": "job_id is required."}
            deleted = bool(store.delete_job(normalized))
            return {"ok": True, "deleted": deleted}

        @governed_tool(
            side_effects=True,
            requires_approval=True,
            approval_reason="Trigger an automation immediately",
            domains=("automations",),
        )
        def automation_run_now(job_id: str) -> Dict[str, Any]:
            """Trigger an immediate run by setting next_run_at to now."""
            if store is None:
                return {"ok": False, "error": "Automations are unavailable (no store)."}
            normalized = str(job_id or "").strip()
            if not normalized:
                return {"ok": False, "error": "job_id is required."}
            now_utc = utcnow()
            updated = store.update_job(
                normalized,
                {
                    "next_run_at": now_utc,
                    "enabled": True,
                    "locked_by": None,
                    "lock_expires_at": None,
                },
            )
            return {
                "ok": True,
                "job": updated.model_dump(),
                "next_run_at": updated.next_run_at.isoformat(),
            }

        automation_create_job.__name__ = "automation_create_job"
        automation_list_jobs.__name__ = "automation_list_jobs"
        automation_pause_job.__name__ = "automation_pause_job"
        automation_resume_job.__name__ = "automation_resume_job"
        automation_delete_job.__name__ = "automation_delete_job"
        automation_run_now.__name__ = "automation_run_now"

        tool_manager.add_tool(automation_create_job)
        tool_manager.add_tool(automation_list_jobs)
        tool_manager.add_tool(automation_pause_job)
        tool_manager.add_tool(automation_resume_job)
        tool_manager.add_tool(automation_delete_job)
        tool_manager.add_tool(automation_run_now)
