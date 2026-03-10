# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Pydantic models for automations (jobs, runs, deliveries)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ScheduleType = Literal["cron", "interval", "one_shot"]
RunStatus = Literal["running", "succeeded", "failed", "canceled"]
DeliveryStatus = Literal["queued", "sent", "failed"]


class AutomationJob(BaseModel):
    job_id: str
    agent_id: str
    name: str

    enabled: bool = True

    schedule_type: ScheduleType
    cron_expr: Optional[str] = None
    interval_seconds: Optional[int] = None
    timezone: str

    start_at: Optional[datetime] = None
    next_run_at: datetime
    last_run_at: Optional[datetime] = None

    misfire_policy: str = "skip"
    max_run_seconds: int = 900
    retry_max_attempts: int = 1
    retry_backoff_seconds: int = 60

    action_type: str
    action_config: Dict[str, Any] = Field(default_factory=dict)

    delivery_type: Optional[str] = None
    delivery_config: Dict[str, Any] = Field(default_factory=dict)

    locked_by: Optional[str] = None
    lock_expires_at: Optional[datetime] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AutomationRun(BaseModel):
    run_id: str
    job_id: str

    scheduled_for: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    status: RunStatus
    attempt: int = 1
    error: Optional[str] = None

    result_summary: Optional[str] = None
    result_payload: Dict[str, Any] = Field(default_factory=dict)

    created_at: Optional[datetime] = None


class AutomationDelivery(BaseModel):
    delivery_id: str
    run_id: str

    channel: str = "whatsapp"
    provider: str = "twilio"
    recipient: str

    status: DeliveryStatus
    provider_message_id: Optional[str] = None
    error: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ActionSpec(BaseModel):
    """Convenience action spec used by tools/UI inputs."""

    action_type: str = "agent_query"
    query_template: str
    memory_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeliverySpec(BaseModel):
    """Convenience delivery spec used by tools/UI inputs."""

    delivery_type: str = "whatsapp_twilio"
    whatsapp_to: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
