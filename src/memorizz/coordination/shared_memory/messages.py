# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Typed message helpers for shared memory coordination."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SharedMemoryMessageType(str, Enum):
    """Supported shared memory message types."""

    COMMAND = "COMMAND"
    STATUS = "STATUS"
    REPORT = "REPORT"
    QUESTION = "QUESTION"


@dataclass
class SharedMemoryMessage:
    """Base shared memory message."""

    message_type: SharedMemoryMessageType
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable payload."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "created_at": self.created_at,
            "payload": self.payload,
        }


def _validate_fields(data: Dict[str, Any], fields: List[str], message_type: str):
    missing = [field for field in fields if not data.get(field)]
    if missing:
        raise ValueError(
            f"{message_type} message missing required fields: {', '.join(missing)}"
        )


def create_command_message(
    command_id: str,
    target_agent_id: str,
    instructions: str,
    priority: int = 3,
    dependencies: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SharedMemoryMessage:
    """Build a validated COMMAND message payload."""
    payload = {
        "command_id": command_id,
        "target_agent_id": target_agent_id,
        "instructions": instructions,
        "priority": priority,
        "dependencies": dependencies or [],
        "metadata": metadata or {},
    }
    _validate_fields(
        payload, ["command_id", "target_agent_id", "instructions"], "COMMAND"
    )
    return SharedMemoryMessage(SharedMemoryMessageType.COMMAND, payload)


def create_status_message(
    command_id: str,
    agent_id: str,
    status: str,
    progress: int,
    blockers: Optional[str] = None,
    summary_ids: Optional[List[str]] = None,
) -> SharedMemoryMessage:
    """Build a validated STATUS message payload."""
    payload = {
        "command_id": command_id,
        "agent_id": agent_id,
        "status": status,
        "progress": max(0, min(progress, 100)),
        "blockers": blockers,
        "summary_ids": summary_ids or [],
    }
    _validate_fields(payload, ["command_id", "agent_id", "status"], "STATUS")
    return SharedMemoryMessage(SharedMemoryMessageType.STATUS, payload)


def create_report_message(
    command_id: str,
    agent_id: str,
    findings: str,
    citations: Optional[List[str]] = None,
    gaps: Optional[List[str]] = None,
    summary_ids: Optional[List[str]] = None,
) -> SharedMemoryMessage:
    """Build a validated REPORT message payload."""
    payload = {
        "command_id": command_id,
        "agent_id": agent_id,
        "findings": findings,
        "citations": citations or [],
        "gaps": gaps or [],
        "summary_ids": summary_ids or [],
    }
    _validate_fields(payload, ["command_id", "agent_id", "findings"], "REPORT")
    return SharedMemoryMessage(SharedMemoryMessageType.REPORT, payload)
