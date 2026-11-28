# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""A next step an agent offered."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Waiting for the user; being carried out; carried out; waiting for a tool
#: call to be approved; the run failed; the user said no; the user said not
#: now; nobody answered in time.
STATUSES = (
    "offered",
    "running",
    "done",
    "awaiting_approval",
    "failed",
    "dismissed",
    "snoozed",
    "expired",
)


@dataclass
class Suggestion:
    """One step the agent thought of.

    ``decision`` is what the agent's level made of it: ``"ask"`` (shown, and
    nothing runs until it is accepted) or ``"auto"`` (the agent carried it out
    on its own). ``reason`` says why in a sentence. ``origin`` is
    ``"grounded"`` for a step that follows from what is in front of the agent,
    ``"predicted"`` for one guessed from the user's habits.
    """

    id: str
    title: str
    message: str
    prompt: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    origin: str = "grounded"
    routine: str = ""
    external: bool = False
    status: str = "offered"
    decision: str = "ask"
    reason: str = ""
    by: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    proposal_id: Optional[str] = None
    memory_id: Optional[str] = None
    thread_id: Optional[str] = None
    trigger: str = "view"
    #: What the user was looking at when the step was thought of.
    about: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "prompt": self.prompt,
            "evidence": list(self.evidence),
            "origin": self.origin,
            "routine": self.routine,
            "external": self.external,
            "status": self.status,
            "decision": self.decision,
            "reason": self.reason,
            "by": self.by,
            "result": self.result,
            "error": self.error,
            "proposal_id": self.proposal_id,
            "memory_id": self.memory_id,
            "thread_id": self.thread_id,
            "trigger": self.trigger,
            "about": dict(self.about or {}),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_event(cls, event: Dict[str, Any]) -> "Suggestion":
        names = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in event.items() if key in names})

    def __str__(self) -> str:
        did = {
            "done": "Done" if self.by == "agent" else "Carried out",
            "awaiting_approval": "Waiting for approval",
            "failed": "Failed",
        }.get(self.status, "Suggestion")
        return f"{did}: {self.title} - {self.message}"
