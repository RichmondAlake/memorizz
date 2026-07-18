# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Skill document model for the Skillbox.

A learned skill is a SKILL.md document that lives in the database with a
lifecycle attached. It is distilled from repeated successful workflow
trajectories and injected into agent context as a strong prior — never a
mandate.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ....embeddings import get_embedding

# Distinguishes "no embedding supplied — generate one" (fresh skill) from
# "document loaded without its embedding" (provider projections exclude the
# vector by default). Loading must NEVER re-bill the embedding API.
_UNSET = object()


class SkillStatus(Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DEMOTED = "demoted"


def _default_stats() -> Dict[str, Any]:
    return {
        "activations": 0,
        "successes": 0,
        "failures": 0,
        # Activated but the run took a different canonical_hash — the agent
        # had the skill in context and went another way. Signal, not noise.
        "deviations": 0,
        "last_activated_at": None,
        # Ring buffer of the most recent activation outcomes (True=success),
        # capped by the monitor's drift window. Avoids re-querying workflow
        # memory to compute the rolling success rate.
        "recent_outcomes": [],
    }


class Skill:
    """A promoted skill document (``MemoryType.SKILLBOX``)."""

    def __init__(
        self,
        name: str,
        description: str,
        content: str,
        preconditions: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        skill_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        source_canonical_hash: Optional[str] = None,
        source_workflow_ids: Optional[List[str]] = None,
        exemplar_workflow_id: Optional[str] = None,
        status: SkillStatus = SkillStatus.CANDIDATE,
        version: int = 1,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        promoted_at: Optional[datetime] = None,
        demoted_at: Optional[datetime] = None,
        demotion_reason: Optional[str] = None,
        baseline: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        embedding: Any = _UNSET,
    ):
        self.name = name
        self.description = description
        self.content = content
        self.preconditions = list(preconditions or [])
        self.tools_used = list(tools_used or [])
        self.queries = list(queries or [])
        self.skill_id = skill_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.user_id = user_id
        self.source_canonical_hash = source_canonical_hash
        self.source_workflow_ids = list(source_workflow_ids or [])
        self.exemplar_workflow_id = exemplar_workflow_id
        self.status = (
            status if isinstance(status, SkillStatus) else SkillStatus(str(status))
        )
        self.version = int(version)
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.promoted_at = promoted_at
        self.demoted_at = demoted_at
        self.demotion_reason = demotion_reason
        self.baseline = dict(baseline) if baseline else {}
        self.stats = {**_default_stats(), **(stats or {})}
        # Reuse the stored embedding on round-trip; only embed fresh docs.
        # ``None`` (doc loaded under an embedding-excluding projection) is
        # preserved as None — regenerating here would bill the embedding API
        # on every list/get.
        self.embedding = (
            self._generate_embedding() if embedding is _UNSET else embedding
        )

    def _generate_embedding(self) -> List[float]:
        """Embed APPLICABILITY semantics only: name + description +
        preconditions + queries.

        Deliberately excludes ``content``, procedure steps, and the tool
        sequence. Workflow embeddings encode *how it was done*; skill
        retrieval must match *when to do it*. Including mechanism here would
        retrieve skills for queries that mention similar tools rather than
        similar intents — the most likely silent-quality bug in the system,
        so a unit test asserts on :meth:`embedding_input`.
        """
        return get_embedding(self.embedding_input())

    def embedding_input(self) -> str:
        """The exact text the skill embedding is generated from."""
        return " ".join(
            [
                self.name,
                self.description,
                " ".join(self.preconditions),
                " ".join(self.queries),
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "preconditions": self.preconditions,
            "tools_used": self.tools_used,
            "queries": self.queries,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "source_canonical_hash": self.source_canonical_hash,
            "source_workflow_ids": self.source_workflow_ids,
            "exemplar_workflow_id": self.exemplar_workflow_id,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "promoted_at": (self.promoted_at.isoformat() if self.promoted_at else None),
            "demoted_at": self.demoted_at.isoformat() if self.demoted_at else None,
            "demotion_reason": self.demotion_reason,
            "baseline": self.baseline,
            "stats": self.stats,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        def _dt(value):
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value))
            except (TypeError, ValueError):
                return None

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            content=data.get("content", ""),
            preconditions=data.get("preconditions"),
            tools_used=data.get("tools_used"),
            queries=data.get("queries"),
            skill_id=data.get("skill_id"),
            agent_id=data.get("agent_id"),
            user_id=data.get("user_id"),
            source_canonical_hash=data.get("source_canonical_hash"),
            source_workflow_ids=data.get("source_workflow_ids"),
            exemplar_workflow_id=data.get("exemplar_workflow_id"),
            status=SkillStatus(data.get("status", SkillStatus.CANDIDATE.value)),
            version=data.get("version", 1),
            created_at=_dt(data.get("created_at")),
            updated_at=_dt(data.get("updated_at")),
            promoted_at=_dt(data.get("promoted_at")),
            demoted_at=_dt(data.get("demoted_at")),
            demotion_reason=data.get("demotion_reason"),
            baseline=data.get("baseline"),
            stats=data.get("stats"),
            embedding=data.get("embedding"),
        )
