# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Governed, reversible forgetting for primary memory records.

Generative Agents never delete: old, unimportant, unused memories simply
sink in the recency/importance/relevance ranking. MemoRizz keeps that as the
always-on tier and adds a governed second tier for records that would
otherwise distort retrieval or grow storage forever. A plan is a dry run; an
approver applies it; the effect is *suppression* (``retention_state =
"suppressed"``), which every retrieval path honours and which can be undone.
Nothing here deletes a record, and audit evidence is never touched.

Retention score (0..1) per record::

    rho = mean(importance, usage, recency)
    usage   = log1p(access_count) / log1p(max access_count in the set)
    recency = decay ** hours_since_last_access   (decay from RetrievalScoring)

A record is a candidate when ``rho < min_retention`` for at least
``grace_days`` since it was created, unless it is pinned, verified, already
suppressed, or cited as a source by another record (explicit lineage only).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..enums.memory_type import MemoryType
from ..memagent.utils.context_dedup import (
    RetrievalScoring,
    _parse_timestamp,
    is_suppressed,
)
from ..memagent.utils.importance import clamp_importance
from .forgetting import ForgettingMechanism
from .models import ForgettingAction, ForgettingCandidate, ForgettingReport, utc_now

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_MEMORY_TYPES: Tuple[str, ...] = (
    MemoryType.CONVERSATION_MEMORY.value,
    MemoryType.KNOWLEDGE_BASE.value,
    MemoryType.ENTITY_MEMORY.value,
    MemoryType.SUMMARIES.value,
    MemoryType.WORKFLOW_MEMORY.value,
)

# Explicit lineage fields: a record referenced by any of these on an active
# record is never suppressed (its evidence is still in use).
_CITATION_FIELDS = (
    "source_id",
    "source_ids",
    "source_record_ids",
    "parent_id",
    "supersedes",
    "supersedes_id",
    "derived_from",
    "source_message_ids",
    "covered_message_ids",
    "message_ids",
    "linked_source_ids",
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RetentionConfig:
    enabled: bool = False
    min_retention: float = 0.15
    grace_days: float = 30.0
    protect_verified: bool = True
    protect_pinned: bool = True
    max_candidates: int = 500
    memory_types: Tuple[str, ...] = DEFAULT_RETENTION_MEMORY_TYPES

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _truthy(self.enabled))
        object.__setattr__(
            self, "min_retention", min(1.0, max(0.0, float(self.min_retention)))
        )
        object.__setattr__(self, "grace_days", max(0.0, float(self.grace_days)))
        object.__setattr__(self, "protect_verified", _truthy(self.protect_verified))
        object.__setattr__(self, "protect_pinned", _truthy(self.protect_pinned))
        object.__setattr__(
            self, "max_candidates", max(1, min(int(self.max_candidates), 10_000))
        )
        types: List[str] = []
        for item in self.memory_types or ():
            value = str(getattr(item, "value", item) or "").strip().lower()
            if value and value not in types:
                try:
                    MemoryType(value)
                except ValueError:
                    continue
                types.append(value)
        object.__setattr__(self, "memory_types", tuple(types))

    @classmethod
    def from_mapping(
        cls, value: Any, *, base: Optional["RetentionConfig"] = None
    ) -> "RetentionConfig":
        current = base or cls()
        if not isinstance(value, dict):
            return current
        updates: Dict[str, Any] = {}
        for name in ("min_retention", "grace_days", "max_candidates"):
            if value.get(name) not in (None, ""):
                try:
                    updates[name] = float(value[name])
                except (TypeError, ValueError):
                    continue
        for name in ("enabled", "protect_verified", "protect_pinned"):
            if value.get(name) not in (None, ""):
                updates[name] = _truthy(value[name])
        types = value.get("memory_types")
        if isinstance(types, str):
            types = [part.strip() for part in types.split(",")]
        if isinstance(types, (list, tuple)) and types:
            updates["memory_types"] = tuple(types)
        return replace(current, **updates) if updates else current

    @classmethod
    def from_env(cls, environ: Optional[Dict[str, str]] = None) -> "RetentionConfig":
        env = os.environ if environ is None else environ
        return cls.from_mapping(
            {
                "enabled": env.get("MEMORIZZ_RETENTION_ENABLED"),
                "min_retention": env.get("MEMORIZZ_RETENTION_MIN_SCORE"),
                "grace_days": env.get("MEMORIZZ_RETENTION_GRACE_DAYS"),
                "protect_verified": env.get("MEMORIZZ_RETENTION_PROTECT_VERIFIED"),
                "protect_pinned": env.get("MEMORIZZ_RETENTION_PROTECT_PINNED"),
                "max_candidates": env.get("MEMORIZZ_RETENTION_MAX_CANDIDATES"),
                "memory_types": env.get("MEMORIZZ_RETENTION_MEMORY_TYPES"),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_retention": self.min_retention,
            "grace_days": self.grace_days,
            "protect_verified": self.protect_verified,
            "protect_pinned": self.protect_pinned,
            "max_candidates": self.max_candidates,
            "memory_types": list(self.memory_types),
        }


def _record_id(row: Mapping[str, Any]) -> str:
    return str(row.get("_id") or row.get("id") or row.get("record_id") or "")


def _owner(row: Mapping[str, Any]) -> str:
    return str(row.get("agent_id") or row.get("owner_agent_id") or "")


def _created_at(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("timestamp", "created_at", "createdAt"):
        value = _parse_timestamp(row.get(key))
        if value is not None:
            return value
    content = row.get("content")
    if isinstance(content, dict):
        return _parse_timestamp(content.get("timestamp"))
    return None


def _last_accessed(row: Mapping[str, Any], created: Optional[float]) -> Optional[float]:
    for key in ("last_accessed_at", "last_accessed", "last_recalled_at"):
        value = _parse_timestamp(row.get(key))
        if value is not None:
            return value
    return created


def _access_count(row: Mapping[str, Any]) -> int:
    for key in ("access_count", "recall_count", "hit_count"):
        try:
            return max(0, int(row.get(key) or 0))
        except (TypeError, ValueError):
            continue
    return 0


def _flag(row: Mapping[str, Any], name: str) -> bool:
    if _truthy(row.get(name)):
        return True
    metadata = row.get("metadata")
    return isinstance(metadata, dict) and _truthy(metadata.get(name))


def _citations(row: Mapping[str, Any]) -> Iterable[str]:
    holders = [row]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        holders.append(metadata)
    for holder in holders:
        for field in _CITATION_FIELDS:
            value = holder.get(field)
            for item in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(item, (str, int)) and str(item):
                    yield str(item)


class RetentionPlanner:
    """Plan and apply reversible suppression of low-retention memory records."""

    def __init__(
        self,
        provider: Any,
        *,
        agent_id: str,
        config: Optional[RetentionConfig] = None,
        scoring: Optional[RetrievalScoring] = None,
        learning_store: Any = None,
    ) -> None:
        if provider is None:
            raise ValueError("RetentionPlanner requires a memory provider")
        self.provider = provider
        self.agent_id = str(agent_id or "").strip()
        self.config = config or RetentionConfig.from_env()
        self.scoring = scoring or RetrievalScoring.from_env()
        self.learning_store = learning_store

    # -- reading -------------------------------------------------------------

    def _rows(self, memory_type: MemoryType, user_id: Any) -> List[Dict[str, Any]]:
        try:
            if user_id is ...:
                rows = self.provider.list_all(memory_type)
            else:
                rows = self.provider.list_all(memory_type, user_id=user_id)
        except TypeError:
            rows = self.provider.list_all(memory_type)
        except Exception as exc:
            logger.warning(
                "Retention planner could not list %s: %s", memory_type.value, exc
            )
            return []
        return [row for row in rows or [] if isinstance(row, dict)]

    def _in_scope(
        self, row: Mapping[str, Any], memory_ids: Optional[Sequence[str]]
    ) -> bool:
        owner = _owner(row)
        if owner:
            if self.agent_id and owner != self.agent_id:
                return False
            if memory_ids and row.get("memory_id") not in set(memory_ids):
                return False
            return True
        if memory_ids:
            return row.get("memory_id") in set(memory_ids)
        return not self.agent_id

    def score(
        self, row: Mapping[str, Any], *, max_access: int, now: float
    ) -> Dict[str, float]:
        created = _created_at(row)
        last = _last_accessed(row, created)
        hours = max(0.0, (now - last) / 3600.0) if last is not None else 0.0
        recency = (
            self.scoring.recency_decay_per_hour**hours if last is not None else 1.0
        )
        count = _access_count(row)
        usage = (math.log1p(count) / math.log1p(max_access)) if max_access > 0 else 0.0
        importance = clamp_importance(
            row.get("importance")
            if row.get("importance") is not None
            else (row.get("metadata") or {}).get("importance")
            if isinstance(row.get("metadata"), dict)
            else None
        )
        if importance is None:
            importance = self.scoring.default_importance
        rho = (recency + usage + importance) / 3.0
        age_days = max(0.0, (now - created) / 86_400.0) if created is not None else 0.0
        return {
            "retention": rho,
            "recency": recency,
            "usage": usage,
            "importance": importance,
            "age_days": age_days,
            "access_count": float(count),
        }

    def plan(
        self,
        *,
        memory_ids: Optional[Sequence[str]] = None,
        user_id: Any = ...,
        now: Optional[float] = None,
    ) -> ForgettingReport:
        """Dry-run plan: which primary memories would be suppressed and why."""
        current = now if now is not None else datetime.now(timezone.utc).timestamp()
        candidates: List[ForgettingCandidate] = []
        retained = 0
        for type_name in self.config.memory_types:
            memory_type = MemoryType(type_name)
            rows = [
                row
                for row in self._rows(memory_type, user_id)
                if _record_id(row) and self._in_scope(row, memory_ids)
            ]
            if not rows:
                continue
            cited = {
                citation
                for row in rows
                if not is_suppressed(row)
                for citation in _citations(row)
            }
            max_access = max((_access_count(row) for row in rows), default=0)
            for row in rows:
                record_id = _record_id(row)
                if is_suppressed(row):
                    continue
                parts = self.score(row, max_access=max_access, now=current)
                verified = _flag(row, "verified")
                pinned = _flag(row, "pinned")
                reason = None
                if parts["age_days"] < self.config.grace_days:
                    reason = None
                elif parts["retention"] < self.config.min_retention:
                    reason = (
                        f"retention {parts['retention']:.2f} below {self.config.min_retention:.2f} "
                        f"after {parts['age_days']:.0f} days "
                        f"(importance {parts['importance']:.2f}, recalls {int(parts['access_count'])})"
                    )
                if reason and verified and self.config.protect_verified:
                    reason = None
                if reason and pinned and self.config.protect_pinned:
                    reason = None
                if reason and record_id in cited:
                    reason = None
                if reason is None:
                    retained += 1
                    continue
                candidates.append(
                    ForgettingCandidate(
                        target_id=record_id,
                        target_type=memory_type.value,
                        action=ForgettingAction.TOMBSTONE,
                        reason=reason,
                        utility=parts["retention"],
                        age_days=parts["age_days"],
                        content_hash=None,
                        superseded_by=None,
                        verified=verified,
                    )
                )
        candidates.sort(key=lambda item: (item.utility, -item.age_days, item.target_id))
        candidates = candidates[: self.config.max_candidates]
        return ForgettingReport(
            plan_id=ForgettingMechanism._plan_id(candidates),
            dry_run=True,
            candidates=tuple(candidates),
            retained=retained,
        )

    # -- applying ------------------------------------------------------------

    def apply(
        self,
        report: ForgettingReport,
        *,
        approved_by: str,
        reason: Optional[str] = None,
    ) -> ForgettingReport:
        """Suppress every candidate of an approved dry-run plan (reversible)."""
        approver = str(approved_by or "").strip()
        if not approver:
            raise ValueError("approved_by is required to apply a retention plan")
        if not report.dry_run:
            raise ValueError("Only a dry-run retention plan can be applied")
        if report.plan_id != ForgettingMechanism._plan_id(report.candidates):
            raise ValueError("Retention plan content does not match its plan_id")
        from ..memory_history import memory_change_context

        applied_at = utc_now()
        suppressed = 0
        errors: List[str] = []
        with memory_change_context(
            actor=approver,
            source=f"forgetting:{report.plan_id}",
            agent_id=self.agent_id or None,
        ):
            for candidate in report.candidates:
                try:
                    memory_type = MemoryType(candidate.target_type)
                    row = self.provider.retrieve_by_id(candidate.target_id, memory_type)
                    if not isinstance(row, dict):
                        raise ValueError("record no longer exists")
                    if is_suppressed(row):
                        raise ValueError("record is already suppressed")
                    if self.config.protect_verified and _flag(row, "verified"):
                        raise ValueError("verified records are retained by policy")
                    if self.config.protect_pinned and _flag(row, "pinned"):
                        raise ValueError("pinned records are retained by policy")
                    ok = self.provider.update_by_id(
                        candidate.target_id,
                        {
                            "retention_state": "suppressed",
                            "suppressed_by": report.plan_id,
                            "suppressed_at": applied_at,
                            "suppression_reason": candidate.reason,
                            "suppression_approver": approver[:240],
                            "suppression_note": str(reason or "")[:2000],
                        },
                        memory_type,
                    )
                    if ok is False:
                        raise ValueError("provider refused the update")
                    if self.learning_store is not None:
                        self.learning_store.put_tombstone(
                            {
                                "target_id": candidate.target_id,
                                "target_type": candidate.target_type,
                                "plan_id": report.plan_id,
                                "policy_reason": candidate.reason,
                                "operator_reason": str(reason or "")[:2000],
                                "approved_by": approver[:240],
                                "approved_at": applied_at,
                                "utility": candidate.utility,
                                "verified": candidate.verified,
                                "agent_id": self.agent_id,
                                "memory_id": row.get("memory_id"),
                                "user_id": row.get("user_id"),
                                "reversible": True,
                            }
                        )
                    suppressed += 1
                except Exception as exc:
                    errors.append(f"{candidate.target_id}: {exc}")
        return ForgettingReport(
            plan_id=report.plan_id,
            dry_run=False,
            candidates=report.candidates,
            tombstoned=suppressed,
            retained=report.retained,
            errors=tuple(errors),
        )

    def unsuppress(
        self,
        record_id: str,
        memory_type: Any,
        *,
        approved_by: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Reverse a suppression; the record returns to retrieval immediately."""
        approver = str(approved_by or "").strip()
        if not approver:
            raise ValueError("approved_by is required")
        kind = MemoryType(getattr(memory_type, "value", memory_type))
        from ..memory_history import memory_change_context

        row = self.provider.retrieve_by_id(str(record_id), kind)
        if not isinstance(row, dict) or not is_suppressed(row):
            return False
        with memory_change_context(
            actor=approver,
            source="forgetting:unsuppress",
            agent_id=self.agent_id or None,
        ):
            ok = self.provider.update_by_id(
                str(record_id),
                {
                    "retention_state": "active",
                    "unsuppressed_by": approver[:240],
                    "unsuppressed_at": utc_now(),
                    "unsuppression_note": str(reason or "")[:2000],
                },
                kind,
            )
        return ok is not False

    def suppressed(
        self, *, memory_ids: Optional[Sequence[str]] = None, user_id: Any = ...
    ) -> List[Dict[str, Any]]:
        """Currently suppressed records in scope (for review pages)."""
        found: List[Dict[str, Any]] = []
        for type_name in self.config.memory_types:
            memory_type = MemoryType(type_name)
            for row in self._rows(memory_type, user_id):
                if not is_suppressed(row) or not self._in_scope(row, memory_ids):
                    continue
                found.append(
                    {
                        "record_id": _record_id(row),
                        "memory_type": memory_type.value,
                        "memory_id": row.get("memory_id"),
                        "suppressed_by": row.get("suppressed_by"),
                        "suppressed_at": row.get("suppressed_at"),
                        "reason": row.get("suppression_reason"),
                        "approver": row.get("suppression_approver"),
                    }
                )
        return found


__all__ = [
    "DEFAULT_RETENTION_MEMORY_TYPES",
    "RetentionConfig",
    "RetentionPlanner",
]
