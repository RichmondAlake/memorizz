"""Governed forgetting for learning artifacts and memory projections."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .models import (
    ForgettingAction,
    ForgettingCandidate,
    ForgettingReport,
    LearningControlPlaneConfig,
    content_digest,
    utc_now,
)
from .store import LearningControlPlaneStore


def _age_days(value: Any) -> float:
    if value is None:
        return 0.0
    text = value.isoformat() if hasattr(value, "isoformat") else str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 86_400)
    except (TypeError, ValueError):
        return 0.0


class ForgettingMechanism:
    """Plan reversible forgetting and require explicit authority to apply it.

    Event history is never hard-deleted by this mechanism.  Rebuildable
    artifacts receive tombstones, which removes them from EvidencePack
    retrieval while preserving provenance and rollback.  Provider-level hard
    deletion remains available only through exact-scope lifecycle APIs.
    """

    def __init__(
        self,
        store: LearningControlPlaneStore,
        *,
        config: LearningControlPlaneConfig,
    ) -> None:
        self.store = store
        self.config = config

    @staticmethod
    def _plan_id(candidates: Sequence[ForgettingCandidate]) -> str:
        return (
            "forget-plan-"
            + hashlib.sha256(
                content_digest([item.to_dict() for item in candidates]).encode()
            ).hexdigest()[:28]
        )

    def plan(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        max_candidates: int = 500,
    ) -> ForgettingReport:
        rows = self.store.list_artifacts(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            limit=max(1, min(int(max_candidates), 10_000)),
        )
        tombstoned = self.store.tombstoned_ids(memory_id=memory_id, user_id=user_id)
        newest_by_hash: Dict[str, Mapping[str, Any]] = {}
        for row in rows:
            digest = str(
                row.get("source_hash")
                or row.get("content_hash")
                or content_digest(row.get("content") or "")
            )
            current = newest_by_hash.get(digest)
            if current is None or str(row.get("updated_at") or "") > str(
                current.get("updated_at") or ""
            ):
                newest_by_hash[digest] = row

        candidates: List[ForgettingCandidate] = []
        retained = 0
        for row in rows:
            target_id = str(row.get("artifact_id") or row.get("record_id") or "")
            if not target_id or target_id in tombstoned:
                continue
            verified = row.get("verified") is True
            age = _age_days(row.get("updated_at") or row.get("created_at"))
            raw_utility = row.get("utility")
            try:
                base_utility = max(0.0, min(1.0, float(raw_utility)))
            except (TypeError, ValueError):
                base_utility = 1.0 if verified else 0.5
            decay = math.pow(0.5, age / self.config.utility_half_life_days)
            utility = base_utility * decay
            digest = str(
                row.get("source_hash")
                or row.get("content_hash")
                or content_digest(row.get("content") or "")
            )
            newest = newest_by_hash.get(digest)
            newest_id = (
                str(newest.get("artifact_id") or newest.get("record_id") or "")
                if newest
                else ""
            )

            reason = None
            superseded_by = None
            if newest_id and newest_id != target_id:
                reason = "duplicate projection superseded by a newer artifact"
                superseded_by = newest_id
            elif (
                age >= self.config.retention_days and utility < self.config.min_utility
            ):
                reason = (
                    f"utility decayed below {self.config.min_utility:.2f} after "
                    f"{age:.1f} days"
                )
            if verified and self.config.preserve_verified_outcomes:
                reason = None
            if reason is None:
                retained += 1
                continue
            candidates.append(
                ForgettingCandidate(
                    target_id=target_id,
                    target_type=str(row.get("artifact_kind") or "learning_artifact"),
                    action=ForgettingAction.TOMBSTONE,
                    reason=reason,
                    utility=utility,
                    age_days=age,
                    content_hash=digest,
                    superseded_by=superseded_by,
                    verified=verified,
                )
            )

        candidates.sort(key=lambda item: (item.utility, -item.age_days))
        plan_id = self._plan_id(candidates)
        return ForgettingReport(
            plan_id=plan_id,
            dry_run=True,
            candidates=tuple(candidates),
            retained=retained,
        )

    def apply(
        self,
        report: ForgettingReport,
        *,
        approved_by: str,
        reason: Optional[str] = None,
    ) -> ForgettingReport:
        approver = str(approved_by or "").strip()
        if not approver:
            raise ValueError("approved_by is required to apply a forgetting plan")
        if not report.dry_run:
            raise ValueError("Only a dry-run forgetting plan can be applied")
        if report.plan_id != self._plan_id(report.candidates):
            raise ValueError("Forgetting plan content does not match its plan_id")
        tombstoned = 0
        errors: List[str] = []
        for candidate in report.candidates:
            if candidate.action != ForgettingAction.TOMBSTONE:
                errors.append(
                    f"Unsupported forgetting action for {candidate.target_id}: "
                    f"{candidate.action.value}"
                )
                continue
            try:
                current = self.store.get(candidate.target_id)
                if not current or current.get("record_type") != "learning_artifact":
                    raise ValueError("target is not a current learning artifact")
                if (
                    self.config.preserve_verified_outcomes
                    and current.get("verified") is True
                ):
                    raise ValueError(
                        "verified outcome artifacts are retained by policy"
                    )
                self.store.put_tombstone(
                    {
                        "target_id": candidate.target_id,
                        "target_type": candidate.target_type,
                        "plan_id": report.plan_id,
                        "policy_reason": candidate.reason,
                        "operator_reason": str(reason or "")[:2000],
                        "approved_by": approver[:240],
                        "approved_at": utc_now(),
                        "utility": candidate.utility,
                        "superseded_by": candidate.superseded_by,
                        "verified": candidate.verified,
                        "agent_id": self.store.agent_id,
                        "memory_id": current.get("memory_id"),
                        "user_id": current.get("user_id"),
                        "thread_id": current.get("thread_id"),
                        "stream_id": current.get("stream_id"),
                    }
                )
                tombstoned += 1
            except Exception as exc:
                errors.append(f"{candidate.target_id}: {exc}")
        return ForgettingReport(
            plan_id=report.plan_id,
            dry_run=False,
            candidates=report.candidates,
            tombstoned=tombstoned,
            retained=report.retained,
            errors=tuple(errors),
        )


__all__ = ["ForgettingMechanism"]
