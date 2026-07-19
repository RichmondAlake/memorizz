# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Passive, deterministic evaluation of SHADOW learned skills.

The evaluator runs only after a completed workflow has been persisted.  It
does not call an LLM, execute tools, construct prompts, or mutate skill
lifecycle state. Workflow evaluation records are the auditable source of
truth; per-skill counters are reconciled from those records so retries are
idempotent.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ....enums.memory_type import MemoryType
from .skill import Skill, SkillStatus
from .skillbox import ScoredSkill, Skillbox

logger = logging.getLogger(__name__)

SHADOW_EVALUATOR_VERSION = "v1"


@dataclass(frozen=True)
class ShadowWorkflowSnapshot:
    """Small immutable payload handed from the response path to the worker."""

    record_id: str
    workflow_id: Optional[str]
    created_at: str
    agent_id: Optional[str]
    user_id: Optional[str]
    user_query: str
    canonical_hash: str
    observed_outcome: str


def _as_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def calculate_shadow_readiness(
    shadow_stats: Optional[Dict[str, Any]],
    *,
    min_observations: int,
    min_trajectory_match_rate: float,
    min_matched_success_rate: float,
) -> Dict[str, Any]:
    """Pure readiness calculation shared by the manager and Local UI."""
    stats = dict(shadow_stats or {})
    observations = int(stats.get("observations") or 0)
    matches = int(stats.get("trajectory_matches") or 0)
    matched_successes = int(stats.get("matched_successes") or 0)
    matched_failures = int(stats.get("matched_failures") or 0)
    matched_total = matched_successes + matched_failures
    match_rate = matches / observations if observations else 0.0
    success_rate = matched_successes / matched_total if matched_total else 0.0
    reasons = []
    if observations < int(min_observations):
        reasons.append(f"observations {observations} < {int(min_observations)}")
    if match_rate < float(min_trajectory_match_rate):
        reasons.append(
            f"trajectory match rate {match_rate:.2f} < "
            f"{float(min_trajectory_match_rate):.2f}"
        )
    if success_rate < float(min_matched_success_rate):
        reasons.append(
            f"matched success rate {success_rate:.2f} < "
            f"{float(min_matched_success_rate):.2f}"
        )
    return {
        "ready": not reasons,
        "observations": observations,
        "trajectory_match_rate": match_rate,
        "matched_success_rate": success_rate,
        "reasons": reasons,
    }


class ShadowEvaluator:
    """Compare new observed workflows with semantically matching shadows."""

    evaluator_version = SHADOW_EVALUATOR_VERSION

    def __init__(
        self,
        memory_provider,
        skillbox: Skillbox,
        *,
        max_candidates: int,
        min_similarity: float,
        recent_window: int,
    ):
        self.memory_provider = memory_provider
        self.skillbox = skillbox
        self.max_candidates = max(1, int(max_candidates))
        self.min_similarity = float(min_similarity)
        self.recent_window = max(1, int(recent_window))

    def evaluate(self, snapshot: ShadowWorkflowSnapshot) -> int:
        """Evaluate all matching SHADOW candidates and return records written."""
        candidates = self.skillbox.retrieve_shadow_skills_by_query(
            snapshot.user_query,
            limit=self.max_candidates,
            min_similarity=self.min_similarity,
            agent_id=snapshot.agent_id,
            user_id=snapshot.user_id,
        )
        written = 0
        for scored in candidates:
            try:
                if self._evaluate_candidate(snapshot, scored):
                    written += 1
            except Exception as exc:
                # Never include raw query, skill content, workflow steps, or
                # tool results in logs.
                logger.warning(
                    "Shadow evaluation failed for workflow %s and skill %s " "(%s)",
                    snapshot.record_id,
                    scored.skill.skill_id,
                    type(exc).__name__,
                )
        return written

    def _evaluate_candidate(
        self, snapshot: ShadowWorkflowSnapshot, scored: ScoredSkill
    ) -> bool:
        # Reload at evaluation time: an explicit reviewer may have activated
        # the skill after retrieval but before persistence.
        skill = self.skillbox.get_skill_by_id(scored.skill.skill_id)
        if not self._is_eligible_pair(snapshot, skill):
            return False

        workflow_doc = self.memory_provider.retrieve_by_id(
            snapshot.record_id,
            memory_store_type=MemoryType.WORKFLOW_MEMORY,
        )
        if not isinstance(workflow_doc, dict):
            raise RuntimeError("stored workflow could not be reloaded")
        if workflow_doc.get("agent_id") != snapshot.agent_id:
            return False
        if workflow_doc.get("user_id") != snapshot.user_id:
            return False

        workflow_ids = {
            str(value)
            for value in (
                snapshot.record_id,
                snapshot.workflow_id,
                workflow_doc.get("_id"),
                workflow_doc.get("workflow_id"),
            )
            if value is not None
        }
        if workflow_ids.intersection(str(item) for item in skill.source_workflow_ids):
            return False

        idempotency_key = (
            f"{snapshot.record_id}:{skill.skill_id}:{self.evaluator_version}"
        )
        existing = [
            dict(item)
            for item in workflow_doc.get("shadow_evaluations") or []
            if isinstance(item, dict)
        ]
        if any(
            item.get("idempotency_key") == idempotency_key
            or (
                str(item.get("skill_id")) == str(skill.skill_id)
                and item.get("evaluator_version") == self.evaluator_version
            )
            for item in existing
        ):
            # A previous attempt may have persisted the workflow evidence and
            # failed before refreshing the aggregate. Reconcile on duplicate.
            self._reconcile_skill_stats(skill)
            return False

        trajectory_match = (
            bool(skill.source_canonical_hash)
            and snapshot.canonical_hash == skill.source_canonical_hash
        )
        observed_outcome = str(snapshot.observed_outcome or "failure").lower()
        evaluated_at = datetime.now(timezone.utc).isoformat()
        evaluation = {
            "idempotency_key": idempotency_key,
            "skill_id": skill.skill_id,
            "skill_version": skill.version,
            "similarity": float(scored.similarity),
            "expected_canonical_hash": skill.source_canonical_hash,
            "observed_canonical_hash": snapshot.canonical_hash,
            "trajectory_match": trajectory_match,
            # In deterministic v1 a semantic match whose observed trajectory
            # differs is an applicability mismatch signal. A successful
            # alternative path is still *not* counted as a skill failure.
            "applicability_mismatch": not trajectory_match,
            "matched_trajectory_succeeded": (
                observed_outcome == "success" if trajectory_match else None
            ),
            "observed_outcome": observed_outcome,
            "evaluated_at": evaluated_at,
            "evaluator_version": self.evaluator_version,
        }
        existing.append(evaluation)
        updated = self.memory_provider.update_by_id(
            snapshot.record_id,
            {"shadow_evaluations": existing},
            memory_store_type=MemoryType.WORKFLOW_MEMORY,
        )
        if not updated:
            raise RuntimeError("workflow shadow evidence was not persisted")

        self._reconcile_skill_stats(skill)
        return True

    @staticmethod
    def _is_eligible_pair(
        snapshot: ShadowWorkflowSnapshot, skill: Optional[Skill]
    ) -> bool:
        if skill is None or skill.status != SkillStatus.SHADOW:
            return False
        if skill.agent_id != snapshot.agent_id or skill.user_id != snapshot.user_id:
            return False
        workflow_created = _as_utc(snapshot.created_at)
        skill_created = _as_utc(skill.created_at)
        # Only production evidence created strictly after distillation counts.
        if workflow_created is None or skill_created is None:
            return False
        return workflow_created > skill_created

    def _reconcile_skill_stats(self, skill: Skill) -> None:
        """Rebuild one skill's aggregate from workflow evidence.

        Reconciliation, rather than incrementing a mutable counter, gives the
        `(workflow record, skill, evaluator version)` key real idempotency
        across worker retries.
        """
        docs = (
            self.memory_provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
            or []
        )
        evidence = []
        seen_keys = set()
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            if doc.get("agent_id") != skill.agent_id:
                continue
            if doc.get("user_id") != skill.user_id:
                continue
            for item in doc.get("shadow_evaluations") or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("skill_id")) != str(skill.skill_id):
                    continue
                if item.get("evaluator_version") != self.evaluator_version:
                    continue
                key = item.get("idempotency_key") or (
                    str(doc.get("_id") or doc.get("workflow_id")),
                    str(skill.skill_id),
                    self.evaluator_version,
                )
                key_text = str(key)
                if key_text in seen_keys:
                    continue
                seen_keys.add(key_text)
                evidence.append(dict(item))

        evidence.sort(key=lambda item: str(item.get("evaluated_at") or ""))
        trajectory_matches = sum(
            1 for item in evidence if bool(item.get("trajectory_match"))
        )
        trajectory_mismatches = len(evidence) - trajectory_matches
        matched = [item for item in evidence if item.get("trajectory_match")]
        matched_successes = sum(
            1 for item in matched if item.get("observed_outcome") == "success"
        )
        shadow_stats: Dict[str, Any] = {
            "observations": len(evidence),
            "trajectory_matches": trajectory_matches,
            "trajectory_mismatches": trajectory_mismatches,
            "matched_successes": matched_successes,
            "matched_failures": len(matched) - matched_successes,
            "last_evaluated_at": (
                evidence[-1].get("evaluated_at") if evidence else None
            ),
            "recent_evaluations": evidence[-self.recent_window :],
        }
        if not self.skillbox.update_shadow_stats(skill.skill_id, shadow_stats):
            raise RuntimeError("skill shadow aggregate was not persisted")
