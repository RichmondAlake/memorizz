# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Promotion engine: workflow trajectory classes → learned skills.

Promotion is gated, never automatic-on-frequency: a trajectory class must
clear execution count, success rate, query diversity, AND recency gates
before it is even ranked. A 50×-repeated 60%-success trajectory is a bug
report, not a skill.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ....enums.memory_type import MemoryType
from ..workflow.canonicalization import TrajectoryStats, aggregate_trajectory_stats
from .distiller import SkillDistiller
from .skill import (
    Skill,
    SkillInjectionRole,
    SkillStatus,
    normalize_skill_injection_role,
)
from .skillbox import Skillbox

logger = logging.getLogger(__name__)


@dataclass
class PromotionConfig:
    """Every knob of the continual-learning loop, in one place."""

    # Eligibility gates — ALL must pass
    min_executions: int = 5
    min_success_rate: float = 0.80
    min_distinct_queries: int = 2
    max_days_since_last_seen: int = 30
    # Ranking (never rescues a gate failure)
    recency_halflife_days: float = 14.0
    max_promotions_per_cycle: int = 3
    # Distillation
    distill_sample_size: int = 5
    include_failure_samples: int = 2
    skill_max_content_chars: int = 4000
    # Validation
    require_shadow: bool = False
    # Retrieval / injection
    retrieval_min_similarity: float = 0.70
    max_skills_in_context: int = 2
    skill_injection_role: SkillInjectionRole = SkillInjectionRole.USER
    # Deprecated compatibility key. Raw exemplars are never co-injected.
    include_exemplar: bool = False
    # Monitoring / demotion
    drift_window_activations: int = 10
    demotion_success_delta: float = 0.25
    min_activations_before_drift_check: int = 5
    # Scheduling: run a promotion cycle every N stored runs (0 = manual only)
    promotion_every_n_runs: int = 25

    def __post_init__(self) -> None:
        self.skill_injection_role = normalize_skill_injection_role(
            self.skill_injection_role
        )
        if (
            self.skill_injection_role == SkillInjectionRole.DEVELOPER
            and not self.require_shadow
        ):
            raise ValueError(
                "developer-role learned skills require require_shadow=True so "
                "generated instructions cannot gain application authority "
                "without an explicit activation review"
            )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PromotionConfig":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PromotionReport:
    """What one promotion cycle did, and why."""

    promoted: List[str] = field(default_factory=list)
    rejected: List[Tuple[str, List[str]]] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    review_flags: List[str] = field(default_factory=list)
    candidates_considered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promoted": self.promoted,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "review_flags": self.review_flags,
            "candidates_considered": self.candidates_considered,
        }


class PromotionEngine:
    """Aggregate → gate → rank → distill → validate → store → stamp."""

    def __init__(
        self,
        memory_provider,
        skillbox: Skillbox,
        llm_provider=None,
        config: Optional[PromotionConfig] = None,
        agent_id: Optional[str] = None,
        resolve_tool: Optional[Callable[[str], bool]] = None,
    ):
        self.memory_provider = memory_provider
        self.skillbox = skillbox
        self.llm_provider = llm_provider
        self.config = config or PromotionConfig()
        self.agent_id = agent_id
        self.resolve_tool = resolve_tool or self._default_tool_resolver
        self.distiller = SkillDistiller(
            llm_provider=llm_provider,
            resolve_tool=self.resolve_tool,
            max_content_chars=self.config.skill_max_content_chars,
        )

    # ------------------------------------------------------------- main flow

    def run_promotion_cycle(self, user_id: Optional[str] = None) -> PromotionReport:
        report = PromotionReport()
        if self.llm_provider is None:
            report.review_flags.append(
                "no LLM provider configured — distillation unavailable"
            )
            return report

        stats = aggregate_trajectory_stats(
            self.memory_provider, agent_id=self.agent_id, user_id=user_id
        )
        skills_by_hash = self._skills_by_hash()

        candidates: List[TrajectoryStats] = []
        for stat in stats:
            eligible, reason = self._eligibility(stat, skills_by_hash)
            if eligible:
                candidates.append(stat)
            elif reason:
                report.skipped.append((stat.canonical_hash, reason))

        report.candidates_considered = len(candidates)
        candidates.sort(key=self._score, reverse=True)

        for candidate in candidates[: self.config.max_promotions_per_cycle]:
            try:
                self._promote_one(candidate, skills_by_hash, report)
            except Exception as exc:
                logger.error(
                    "Promotion failed for %s: %s", candidate.canonical_hash, exc
                )
                report.rejected.append((candidate.canonical_hash, [f"error: {exc}"]))

        report.review_flags.extend(self._deviation_review_flags())
        return report

    def _promote_one(
        self,
        candidate: TrajectoryStats,
        skills_by_hash: Dict[str, List[Skill]],
        report: PromotionReport,
    ) -> None:
        success_docs = self._fetch_docs(
            candidate.sample_workflow_ids[: self.config.distill_sample_size],
            candidate,
        )
        failure_docs = self._fetch_docs(
            candidate.failure_workflow_ids[: self.config.include_failure_samples],
            candidate,
        )
        if not success_docs:
            report.rejected.append(
                (candidate.canonical_hash, ["no sample runs retrievable"])
            )
            return

        previous = self._latest_retired_skill(
            skills_by_hash.get(candidate.canonical_hash, [])
        )
        skill_md = self.distiller.distill(
            success_docs,
            failure_docs,
            previous_demotion_reason=(previous.demotion_reason if previous else None),
        )
        verdict, parsed = self.distiller.validate(
            skill_md,
            success_docs,
            failure_docs=failure_docs,
        )
        if not verdict.ok or parsed is None:
            report.rejected.append((candidate.canonical_hash, verdict.reasons))
            return

        status = (
            SkillStatus.SHADOW if self.config.require_shadow else SkillStatus.ACTIVE
        )
        skill = Skill(
            name=parsed.name,
            description=parsed.description,
            content=parsed.content,
            preconditions=parsed.preconditions,
            tools_used=parsed.tools,
            queries=self._skill_queries(candidate),
            agent_id=self.agent_id,
            source_canonical_hash=candidate.canonical_hash,
            source_workflow_ids=[
                str(doc.get("workflow_id") or doc.get("_id")) for doc in success_docs
            ],
            # Provider record id first — this is the id retrieve_by_id
            # accepts on every provider (Mongo requires the ObjectId).
            exemplar_workflow_id=(
                str(success_docs[0].get("_id") or success_docs[0].get("workflow_id"))
            ),
            status=status,
            version=(previous.version + 1) if previous else 1,
            promoted_at=datetime.now() if status == SkillStatus.ACTIVE else None,
            injection_role=self.config.skill_injection_role,
            baseline=candidate.baseline_snapshot(),
        )
        self.skillbox.add_skill(skill)
        if status == SkillStatus.ACTIVE:
            # SHADOW skills deliberately do NOT stamp: they are never
            # injected, so suppressing workflow retrieval for them would
            # remove context without providing the compiled replacement.
            self.stamp_workflows(candidate, skill.skill_id)
        report.promoted.append(skill.skill_id)

    def promote_class(
        self, canonical_hash: str, user_id: Optional[str] = None
    ) -> PromotionReport:
        """Run a gated promotion for ONE trajectory class.

        Human-in-the-loop entry point (the UI's "distill now"): targets a
        single class instead of ranking the whole log, but the eligibility
        gates are STILL enforced — hand-picking a class never buys it out
        of the evidence requirements (invariant 2).
        """
        report = PromotionReport()
        if self.llm_provider is None:
            report.review_flags.append(
                "no LLM provider configured — distillation unavailable"
            )
            return report

        stats = aggregate_trajectory_stats(
            self.memory_provider, agent_id=self.agent_id, user_id=user_id
        )
        stat = next((s for s in stats if s.canonical_hash == canonical_hash), None)
        if stat is None:
            report.skipped.append((canonical_hash, "unknown trajectory class"))
            return report

        skills_by_hash = self._skills_by_hash()
        eligible, reason = self._eligibility(stat, skills_by_hash)
        report.candidates_considered = 1
        if not eligible:
            report.skipped.append((canonical_hash, reason or "not eligible"))
            return report

        try:
            self._promote_one(stat, skills_by_hash, report)
        except Exception as exc:
            logger.error("Promotion failed for %s: %s", canonical_hash, exc)
            report.rejected.append((canonical_hash, [f"error: {exc}"]))
        return report

    def activate_skill(
        self,
        skill_id: str,
        injection_role: Optional[Any] = None,
    ) -> bool:
        """SHADOW/CANDIDATE → ACTIVE, with the workflow stamp backfill the
        write-time path can't provide retroactively."""
        skill = self.skillbox.get_skill_by_id(skill_id)
        if not skill:
            return False
        if skill.status not in (SkillStatus.SHADOW, SkillStatus.CANDIDATE):
            return False
        if injection_role is not None:
            skill.injection_role = normalize_skill_injection_role(injection_role)
        skill.status = SkillStatus.ACTIVE
        skill.promoted_at = skill.promoted_at or datetime.now()
        if not self.skillbox.update_skill(skill):
            return False
        if skill.source_canonical_hash:
            stats = aggregate_trajectory_stats(
                self.memory_provider, agent_id=self.agent_id
            )
            for stat in stats:
                if stat.canonical_hash == skill.source_canonical_hash:
                    self.stamp_workflows(stat, skill_id)
                    break
        return True

    # ----------------------------------------------------------------- gates

    def _eligibility(
        self,
        stat: TrajectoryStats,
        skills_by_hash: Dict[str, List[Skill]],
    ) -> Tuple[bool, Optional[str]]:
        existing = skills_by_hash.get(stat.canonical_hash, [])
        live = [
            s
            for s in existing
            if s.status
            in (SkillStatus.ACTIVE, SkillStatus.SHADOW, SkillStatus.CANDIDATE)
        ]
        if live or stat.already_promoted:
            return False, "already covered by a live skill"

        # Re-promotion after demotion: only runs AFTER the demotion count
        # toward the gates, so a demoted skill has to re-earn its place on
        # fresh evidence rather than the evidence that produced the bad
        # version.
        retired = self._latest_retired_skill(existing)
        runs = stat.runs
        if retired and retired.demoted_at:
            runs = [
                run
                for run in stat.runs
                if run.created_at and run.created_at > retired.demoted_at
            ]

        executions = len(runs)
        successes = sum(1 for run in runs if run.outcome == "success")
        if executions < self.config.min_executions:
            return False, f"executions {executions} < {self.config.min_executions}"
        success_rate = successes / executions
        if success_rate < self.config.min_success_rate:
            return (
                False,
                f"success_rate {success_rate:.2f} < {self.config.min_success_rate}",
            )
        if stat.distinct_query_count < self.config.min_distinct_queries:
            # A single verbatim query repeated N times is a cache candidate
            # (SEMANTIC_CACHE exists for that), not a generalizable skill.
            return (
                False,
                f"distinct_queries {stat.distinct_query_count} < "
                f"{self.config.min_distinct_queries}",
            )
        if stat.last_seen is None:
            return False, "no timestamps on runs"
        age_days = (datetime.now() - stat.last_seen).days
        if age_days > self.config.max_days_since_last_seen:
            return False, f"stale: last seen {age_days}d ago"
        return True, None

    def _score(self, stat: TrajectoryStats) -> float:
        days_since = 0.0
        if stat.last_seen:
            days_since = max(
                0.0, (datetime.now() - stat.last_seen).total_seconds() / 86400.0
            )
        recency = 0.5 ** (days_since / self.config.recency_halflife_days)
        return stat.success_rate * math.log1p(stat.executions) * recency

    # ------------------------------------------------------------- stamping

    def stamp_workflows(self, stat: TrajectoryStats, skill_id: str) -> int:
        """Set ``promoted_skill_id`` on every stored run of the class."""
        stamped = 0
        for run in stat.runs:
            if not run.record_id or run.promoted_skill_id == skill_id:
                continue
            try:
                if self.memory_provider.update_by_id(
                    run.record_id,
                    {"promoted_skill_id": skill_id},
                    memory_store_type=MemoryType.WORKFLOW_MEMORY,
                ):
                    stamped += 1
            except Exception as exc:
                logger.debug("Stamp failed for %s: %s", run.record_id, exc)
        return stamped

    # -------------------------------------------------------------- helpers

    def _skills_by_hash(self) -> Dict[str, List[Skill]]:
        grouped: Dict[str, List[Skill]] = {}
        for skill in self.skillbox.list_skills():
            if skill.source_canonical_hash:
                grouped.setdefault(skill.source_canonical_hash, []).append(skill)
        return grouped

    @staticmethod
    def _latest_retired_skill(skills: List[Skill]) -> Optional[Skill]:
        retired = [
            s
            for s in skills
            if s.status in (SkillStatus.DEMOTED, SkillStatus.DEPRECATED)
        ]
        if not retired:
            return None
        return max(retired, key=lambda s: s.version)

    def _fetch_docs(
        self, workflow_ids: List[str], stat: TrajectoryStats
    ) -> List[Dict[str, Any]]:
        by_workflow_id = {
            run.workflow_id: run.record_id for run in stat.runs if run.workflow_id
        }
        docs = []
        for workflow_id in workflow_ids:
            record_id = by_workflow_id.get(workflow_id) or workflow_id
            try:
                doc = self.memory_provider.retrieve_by_id(
                    record_id, memory_store_type=MemoryType.WORKFLOW_MEMORY
                )
            except Exception:
                doc = None
            if doc:
                docs.append(doc)
        return docs

    def _skill_queries(self, stat: TrajectoryStats) -> List[str]:
        """Sampled real user queries — the Toolbox synthetic-queries idea,
        grounded in what actually triggered the trajectory."""
        seen = set()
        queries = []
        for run in stat.runs:
            text = (run.user_query or "").strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                queries.append(text)
            if len(queries) >= 8:
                break
        return queries

    def _deviation_review_flags(self) -> List[str]:
        """Skills that retrieve on queries they don't fit (agent had them in
        context but went another way >50% of the time). Not auto-demoted —
        surfaced for review: the description/embedding is miscalibrated."""
        flags = []
        for skill in self.skillbox.list_skills(
            statuses=(SkillStatus.ACTIVE, SkillStatus.SHADOW)
        ):
            activations = skill.stats.get("activations", 0)
            deviations = skill.stats.get("deviations", 0)
            if (
                activations >= self.config.min_activations_before_drift_check
                and deviations / activations > 0.5
            ):
                flags.append(
                    f"skill {skill.name} ({skill.skill_id}): deviation rate "
                    f"{deviations}/{activations} — applicability likely "
                    "miscalibrated"
                )
        return flags

    def _default_tool_resolver(self, tool_name: str) -> bool:
        try:
            return (
                self.memory_provider.retrieve_by_name(
                    tool_name, memory_store_type=MemoryType.TOOLBOX
                )
                is not None
            )
        except Exception:
            return False
