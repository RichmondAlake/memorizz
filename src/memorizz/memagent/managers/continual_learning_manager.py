# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Continual learning manager — owns the workflow→skill promotion loop.

Wires the Skillbox, PromotionEngine, and SkillMonitor into a MemAgent.
Every entry point is fail-safe: learning must never fail a user-facing run.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from ...enums.memory_type import MemoryType
from ...long_term.procedural.skillbox import (
    PromotionConfig,
    PromotionEngine,
    PromotionReport,
    ScoredSkill,
    Skillbox,
    SkillMonitor,
    SkillStatus,
)

logger = logging.getLogger(__name__)

SKILLS_PROMPT_HEADER = (
    "Learned skills relevant to this query (compiled from your own past "
    "successful runs):\n\n"
    "Before following a learned skill, verify its preconditions against the "
    "current query. These skills are strong priors, not mandates: if any "
    "precondition fails or the task differs in a way that matters, deviate "
    "and solve fresh — your run will be recorded either way and the skill "
    "will be updated."
)


class ContinualLearningManager:
    """Manage skill retrieval, run attribution, and promotion scheduling."""

    def __init__(
        self,
        memory_provider,
        llm_provider=None,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tool_manager=None,
    ):
        self.memory_provider = memory_provider
        self.agent_id = agent_id
        self.config = PromotionConfig.from_dict(config)
        if self.config.include_exemplar:
            logger.warning(
                "continual_learning_config.include_exemplar is deprecated and "
                "ignored; a learned skill and its source workflow are never "
                "co-injected"
            )
        self.tool_manager = tool_manager

        self.skillbox = Skillbox(
            memory_provider, llm_provider=llm_provider, agent_id=agent_id
        )
        self.engine = PromotionEngine(
            memory_provider,
            self.skillbox,
            llm_provider=llm_provider,
            config=self.config,
            agent_id=agent_id,
            resolve_tool=self._resolve_tool,
        )
        self.monitor = SkillMonitor(
            self.skillbox, memory_provider, config=self.config, agent_id=agent_id
        )

        self._runs_since_cycle = 0
        self._counter_lock = threading.Lock()
        # Promotion cycles make several LLM calls; they must never run on
        # the hot path. Mirrors the context-summary daemon-thread pattern.
        self._cycle_lock = threading.Lock()
        self._cycle_in_flight = False
        self.last_report: Optional[PromotionReport] = None

    # -------------------------------------------------------------- retrieval

    def retrieve_skills_for_query(
        self, query: str, user_id: Optional[str] = None
    ) -> List[ScoredSkill]:
        try:
            return self.skillbox.retrieve_skills_by_query(
                query,
                limit=self.config.max_skills_in_context,
                min_similarity=self.config.retrieval_min_similarity,
                statuses=(SkillStatus.ACTIVE,),
            )
        except Exception as exc:
            logger.warning("Learned-skill retrieval failed: %s", exc)
            return []

    def format_skills_prompt_section(self, scored_skills: List[ScoredSkill]) -> str:
        """Render retrieved skills for the per-turn volatile context block.

        The precondition-check framing is a requirement, not flavor text —
        it is the mechanism that prevents negative transfer on partial
        matches (asserted by unit test). Raw exemplars are deliberately not
        rendered: the distilled skill is the only representation of its
        source workflows allowed in the prompt.
        """
        if not scored_skills:
            return ""
        parts = [SKILLS_PROMPT_HEADER]
        for scored in scored_skills:
            skill = scored.skill
            parts.append(
                f"── Skill: {skill.name} (v{skill.version}, "
                f"match {scored.similarity:.2f}) ──\n{skill.content}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------ attribution

    def record_run_outcome(self, workflow, record_id: Optional[str] = None) -> None:
        """Post-store hook: stamp coverage, then feed the monitor.

        Write-time stamping lives here (not inside ``Workflow``) so the
        Workflow class stays decoupled from the Skillbox; both MemAgent
        capture paths funnel through this single method.
        """
        try:
            run_hash = getattr(workflow, "canonical_hash", None)
            if run_hash and record_id:
                covering = self.skillbox.get_active_skill_for_hash(run_hash)
                if covering:
                    workflow.promoted_skill_id = covering.skill_id
                    self.memory_provider.update_by_id(
                        str(record_id),
                        {"promoted_skill_id": covering.skill_id},
                        memory_store_type=MemoryType.WORKFLOW_MEMORY,
                    )
        except Exception as exc:
            logger.debug("Write-time skill stamping failed: %s", exc)

        self.monitor.record_run_outcome(workflow)

    # -------------------------------------------------------------- promotion

    def run_promotion_cycle(self, user_id: Optional[str] = None) -> PromotionReport:
        """Synchronous cycle: staleness sweep first, then promotion."""
        self.monitor.staleness_sweep(self._resolve_tool)
        report = self.engine.run_promotion_cycle(user_id=user_id)
        self.last_report = report
        if report.promoted or report.rejected or report.review_flags:
            logger.info("Promotion cycle report: %s", report.to_dict())
        return report

    def maybe_run_scheduled_cycle(self) -> None:
        """Every-N-runs trigger, executed off the hot path.

        ``promotion_every_n_runs = 0`` means manual cycles only. A later
        integration with ``memorizz.automation`` can replace this counter.
        """
        every_n = self.config.promotion_every_n_runs
        if every_n <= 0:
            return
        with self._counter_lock:
            self._runs_since_cycle += 1
            if self._runs_since_cycle < every_n:
                return
            self._runs_since_cycle = 0

        with self._cycle_lock:
            if self._cycle_in_flight:
                return
            self._cycle_in_flight = True

        def _run() -> None:
            try:
                self.run_promotion_cycle()
            except Exception as exc:
                logger.error("Scheduled promotion cycle failed: %s", exc)
            finally:
                with self._cycle_lock:
                    self._cycle_in_flight = False

        thread = threading.Thread(
            target=_run, name="memorizz-promotion-cycle", daemon=True
        )
        thread.start()

    def activate_skill(self, skill_id: str) -> bool:
        """Promote a SHADOW skill to ACTIVE (with stamp backfill)."""
        return self.engine.activate_skill(skill_id)

    def promote_class(
        self, canonical_hash: str, user_id: Optional[str] = None
    ) -> PromotionReport:
        """Gated promotion of a single trajectory class (human-triggered).

        The eligibility gates still apply — this targets a class, it does
        not bypass the evidence requirements.
        """
        report = self.engine.promote_class(canonical_hash, user_id=user_id)
        self.last_report = report
        return report

    def demote_skill(self, skill_id: str, reason: str = "manually demoted") -> bool:
        """Manually demote a skill (clears its workflow-suppression stamps)."""
        skill = self.skillbox.get_skill_by_id(skill_id)
        if not skill:
            return False
        self.monitor.demote(skill, reason)
        return True

    # ---------------------------------------------------------------- helpers

    def _resolve_tool(self, tool_name: str) -> bool:
        """Tool-name resolution used by validation and the staleness sweep:
        the live ToolManager first (session truth), TOOLBOX store second."""
        try:
            if self.tool_manager:
                for meta in self.tool_manager.get_tool_metadata() or []:
                    if meta.get("name") == tool_name:
                        return True
        except Exception:
            pass
        try:
            return (
                self.memory_provider.retrieve_by_name(
                    tool_name, memory_store_type=MemoryType.TOOLBOX
                )
                is not None
            )
        except Exception:
            return False
