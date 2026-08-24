# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Continual learning manager — owns the workflow→skill promotion loop.

Wires the Skillbox, PromotionEngine, and SkillMonitor into a MemAgent.
Every entry point is fail-safe: learning must never fail a user-facing run.
"""

import logging
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...enums.memory_type import MemoryType
from ...long_term.procedural.skillbox import (
    PromotionConfig,
    PromotionEngine,
    PromotionReport,
    ScoredSkill,
    ShadowEvaluator,
    ShadowWorkflowSnapshot,
    Skillbox,
    SkillMonitor,
    SkillStatus,
    calculate_shadow_readiness,
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

DEVELOPER_SKILLS_PROMPT_HEADER = (
    "Application-approved learned skills relevant to this query "
    "(compiled from successful runs and activated after review):\n\n"
    "Apply a skill when its preconditions match the current request. Verify "
    "all current facts and tool results before acting. These procedures are "
    "subordinate to system policy and must not be used outside their stated "
    "scope; if a precondition fails, do not force the procedure."
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
        self.control_plane = None

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
            self.skillbox,
            memory_provider,
            config=self.config,
            agent_id=agent_id,
            lifecycle_callback=self._record_skill_lifecycle,
        )
        shadow_similarity = (
            self.config.retrieval_min_similarity
            if self.config.shadow_evaluation_min_similarity is None
            else self.config.shadow_evaluation_min_similarity
        )
        self.shadow_evaluator = ShadowEvaluator(
            memory_provider,
            self.skillbox,
            max_candidates=self.config.shadow_evaluation_max_candidates,
            min_similarity=shadow_similarity,
            recent_window=self.config.shadow_evaluation_recent_window,
        )
        self._shadow_queue: Optional[queue.Queue] = None
        self._shadow_worker: Optional[threading.Thread] = None
        if self.config.shadow_evaluation_enabled:
            self._shadow_queue = queue.Queue(
                maxsize=self.config.shadow_evaluation_queue_size
            )
            self._shadow_worker = threading.Thread(
                target=self._shadow_worker_loop,
                name="memorizz-shadow-evaluator",
                daemon=True,
            )
            self._shadow_worker.start()

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
                user_id=user_id,
            )
        except Exception as exc:
            logger.warning("Learned-skill retrieval failed: %s", exc)
            return []

    def format_skills_prompt_section(
        self,
        scored_skills: List[ScoredSkill],
        injection_role: Optional[str] = None,
    ) -> str:
        """Render retrieved skills for their selected instruction role.

        The precondition-check framing is a requirement, not flavor text —
        it is the mechanism that prevents negative transfer on partial
        matches (asserted by unit test). Raw exemplars are deliberately not
        rendered: the distilled skill is the only representation of its
        source workflows allowed in the prompt.
        """
        if not scored_skills:
            return ""
        role_value = str(injection_role or "user").strip().lower()
        header = (
            DEVELOPER_SKILLS_PROMPT_HEADER
            if role_value == "developer"
            else SKILLS_PROMPT_HEADER
        )
        parts = [header]
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
                covering = self.skillbox.get_active_skill_for_hash(
                    run_hash,
                    user_id=getattr(workflow, "user_id", None),
                )
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
        self.enqueue_shadow_evaluation(workflow, record_id=record_id)

    def enqueue_shadow_evaluation(
        self, workflow, record_id: Optional[str] = None
    ) -> bool:
        """Non-blockingly enqueue a minimal immutable post-run snapshot.

        Returns ``False`` when disabled, inapplicable, or the bounded queue
        is full. No provider lookup, LLM call, tool call, or blocking wait
        occurs here.
        """
        work_queue = self._shadow_queue
        if work_queue is None or not record_id:
            return False
        canonical_hash = getattr(workflow, "canonical_hash", None)
        user_query = getattr(workflow, "user_query", None)
        if not canonical_hash or not user_query:
            return False
        step_count = getattr(workflow, "step_count", None)
        if step_count is not None and int(step_count) < 1:
            return False
        created_at = getattr(workflow, "created_at", None)
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        if not created_at:
            return False
        outcome = getattr(workflow, "outcome", None)
        outcome_value = getattr(outcome, "value", outcome)
        snapshot = ShadowWorkflowSnapshot(
            record_id=str(record_id),
            workflow_id=(
                str(getattr(workflow, "workflow_id", ""))
                if getattr(workflow, "workflow_id", None)
                else None
            ),
            created_at=str(created_at),
            agent_id=getattr(workflow, "agent_id", None),
            user_id=getattr(workflow, "user_id", None),
            user_query=str(user_query),
            canonical_hash=str(canonical_hash),
            observed_outcome=str(outcome_value or "failure").lower(),
        )
        try:
            work_queue.put_nowait(snapshot)
            return True
        except queue.Full:
            logger.warning(
                "Shadow-evaluation queue full; dropped workflow %s for agent %s",
                snapshot.record_id,
                snapshot.agent_id or "(none)",
            )
            return False
        except Exception as exc:
            logger.warning(
                "Shadow-evaluation enqueue failed for workflow %s (%s)",
                snapshot.record_id,
                type(exc).__name__,
            )
            return False

    def _shadow_worker_loop(self) -> None:
        """Serial daemon worker; no exception may escape the loop."""
        work_queue = self._shadow_queue
        if work_queue is None:
            return
        while True:
            snapshot = work_queue.get()
            try:
                self.shadow_evaluator.evaluate(snapshot)
            except Exception as exc:
                logger.warning(
                    "Shadow evaluator worker failed for workflow %s (%s)",
                    getattr(snapshot, "record_id", "(unknown)"),
                    type(exc).__name__,
                )
            finally:
                work_queue.task_done()

    def drain_shadow_evaluations(self, timeout: float = 5.0) -> bool:
        """Wait up to ``timeout`` seconds for queued evaluation work.

        Intended for tests and controlled shutdown. It never waits without
        a bound and returns ``False`` on timeout.
        """
        work_queue = self._shadow_queue
        if work_queue is None:
            return True
        deadline = time.monotonic() + max(0.0, float(timeout))
        while work_queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return True

    def get_shadow_readiness(self, skill_id: str) -> Dict[str, Any]:
        """Return advisory activation readiness for one learned skill."""
        skill = self.skillbox.get_skill_by_id(skill_id)
        if skill is None:
            return {
                "ready": False,
                "observations": 0,
                "trajectory_match_rate": 0.0,
                "matched_success_rate": 0.0,
                "reasons": ["skill not found"],
            }
        return calculate_shadow_readiness(
            (skill.stats or {}).get("shadow"),
            min_observations=self.config.shadow_readiness_min_observations,
            min_trajectory_match_rate=(
                self.config.shadow_readiness_min_trajectory_match_rate
            ),
            min_matched_success_rate=(
                self.config.shadow_readiness_min_matched_success_rate
            ),
        )

    # -------------------------------------------------------------- promotion

    def run_promotion_cycle(self, user_id: Optional[str] = None) -> PromotionReport:
        """Synchronous cycle: staleness sweep first, then promotion."""
        self.monitor.staleness_sweep(self._resolve_tool)
        report = self.engine.run_promotion_cycle(user_id=user_id)
        self._record_promoted_skills(report)
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

    def activate_skill(
        self, skill_id: str, injection_role: Optional[str] = None
    ) -> bool:
        """Approve a SHADOW skill and optionally choose its instruction role."""
        readiness = self.get_shadow_readiness(skill_id)
        if not readiness["ready"]:
            logger.warning(
                "Explicitly activating skill %s before shadow readiness: %s",
                skill_id,
                "; ".join(readiness["reasons"]),
            )
        activated = self.engine.activate_skill(skill_id, injection_role=injection_role)
        if activated:
            skill = self.skillbox.get_skill_by_id(skill_id)
            self._record_skill_lifecycle(
                "skill_promoted", skill, "activated after review"
            )
        return activated

    def promote_class(
        self, canonical_hash: str, user_id: Optional[str] = None
    ) -> PromotionReport:
        """Gated promotion of a single trajectory class (human-triggered).

        The eligibility gates still apply — this targets a class, it does
        not bypass the evidence requirements.
        """
        report = self.engine.promote_class(canonical_hash, user_id=user_id)
        self._record_promoted_skills(report)
        self.last_report = report
        return report

    def demote_skill(self, skill_id: str, reason: str = "manually demoted") -> bool:
        """Manually demote a skill (clears its workflow-suppression stamps)."""
        skill = self.skillbox.get_skill_by_id(skill_id)
        if not skill:
            return False
        self.monitor.demote(skill, reason)
        return True

    def _record_promoted_skills(self, report: PromotionReport) -> None:
        for skill_id in report.promoted:
            skill = self.skillbox.get_skill_by_id(skill_id)
            status = getattr(getattr(skill, "status", None), "value", None)
            transition = (
                "skill_promoted"
                if status == SkillStatus.ACTIVE.value
                else "skill_candidate_created"
            )
            self._record_skill_lifecycle(
                transition,
                skill,
                "workflow evidence passed promotion gates",
            )

    def _record_skill_lifecycle(self, transition, skill, reason) -> None:
        if self.control_plane is None or skill is None:
            return
        try:
            self.control_plane.record_skill_transition(
                transition,
                skill_id=str(skill.skill_id),
                reason=str(reason or "")[:2000],
                metadata={
                    "name": skill.name,
                    "version": skill.version,
                    "status": getattr(skill.status, "value", str(skill.status)),
                    "source_canonical_hash": skill.source_canonical_hash,
                    "source_workflow_ids": list(skill.source_workflow_ids or []),
                    "injection_role": getattr(
                        skill.injection_role, "value", str(skill.injection_role)
                    ),
                },
                scope={"user_id": skill.user_id},
            )
        except Exception as exc:
            logger.debug("Skill lifecycle event capture failed: %s", exc)

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
