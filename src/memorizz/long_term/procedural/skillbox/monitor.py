# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""SkillMonitor — attribution, drift detection, and demotion.

Every promoted skill is monitored for its whole life and has a demotion
path. A promotion-only system monotonically accumulates authority in
artifacts whose ground truth decays; forgetting is part of the learning
loop, not an ops afterthought.

Nothing in this module may raise into a user-facing run: every entry point
swallows and logs.
"""

import logging
from typing import Callable, List, Optional

from ...._time import utc_now_iso
from ....enums.memory_type import MemoryType
from .promotion import PromotionConfig
from .skill import Skill, SkillStatus
from .skillbox import Skillbox

logger = logging.getLogger(__name__)


class SkillMonitor:
    def __init__(
        self,
        skillbox: Skillbox,
        memory_provider,
        config: Optional[PromotionConfig] = None,
        agent_id: Optional[str] = None,
        lifecycle_callback: Optional[Callable[..., None]] = None,
    ):
        self.skillbox = skillbox
        self.memory_provider = memory_provider
        self.config = config or PromotionConfig()
        self.agent_id = agent_id
        self.lifecycle_callback = lifecycle_callback

    # ------------------------------------------------------------ attribution

    def record_run_outcome(self, workflow) -> None:
        """Update stats for every skill that was in this run's context.

        Attribution is by ``skills_activated`` (what was injected), not by
        hash match: a run that had skill X in context but took a different
        canonical path is a *deviation* — the skill retrieved on a query it
        didn't fit. That's tracked, not discarded.
        """
        try:
            skill_ids = list(getattr(workflow, "skills_activated", None) or [])
            held_out = list(getattr(workflow, "skills_held_out", None) or [])
            if not skill_ids and not held_out:
                return
            outcome = getattr(getattr(workflow, "outcome", None), "value", "success")
            succeeded = outcome == "success"
            verified = bool(getattr(workflow, "outcome_verified", False))
            run_hash = getattr(workflow, "canonical_hash", None)

            # The control side: the skill matched and was withheld. Only the
            # with-and-without figure moves; activations and drift do not.
            for skill_id in held_out:
                skill = self.skillbox.get_skill_by_id(skill_id)
                if not skill or skill.status != SkillStatus.ACTIVE:
                    continue
                self._credit_holdout(skill, "without", succeeded, verified)
                self.skillbox.update_skill(skill)

            for skill_id in skill_ids:
                skill = self.skillbox.get_skill_by_id(skill_id)
                # Passive shadow evidence has its own evaluator and storage
                # channel.  A shadow ID supplied manually (or by stale
                # application code) must never contaminate active activation
                # counts, rolling drift, or demotion decisions.
                if not skill or skill.status != SkillStatus.ACTIVE:
                    continue
                stats = skill.stats
                stats["activations"] = stats.get("activations", 0) + 1
                if succeeded:
                    stats["successes"] = stats.get("successes", 0) + 1
                else:
                    stats["failures"] = stats.get("failures", 0) + 1
                if run_hash and run_hash != skill.source_canonical_hash:
                    stats["deviations"] = stats.get("deviations", 0) + 1
                stats["last_activated_at"] = utc_now_iso()
                recent = list(stats.get("recent_outcomes") or [])
                recent.append(bool(succeeded))
                stats["recent_outcomes"] = recent[
                    -self.config.drift_window_activations :
                ]
                skill.stats = stats
                self._credit_holdout(skill, "with", succeeded, verified)
                self.skillbox.update_skill(skill)
                self._maybe_demote(skill)
        except Exception as exc:
            logger.error("Skill monitor attribution failed: %s", exc)

    @staticmethod
    def _credit_holdout(
        skill: Skill,
        side: str,
        succeeded: bool,
        verified: bool,
        count_run: bool = True,
    ) -> None:
        """Add one run to a side of the skill's with-and-without figure.

        ``count_run=False`` credits only the verified counters: a host that
        confirms an outcome after the run was stored must not count the run
        twice."""
        holdout = skill.stats.setdefault("holdout", {})
        counters = holdout.setdefault(side, {})
        for key in ("runs", "successes", "verified_runs", "verified_successes"):
            counters.setdefault(key, 0)
        if count_run:
            counters["runs"] += 1
            counters["successes"] += 1 if succeeded else 0
        if verified:
            counters["verified_runs"] += 1
            counters["verified_successes"] += 1 if succeeded else 0

    def record_verified_outcome(self, workflow_doc: dict, succeeded: bool) -> None:
        """A host confirmed the outcome of a run already stored: credit the
        verified counters of the skills it ran with and without."""
        try:
            for side, key in (
                ("with", "skills_activated"),
                ("without", "skills_held_out"),
            ):
                for skill_id in workflow_doc.get(key) or []:
                    skill = self.skillbox.get_skill_by_id(skill_id)
                    if not skill or skill.status != SkillStatus.ACTIVE:
                        continue
                    self._credit_holdout(
                        skill, side, succeeded, verified=True, count_run=False
                    )
                    self.skillbox.update_skill(skill)
        except Exception as exc:
            logger.error("Verified-outcome attribution failed: %s", exc)

    # ----------------------------------------------------------------- drift

    def _maybe_demote(self, skill: Skill) -> None:
        try:
            if skill.status != SkillStatus.ACTIVE:
                return
            stats = skill.stats
            if (
                stats.get("activations", 0)
                < self.config.min_activations_before_drift_check
            ):
                return
            rolling = self._rolling_success_rate(skill)
            if rolling is None:
                return
            baseline_rate = (skill.baseline or {}).get("success_rate")
            if baseline_rate is None:
                return
            if rolling < baseline_rate - self.config.demotion_success_delta:
                self.demote(
                    skill,
                    reason=(
                        f"drift: rolling {rolling:.2f} vs baseline "
                        f"{baseline_rate:.2f}"
                    ),
                )
        except Exception as exc:
            logger.error("Drift check failed for %s: %s", skill.skill_id, exc)

    def _rolling_success_rate(self, skill: Skill) -> Optional[float]:
        recent = skill.stats.get("recent_outcomes") or []
        if not recent:
            return None
        window = recent[-self.config.drift_window_activations :]
        return sum(1 for outcome in window if outcome) / len(window)

    # -------------------------------------------------------------- demotion

    def demote(self, skill: Skill, reason: str) -> None:
        logger.warning("Demoting skill %s (%s): %s", skill.name, skill.skill_id, reason)
        self.skillbox.set_status(skill.skill_id, SkillStatus.DEMOTED, reason)
        # Release the raw trajectories back to workflow retrieval — the
        # compiled form no longer speaks for them.
        self._clear_promoted_stamps(skill)
        self._notify_lifecycle("skill_demoted", skill, reason)

    def staleness_sweep(self, resolve_tool) -> List[str]:
        """Deprecate ACTIVE skills whose tools no longer resolve.

        Run at the top of every promotion cycle. ``resolve_tool`` is the
        same ``(name) -> bool`` callable the validation gate uses, so a
        skill is held to the identical standard for its whole life.
        """
        deprecated = []
        try:
            for skill in self.skillbox.list_skills(statuses=(SkillStatus.ACTIVE,)):
                missing = [
                    tool_name
                    for tool_name in skill.tools_used
                    if not self._safe_resolve(resolve_tool, tool_name)
                ]
                if missing:
                    reason = f"tool removed: {', '.join(missing)}"
                    logger.warning(
                        "Deprecating skill %s (%s): %s",
                        skill.name,
                        skill.skill_id,
                        reason,
                    )
                    self.skillbox.set_status(
                        skill.skill_id, SkillStatus.DEPRECATED, reason
                    )
                    self._clear_promoted_stamps(skill)
                    deprecated.append(skill.skill_id)
                    self._notify_lifecycle("skill_deprecated", skill, reason)
        except Exception as exc:
            logger.error("Staleness sweep failed: %s", exc)
        return deprecated

    def _notify_lifecycle(self, transition: str, skill: Skill, reason: str) -> None:
        if not callable(self.lifecycle_callback):
            return
        try:
            self.lifecycle_callback(transition, skill, reason)
        except Exception as exc:
            logger.debug("Skill lifecycle callback failed: %s", exc)

    @staticmethod
    def _safe_resolve(resolve_tool, tool_name: str) -> bool:
        try:
            return bool(resolve_tool(tool_name))
        except Exception:
            return False

    def _clear_promoted_stamps(self, skill: Skill) -> None:
        try:
            documents = (
                self.memory_provider.list_all(
                    memory_store_type=MemoryType.WORKFLOW_MEMORY
                )
                or []
            )
            for doc in documents:
                if not isinstance(doc, dict):
                    continue
                if str(doc.get("promoted_skill_id")) != str(skill.skill_id):
                    continue
                record_id = doc.get("_id") or doc.get("workflow_id")
                if not record_id:
                    continue
                self.memory_provider.update_by_id(
                    str(record_id),
                    {"promoted_skill_id": None},
                    memory_store_type=MemoryType.WORKFLOW_MEMORY,
                )
        except Exception as exc:
            logger.error("Failed clearing stamps for %s: %s", skill.skill_id, exc)
