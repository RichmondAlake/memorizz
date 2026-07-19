# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Unit tests for the promotion gates, validation gate, suppression
contract, and the skill monitor."""

from datetime import datetime, timedelta

import pytest

from memorizz.long_term.procedural.skillbox import (
    PromotionConfig,
    PromotionEngine,
    Skill,
    SkillInjectionRole,
    SkillMonitor,
    SkillStatus,
)
from memorizz.long_term.procedural.skillbox.distiller import SkillDistiller
from memorizz.long_term.procedural.workflow.canonicalization import (
    RunRef,
    TrajectoryStats,
)
from memorizz.long_term.procedural.workflow.workflow import Workflow, WorkflowOutcome

NOW = datetime(2026, 7, 4, 12, 0, 0)


def _stats(
    executions=6,
    successes=5,
    distinct_queries=3,
    last_seen_days_ago=1,
    already_promoted=False,
    chash="hash-1",
):
    runs = []
    for index in range(executions):
        runs.append(
            RunRef(
                workflow_id=f"wf-{index}",
                outcome="success" if index < successes else "failure",
                created_at=NOW - timedelta(days=last_seen_days_ago, hours=index),
                user_query=f"query variant {index % max(distinct_queries, 1)}",
                record_id=f"rec-{index}",
            )
        )
    return TrajectoryStats(
        canonical_hash=chash,
        executions=executions,
        successes=successes,
        success_rate=successes / executions,
        distinct_query_count=distinct_queries,
        first_seen=NOW - timedelta(days=10),
        last_seen=NOW - timedelta(days=last_seen_days_ago),
        sample_workflow_ids=[r.workflow_id for r in runs if r.outcome == "success"],
        failure_workflow_ids=[r.workflow_id for r in runs if r.outcome != "success"],
        already_promoted=already_promoted,
        runs=runs,
    )


class _StubSkillbox:
    def __init__(self, skills=None):
        self.skills = list(skills or [])

    def list_skills(self, statuses=None):
        wanted = {s for s in statuses} if statuses else None
        return [s for s in self.skills if wanted is None or s.status in wanted]

    def add_skill(self, skill):
        self.skills.append(skill)
        return skill.skill_id

    def get_skill_by_id(self, skill_id):
        return next((s for s in self.skills if s.skill_id == skill_id), None)

    def set_status(self, skill_id, status, reason=None):
        skill = self.get_skill_by_id(skill_id)
        if not skill:
            return False
        skill.status = status
        if status in (SkillStatus.DEMOTED, SkillStatus.DEPRECATED):
            skill.demoted_at = datetime.now()
            skill.demotion_reason = reason
        return True

    def update_skill(self, skill):
        return True

    def get_active_skill_for_hash(self, canonical_hash):
        for skill in self.skills:
            if (
                skill.status == SkillStatus.ACTIVE
                and skill.source_canonical_hash == canonical_hash
            ):
                return skill
        return None


def _engine(skillbox=None, config=None, provider=None):
    return PromotionEngine(
        memory_provider=provider,
        skillbox=skillbox or _StubSkillbox(),
        llm_provider=object(),  # only checked for None in run_promotion_cycle
        config=config or PromotionConfig(),
        agent_id="agent-1",
        resolve_tool=lambda name: True,
    )


class TestSkillAuthority:
    def test_legacy_default_remains_user_authority(self):
        assert PromotionConfig().skill_injection_role == SkillInjectionRole.USER
        assert PromotionConfig().shadow_evaluation_enabled is False
        assert (
            PromotionConfig.from_dict({}).skill_injection_role
            == SkillInjectionRole.USER
        )

    def test_developer_authority_requires_shadow_review(self):
        with pytest.raises(ValueError, match="require_shadow=True"):
            PromotionConfig(skill_injection_role="developer")

        config = PromotionConfig(
            skill_injection_role="developer",
            require_shadow=True,
        )
        assert config.skill_injection_role == SkillInjectionRole.DEVELOPER

    def test_shadow_evaluation_configuration_is_validated(self):
        config = PromotionConfig.from_dict(
            {
                "shadow_evaluation_enabled": True,
                "shadow_evaluation_max_candidates": 4,
                "shadow_evaluation_min_similarity": 0.65,
                "shadow_evaluation_queue_size": 25,
                "shadow_evaluation_recent_window": 12,
                "shadow_readiness_min_observations": 8,
            }
        )
        assert config.shadow_evaluation_enabled is True
        assert config.shadow_evaluation_max_candidates == 4
        assert config.shadow_evaluation_min_similarity == 0.65
        with pytest.raises(ValueError, match="queue_size"):
            PromotionConfig(shadow_evaluation_queue_size=0)

    def test_review_activation_can_set_developer_authority(self):
        skill = Skill(
            name="review me",
            description="d",
            content="c",
            source_canonical_hash=None,
            status=SkillStatus.SHADOW,
            embedding=[],
        )
        box = _StubSkillbox([skill])
        engine = _engine(
            skillbox=box,
            config=PromotionConfig(
                require_shadow=True,
                skill_injection_role="developer",
            ),
            provider=_RecordingProvider(),
        )

        assert engine.activate_skill(skill.skill_id, injection_role="developer")
        assert skill.status == SkillStatus.ACTIVE
        assert skill.injection_role == SkillInjectionRole.DEVELOPER


class TestGates:
    """Each gate must independently block promotion — frequency alone
    never promotes."""

    def test_all_gates_pass(self):
        engine = _engine()
        ok, reason = engine._eligibility(_stats(), {})
        assert ok, reason

    def test_min_executions_blocks(self):
        engine = _engine()
        ok, reason = engine._eligibility(_stats(executions=4, successes=4), {})
        assert not ok and "executions" in reason

    def test_min_success_rate_blocks(self):
        engine = _engine()
        # 79% < 80% even with plenty of volume
        ok, reason = engine._eligibility(_stats(executions=100, successes=79), {})
        assert not ok and "success_rate" in reason

    def test_min_distinct_queries_blocks(self):
        engine = _engine()
        ok, reason = engine._eligibility(_stats(distinct_queries=1), {})
        assert not ok and "distinct_queries" in reason

    def test_staleness_blocks(self):
        engine = _engine()
        ok, reason = engine._eligibility(_stats(last_seen_days_ago=45), {})
        assert not ok and "stale" in reason

    def test_already_promoted_blocks(self):
        engine = _engine()
        ok, reason = engine._eligibility(_stats(already_promoted=True), {})
        assert not ok

    def test_live_shadow_skill_blocks_repromotion(self):
        shadow = Skill(
            name="s",
            description="d",
            content="c",
            source_canonical_hash="hash-1",
            status=SkillStatus.SHADOW,
            embedding=[],
        )
        engine = _engine()
        ok, reason = engine._eligibility(_stats(), {"hash-1": [shadow]})
        assert not ok and "live skill" in reason

    def test_repromotion_counts_only_post_demotion_runs(self):
        demoted = Skill(
            name="s",
            description="d",
            content="c",
            source_canonical_hash="hash-1",
            status=SkillStatus.DEMOTED,
            demoted_at=NOW - timedelta(days=0, hours=1),
            embedding=[],
        )
        engine = _engine()
        # 6 runs, but all created before the demotion → 0 count toward gates
        stale_stats = _stats(last_seen_days_ago=2)
        ok, reason = engine._eligibility(stale_stats, {"hash-1": [demoted]})
        assert not ok and "executions" in reason


class TestScore:
    def test_recency_decay_orders_candidates(self):
        engine = _engine()
        fresh = _stats(last_seen_days_ago=0)
        old = _stats(last_seen_days_ago=28)
        assert engine._score(fresh) > engine._score(old)

    def test_success_rate_scales_score(self):
        engine = _engine()
        strong = _stats(executions=10, successes=10)
        weak = _stats(executions=10, successes=8)
        assert engine._score(strong) > engine._score(weak)


class _RecordingProvider:
    """update_by_id recorder + canned retrieve for stamping tests."""

    def __init__(self):
        self.updates = []

    def update_by_id(self, id, data, memory_store_type=None):
        self.updates.append((id, data))
        return True

    def retrieve_by_name(self, name, memory_store_type=None):
        return {"name": name}

    def list_all(self, memory_store_type=None):
        return []


class TestStamping:
    def test_stamp_workflows_uses_record_ids(self):
        provider = _RecordingProvider()
        engine = _engine(provider=provider)
        stat = _stats(executions=3, successes=3)
        stamped = engine.stamp_workflows(stat, "skill-42")
        assert stamped == 3
        assert all(
            data == {"promoted_skill_id": "skill-42"} for _, data in provider.updates
        )
        assert {rid for rid, _ in provider.updates} == {
            "rec-0",
            "rec-1",
            "rec-2",
        }


VALID_SKILL_MD = """---
name: Refund lookup and issue
description: Use when the user asks to refund a completed order.
preconditions:
  - Order identifier is present in the query
tools: [lookup_order, issue_refund]
---

# Refund lookup and issue

## When to apply
When the user requests a refund and an order id is available.
When NOT to apply: pending or already-refunded orders.

## Procedure
1. Call `lookup_order` with the order identifier.
2. Call `issue_refund` with order_id and amount.
"""


class _JudgeLLM:
    def __init__(self, answer="YES — plausible."):
        self.answer = answer

    def generate_text(self, prompt, instructions=None):
        return self.answer


class _RecordingJudge(_JudgeLLM):
    def __init__(self, answer="YES — plausible."):
        super().__init__(answer)
        self.prompt = ""
        self.instructions = ""

    def generate_text(self, prompt, instructions=None):
        self.prompt = prompt
        self.instructions = instructions or ""
        return self.answer


def _distiller(resolve=lambda name: True, llm=None):
    return SkillDistiller(
        llm_provider=llm or _JudgeLLM(),
        resolve_tool=resolve,
        max_content_chars=4000,
    )


def _sample_docs():
    return [
        {
            "user_query": "refund order A-12345678",
            "outcome": "success",
            "steps": {
                "Step 1: lookup_order": {
                    "arguments": {"order_id": "A-12345678"},
                    "result": "found",
                    "error": None,
                }
            },
        }
    ]


class TestValidationGate:
    def test_valid_document_passes(self):
        verdict, parsed = _distiller().validate(VALID_SKILL_MD, _sample_docs())
        assert verdict.ok, verdict.reasons
        assert parsed.tools == ["lookup_order", "issue_refund"]

    def test_insufficient_rejected(self):
        verdict, parsed = _distiller().validate("INSUFFICIENT", _sample_docs())
        assert not verdict.ok and parsed is None

    def test_unknown_tool_rejected(self):
        verdict, _ = _distiller(resolve=lambda name: name != "issue_refund").validate(
            VALID_SKILL_MD, _sample_docs()
        )
        assert not verdict.ok
        assert any("issue_refund" in reason for reason in verdict.reasons)

    def test_hallucinated_tool_in_procedure_rejected(self):
        bad = VALID_SKILL_MD.replace(
            "Call `issue_refund` with order_id and amount.",
            "Call `magic_undo` with order_id.",
        )
        verdict, _ = _distiller().validate(bad, _sample_docs())
        assert not verdict.ok
        assert any("magic_undo" in reason for reason in verdict.reasons)

    def test_tool_dump_description_rejected(self):
        bad = VALID_SKILL_MD.replace(
            "description: Use when the user asks to refund a completed order.",
            "description: lookup_order issue_refund lookup_order",
        )
        verdict, _ = _distiller().validate(bad, _sample_docs())
        assert not verdict.ok
        assert any("tool-sequence" in reason for reason in verdict.reasons)

    def test_oversize_content_rejected(self):
        distiller = SkillDistiller(
            llm_provider=_JudgeLLM(),
            resolve_tool=lambda n: True,
            max_content_chars=100,
        )
        verdict, _ = distiller.validate(VALID_SKILL_MD, _sample_docs())
        assert not verdict.ok
        assert any("exceeds" in reason for reason in verdict.reasons)

    def test_literal_argument_leak_rejected(self):
        leaked = VALID_SKILL_MD.replace(
            "1. Call `lookup_order` with the order identifier.",
            "1. Call `lookup_order` with order_id A-12345678.",
        )
        verdict, _ = _distiller().validate(leaked, _sample_docs())
        assert not verdict.ok
        assert any("leaked" in reason for reason in verdict.reasons)

    def test_llm_judge_no_rejects(self):
        verdict, _ = _distiller(llm=_JudgeLLM("NO — steps contradict runs.")).validate(
            VALID_SKILL_MD, _sample_docs()
        )
        assert not verdict.ok
        assert any("judge" in reason.lower() for reason in verdict.reasons)

    def test_llm_judge_receives_failed_runs(self):
        judge = _RecordingJudge()
        failed = {
            "user_query": "refund pending order P-87654321",
            "outcome": "failure",
            "steps": {
                "Step 1: lookup_order": {
                    "arguments": {"order_id": "P-87654321"},
                    "result": "pending",
                    "error": None,
                }
            },
        }

        verdict, _ = _distiller(llm=judge).validate(
            VALID_SKILL_MD,
            _sample_docs(),
            failure_docs=[failed],
        )

        assert verdict.ok, verdict.reasons
        assert "OBSERVED FAILED RUNS" in judge.prompt
        assert "refund pending order" in judge.prompt
        assert "avoiding or correctly handling" in judge.instructions

    def test_literal_leak_from_failed_run_is_rejected(self):
        failed = {
            "user_query": "refund pending order P-87654321",
            "outcome": "failure",
            "steps": {
                "Step 1: lookup_order": {
                    "arguments": {"order_id": "P-87654321"},
                    "result": "pending",
                    "error": None,
                }
            },
        }
        leaked = VALID_SKILL_MD.replace(
            "pending or already-refunded orders.",
            "pending order P-87654321.",
        )

        verdict, _ = _distiller().validate(
            leaked,
            _sample_docs(),
            failure_docs=[failed],
        )

        assert not verdict.ok
        assert any("leaked" in reason for reason in verdict.reasons)


class TestSuppression:
    """retrieve_workflows_by_query hides skill-covered trajectories."""

    class _QueryProvider:
        def __init__(self, docs):
            self.docs = docs

        def retrieve_by_query(self, query, memory_store_type=None, limit=5):
            return self.docs[:limit]

    def _doc(self, idx, promoted=None):
        return {
            "name": f"wf {idx}",
            "workflow_id": f"wf-{idx}",
            "steps": {},
            "outcome": "success",
            "promoted_skill_id": promoted,
            "embedding": [0.0],
        }

    def test_default_excludes_promoted(self):
        provider = self._QueryProvider(
            [self._doc(1, promoted="skill-1"), self._doc(2), self._doc(3)]
        )
        results = Workflow.retrieve_workflows_by_query("q", provider, limit=2)
        assert [w.workflow_id for w in results] == ["wf-2", "wf-3"]

    def test_opt_out_includes_promoted(self):
        provider = self._QueryProvider([self._doc(1, promoted="skill-1"), self._doc(2)])
        results = Workflow.retrieve_workflows_by_query(
            "q", provider, limit=2, exclude_promoted=False
        )
        assert [w.workflow_id for w in results] == ["wf-1", "wf-2"]


def _monitored_skill(baseline_rate=0.9, status=SkillStatus.ACTIVE):
    return Skill(
        name="Refund flow",
        description="d",
        content="c",
        source_canonical_hash="hash-1",
        status=status,
        baseline={"executions": 10, "success_rate": baseline_rate},
        embedding=[],
    )


def _workflow(outcome, skills, chash="hash-1"):
    wf = Workflow.__new__(Workflow)
    wf.skills_activated = skills
    wf.outcome = outcome
    wf.canonical_hash = chash
    return wf


class TestMonitor:
    def _monitor(self, skill, config=None):
        box = _StubSkillbox([skill])
        monitor = SkillMonitor(
            box,
            memory_provider=_RecordingProvider(),
            config=config
            or PromotionConfig(
                min_activations_before_drift_check=5,
                drift_window_activations=10,
                demotion_success_delta=0.2,
            ),
        )
        return monitor, box

    def test_attribution_counts_and_deviations(self):
        skill = _monitored_skill()
        monitor, _ = self._monitor(skill)
        monitor.record_run_outcome(_workflow(WorkflowOutcome.SUCCESS, [skill.skill_id]))
        monitor.record_run_outcome(
            _workflow(WorkflowOutcome.FAILURE, [skill.skill_id], chash="other")
        )
        assert skill.stats["activations"] == 2
        assert skill.stats["successes"] == 1
        assert skill.stats["failures"] == 1
        assert skill.stats["deviations"] == 1

    def test_demotion_does_not_fire_at_exact_boundary(self):
        # baseline 0.9, delta 0.2 → threshold 0.7. Rolling exactly 0.7
        # (7/10) must NOT demote — the contract is strictly below.
        skill = _monitored_skill(baseline_rate=0.9)
        monitor, _ = self._monitor(skill)
        outcomes = [True] * 7 + [False] * 3
        for succeeded in outcomes:
            monitor.record_run_outcome(
                _workflow(
                    WorkflowOutcome.SUCCESS if succeeded else WorkflowOutcome.FAILURE,
                    [skill.skill_id],
                )
            )
        assert skill.status == SkillStatus.ACTIVE

    def test_demotion_fires_below_boundary(self):
        skill = _monitored_skill(baseline_rate=0.9)
        monitor, _ = self._monitor(skill)
        outcomes = [True] * 6 + [False] * 4  # rolling 0.6 < 0.7
        for succeeded in outcomes:
            monitor.record_run_outcome(
                _workflow(
                    WorkflowOutcome.SUCCESS if succeeded else WorkflowOutcome.FAILURE,
                    [skill.skill_id],
                )
            )
        assert skill.status == SkillStatus.DEMOTED
        assert "drift" in (skill.demotion_reason or "")

    def test_demotion_clears_stamps(self):
        skill = _monitored_skill()
        box = _StubSkillbox([skill])
        provider = _RecordingProvider()
        provider.list_all = lambda memory_store_type=None: [
            {"_id": "rec-1", "promoted_skill_id": skill.skill_id},
            {"_id": "rec-2", "promoted_skill_id": "someone-else"},
        ]
        monitor = SkillMonitor(box, provider, config=PromotionConfig())
        monitor.demote(skill, reason="test")
        assert provider.updates == [("rec-1", {"promoted_skill_id": None})]

    def test_staleness_sweep_deprecates_on_tool_removal(self):
        skill = _monitored_skill()
        skill.tools_used = ["lookup_order", "gone_tool"]
        monitor, box = self._monitor(skill)
        deprecated = monitor.staleness_sweep(lambda name: name != "gone_tool")
        assert deprecated == [skill.skill_id]
        assert skill.status == SkillStatus.DEPRECATED
        assert "gone_tool" in (skill.demotion_reason or "")

    def test_monitor_exceptions_never_propagate(self):
        skill = _monitored_skill()
        box = _StubSkillbox([skill])
        box.update_skill = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
        monitor = SkillMonitor(box, _RecordingProvider(), config=PromotionConfig())
        # Must not raise into the user-facing run.
        monitor.record_run_outcome(_workflow(WorkflowOutcome.SUCCESS, [skill.skill_id]))

    def test_shadow_ids_never_enter_active_attribution_or_drift(self):
        skill = _monitored_skill(status=SkillStatus.SHADOW)
        monitor, _ = self._monitor(skill)
        for _ in range(10):
            monitor.record_run_outcome(
                _workflow(WorkflowOutcome.FAILURE, [skill.skill_id])
            )
        assert skill.status == SkillStatus.SHADOW
        assert skill.stats["activations"] == 0
        assert skill.stats["successes"] == 0
        assert skill.stats["failures"] == 0
        assert skill.stats["recent_outcomes"] == []


class TestPromoteClass:
    """promote_class targets one trajectory class but never bypasses gates."""

    class _AggProvider:
        """Provider stub: workflow docs for aggregate + recording updates."""

        def __init__(self, docs):
            self._docs = docs
            self.updates = []

        def list_all(self, memory_store_type=None):
            return self._docs

        def retrieve_by_id(self, id, memory_store_type=None):
            return next((d for d in self._docs if str(d.get("_id")) == str(id)), None)

        def update_by_id(self, id, data, memory_store_type=None):
            self.updates.append((id, data))
            return True

        def retrieve_by_name(self, name, memory_store_type=None):
            return {"name": name}

    @staticmethod
    def _docs(count, chash="class-1"):
        return [
            {
                "_id": f"rec-{i}",
                "workflow_id": f"wf-{i}",
                "agent_id": "agent-1",
                "canonical_hash": chash,
                "outcome": "success",
                "user_query": f"variant {i}",
                "created_at": (NOW - timedelta(hours=i)).isoformat(),
                "steps": {},
            }
            for i in range(count)
        ]

    class _LLM:
        def generate_text(self, prompt, instructions=None):
            if instructions and "YES or NO" in instructions:
                return "YES"
            return VALID_SKILL_MD

    def _engine(self, provider, box):
        return PromotionEngine(
            memory_provider=provider,
            skillbox=box,
            llm_provider=self._LLM(),
            config=PromotionConfig(min_executions=5, min_distinct_queries=2),
            agent_id="agent-1",
            resolve_tool=lambda name: True,
        )

    def test_eligible_class_is_promoted(self):
        provider = self._AggProvider(self._docs(5))
        box = _StubSkillbox()
        report = self._engine(provider, box).promote_class("class-1")
        assert len(report.promoted) == 1
        assert box.skills and box.skills[0].status == SkillStatus.ACTIVE
        # source runs stamped
        assert all(
            data == {"promoted_skill_id": box.skills[0].skill_id}
            for _, data in provider.updates
        )

    def test_gates_still_enforced(self):
        provider = self._AggProvider(self._docs(2))  # below min_executions
        box = _StubSkillbox()
        report = self._engine(provider, box).promote_class("class-1")
        assert report.promoted == []
        assert report.skipped and "executions" in report.skipped[0][1]
        assert box.skills == []

    def test_unknown_class_is_skipped(self):
        provider = self._AggProvider(self._docs(5))
        box = _StubSkillbox()
        report = self._engine(provider, box).promote_class("no-such-hash")
        assert report.skipped == [("no-such-hash", "unknown trajectory class")]
