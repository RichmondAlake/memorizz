"""Passive SHADOW-traffic evaluation and readiness tests."""

import queue
import threading
import time
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.procedural.skillbox import (
    ShadowWorkflowSnapshot,
    Skill,
    Skillbox,
    SkillStatus,
)
from memorizz.long_term.procedural.workflow import Workflow, WorkflowOutcome
from memorizz.memagent.managers.continual_learning_manager import (
    ContinualLearningManager,
)


class MemoryProviderStub:
    """Small provider with a true pre-top-k skill filter."""

    def __init__(self):
        self.docs = {kind: [] for kind in MemoryType}
        self.shadow_retrieval_calls = []
        self.block_shadow_retrieval = None
        self.shadow_retrieval_started = None
        self.raise_on_shadow_retrieval = False
        self.fail_workflow_updates = False

    def store(self, data=None, memory_store_type=None, **_kwargs):
        kind = MemoryType(memory_store_type)
        doc = dict(data or {})
        record_id = str(
            doc.get("_id")
            or doc.get("skill_id")
            or doc.get("workflow_id")
            or f"record-{len(self.docs[kind])}"
        )
        doc["_id"] = record_id
        self.docs[kind].append(doc)
        return record_id

    def list_all(self, memory_store_type=None, **_kwargs):
        return [dict(doc) for doc in self.docs[MemoryType(memory_store_type)]]

    def retrieve_by_id(self, record_id, memory_store_type=None):
        for doc in self.docs[MemoryType(memory_store_type)]:
            identities = {
                str(doc.get("_id")),
                str(doc.get("workflow_id")),
                str(doc.get("skill_id")),
            }
            if str(record_id) in identities:
                return dict(doc)
        return None

    def update_by_id(self, record_id, patch, memory_store_type=None):
        kind = MemoryType(memory_store_type)
        if kind == MemoryType.WORKFLOW_MEMORY and self.fail_workflow_updates:
            raise RuntimeError("persistence unavailable")
        for doc in self.docs[kind]:
            identities = {
                str(doc.get("_id")),
                str(doc.get("workflow_id")),
                str(doc.get("skill_id")),
            }
            if str(record_id) in identities:
                doc.update(dict(patch))
                return True
        return False

    def retrieve_by_query(self, query, memory_store_type=None, limit=1, **_kwargs):
        docs = sorted(
            self.docs[MemoryType(memory_store_type)],
            key=lambda doc: float(doc.get("_test_score", 0.9)),
            reverse=True,
        )
        return [
            {**dict(doc), "score": float(doc.get("_test_score", 0.9))}
            for doc in docs[:limit]
        ]

    def retrieve_skillbox_candidates(
        self, query, *, limit, statuses, agent_id, user_id
    ):
        self.shadow_retrieval_calls.append(
            {
                "limit": limit,
                "statuses": list(statuses),
                "agent_id": agent_id,
                "user_id": user_id,
            }
        )
        if self.shadow_retrieval_started:
            self.shadow_retrieval_started.set()
        if self.block_shadow_retrieval:
            self.block_shadow_retrieval.wait(timeout=2)
        if self.raise_on_shadow_retrieval:
            raise RuntimeError("vector store unavailable")
        eligible = [
            doc
            for doc in self.docs[MemoryType.SKILLBOX]
            if doc.get("status") in set(statuses)
            and doc.get("agent_id") == agent_id
            and doc.get("user_id") == user_id
        ]
        eligible.sort(key=lambda doc: float(doc.get("_test_score", 0.9)), reverse=True)
        return [
            {**dict(doc), "score": float(doc.get("_test_score", 0.9))}
            for doc in eligible[:limit]
        ]

    def retrieve_by_name(self, *_args, **_kwargs):
        return None


def _manager(provider, **overrides):
    config = {
        "shadow_evaluation_enabled": True,
        "shadow_evaluation_min_similarity": 0.1,
        "shadow_evaluation_recent_window": 2,
        "shadow_readiness_min_observations": 3,
        "shadow_readiness_min_trajectory_match_rate": 0.75,
        "shadow_readiness_min_matched_success_rate": 0.75,
        "promotion_every_n_runs": 0,
    }
    config.update(overrides)
    return ContinualLearningManager(
        provider,
        agent_id="agent-a",
        config=config,
    )


def _skill(
    *,
    status=SkillStatus.SHADOW,
    agent_id="agent-a",
    user_id="user-a",
    canonical_hash="expected-hash",
    created_at=None,
    source_ids=None,
    score=0.9,
):
    skill = Skill(
        name="Refund completed order",
        description="Use when a user requests an eligible order refund.",
        content="Verify eligibility, then refund.",
        queries=["refund my order"],
        agent_id=agent_id,
        user_id=user_id,
        source_canonical_hash=canonical_hash,
        source_workflow_ids=source_ids or [],
        status=status,
        created_at=created_at or datetime.now() - timedelta(minutes=1),
        embedding=[],
    )
    payload = skill.to_dict()
    payload["_test_score"] = score
    return skill, payload


def _store_skill(provider, skill, payload):
    provider.store(payload, memory_store_type=MemoryType.SKILLBOX)
    return skill


def _workflow(
    provider,
    *,
    workflow_id,
    canonical_hash="expected-hash",
    outcome=WorkflowOutcome.SUCCESS,
    created_at=None,
    agent_id="agent-a",
    user_id="user-a",
):
    workflow = Workflow(
        name="Tool run",
        description="Observed production workflow",
        workflow_id=workflow_id,
        agent_id=agent_id,
        user_id=user_id,
        user_query="please refund this completed order",
        steps={
            "Step 1: lookup_order": {
                "arguments": {"order_id": "current"},
                "result": "completed",
                "error": None,
            }
        },
        outcome=outcome,
        created_at=created_at or datetime.now(),
        embedding=[],
    )
    workflow.canonical_hash = canonical_hash
    workflow.canonical_signature = [{"tool": "lookup_order"}]
    workflow.step_count = 1
    record_id = workflow.store_workflow(provider)
    return workflow, record_id


def test_snapshot_is_immutable_and_contains_no_workflow_steps():
    snapshot = ShadowWorkflowSnapshot(
        record_id="record-1",
        workflow_id="workflow-1",
        created_at=datetime.now().isoformat(),
        agent_id="agent-a",
        user_id="user-a",
        user_query="refund",
        canonical_hash="hash",
        observed_outcome="success",
    )
    assert not hasattr(snapshot, "steps")
    with pytest.raises(FrozenInstanceError):
        snapshot.canonical_hash = "other"


def test_none_shadow_threshold_uses_active_retrieval_threshold():
    provider = MemoryProviderStub()
    manager = _manager(
        provider,
        retrieval_min_similarity=0.63,
        shadow_evaluation_min_similarity=None,
    )
    assert manager.shadow_evaluator.min_similarity == 0.63


def test_passive_evaluation_is_opt_in():
    provider = MemoryProviderStub()
    manager = ContinualLearningManager(
        provider,
        agent_id="agent-a",
        config={"shadow_evaluation_enabled": False},
    )
    workflow, record_id = _workflow(provider, workflow_id="disabled")
    manager.record_run_outcome(workflow, record_id)
    assert manager._shadow_queue is None
    assert provider.shadow_retrieval_calls == []


def test_dedicated_retrieval_returns_only_exact_scope_shadows():
    provider = MemoryProviderStub()
    box = Skillbox(provider, agent_id="agent-a")
    for status, agent_id, user_id, score in (
        (SkillStatus.ACTIVE, "agent-a", "user-a", 1.0),
        (SkillStatus.SHADOW, "agent-b", "user-a", 0.99),
        (SkillStatus.SHADOW, "agent-a", "user-b", 0.98),
        (SkillStatus.SHADOW, "agent-a", "user-a", 0.80),
    ):
        skill, payload = _skill(
            status=status,
            agent_id=agent_id,
            user_id=user_id,
            score=score,
        )
        _store_skill(provider, skill, payload)

    hits = box.retrieve_shadow_skills_by_query(
        "refund",
        limit=1,
        min_similarity=0.1,
        agent_id="agent-a",
        user_id="user-a",
    )
    assert len(hits) == 1
    assert hits[0].skill.status == SkillStatus.SHADOW
    assert hits[0].skill.agent_id == "agent-a"
    assert hits[0].skill.user_id == "user-a"
    assert provider.shadow_retrieval_calls[-1]["statuses"] == ["shadow"]


def test_third_party_provider_uses_overfetch_and_filter_fallback():
    shadow, shadow_doc = _skill(score=0.7)

    class LegacyProvider:
        def __init__(self):
            self.limit = None

        def retrieve_by_query(self, query, memory_store_type=None, limit=1):
            self.limit = limit
            active_docs = []
            for index in range(8):
                _active, doc = _skill(
                    status=SkillStatus.ACTIVE,
                    score=1.0 - index / 100,
                )
                doc["score"] = doc["_test_score"]
                active_docs.append(doc)
            shadow_doc["score"] = 0.7
            return [*active_docs, shadow_doc]

    provider = LegacyProvider()
    box = Skillbox(provider, agent_id="agent-a")
    hits = box.retrieve_shadow_skills_by_query(
        "refund",
        limit=1,
        min_similarity=0.1,
        agent_id="agent-a",
        user_id="user-a",
    )
    assert provider.limit == 30
    assert [hit.skill.skill_id for hit in hits] == [shadow.skill_id]


def test_matching_mismatching_and_failed_runs_update_separate_shadow_stats():
    provider = MemoryProviderStub()
    manager = _manager(provider)
    skill, payload = _skill()
    _store_skill(provider, skill, payload)

    cases = (
        ("match-success", "expected-hash", WorkflowOutcome.SUCCESS),
        ("match-failure", "expected-hash", WorkflowOutcome.FAILURE),
        ("other-success", "alternative-hash", WorkflowOutcome.SUCCESS),
        ("other-failure", "another-hash", WorkflowOutcome.FAILURE),
    )
    records = []
    for workflow_id, canonical_hash, outcome in cases:
        workflow, record_id = _workflow(
            provider,
            workflow_id=workflow_id,
            canonical_hash=canonical_hash,
            outcome=outcome,
        )
        manager.record_run_outcome(workflow, record_id)
        records.append((workflow, record_id))

    assert manager.drain_shadow_evaluations(timeout=2)
    loaded = manager.skillbox.get_skill_by_id(skill.skill_id)
    shadow = loaded.stats["shadow"]
    assert shadow["observations"] == 4
    assert shadow["trajectory_matches"] == 2
    assert shadow["trajectory_mismatches"] == 2
    assert shadow["matched_successes"] == 1
    assert shadow["matched_failures"] == 1
    assert len(shadow["recent_evaluations"]) == 2
    assert loaded.stats["activations"] == 0

    for workflow, record_id in records:
        doc = provider.retrieve_by_id(record_id, MemoryType.WORKFLOW_MEMORY)
        assert doc["skills_activated"] == []
        assert len(doc["shadow_evaluations"]) == 1
        evaluation = doc["shadow_evaluations"][0]
        if workflow.canonical_hash != "expected-hash":
            assert evaluation["trajectory_match"] is False
            assert evaluation["matched_trajectory_succeeded"] is None

    # Retrying the same post-store hook reconciles but does not double count.
    workflow, record_id = records[0]
    manager.record_run_outcome(workflow, record_id)
    assert manager.drain_shadow_evaluations(timeout=2)
    loaded = manager.skillbox.get_skill_by_id(skill.skill_id)
    assert loaded.stats["shadow"]["observations"] == 4
    doc = provider.retrieve_by_id(record_id, MemoryType.WORKFLOW_MEMORY)
    assert len(doc["shadow_evaluations"]) == 1


def test_source_pre_skill_and_cross_tenant_workflows_are_excluded():
    provider = MemoryProviderStub()
    manager = _manager(provider)
    skill_created = datetime.now()
    skill, payload = _skill(
        created_at=skill_created,
        source_ids=["source-workflow"],
    )
    _store_skill(provider, skill, payload)

    excluded = (
        _workflow(
            provider,
            workflow_id="source-workflow",
            created_at=skill_created + timedelta(seconds=1),
        ),
        _workflow(
            provider,
            workflow_id="pre-skill",
            created_at=skill_created - timedelta(seconds=1),
        ),
        _workflow(
            provider,
            workflow_id="other-agent",
            created_at=skill_created + timedelta(seconds=1),
            agent_id="agent-b",
        ),
        _workflow(
            provider,
            workflow_id="other-user",
            created_at=skill_created + timedelta(seconds=1),
            user_id="user-b",
        ),
    )
    for workflow, record_id in excluded:
        manager.record_run_outcome(workflow, record_id)

    accepted, accepted_id = _workflow(
        provider,
        workflow_id="fresh-production",
        created_at=skill_created + timedelta(seconds=2),
    )
    manager.record_run_outcome(accepted, accepted_id)
    assert manager.drain_shadow_evaluations(timeout=2)

    for workflow, record_id in excluded:
        assert (
            provider.retrieve_by_id(record_id, MemoryType.WORKFLOW_MEMORY)[
                "shadow_evaluations"
            ]
            == []
        )
    assert (
        len(
            provider.retrieve_by_id(accepted_id, MemoryType.WORKFLOW_MEMORY)[
                "shadow_evaluations"
            ]
        )
        == 1
    )
    assert manager.get_shadow_readiness(skill.skill_id)["observations"] == 1


def test_readiness_is_advisory_and_reports_threshold_reasons():
    provider = MemoryProviderStub()
    manager = _manager(provider)
    skill, payload = _skill()
    payload["stats"]["shadow"] = {
        "observations": 2,
        "trajectory_matches": 2,
        "trajectory_mismatches": 0,
        "matched_successes": 2,
        "matched_failures": 0,
        "last_evaluated_at": None,
        "recent_evaluations": [],
    }
    _store_skill(provider, skill, payload)

    readiness = manager.get_shadow_readiness(skill.skill_id)
    assert readiness == {
        "ready": False,
        "observations": 2,
        "trajectory_match_rate": 1.0,
        "matched_success_rate": 1.0,
        "reasons": ["observations 2 < 3"],
    }
    # Explicit review remains authoritative even before advisory readiness.
    assert manager.activate_skill(skill.skill_id)
    assert manager.skillbox.get_skill_by_id(skill.skill_id).status == SkillStatus.ACTIVE


def test_queue_full_drops_immediately_without_provider_work():
    provider = MemoryProviderStub()
    manager = _manager(provider, shadow_evaluation_queue_size=1)
    replacement = queue.Queue(maxsize=1)
    replacement.put_nowait(object())
    manager._shadow_queue = replacement
    workflow, record_id = _workflow(provider, workflow_id="queue-full")

    started = time.perf_counter()
    enqueued = manager.enqueue_shadow_evaluation(workflow, record_id)
    elapsed = time.perf_counter() - started
    assert enqueued is False
    assert elapsed < 0.05
    assert provider.shadow_retrieval_calls == []


def test_blocked_shadow_retrieval_does_not_delay_post_run_path():
    provider = MemoryProviderStub()
    release = threading.Event()
    started = threading.Event()
    provider.block_shadow_retrieval = release
    provider.shadow_retrieval_started = started
    manager = _manager(provider)
    skill, payload = _skill()
    _store_skill(provider, skill, payload)
    workflow, record_id = _workflow(provider, workflow_id="latency")

    before = time.perf_counter()
    manager.record_run_outcome(workflow, record_id)
    elapsed = time.perf_counter() - before
    assert elapsed < 0.05
    assert started.wait(timeout=1)
    assert not manager.drain_shadow_evaluations(timeout=0.01)
    release.set()
    assert manager.drain_shadow_evaluations(timeout=2)


@pytest.mark.parametrize("failure_mode", ["retrieval", "persistence", "worker"])
def test_background_failures_never_reach_the_user_path(failure_mode, monkeypatch):
    provider = MemoryProviderStub()
    manager = _manager(provider)
    skill, payload = _skill()
    _store_skill(provider, skill, payload)
    if failure_mode == "retrieval":
        provider.raise_on_shadow_retrieval = True
    elif failure_mode == "persistence":
        provider.fail_workflow_updates = True
    else:
        monkeypatch.setattr(
            manager.shadow_evaluator,
            "evaluate",
            lambda _snapshot: (_ for _ in ()).throw(RuntimeError("worker failed")),
        )
    workflow, record_id = _workflow(provider, workflow_id=f"failure-{failure_mode}")

    # The user-facing post-run hook returns normally in every failure mode.
    manager.record_run_outcome(workflow, record_id)
    assert manager.drain_shadow_evaluations(timeout=2)
