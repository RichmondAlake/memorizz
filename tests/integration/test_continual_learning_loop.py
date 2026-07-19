# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""End-to-end continual learning lifecycle on the filesystem provider.

One scripted life of a skill:
seed runs → promotion cycle → ACTIVE skill → retrieval/injection →
guided-run attribution → forced failures → drift demotion → workflows
retrievable again → trajectory re-qualifies → v2 skill.
"""

import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.procedural.skillbox import SkillStatus
from memorizz.long_term.procedural.workflow.workflow import Workflow, WorkflowOutcome
from memorizz.memagent.managers.continual_learning_manager import (
    SKILLS_PROMPT_HEADER,
    ContinualLearningManager,
)
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


def stable_embedding(text: str, dims: int = 16) -> List[float]:
    vector = [0.0] * dims
    for token in re.findall(r"[a-z]+", str(text).lower()):
        digest = int(hashlib.sha1(token.encode()).hexdigest(), 16)
        vector[digest % dims] += 1.0
    norm = sum(component**2 for component in vector) ** 0.5 or 1.0
    return [component / norm for component in vector]


class StableEmbeddingProvider:
    def get_embedding(self, text: str, **kwargs):
        return stable_embedding(text)

    def get_provider_info(self):
        return {"provider": "stable-test", "model": "bow", "dimensions": 16}


DISTILLED_SKILL_MD = """---
name: Refund lookup and issue
description: Use when the user asks to refund a specific completed order.
preconditions:
  - Order identifier is present in the query or retrievable
tools: [lookup_order, issue_refund]
---

# Refund lookup and issue

## When to apply
Apply when the user requests a refund for an identified order.
When NOT to apply: orders that are pending or already refunded.

## Procedure
1. Call `lookup_order` with the order identifier. Expect order_id, amount, status.
2. If status is not completed, stop and inform the user.
3. Call `issue_refund` with order_id and amount.

## Failure modes observed
- `lookup_order` returns empty when the identifier includes whitespace.
"""


class ScriptedLLM:
    """Distillation returns a canned SKILL.md; the judge always says YES."""

    def __init__(self):
        self.distill_calls = 0

    def generate_text(self, prompt, instructions=None):
        if instructions and "YES or NO" in instructions:
            return "YES — the procedure matches the observed runs."
        self.distill_calls += 1
        return DISTILLED_SKILL_MD


@pytest.fixture()
def provider(tmp_path):
    config = FileSystemConfig(
        root_path=Path(tmp_path) / "cl-loop",
        embedding_provider=StableEmbeddingProvider(),
        lazy_vector_indexes=True,
    )
    return FileSystemProvider(config)


@pytest.fixture(autouse=True)
def _patch_embeddings(monkeypatch):
    from memorizz.long_term.procedural.skillbox import skill as skill_mod
    from memorizz.long_term.procedural.skillbox import skillbox as skillbox_mod
    from memorizz.long_term.procedural.workflow import workflow as workflow_mod

    monkeypatch.setattr(skill_mod, "get_embedding", stable_embedding)
    monkeypatch.setattr(skillbox_mod, "get_embedding", stable_embedding)
    monkeypatch.setattr(workflow_mod, "get_embedding", stable_embedding)


def _seed_tools(provider):
    for name in ("lookup_order", "issue_refund"):
        provider.store(
            {"name": name, "docstring": f"{name} tool", "embedding": [0.0]},
            memory_store_type=MemoryType.TOOLBOX,
        )


def _run(
    provider,
    manager,
    query,
    order_id,
    outcome=WorkflowOutcome.SUCCESS,
    created_at=None,
    skills=None,
    fail_step=False,
):
    """Capture one agent run the way MemAgent's funnel does."""
    steps = {
        "Step 1: lookup_order": {
            "arguments": {"order_id": order_id},
            "result": f"order {order_id} completed",
            "timestamp": (created_at or datetime.now()).isoformat(),
            "error": None,
        },
        "Step 2: issue_refund": {
            "arguments": {"order_id": order_id, "amount": 10},
            "result": "refunded" if not fail_step else "error",
            "timestamp": (
                (created_at or datetime.now()) + timedelta(seconds=1)
            ).isoformat(),
            "error": "gateway down" if fail_step else None,
        },
    }
    workflow = Workflow(
        name="Tool Execution for Query",
        description=f"Workflow tracking tool usage for: {query[:100]}",
        agent_id="agent-1",
        user_query=query,
        steps=steps,
        outcome=outcome,
        created_at=created_at,
        skills_activated=list(skills or []),
    )
    record_id = workflow.store_workflow(provider)
    manager.record_run_outcome(workflow, record_id=record_id)
    return workflow


@pytest.mark.integration
def test_full_lifecycle(provider):
    _seed_tools(provider)
    llm = ScriptedLLM()
    manager = ContinualLearningManager(
        provider,
        llm_provider=llm,
        agent_id="agent-1",
        config={
            "min_executions": 5,
            "min_distinct_queries": 2,
            "min_success_rate": 0.8,
            "promotion_every_n_runs": 0,  # manual cycles in this test
            "retrieval_min_similarity": 0.3,
            "min_activations_before_drift_check": 5,
            "drift_window_activations": 10,
            "demotion_success_delta": 0.25,
        },
    )

    # ---- 1. Seed: 5 successes + 1 failure of the same canonical procedure,
    #         varying argument values and query phrasings.
    base = datetime.now() - timedelta(days=1)
    queries = [
        "refund order A1",
        "please refund my order B2",
        "refund order C3 now",
        "I want a refund for order D4",
        "refund order E5",
    ]
    for index, query in enumerate(queries):
        _run(
            provider,
            manager,
            query,
            f"X{index}",
            created_at=base + timedelta(minutes=index),
        )
    _run(
        provider,
        manager,
        "refund order F6",
        "X9",
        outcome=WorkflowOutcome.FAILURE,
        fail_step=True,
        created_at=base + timedelta(minutes=9),
    )

    # ---- 2. Promotion cycle → one ACTIVE skill.
    report = manager.run_promotion_cycle()
    assert len(report.promoted) == 1, (report.rejected, report.skipped)
    active = manager.skillbox.list_skills(statuses=(SkillStatus.ACTIVE,))
    assert len(active) == 1
    skill = active[0]
    assert skill.version == 1
    # Description encodes applicability, not mechanism.
    assert "Use when" in skill.description
    assert "lookup_order" not in skill.description

    # Successful-run docs are stamped (suppression contract).
    success_hash = skill.source_canonical_hash
    workflow_docs = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
    stamped = [d for d in workflow_docs if d.get("promoted_skill_id")]
    assert all(d["promoted_skill_id"] == skill.skill_id for d in stamped)
    assert {d["canonical_hash"] for d in stamped} <= {
        success_hash,
        # the failure run has a different terminal-error signature, so it
        # may or may not share the success hash — stamped rows must all
        # belong to the promoted class either way
    }

    # ---- 3. A NEW run of the covered class is still written (always
    #         write) and auto-stamped at write time (selective retrieve).
    fresh = _run(provider, manager, "refund order Z9", "Z9")
    docs = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
    fresh_doc = next(d for d in docs if d.get("workflow_id") == fresh.workflow_id)
    assert fresh_doc.get("promoted_skill_id") == skill.skill_id

    # ---- 4. Retrieval suppression: promoted trajectories are hidden by
    #         default, visible on opt-out.
    hidden = Workflow.retrieve_workflows_by_query("refund order", provider, limit=5)
    assert all(w.canonical_hash != success_hash for w in hidden)
    visible = Workflow.retrieve_workflows_by_query(
        "refund order", provider, limit=5, exclude_promoted=False
    )
    assert any(w.canonical_hash == success_hash for w in visible)

    # ---- 5. Skill retrieval + prompt injection framing.
    scored = manager.retrieve_skills_for_query("please refund order Q7")
    assert scored and scored[0].skill.skill_id == skill.skill_id
    section = manager.format_skills_prompt_section(scored)
    assert "strong priors, not mandates" in section
    assert "verify its preconditions" in section
    assert skill.content in section
    assert SKILLS_PROMPT_HEADER in section

    # ---- 6. Guided-run attribution: runs with the skill in context update
    #         its stats; forced failures trip drift demotion.
    for index in range(6):
        _run(
            provider,
            manager,
            f"refund order FAIL{index}",
            f"F{index}",
            outcome=WorkflowOutcome.FAILURE,
            fail_step=True,
            skills=[skill.skill_id],
        )
    demoted = manager.skillbox.get_skill_by_id(skill.skill_id)
    assert demoted.status == SkillStatus.DEMOTED
    assert "drift" in demoted.demotion_reason
    assert demoted.stats["activations"] >= 5

    # ---- 7. Demotion releases the workflows back to retrieval.
    docs = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
    assert all(not d.get("promoted_skill_id") for d in docs)
    visible_again = Workflow.retrieve_workflows_by_query(
        "refund order", provider, limit=5
    )
    assert any(w.canonical_hash == success_hash for w in visible_again)

    # And the demoted skill no longer retrieves.
    assert manager.retrieve_skills_for_query("please refund order Q7") == []

    # ---- 8. Re-qualification on POST-demotion evidence only → v2 skill.
    rebase = datetime.now() + timedelta(seconds=1)
    for index in range(5):
        _run(
            provider,
            manager,
            f"refund order NEW{index} please",
            f"N{index}",
            created_at=rebase + timedelta(minutes=index),
        )
    report2 = manager.run_promotion_cycle()
    assert len(report2.promoted) == 1, (report2.rejected, report2.skipped)
    v2 = manager.skillbox.get_skill_by_id(report2.promoted[0])
    assert v2.version == 2
    assert v2.status == SkillStatus.ACTIVE


@pytest.mark.integration
def test_shadow_mode_never_stamps_or_injects(provider):
    _seed_tools(provider)
    manager = ContinualLearningManager(
        provider,
        llm_provider=ScriptedLLM(),
        agent_id="agent-1",
        config={
            "min_executions": 5,
            "min_distinct_queries": 2,
            "promotion_every_n_runs": 0,
            "require_shadow": True,
            "skill_injection_role": "developer",
            "retrieval_min_similarity": 0.3,
            "shadow_evaluation_enabled": True,
            "shadow_evaluation_min_similarity": 0.3,
            "shadow_readiness_min_observations": 1,
            "shadow_readiness_min_trajectory_match_rate": 1.0,
            "shadow_readiness_min_matched_success_rate": 1.0,
        },
    )
    base = datetime.now() - timedelta(hours=2)
    for index in range(5):
        _run(
            provider,
            manager,
            f"refund order S{index} variant {index}",
            f"S{index}",
            created_at=base + timedelta(minutes=index),
        )

    report = manager.run_promotion_cycle()
    assert len(report.promoted) == 1
    shadow = manager.skillbox.get_skill_by_id(report.promoted[0])
    assert shadow.status == SkillStatus.SHADOW
    assert shadow.injection_role.value == "developer"

    # Shadow skills never suppress workflow retrieval…
    docs = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
    assert all(not d.get("promoted_skill_id") for d in docs)
    assert all(not d.get("shadow_evaluations") for d in docs)
    # …and never inject.
    assert manager.retrieve_skills_for_query("refund order S1 variant") == []

    # A fresh production run is evaluated only after it is stored. Source
    # workflows remain untouched, and passive evidence is not attribution.
    fresh = _run(provider, manager, "refund order NEW shadow variant", "NEW")
    assert manager.drain_shadow_evaluations(timeout=2)
    docs = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
    source_ids = set(shadow.source_workflow_ids)
    source_docs = [doc for doc in docs if str(doc.get("workflow_id")) in source_ids]
    assert source_docs and all(not doc.get("shadow_evaluations") for doc in source_docs)
    fresh_doc = next(doc for doc in docs if doc.get("workflow_id") == fresh.workflow_id)
    assert fresh_doc["skills_activated"] == []
    assert len(fresh_doc["shadow_evaluations"]) == 1
    assert fresh_doc["shadow_evaluations"][0]["skill_id"] == shadow.skill_id
    readiness = manager.get_shadow_readiness(shadow.skill_id)
    assert readiness["ready"] is True
    assert readiness["observations"] == 1

    # Manual activation flips both.
    assert manager.activate_skill(shadow.skill_id)
    assert manager.skillbox.get_skill_by_id(shadow.skill_id).injection_role.value == (
        "developer"
    )
    docs = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
    assert any(d.get("promoted_skill_id") == shadow.skill_id for d in docs)
    assert manager.retrieve_skills_for_query("refund order S1 variant")


@pytest.mark.integration
def test_insufficient_distillation_is_rejected_not_stored(provider):
    _seed_tools(provider)

    class InsufficientLLM:
        def generate_text(self, prompt, instructions=None):
            return "INSUFFICIENT"

    manager = ContinualLearningManager(
        provider,
        llm_provider=InsufficientLLM(),
        agent_id="agent-1",
        config={
            "min_executions": 5,
            "min_distinct_queries": 2,
            "promotion_every_n_runs": 0,
        },
    )
    base = datetime.now() - timedelta(hours=1)
    for index in range(5):
        _run(
            provider,
            manager,
            f"do the thing {index}",
            f"T{index}",
            created_at=base + timedelta(minutes=index),
        )

    report = manager.run_promotion_cycle()
    assert report.promoted == []
    assert report.rejected and "INSUFFICIENT" in report.rejected[0][1][0]
    assert manager.skillbox.list_skills() == []
