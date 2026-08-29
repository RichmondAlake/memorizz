"""Learning control-plane contracts across the filesystem provider and agent SDK."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from memorizz import (
    FileSystemConfig,
    FileSystemProvider,
    LearningControlPlane,
    OutcomeEvidence,
)
from memorizz.enums import MemoryType
from memorizz.learning import LearningEventType, LearningRecordConflictError
from memorizz.memagent import MemAgent
from memorizz.memagent.builders import MemAgentBuilder
from tests.mocks.mock_providers import MockLLMProvider


class _Embeddings:
    def get_embedding(self, text):
        value = str(text).lower()
        return [
            float("release" in value),
            float("tuesday" in value),
            float("london" in value),
        ]

    def get_provider_info(self):
        return "unit-test"


@pytest.fixture()
def provider(tmp_path):
    return FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "learning-control-plane",
            embedding_provider=_Embeddings(),
            lazy_vector_indexes=True,
        )
    )


def _plane(provider):
    return LearningControlPlane(
        provider,
        agent_id="agent-1",
        config={
            "enabled": True,
            "compile_async": False,
            "compile_every_n_events": 0,
            "evidence_sources": ["knowledge_base", "learning_artifacts"],
            "evidence_token_budget": 80,
        },
    )


@pytest.mark.unit
def test_events_are_idempotent_immutable_and_incrementally_compiled(provider):
    plane = _plane(provider)
    scope = {
        "memory_id": "memory-1",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "trace_id": "trace-1",
    }
    first = plane.begin_run("Ship the release", scope=scope)
    again = plane.begin_run("Ship the release", scope=scope)
    assert first.event_id == again.event_id

    with pytest.raises(LearningRecordConflictError):
        plane.begin_run("A different request", scope=scope)

    plane.record_tool(
        tool_name="deploy",
        arguments={"region": "London"},
        result={"ok": True},
        success=True,
        outcome={
            "status": "fallback",
            "ok": True,
            "fallback_used": True,
            "reason_code": "primary_timeout",
        },
        scope=scope,
    )
    plane.complete_run("done", status="success", tool_call_count=1, scope=scope)
    first_compile = plane.compile(
        memory_id="memory-1", user_id="user-1", thread_id="thread-1"
    )
    second_compile = plane.compile(
        memory_id="memory-1", user_id="user-1", thread_id="thread-1"
    )

    assert first_compile.compiled_events == 3
    assert first_compile.artifacts_written == 1
    assert second_compile.compiled_events == 0
    artifacts = plane.store.list_artifacts(
        memory_id="memory-1", user_id="user-1", thread_id="thread-1"
    )
    assert len(artifacts) == 1
    assert "Tool deploy completed via fallback" in artifacts[0]["content"]


@pytest.mark.unit
def test_evidence_pack_is_scoped_deduplicated_and_token_bounded(provider):
    embedder = provider.config.embedding_provider
    for user, suffix in (("user-1", "Tuesday"), ("user-2", "Friday")):
        content = f"The London release train runs on {suffix}."
        provider.store(
            {
                "_id": f"kb-{user}",
                "memory_id": "memory-1",
                "user_id": user,
                "agent_id": "agent-1",
                "content": content,
                "embedding": embedder.get_embedding(content),
            },
            MemoryType.KNOWLEDGE_BASE,
        )
    plane = _plane(provider)
    pack = plane.retrieve_evidence(
        "When is the London release?",
        memory_id="memory-1",
        user_id="user-1",
        thread_id="thread-1",
    )
    assert pack.items
    assert pack.tokens_used <= pack.token_budget
    assert all(item.metadata.get("user_id") == "user-1" for item in pack.items)
    assert "Tuesday" in pack.render()
    assert "Friday" not in pack.render()

    duplicate = plane.retrieve_evidence(
        "When is the London release?",
        memory_id="memory-1",
        user_id="user-1",
        thread_id="thread-1",
        history_texts=["The London release train runs on Tuesday."],
    )
    assert not duplicate.items


@pytest.mark.unit
def test_entity_evidence_uses_exact_fallback_when_vector_search_is_missing(
    provider, monkeypatch
):
    provider.store(
        {
            "_id": "entity-user-1",
            "entity_id": "entity-user-1",
            "name": "user",
            "entity_type": "person",
            "attributes": [{"name": "role", "value": "release engineer"}],
            "relations": [],
            "metadata": {"is_self": True},
            "memory_id": "shared-memory",
            "user_id": "user-1",
        },
        MemoryType.ENTITY_MEMORY,
    )
    monkeypatch.setattr(
        provider,
        "get_vector_search_status",
        lambda _memory_type: {
            "available": False,
            "queryable": False,
            "reason": "vector_index_missing",
        },
        raising=False,
    )
    plane = LearningControlPlane(
        provider,
        agent_id="agent-1",
        config={
            "enabled": True,
            "compile_async": False,
            "compile_every_n_events": 0,
            "evidence_sources": ["entity_memory"],
        },
    )

    pack = plane.retrieve_evidence(
        "What do you know about my user profile?",
        memory_id="shared-memory",
        user_id="user-1",
        thread_id="thread-1",
    )

    assert len(pack.items) == 1
    assert "release engineer" in pack.items[0].content
    assert pack.items[0].metadata["retrieval_mode"] == "exact_fallback"
    assert pack.items[0].metadata["retrieval_degraded"] is True
    assert pack.items[0].metadata["retrieval_reason"] == "vector_index_missing"
    assert pack.warnings == ("entity_memory retrieval degraded: vector_index_missing",)


@pytest.mark.unit
def test_verified_outcome_and_reversible_forgetting(provider):
    plane = _plane(provider)
    scope = {
        "memory_id": "memory-1",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "trace_id": "trace-1",
    }
    outcome = plane.record_outcome(
        OutcomeEvidence.from_value(
            "success", verified=True, source="integration_test", score=1.0
        ),
        scope=scope,
        external_id="outcome-1",
    )
    assert outcome["record_type"] == "observability_outcome"
    assert outcome["outcome_evidence"]["learning_authoritative"] is True

    common = {
        "agent_id": "agent-1",
        "memory_id": "memory-1",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "artifact_kind": "memory_fact",
        "content": "Duplicate low-utility projection",
        "source_hash": "same-source",
        "verified": False,
        "utility": 0.1,
    }
    plane.store.put_artifact({**common, "record_id": "artifact-old"})
    plane.store.put_artifact({**common, "record_id": "artifact-new"})
    plane.store.put_artifact(
        {
            **common,
            "record_id": "artifact-verified",
            "source_hash": "verified-source",
            "verified": True,
        }
    )

    plan = plane.plan_forgetting(
        memory_id="memory-1", user_id="user-1", thread_id="thread-1"
    )
    assert [item.target_id for item in plan.candidates] == ["artifact-old"]
    stored_plan = plane.get_forgetting_plan(
        plan.plan_id,
        memory_id="memory-1",
        user_id="user-1",
        thread_id="thread-1",
    )
    applied = plane.apply_forgetting(
        stored_plan, approved_by="operator@example.test", scope=scope
    )
    assert applied.tombstoned == 1
    assert "artifact-old" in plane.store.tombstoned_ids(
        memory_id="memory-1", user_id="user-1"
    )
    with pytest.raises(ValueError, match="already been applied"):
        plane.apply_forgetting(
            stored_plan, approved_by="operator@example.test", scope=scope
        )


@pytest.mark.unit
def test_agent_can_reload_and_apply_a_durable_forgetting_plan(provider):
    agent = MemAgent(
        model=MockLLMProvider(),
        memory_provider=provider,
        agent_id="forgetting-agent",
        learning_control_plane={"enabled": True, "compile_async": False},
    )
    agent.learning_control_plane.store.put_artifact(
        {
            "record_id": "old-artifact",
            "artifact_kind": "memory_fact",
            "agent_id": agent.agent_id,
            "memory_id": "memory-1",
            "user_id": "user-1",
            "thread_id": "thread-1",
            "content": "old low utility fact",
            "source_hash": "duplicate-hash",
            "utility": 0.0,
        }
    )
    agent.learning_control_plane.store.put_artifact(
        {
            "record_id": "new-artifact",
            "artifact_kind": "memory_fact",
            "agent_id": agent.agent_id,
            "memory_id": "memory-1",
            "user_id": "user-1",
            "thread_id": "thread-1",
            "content": "newer duplicate projection",
            "source_hash": "duplicate-hash",
            "utility": 0.5,
        }
    )
    plan = agent.plan_forgetting(
        memory_id="memory-1", user_id="user-1", thread_id="thread-1"
    )
    restored = agent.get_forgetting_plan(
        plan.plan_id,
        memory_id="memory-1",
        user_id="user-1",
        thread_id="thread-1",
    )
    applied = agent.apply_forgetting(
        restored,
        approved_by="operator@example.test",
        memory_id="memory-1",
        user_id="user-1",
        thread_id="thread-1",
    )
    assert applied.tombstoned == 1
    forgotten = agent.learning_control_plane.store.list_events(
        memory_id="memory-1", user_id="user-1", thread_id="thread-1"
    )[-1]
    assert forgotten.event_type == LearningEventType.MEMORY_FORGOTTEN


@pytest.mark.unit
def test_builder_agent_context_and_persistence_round_trip(provider):
    content = "The London release train runs on Tuesday."
    agent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "memorizz:default:Learning agent"))
    provider.store(
        {
            "_id": "kb-release",
            "memory_id": "memory-1",
            "user_id": "user-1",
            "agent_id": agent_id,
            "content": content,
            "embedding": provider.config.embedding_provider.get_embedding(content),
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    model = MockLLMProvider(["It runs Tuesday."])
    agent = (
        MemAgentBuilder()
        .with_name("Learning agent")
        .with_model(model)
        .with_memory_provider(provider)
        .with_memory_ids("memory-1")
        .with_learning_control_plane(
            evidence_sources=["knowledge_base"], evidence_token_budget=120
        )
        .build()
    )
    answer = agent.run(
        "When is the London release?",
        memory_id="memory-1",
        user_id="user-1",
    )
    assert answer == "It runs Tuesday."
    assert "MEMORIZZ EVIDENCE PACK" in str(model.last_messages)
    agent.save()

    loaded = MemAgent.load(agent_id, memory_provider=provider, model=MockLLMProvider())
    assert loaded.learning_control_plane_enabled is True
    assert loaded.learning_control_plane_config.evidence_token_budget == 120
    assert loaded.learning_control_plane is not None
    assert (
        loaded.learning_report(memory_id="memory-1", user_id="user-1")["enabled"]
        is True
    )
    assert any(
        event.event_type == LearningEventType.RUN_COMPLETED
        for event in loaded.learning_control_plane.store.list_events(
            memory_id="memory-1", user_id="user-1"
        )
    )
    agent.close(close_memory_provider=False)
    loaded.close(close_memory_provider=False)
