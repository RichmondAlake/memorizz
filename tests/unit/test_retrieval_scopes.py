"""RetrievalPolicy scopes: "thread" keeps the knowledge base, sealed threads,
a per-call policy, the evidence path honouring the policy, and recall counts
split by source."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from memorizz.enums import MemoryType
from memorizz.learning.evidence import EvidencePlanner
from memorizz.learning.models import LearningControlPlaneConfig
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.retrieval import RetrievalPolicy, thread_entity_memory_id

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-scopes-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _offline_embeddings(monkeypatch):
    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda text: list(VECTOR))
    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        lambda text: list(VECTOR),
    )
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")


class _Embedder:
    def get_embedding(self, text: str) -> List[float]:
        return list(VECTOR)


class _Model:
    model = "stub"

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        return "ok"

    def get_config(self):
        return {"provider": "openai", "model": "stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


def _agent(work_dir: Path, **kwargs) -> MemAgent:
    kwargs.setdefault("model", _Model())
    kwargs.setdefault(
        "memory_provider",
        FileSystemProvider(
            FileSystemConfig(
                root_path=work_dir / "memory",
                embedding_provider=_Embedder(),
                use_faiss=False,
                lazy_vector_indexes=True,
            )
        ),
    )
    kwargs.setdefault("memory_ids", ["memory-1"])
    kwargs.setdefault("auto_register", False)
    kwargs.setdefault("agent_id", "agent-1")
    return MemAgent(**kwargs)


# -- the policy object -------------------------------------------------------


def test_the_thread_shortcut_narrows_conversation_recall_and_keeps_the_knowledge_base():
    policy = RetrievalPolicy.from_value("thread")
    assert policy.conversation_scope == "thread"
    assert policy.knowledge_base_scope == "memory"
    assert policy.entity_scope == "memory"
    assert not policy.is_sealed


def test_sealed_keeps_conversations_and_entity_facts_in_the_thread():
    sealed = RetrievalPolicy.sealed()
    assert sealed == RetrievalPolicy.from_value("sealed")
    assert (sealed.conversation_scope, sealed.entity_scope) == ("thread", "thread")
    assert sealed.knowledge_base_scope == "memory" and sealed.is_sealed
    assert sealed.to_dict()["entity_scope"] == "thread"
    assert RetrievalPolicy.from_value(sealed.to_dict()) == sealed
    # The default stays out of saved records, so older records compare equal.
    assert "entity_scope" not in RetrievalPolicy().to_dict()
    with pytest.raises(ValueError):
        RetrievalPolicy(entity_scope="everywhere")


def test_a_threads_entity_scope_fits_every_provider():
    assert thread_entity_memory_id("m", "t") == "m:thread:t"
    long_scope = thread_entity_memory_id("m" * 240, "t" * 40)
    assert len(long_scope) <= 255 and ":thread:" in long_scope
    assert long_scope == thread_entity_memory_id("m" * 240, "t" * 40)


# -- per-call policy ---------------------------------------------------------


def test_a_policy_passed_to_run_applies_to_that_call_only(work_dir):
    agent = _agent(work_dir)
    seen: List[RetrievalPolicy] = []
    build = agent._build_context

    def spy(*args, **kwargs):
        seen.append(agent._active_retrieval_policy())
        return build(*args, **kwargs)

    agent._build_context = spy
    agent.run("hi", memory_id="memory-1", thread_id="t1", retrieval_policy="disabled")
    agent.run("hi again", memory_id="memory-1", thread_id="t1")
    assert seen == [RetrievalPolicy.disabled(), RetrievalPolicy()]
    assert agent.retrieval_policy == RetrievalPolicy()
    agent.close()


def test_concurrent_calls_keep_their_own_policy(work_dir):
    agent = _agent(work_dir)
    barrier = threading.Barrier(2, timeout=10)
    seen: Dict[str, RetrievalPolicy] = {}
    build = agent._build_context

    def spy(query, *args, **kwargs):
        barrier.wait()
        seen[query] = agent._active_retrieval_policy()
        return build(query, *args, **kwargs)

    agent._build_context = spy
    workers = [
        threading.Thread(
            target=agent.run,
            args=(name,),
            kwargs={
                "memory_id": "memory-1",
                "thread_id": name,
                "retrieval_policy": policy,
            },
        )
        for name, policy in (("sealed", "sealed"), ("open", None))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=20)
    assert seen["sealed"].is_sealed
    assert seen["open"] == RetrievalPolicy()
    agent.close()


# -- sealed threads ----------------------------------------------------------


def _upsert(agent: MemAgent, thread_id: str, policy: Any, value: str) -> None:
    agent._current_memory_id = "memory-1"
    agent._current_thread_id = thread_id
    agent._retrieval_policy_override = RetrievalPolicy.from_value(policy)
    agent.tool_manager.get_tool_callable("entity_memory_upsert")(
        name="Ada", entity_type="person", attributes=[{"name": "city", "value": value}]
    )
    agent._retrieval_policy_override = None


def _lookup(agent: MemAgent, thread_id: str, policy: Any) -> List[str]:
    agent._current_memory_id = "memory-1"
    agent._current_thread_id = thread_id
    agent._retrieval_policy_override = RetrievalPolicy.from_value(policy)
    try:
        found = agent.tool_manager.get_tool_callable("entity_memory_lookup")(name="Ada")
    finally:
        agent._retrieval_policy_override = None
    return [
        value for entity in found["matches"] for value in entity["attributes"].values()
    ]


def test_entity_facts_written_in_a_sealed_thread_stay_in_that_thread(work_dir):
    agent = _agent(work_dir, application_mode="assistant")
    _upsert(agent, "t1", "sealed", "Lisbon")
    _upsert(agent, "t0", None, "Lagos")
    assert _lookup(agent, "t1", "sealed") == ["Lisbon"]
    assert _lookup(agent, "t2", "sealed") == []
    assert _lookup(agent, "t2", None) == ["Lagos"]
    agent.close()


def test_a_sealed_thread_reuses_only_its_own_cached_answers(work_dir):
    agent = _agent(work_dir)
    agent._current_thread_id = "t1"
    open_fingerprints = agent._semantic_cache_metadata()["fingerprints"]
    agent._retrieval_policy_override = RetrievalPolicy.sealed()
    sealed_t1 = agent._semantic_cache_metadata()["fingerprints"]
    agent._current_thread_id = "t2"
    sealed_t2 = agent._semantic_cache_metadata()["fingerprints"]
    assert "sealed_thread" not in open_fingerprints
    assert sealed_t1 != sealed_t2 != open_fingerprints
    agent.close()


def test_a_sealed_thread_cannot_change_the_shared_persona(work_dir):
    agent = _agent(work_dir, persona={"name": "Rizz", "role": "assistant"})
    update = agent.tool_manager.get_tool_callable("update_persona")
    if update is None:
        pytest.skip("persona tools are not registered for this agent")
    agent._retrieval_policy_override = RetrievalPolicy.sealed()
    result = update(updates={"role": "pirate"}, reason="the user asked")
    assert result["updated"] is False and "sealed" in result["reason"]
    agent.close()


# -- the evidence path -------------------------------------------------------


class _Provider:
    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self.rows = {
            MemoryType.CONVERSATION_MEMORY: [
                {
                    "_id": "c1",
                    "thread_id": "here",
                    "role": "user",
                    "content": "here row",
                },
                {
                    "_id": "c2",
                    "thread_id": "there",
                    "role": "user",
                    "content": "there row",
                },
            ],
            MemoryType.KNOWLEDGE_BASE: [
                {
                    "_id": "k1",
                    "memory_id": "m",
                    "namespace": "docs",
                    "content": "kb row",
                },
                {
                    "_id": "k2",
                    "memory_id": "m",
                    "namespace": "other",
                    "content": "kb other",
                },
            ],
            MemoryType.SUMMARIES: [
                {
                    "_id": "s1",
                    "memory_id": "m",
                    "thread_id": "there",
                    "content": "sum there",
                },
            ],
        }

    def retrieve_by_query(self, query, memory_store_type=None, **kwargs):
        self.calls.append((memory_store_type, dict(kwargs)))
        rows = self.rows.get(memory_store_type, [])
        if kwargs.get("thread_id") is not None:
            rows = [row for row in rows if row.get("thread_id") == kwargs["thread_id"]]
        if kwargs.get("namespace") is not None:
            rows = [row for row in rows if row.get("namespace") == kwargs["namespace"]]
        return [dict(row, score=0.9) for row in rows]


def _pack_ids(policy: Optional[RetrievalPolicy]) -> set:
    planner = EvidencePlanner(
        _Provider(),
        agent_id="a",
        config=LearningControlPlaneConfig(
            enabled=True,
            evidence_sources=("conversation_memory", "knowledge_base", "summaries"),
            evidence_max_items=10,
            evidence_max_per_source=10,
        ),
    )
    pack = planner.build(
        "row",
        memory_id="m",
        thread_id="here",
        knowledge_base_ids=[],
        retrieval_policy=policy,
    )
    return {item.source_id for item in pack.items}


def test_without_a_policy_evidence_keeps_its_current_thread_default():
    assert _pack_ids(None) == {"c1", "k1", "k2", "s1"}


def test_evidence_follows_the_agents_conversation_scope():
    assert _pack_ids(RetrievalPolicy()) == {"c1", "c2", "k1", "k2", "s1"}
    # Thread scope also keeps other threads' summaries out.
    assert _pack_ids(RetrievalPolicy(conversation_scope="thread")) == {"c1", "k1", "k2"}
    assert _pack_ids(RetrievalPolicy(conversation_scope="disabled")) == {
        "k1",
        "k2",
        "s1",
    }


def test_evidence_follows_the_knowledge_base_scope():
    assert _pack_ids(RetrievalPolicy.disabled()) == {"s1"}
    namespaced = RetrievalPolicy(
        conversation_scope="disabled",
        knowledge_base_scope="namespace",
        knowledge_base_namespaces=("docs",),
    )
    assert _pack_ids(namespaced) == {"k1", "s1"}


# -- recall counts -----------------------------------------------------------


def test_recall_counts_say_which_source_each_memory_came_from(work_dir):
    agent = _agent(work_dir)
    events: List[Dict[str, Any]] = []
    agent.set_stream_event_callback(lambda event: events.append(dict(event)))
    agent._current_thread_id = "now"

    def retrieve(**kwargs):
        if kwargs.get("memory_type") == MemoryType.KNOWLEDGE_BASE:
            return [
                {
                    "_id": "k",
                    "content": "the launch is in May",
                    "embedding": [0.0, 1.0, 0.0],
                    "score": 0.9,
                }
            ]
        if kwargs.get("memory_type") == MemoryType.CONVERSATION_MEMORY:
            return [
                {
                    "_id": "c",
                    "role": "user",
                    "content": "I prefer short answers",
                    "embedding": [0.0, 0.0, 1.0],
                    "score": 0.9,
                }
            ]
        return []

    agent.memory_manager.retrieve_relevant_memories = retrieve
    agent.active_memory_types = [
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.KNOWLEDGE_BASE,
    ]
    agent.run("when is the launch?", memory_id="memory-1", thread_id="now")
    supplied = [
        json.loads(event["content"])
        for event in events
        if event.get("trace_kind") == "memory_context"
    ]
    payload = [item for item in supplied if item.get("stage") == "supplied"][-1]
    assert payload["semantic_source_counts"] == {"knowledge_base": 1, "conversation": 1}
    assert payload["source_counts"]["semantic_memories"] == 2
    agent.close()


def test_describe_memory_names_types_tools_and_policy(work_dir):
    agent = _agent(work_dir, application_mode="assistant", retrieval_policy="sealed")
    overview = agent.describe_memory()
    assert "entity_memory" in overview["memory_types"]
    assert overview["entity_memory"] is True
    assert "entity_memory_upsert" in overview["memory_tools"]
    assert overview["retrieval_policy"]["entity_scope"] == "thread"
    assert overview["background_model"] is None
    agent.close()


# -- a sealed thread's conversation never reaches other threads ---------------


def test_rows_a_sealed_turn_writes_are_recalled_only_in_that_thread(work_dir):
    agent = _agent(work_dir)
    agent._current_memory_id = "memory-1"
    agent._retrieval_policy_override = RetrievalPolicy.sealed()
    agent._record_interaction(
        "my secret project is Falcon", "noted", "memory-1", "sealed"
    )
    agent._retrieval_policy_override = None
    agent._record_interaction("I like tea", "noted", "memory-1", "open")
    rows = agent.memory_provider.list_all(MemoryType.CONVERSATION_MEMORY)
    states = {(row["thread_id"], row.get("retention_state")) for row in rows}
    assert ("sealed", "sealed") in states and ("open", "sealed") not in states

    def retrieve(**kwargs):
        if kwargs.get("memory_type") == MemoryType.CONVERSATION_MEMORY:
            return [
                # Distinct vectors, so near-duplicate removal keeps every row.
                dict(
                    row,
                    score=0.9,
                    embedding=[float(i == index) for i in range(len(rows))],
                )
                for index, row in enumerate(rows)
            ]
        return []

    agent.memory_manager.retrieve_relevant_memories = retrieve
    agent.active_memory_types = [MemoryType.CONVERSATION_MEMORY]
    agent._current_thread_id = "elsewhere"
    context = agent._build_context("what is my secret project?", "memory-1")
    assert "Falcon" not in json.dumps(context, default=str)
    assert "I like tea" in json.dumps(context.get("retrieved_memories"), default=str)
    # In the sealed thread itself it is part of the history (so not recalled twice).
    agent._current_thread_id = "sealed"
    context = agent._build_context("what is my secret project?", "memory-1")
    assert "Falcon" in json.dumps(context.get("conversation_history"), default=str)
    agent.close()


def test_evidence_never_includes_another_threads_sealed_rows():
    provider = _Provider()
    provider.rows[MemoryType.CONVERSATION_MEMORY].append(
        {
            "_id": "c3",
            "thread_id": "there",
            "role": "user",
            "content": "sealed row",
            "retention_state": "sealed",
        }
    )
    planner = EvidencePlanner(
        provider,
        agent_id="a",
        config=LearningControlPlaneConfig(
            enabled=True,
            evidence_sources=("conversation_memory",),
            evidence_max_items=10,
            evidence_max_per_source=10,
        ),
    )
    pack = planner.build(
        "row", memory_id="m", thread_id="here", retrieval_policy=RetrievalPolicy()
    )
    assert {item.source_id for item in pack.items} == {"c1", "c2"}
    pack = planner.build(
        "row", memory_id="m", thread_id="there", retrieval_policy=RetrievalPolicy()
    )
    assert {item.source_id for item in pack.items} == {"c1", "c2", "c3"}


def test_unsuppressing_a_sealed_row_keeps_it_sealed(work_dir):
    from memorizz.learning.forgetting import ForgettingMechanism
    from memorizz.learning.models import (
        ForgettingAction,
        ForgettingCandidate,
        ForgettingReport,
    )
    from memorizz.learning.retention import RetentionPlanner

    agent = _agent(work_dir)
    agent._retrieval_policy_override = RetrievalPolicy.sealed()
    agent._record_interaction("secret", "noted", "memory-1", "sealed")
    provider = agent.memory_provider
    row = provider.list_all(MemoryType.CONVERSATION_MEMORY)[0]
    candidates = (
        ForgettingCandidate(
            target_id=row["_id"],
            target_type=MemoryType.CONVERSATION_MEMORY.value,
            action=ForgettingAction.TOMBSTONE,
            reason="test",
            utility=0.0,
            age_days=99,
        ),
    )
    plan = ForgettingReport(
        plan_id=ForgettingMechanism._plan_id(candidates),
        dry_run=True,
        candidates=candidates,
    )
    planner = RetentionPlanner(provider, agent_id="agent-1")
    planner.apply(plan, approved_by="operator-7")
    assert (
        provider.retrieve_by_id(row["_id"], MemoryType.CONVERSATION_MEMORY)[
            "retention_state"
        ]
        == "suppressed"
    )
    planner.unsuppress(
        row["_id"], MemoryType.CONVERSATION_MEMORY, approved_by="operator-7"
    )
    assert (
        provider.retrieve_by_id(row["_id"], MemoryType.CONVERSATION_MEMORY)[
            "retention_state"
        ]
        == "sealed"
    )
    agent.close()


def test_a_scope_string_for_one_call_keeps_the_agents_other_settings(work_dir):
    agent = _agent(
        work_dir,
        retrieval_policy={"scoring": {"alpha_recency": 0.5}, "max_items": 2},
    )
    sealed = agent._call_policy("sealed")
    assert sealed.is_sealed and sealed.scoring == {"alpha_recency": 0.5}
    assert sealed.max_items == 2
    assert agent._call_policy("thread").entity_scope == "memory"
    # A policy object replaces the agent's for the call.
    assert agent._call_policy(RetrievalPolicy.disabled()) == RetrievalPolicy.disabled()
    agent.close()
