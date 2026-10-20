"""Memory consolidation: plan, apply and undo, on the filesystem provider."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from memorizz.enums import MemoryType
from memorizz.learning.consolidation import (
    ADD_FACT,
    MERGE_ENTITIES,
    RETIRE_ATTRIBUTE,
    SUPPRESS_DUPLICATE,
    ConsolidationConfig,
    ConsolidationPlan,
    MemoryConsolidator,
)
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.retrieval import RetrievalPolicy

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-consolidation-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda text: list(VECTOR))
    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        lambda text: list(VECTOR),
    )
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")


class _Embedder:
    def get_embedding(self, text: str) -> List[float]:
        return list(VECTOR)


def _provider(work_dir: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=work_dir / "memory",
            embedding_provider=_Embedder(),
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )


class _Model:
    """Answers the review and extraction prompts with fixed JSON."""

    model = "background"

    def __init__(self, same_fact=None, same_entity=None, facts=None):
        self.review = {"same_fact": same_fact or [], "same_entity": same_entity or []}
        self.facts = facts or []
        self.prompts: List[str] = []

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        self.prompts.append(prompt)
        if "same_fact" in prompt:
            return "```json\n" + json.dumps(self.review) + "\n```"
        if "durable facts" in prompt:
            return json.dumps({"facts": self.facts})
        return "ok"

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def get_config(self):
        return {"provider": "openai", "model": "background"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


def _entity(provider, entity_id: str, name: str, **attributes) -> None:
    provider.store(
        data={
            "_id": entity_id,
            "entity_id": entity_id,
            "name": name,
            "entity_type": "person",
            "attributes": [
                {"name": key, "value": value, "updated_at": "2026-01-01T00:00:00+00:00"}
                for key, value in attributes.items()
            ],
            "relations": [],
            "metadata": {},
            "memory_id": "memory-1",
            "user_id": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        memory_store_type=MemoryType.ENTITY_MEMORY,
    )


def _summary(provider, summary_id: str, text: str, created_at: float) -> None:
    provider.store(
        data={
            "_id": summary_id,
            "memory_id": "memory-1",
            "agent_id": "agent-1",
            "thread_id": "t1",
            "user_id": None,
            "content": text,
            "created_at": created_at,
        },
        memory_store_type=MemoryType.SUMMARIES,
    )


def _facts(provider, name: str) -> Dict[str, Any]:
    record = EntityMemory(provider).get_entity_by_name(name, memory_id="memory-1")
    return {
        item["name"]: item["value"] for item in (record or {}).get("attributes") or []
    }


def test_duplicate_entities_and_summaries_are_merged_then_restored(work_dir):
    provider = _provider(work_dir)
    _entity(provider, "ada-1", "Ada", role="CTO")
    _entity(provider, "ada-2", "ada", city="Lagos")
    _summary(provider, "s-old", "The user likes tea.", time.time() - 100)
    _summary(provider, "s-new", "the user  likes TEA.", time.time())
    consolidator = MemoryConsolidator(provider, agent_id="agent-1")

    plan = consolidator.plan(memory_id="memory-1")
    assert plan.counts[MERGE_ENTITIES] == 1
    suppress = [op for op in plan.ops if op.kind == SUPPRESS_DUPLICATE]
    assert [(op.target_id, op.details["duplicate_of"]) for op in suppress] == [
        ("s-new", "s-old")
    ]
    assert "no model" in " ".join(plan.warnings)

    result = consolidator.apply(plan, approved_by="operator-7")
    assert result.errors == ()
    assert (
        result.applied[MERGE_ENTITIES] == 1 and result.applied[SUPPRESS_DUPLICATE] == 1
    )
    entities = EntityMemory(provider).list_entities(memory_id="memory-1")
    assert len(entities) == 1
    assert {a["name"] for a in entities[0]["attributes"]} == {"role", "city"}
    assert (
        provider.retrieve_by_id("s-new", MemoryType.SUMMARIES)["retention_state"]
        == "suppressed"
    )

    # Nothing left to do on a second pass.
    assert consolidator.plan(memory_id="memory-1").ops == ()

    counts = consolidator.restore(
        plan.plan_id, memory_id="memory-1", approved_by="operator-7"
    )
    assert counts["entities"] == 1 and counts["rows"] == 1
    assert len(EntityMemory(provider).list_entities(memory_id="memory-1")) == 2
    # Each record holds exactly what it held before the merge.
    held = {
        r["entity_id"]: {a["name"] for a in r["attributes"]}
        for r in EntityMemory(provider).list_entities(memory_id="memory-1")
    }
    assert held == {"ada-1": {"role"}, "ada-2": {"city"}}
    assert (
        provider.retrieve_by_id("s-new", MemoryType.SUMMARIES)["retention_state"]
        == "active"
    )


def test_the_model_retires_a_fact_stored_twice_and_it_can_come_back(work_dir):
    provider = _provider(work_dir)
    entities = EntityMemory(provider)
    entities.upsert_entity(
        name="Ada",
        entity_type="person",
        attributes=[{"name": "favorite_drink", "value": "tea"}],
        memory_id="memory-1",
    )
    time.sleep(0.01)
    entity_id = entities.upsert_entity(
        name="Ada",
        attributes=[{"name": "preferred_beverage", "value": "coffee"}],
        memory_id="memory-1",
    )
    # The model lists the pair newest first; the older fact still goes.
    model = _Model(
        same_fact=[
            {
                "entity_id": entity_id,
                "attributes": ["preferred_beverage", "favorite_drink"],
            },
            {"entity_id": "someone-else", "attributes": ["x", "y"]},
            {"entity_id": entity_id, "attributes": ["favorite_drink", "missing"]},
        ]
    )
    consolidator = MemoryConsolidator(provider, model=model)
    plan = consolidator.plan(memory_id="memory-1")
    retire = [op for op in plan.ops if op.kind == RETIRE_ATTRIBUTE]
    assert [op.details["attribute"] for op in retire] == ["favorite_drink"]
    assert retire[0].details["why"] == "outdated"
    assert retire[0].reason == "replaced by the newer preferred_beverage"
    assert plan.model == "openai/background"

    consolidator.apply(plan, approved_by="operator-7")
    assert _facts(provider, "Ada") == {"preferred_beverage": "coffee"}
    record = entities.get_entity_by_name("Ada", memory_id="memory-1")
    assert record["metadata"]["retired_attributes"][0]["value"] == "tea"

    consolidator.restore(plan.plan_id, memory_id="memory-1", approved_by="operator-7")
    assert _facts(provider, "Ada") == {
        "favorite_drink": "tea",
        "preferred_beverage": "coffee",
    }


def _agent(work_dir: Path, model: _Model, **kwargs) -> MemAgent:
    return MemAgent(
        model=_Model(),
        background_model=model,
        memory_provider=_provider(work_dir),
        memory_ids=["memory-1"],
        application_mode="assistant",
        auto_register=False,
        agent_id="agent-1",
        **kwargs,
    )


def test_new_facts_must_quote_what_the_user_said(work_dir):
    model = _Model(
        facts=[
            {
                "entity": "user",
                "entity_type": "person",
                "attribute": "City",
                "value": "Lisbon",
                "evidence": "I moved to Lisbon",
            },
            # The assistant said this, not the user.
            {
                "entity": "user",
                "attribute": "job",
                "value": "chef",
                "evidence": "you are a chef",
            },
            {
                "entity": "user",
                "attribute": "pet",
                "value": "cat",
                "evidence": "my cat Mimi",
            },
        ]
    )
    agent = _agent(work_dir, model)
    agent._record_interaction(
        "I moved to Lisbon last month.", "So you are a chef now?", "memory-1", "t1"
    )
    agent._record_interaction(
        "No. But my cat Mimi loves it here.", "Lovely.", "memory-1", "t1"
    )

    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    facts = {
        op.details["attribute"]: op.details["value"]
        for op in plan.ops
        if op.kind == ADD_FACT
    }
    assert facts == {"city": "Lisbon", "pet": "cat"}
    assert (
        "durable facts" in model.prompts[-1]
        and "I moved to Lisbon" in model.prompts[-1]
    )

    result = agent.apply_memory_consolidation(plan.plan_id, approved_by="operator-7")
    assert result.applied[ADD_FACT] == 2
    assert _facts(agent.memory_provider, "user") == {"city": "Lisbon", "pet": "cat"}

    # The same conversation adds nothing the second time.
    assert [
        op
        for op in agent.plan_memory_consolidation(thread_ids=["t1"]).ops
        if op.kind == ADD_FACT
    ] == []

    agent.restore_memory_consolidation(plan.plan_id, approved_by="operator-7")
    assert _facts(agent.memory_provider, "user") == {}
    agent.close()


def test_a_changed_value_replaces_the_old_one_reversibly(work_dir):
    model = _Model(
        facts=[
            {
                "entity": "user",
                "attribute": "city",
                "value": "Porto",
                "evidence": "now I live in Porto",
            }
        ]
    )
    agent = _agent(work_dir, model)
    EntityMemory(agent.memory_provider).upsert_entity(
        name="user",
        attributes=[{"name": "city", "value": "Lisbon"}],
        memory_id="memory-1",
    )
    agent._record_interaction(
        "Big news: now I live in Porto.", "Congratulations!", "memory-1", "t1"
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    (op,) = [op for op in plan.ops if op.kind == ADD_FACT]
    assert op.details["replaces"] == "Lisbon"
    agent.apply_memory_consolidation(plan, approved_by="operator-7")
    assert _facts(agent.memory_provider, "user") == {"city": "Porto"}
    agent.restore_memory_consolidation(plan.plan_id, approved_by="operator-7")
    assert _facts(agent.memory_provider, "user") == {"city": "Lisbon"}
    agent.close()


def test_a_plan_that_was_edited_is_refused(work_dir):
    provider = _provider(work_dir)
    _summary(provider, "a", "same", 1.0)
    _summary(provider, "b", "same", 2.0)
    consolidator = MemoryConsolidator(provider)
    plan = consolidator.plan(memory_id="memory-1").to_dict()
    plan["ops"][0]["target_id"] = "a"
    with pytest.raises(ValueError, match="does not match"):
        consolidator.apply(ConsolidationPlan.from_dict(plan), approved_by="operator-7")
    with pytest.raises(ValueError, match="approved_by"):
        consolidator.apply(consolidator.plan(memory_id="memory-1"), approved_by=" ")


def test_scheduled_passes_run_in_the_background_and_skip_sealed_threads(work_dir):
    model = _Model(
        facts=[
            {
                "entity": "user",
                "attribute": "city",
                "value": "Lisbon",
                "evidence": "I live in Lisbon",
            }
        ]
    )
    agent = _agent(
        work_dir, model, memory_consolidation={"every_n_turns": 2, "every_hours": 0}
    )
    agent._record_interaction("I live in Lisbon.", "Noted.", "memory-1", "t1")
    agent._record_interaction("I live in Lisbon.", "Noted.", "memory-1", "secret")

    agent._retrieval_policy_override = RetrievalPolicy.sealed()
    agent._note_turn_for_consolidation("memory-1", None, "secret")
    agent._retrieval_policy_override = None
    agent._note_turn_for_consolidation("memory-1", None, "t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _facts(agent.memory_provider, "user") == {}  # one counted turn of two

    agent._note_turn_for_consolidation("memory-1", None, "t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _facts(agent.memory_provider, "user") == {"city": "Lisbon"}
    # Only the open thread was read.
    assert all("secret" not in prompt for prompt in model.prompts)
    record = EntityMemory(agent.memory_provider).get_entity_by_name(
        "user", memory_id="memory-1"
    )
    assert record["attributes"][0]["source"].startswith("consolidation:")
    agent.close()


def test_review_mode_leaves_the_plan_for_an_approver(work_dir):
    model = _Model(
        facts=[
            {
                "entity": "user",
                "attribute": "city",
                "value": "Lisbon",
                "evidence": "I live in Lisbon",
            }
        ]
    )
    agent = _agent(
        work_dir,
        model,
        memory_consolidation={
            "every_n_turns": 1,
            "auto_apply": False,
            "every_hours": 0,
        },
    )
    agent._record_interaction("I live in Lisbon.", "Noted.", "memory-1", "t1")
    agent._note_turn_for_consolidation("memory-1", None, "t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _facts(agent.memory_provider, "user") == {}
    (pending,) = agent.pending_memory_consolidations()
    agent.apply_memory_consolidation(pending.plan_id, approved_by="operator-7")
    assert _facts(agent.memory_provider, "user") == {"city": "Lisbon"}
    assert agent.pending_memory_consolidations() == []
    agent.close()


def test_the_schedule_is_saved_with_the_agent(work_dir):
    agent = _agent(work_dir, _Model()).with_memory_consolidation(every_n_turns=5)
    agent.save()
    loaded = MemAgent.load(
        "agent-1", memory_provider=agent.memory_provider, model=_Model()
    )
    assert loaded.memory_consolidation == ConsolidationConfig(
        enabled=True, every_n_turns=5
    )
    assert loaded.describe_memory()["memory_consolidation"]["every_n_turns"] == 5
    loaded.close()
    agent.close()


def test_two_facts_that_disagree_with_no_newer_one_both_stay(work_dir):
    provider = _provider(work_dir)
    _entity(provider, "ada", "Ada", home_city="Lagos", current_city="Lisbon")
    model = _Model(
        same_fact=[{"entity_id": "ada", "attributes": ["home_city", "current_city"]}]
    )
    plan = MemoryConsolidator(provider, model=model).plan(memory_id="memory-1")
    assert plan.ops == ()
    assert "neither is newer" in " ".join(plan.warnings)


def test_one_person_under_two_names_merges_only_with_shared_facts(work_dir):
    provider = _provider(work_dir)
    _entity(provider, "self", "user", drink="tea", city="Lisbon")
    _entity(provider, "dana", "Dana", drink="tea", city="Lisbon", job="nurse")
    _entity(provider, "sam", "Sam", drink="tea", job="chef")
    model = _Model(
        same_entity=[{"entity_ids": ["self", "dana"]}, {"entity_ids": ["dana", "sam"]}]
    )
    consolidator = MemoryConsolidator(provider, model=model)
    plan = consolidator.plan(memory_id="memory-1")
    (merge,) = plan.ops
    assert merge.kind == MERGE_ENTITIES and merge.target_id == "dana"
    assert merge.details["duplicate_entity_ids"] == ["self"]

    consolidator.apply(plan, approved_by="operator-7")
    names = {
        r["name"] for r in EntityMemory(provider).list_entities(memory_id="memory-1")
    }
    assert names == {"Dana", "Sam"}
    consolidator.restore(plan.plan_id, memory_id="memory-1", approved_by="operator-7")
    names = {
        r["name"] for r in EntityMemory(provider).list_entities(memory_id="memory-1")
    }
    assert names == {"user", "Dana", "Sam"}


def test_a_suppressed_sealed_summary_comes_back_sealed(work_dir):
    provider = _provider(work_dir)
    _summary(provider, "a", "same text", 1.0)
    _summary(provider, "b", "same text", 2.0)
    provider.update_by_id("b", {"retention_state": "sealed"}, MemoryType.SUMMARIES)
    consolidator = MemoryConsolidator(provider)
    plan = consolidator.plan(memory_id="memory-1")
    consolidator.apply(plan, approved_by="operator-7")
    assert (
        provider.retrieve_by_id("b", MemoryType.SUMMARIES)["retention_state"]
        == "suppressed"
    )
    consolidator.restore(plan.plan_id, memory_id="memory-1", approved_by="operator-7")
    assert (
        provider.retrieve_by_id("b", MemoryType.SUMMARIES)["retention_state"]
        == "sealed"
    )


def test_a_fact_needs_a_quote_that_says_it_and_is_not_stored_elsewhere(work_dir):
    model = _Model(
        facts=[
            # A real quote that says nothing about drinks.
            {
                "entity": "user",
                "attribute": "drink",
                "value": "tea",
                "evidence": "my sister Ines is visiting",
            },
            # Already stored under the user's name.
            {
                "entity": "user",
                "attribute": "job",
                "value": "nurse",
                "evidence": "I work as a nurse",
            },
            {
                "entity": "user",
                "attribute": "sister",
                "value": "Ines",
                "evidence": "my sister Ines is visiting",
            },
        ]
    )
    agent = _agent(work_dir, model)
    EntityMemory(agent.memory_provider).upsert_entity(
        name="Dana",
        attributes=[{"name": "profession", "value": "nurse"}],
        memory_id="memory-1",
    )
    agent._record_interaction(
        "I work as a nurse and my sister Ines is visiting.", "Nice.", "memory-1", "t1"
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    assert [op.details["attribute"] for op in plan.ops if op.kind == ADD_FACT] == [
        "sister"
    ]
    agent.close()


def test_the_journal_says_what_each_consolidation_change_did(work_dir):
    from memorizz.memory_history import MemoryHistory

    provider = _provider(work_dir)
    history = MemoryHistory(provider, record_changes=True)
    _entity(provider, "ada-1", "Ada Lovelace", role="mathematician")
    _entity(provider, "ada-2", "Ada Lovelace", known_for="notes")
    _summary(provider, "s-old", "Same summary.", 1.0)
    _summary(provider, "s-new", "Same summary.", 2.0)
    consolidator = MemoryConsolidator(provider, agent_id="agent-1")
    plan = consolidator.plan(memory_id="memory-1")
    consolidator.apply(plan, approved_by="operator-7")
    consolidator.restore(plan.plan_id, memory_id="memory-1", approved_by="operator-7")

    events = history.timeline(memory_id="memory-1", limit=100)["events"]
    made = [e for e in events if e["source"] == f"consolidation:{plan.plan_id}"]
    assert {e["label"] for e in made} >= {
        "Merged duplicates of Ada Lovelace",
        "Hid a duplicate summary",
    }
    assert {e["actor"] for e in made} == {"operator-7"}
    undone = [
        e for e in events if e["source"] == f"consolidation:{plan.plan_id}:restore"
    ]
    assert undone and {e["label"] for e in undone} == {"Undid memory consolidation"}


def test_instances_of_one_agent_share_the_schedule(work_dir):
    """A server that loads the agent per request still reaches every_n_turns."""
    model = _Model(
        facts=[
            {
                "entity": "user",
                "attribute": "city",
                "value": "Lisbon",
                "evidence": "I live in Lisbon",
            }
        ]
    )
    provider = _provider(work_dir)

    def load():
        return MemAgent(
            model=_Model(),
            background_model=model,
            memory_provider=provider,
            memory_ids=["memory-1"],
            application_mode="assistant",
            auto_register=False,
            agent_id="agent-1",
            memory_consolidation={"every_n_turns": 2, "every_hours": 0},
        )

    first = load()
    first._record_interaction("I live in Lisbon.", "Noted.", "memory-1", "t1")
    first._note_turn_for_consolidation("memory-1", None, "t1")
    second = load()
    second._note_turn_for_consolidation("memory-1", None, "t1")
    assert second.wait_for_memory_upkeep(timeout=10)
    assert _facts(provider, "user") == {"city": "Lisbon"}
    second.close()
    first.close()
