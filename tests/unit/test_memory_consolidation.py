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
        source_level=2,  # what the user said before
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


def test_what_the_user_says_never_replaces_a_value_the_host_wrote(work_dir):
    from memorizz.learning.consolidation import FACT_NOT_APPLIED
    from memorizz.long_term.semantic.entity_memory.entity_memory import (
        attribute_timeline,
    )

    model = _Model(
        facts=[
            {
                "entity": "Counter Fraud Unit",
                "attribute": "threshold",
                "value": "3,000",
                "evidence": "the threshold is 3,000 now",
            }
        ]
    )
    agent = _agent(work_dir, model)
    entities = EntityMemory(agent.memory_provider)
    entities.upsert_entity(  # the host, from its policy system (level 3)
        name="Counter Fraud Unit",
        attributes=[{"name": "threshold", "value": "2,500"}],
        memory_id="memory-1",
    )
    agent._record_interaction(
        "I think the threshold is 3,000 now.", "Noted.", "memory-1", "t1"
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    (op,) = [op for op in plan.ops if op.kind == ADD_FACT]
    assert op.details["beside"] == "2,500" and "replaces" not in op.details
    result = agent.apply_memory_consolidation(plan, approved_by="operator-7")
    assert result.applied[FACT_NOT_APPLIED] == 1 and result.applied[ADD_FACT] == 0
    record = entities.get_entity_by_name("Counter Fraud Unit", memory_id="memory-1")
    (threshold,) = record["attributes"]
    assert threshold["value"] == "2,500"
    assert [
        (row["value"], row["applied"]) for row in attribute_timeline(threshold)[1:]
    ] == [("3,000", False)]

    # A plan stored before the rule may still say "replaces": the host's
    # value stays all the same.
    from memorizz.learning.consolidation import (
        ConsolidationOp,
        ConsolidationPlan,
        MemoryConsolidator,
    )

    old_op = ConsolidationOp(
        ADD_FACT,
        "entity_memory",
        record["entity_id"],
        "the user said this; memory still had the old value",
        {
            "entity": "Counter Fraud Unit",
            "attribute": "threshold",
            "value": "9,999",
            "evidence": "x",
            "replaces": "2,500",
        },
    )
    old_plan = ConsolidationPlan(
        plan_id=ConsolidationPlan.make_id("memory-1", None, (old_op,)),
        memory_id="memory-1",
        user_id=None,
        ops=(old_op,),
    )
    MemoryConsolidator(agent.memory_provider).apply(old_plan, approved_by="operator-7")
    assert _facts(agent.memory_provider, "Counter Fraud Unit") == {"threshold": "2,500"}
    agent.close()


def test_a_fact_added_to_a_record_does_not_rename_it(work_dir):
    model = _Model(
        facts=[
            {
                "entity": "tomasz nowak",
                "entity_type": "human",
                "attribute": "handles",
                "value": "subsidence claims",
                "evidence": "handles subsidence claims",
            }
        ]
    )
    agent = _agent(work_dir, model)
    entities = EntityMemory(agent.memory_provider)
    entities.upsert_entity(
        name="Tomasz Nowak",
        entity_type="person",
        attributes=[{"name": "team", "value": "property"}],
        memory_id="memory-1",
        source_level=2,
    )
    agent._record_interaction(
        "Tomasz Nowak handles subsidence claims.", "Noted.", "memory-1", "t1"
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    agent.apply_memory_consolidation(plan, approved_by="operator-7")
    rows = entities.list_entities(memory_id="memory-1")
    (record,) = [
        row
        for row in rows
        if any(item["name"] == "handles" for item in row["attributes"])
    ]
    assert (record["name"], record["entity_type"]) == ("Tomasz Nowak", "person")
    assert {item["name"] for item in record["attributes"]} == {"team", "handles"}
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


def test_names_that_agree_need_one_shared_fact_and_never_none(work_dir):
    from memorizz.learning.consolidation import MERGE_ENTITIES, _names_agree

    assert _names_agree("CFU", "Counter Fraud Unit")
    assert _names_agree("counter  fraud unit", "Counter Fraud Unit")
    assert not _names_agree("CFU", "Customer Feedback")
    assert not _names_agree("Ada", "Ada Lovelace")
    assert not _names_agree("C", "Counter")

    provider = _provider(work_dir)
    _entity(provider, "cfu-long", "Counter Fraud Unit", threshold="2,500")
    _entity(provider, "cfu-short", "CFU", lead="Marcus Webb", office="Leeds")
    reply = [{"entity_ids": ["cfu-long", "cfu-short"]}]

    def plan():
        consolidator = MemoryConsolidator(provider, model=_Model(same_entity=reply))
        return [
            op
            for op in consolidator.plan(memory_id="memory-1").ops
            if op.kind == MERGE_ENTITIES
        ]

    # The names agree and the model says so, but they share no fact: no merge.
    assert plan() == []
    # One shared fact is enough with names that agree, across spellings of
    # the type.
    EntityMemory(provider).upsert_entity(
        entity_id="cfu-long",
        name="Counter Fraud Unit",
        entity_type="organisation",
        attributes=[{"name": "lead", "value": "Marcus Webb"}],
        memory_id="memory-1",
    )
    EntityMemory(provider).upsert_entity(
        entity_id="cfu-short",
        name="CFU",
        entity_type="Organization",
        memory_id="memory-1",
    )
    (merge,) = plan()
    assert "matching name" in merge.reason


def test_another_spelling_of_a_name_does_not_rename_the_record(work_dir):
    entities = EntityMemory(_provider(work_dir))
    entities.upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "threshold", "value": "2,500"}],
        memory_id="memory-1",
    )
    entities.upsert_entity(
        name="counter  fraud unit",
        attributes=[{"name": "lead", "value": "Marcus Webb"}],
        memory_id="memory-1",
        source_level=2,
    )
    (record,) = entities.list_entities(memory_id="memory-1")
    assert record["name"] == "Counter Fraud Unit"
    assert {item["name"] for item in record["attributes"]} == {"threshold", "lead"}


def test_a_merge_keeps_the_copy_of_a_fact_from_the_stronger_source(work_dir):
    from memorizz.long_term.semantic.entity_memory.entity_memory import source_level

    entities = EntityMemory(_provider(work_dir))
    scope = {"memory_id": "memory-1", "user_id": "user-1"}
    host = entities.upsert_entity(  # host code: level 3
        name="Counter Fraud Unit",
        entity_type="organisation",
        attributes=[{"name": "lead", "value": "Marcus Webb", "confidence": 0.5}],
        **scope,
    )
    said = entities.upsert_entity(  # a conversation, later and more confident
        name="CFU",
        entity_type="organization",
        attributes=[{"name": "lead", "value": "Marcus Webb", "confidence": 0.9}],
        source_level=2,
        **scope,
    )
    entities.consolidate_duplicate_entities(
        entity_ids=[host, said], apply=True, provenance="plan-1", **scope
    )
    (record,) = entities.list_entities(**scope)
    (lead,) = record["attributes"]
    assert source_level(lead) == 3
    # A spelling of the type is not a new type either.
    entities.upsert_entity(
        entity_id=record["entity_id"],
        name=record["name"],
        entity_type="Organization"
        if record["entity_type"] == "organisation"
        else "Organisation",
        source_level=2,
        **scope,
    )
    kept = entities.get_entity(record["entity_id"], **scope)
    assert kept["entity_type"] == record["entity_type"]


def test_an_older_statement_never_replaces_a_newer_value(work_dir):
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
    agent = _agent(work_dir, model)
    agent._record_interaction("I live in Lisbon.", "Noted.", "memory-1", "t1")
    time.sleep(0.02)
    # Said later, and already in memory: the stored value is the newer one.
    EntityMemory(agent.memory_provider).upsert_entity(
        name="user",
        attributes=[{"name": "city", "value": "Porto"}],
        memory_id="memory-1",
        source_level=2,
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    assert [op for op in plan.ops if op.kind == ADD_FACT] == []
    assert _facts(agent.memory_provider, "user")["city"] == "Porto"
    agent.close()


def test_a_fact_aimed_at_a_record_the_plan_merges_lands_on_the_survivor(work_dir):
    from memorizz.learning.consolidation import ConsolidationOp, ConsolidationPlan

    provider = _provider(work_dir)
    entities = EntityMemory(provider)
    scope = {"memory_id": "memory-1", "user_id": "user-1"}
    host = entities.upsert_entity(
        name="Counter Fraud Unit",
        entity_type="team",
        attributes=[{"name": "led_by", "value": "Marcus Webb"}],
        **scope,
    )
    short = entities.upsert_entity(
        name="CFU",
        entity_type="team",
        attributes=[{"name": "led_by", "value": "Marcus Webb"}],
        source_level=2,
        **scope,
    )

    def fact(target):
        return ConsolidationOp(
            ADD_FACT,
            MemoryType.ENTITY_MEMORY.value,
            target,
            "The user said so.",
            {
                "entity": "cfu",
                "entity_type": "team",
                "attribute": "office",
                "value": "Manchester",
                "evidence": "the CFU moved to Manchester",
            },
        )

    ops = (
        ConsolidationOp(
            MERGE_ENTITIES,
            MemoryType.ENTITY_MEMORY.value,
            host,
            "Same team under two names.",
            {"duplicate_entity_ids": [short], "names": ["Counter Fraud Unit", "CFU"]},
        ),
        # The model asked for the fact once per record, as it saw two.
        fact(short),
        fact(host),
    )
    plan = ConsolidationPlan(
        plan_id=ConsolidationPlan.make_id("memory-1", "user-1", ops),
        memory_id="memory-1",
        user_id="user-1",
        ops=ops,
    )
    result = MemoryConsolidator(provider, agent_id="agent-1").apply(
        plan, approved_by="operator-7"
    )

    assert result.errors == ()
    (record,) = entities.list_entities(**scope)
    offices = [a for a in record["attributes"] if a["name"] == "office"]
    assert [a["value"] for a in offices] == ["Manchester"]


def test_the_models_spelling_of_a_name_is_still_the_stored_entity(work_dir):
    from memorizz.learning.consolidation import FACT_NOT_APPLIED

    model = _Model(
        facts=[
            {
                "entity": "counter_fraud_unit",
                "attribute": "threshold",
                "value": "3,000",
                "evidence": "the threshold is 3,000 now",
            }
        ]
    )
    agent = _agent(work_dir, model)
    entities = EntityMemory(agent.memory_provider)
    entities.upsert_entity(  # the host, from its policy system (level 3)
        name="Counter Fraud Unit",
        attributes=[{"name": "threshold", "value": "2,500"}],
        memory_id="memory-1",
    )
    agent._record_interaction(
        "I think the threshold is 3,000 now.", "Noted.", "memory-1", "t1"
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    (op,) = [op for op in plan.ops if op.kind == ADD_FACT]
    assert op.details["beside"] == "2,500"
    result = agent.apply_memory_consolidation(plan, approved_by="operator-7")
    assert result.applied[FACT_NOT_APPLIED] == 1
    (record,) = entities.list_entities(memory_id="memory-1")
    assert record["name"] == "Counter Fraud Unit"
    assert [a["value"] for a in record["attributes"]] == ["2,500"]
    agent.close()


def test_an_older_statement_loses_whatever_the_model_calls_the_entity(work_dir):
    # Monday: "sits in Leeds". Wednesday, already in memory: Manchester. The
    # model names the team in snake_case; Monday's words still must not win.
    model = _Model(
        facts=[
            {
                "entity": "counter_fraud_unit",
                "attribute": "office",
                "value": "Leeds",
                "evidence": "The Counter Fraud Unit sits in Leeds",
            }
        ]
    )
    agent = _agent(work_dir, model)
    agent._record_interaction(
        "The Counter Fraud Unit sits in Leeds.", "Noted.", "memory-1", "t1"
    )
    time.sleep(0.02)
    EntityMemory(agent.memory_provider).upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "office", "value": "Manchester"}],
        memory_id="memory-1",
        source_level=2,
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    assert [op for op in plan.ops if op.kind == ADD_FACT] == []
    assert _facts(agent.memory_provider, "Counter Fraud Unit")["office"] == "Manchester"
    agent.close()


def test_a_replaced_value_does_not_come_back_under_another_attribute_name(work_dir):
    # Tuesday: "is in the Leeds office". Wednesday, in memory: Manchester, with
    # Leeds in the history. The model files Tuesday's words as a new attribute.
    model = _Model(
        facts=[
            {
                "entity": "Counter Fraud Unit",
                "attribute": "location",
                "value": "Leeds office",
                "evidence": "the Counter Fraud Unit is in the Leeds office",
            }
        ]
    )
    agent = _agent(work_dir, model)
    entities = EntityMemory(agent.memory_provider)
    entities.upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "office_location", "value": "Leeds office"}],
        memory_id="memory-1",
        source_level=2,
    )
    agent._record_interaction(
        "Confirmed: the Counter Fraud Unit is in the Leeds office.",
        "Noted.",
        "memory-1",
        "t1",
    )
    time.sleep(0.02)
    entities.upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "office_location", "value": "Manchester office"}],
        memory_id="memory-1",
        source_level=2,
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    assert [op for op in plan.ops if op.kind == ADD_FACT] == []
    facts = _facts(agent.memory_provider, "Counter Fraud Unit")
    assert facts == {"office_location": "Manchester office"}
    agent.close()


def test_a_value_the_user_states_again_is_confirmed_and_undone(work_dir):
    from memorizz.learning.consolidation import REINFORCE_FACT

    model = _Model()
    agent = _agent(work_dir, model)
    entities = EntityMemory(agent.memory_provider)
    entities.upsert_entity(  # the host's record
        name="Counter Fraud Unit",
        attributes=[{"name": "office", "value": "Leeds"}],
        memory_id="memory-1",
    )
    time.sleep(0.02)
    agent._record_interaction(
        "Confirming the Counter Fraud Unit is in Leeds.", "Noted.", "memory-1", "t1"
    )
    # The model lists it as confirmed, naming the entity its own way.
    model.generate_text = lambda prompt, instructions=None: (
        json.dumps(
            {
                "facts": [],
                "confirmed": [
                    {
                        "entity": "counter_fraud_unit",
                        "attribute": "office",
                        "evidence": "the Counter Fraud Unit is in Leeds",
                    }
                ],
            }
        )
        if "durable facts" in prompt
        else json.dumps({"same_fact": [], "same_entity": []})
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    (op,) = plan.ops
    assert op.kind == REINFORCE_FACT and plan.counts[REINFORCE_FACT] == 1
    assert op.details["value"] == "Leeds" and op.details["thread_id"] == "t1"
    result = agent.apply_memory_consolidation(plan, approved_by="operator-7")
    assert result.applied[REINFORCE_FACT] == 1 and not result.errors
    (record,) = entities.list_entities(memory_id="memory-1")
    (office,) = record["attributes"]
    assert office["value"] == "Leeds" and office["level"] == 3
    assert office["confirmations"] == 1
    assert office["confirmed_by"][0]["thread_id"] == "t1"
    # The same conversation does not confirm it a second time.
    assert not agent.plan_memory_consolidation(thread_ids=["t1"]).ops
    counts = MemoryConsolidator(agent.memory_provider, agent_id="agent-1").restore(
        plan.plan_id, memory_id="memory-1", approved_by="operator-7"
    )
    assert counts["confirmations_removed"] == 1
    (record,) = entities.list_entities(memory_id="memory-1")
    assert "confirmations" not in record["attributes"][0]
    agent.close()


def test_the_conversation_a_value_came_from_does_not_confirm_it(work_dir):
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
    agent = _agent(work_dir, model)
    agent._record_interaction("I live in Lisbon.", "Noted.", "memory-1", "t1")
    time.sleep(0.02)
    EntityMemory(agent.memory_provider).upsert_entity(
        name="user",
        attributes=[{"name": "city", "value": "Lisbon", "thread_id": "t1"}],
        memory_id="memory-1",
        source_level=2,
    )
    assert not agent.plan_memory_consolidation(thread_ids=["t1"]).ops
    agent.close()
