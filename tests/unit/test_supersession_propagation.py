"""Propagation: flag skills and notes that still state a superseded value."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from memorizz.enums import MemoryType
from memorizz.learning.propagation import SupersessionPropagator, mentions
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda text: list(VECTOR))
    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        lambda text: list(VECTOR),
    )


class _Model:
    model = "chat"
    facts: list = []

    def generate_text(self, prompt, instructions=None):
        if "durable facts" in prompt:
            return json.dumps({"facts": self.facts})
        if "same_fact" in prompt:
            return json.dumps({"same_fact": [], "same_entity": []})
        return "ok"

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def get_config(self):
        return {"provider": "openai", "model": "chat"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


class _Embedder:
    def get_embedding(self, text):
        return list(VECTOR)


@pytest.fixture()
def agent():
    path = Path(tempfile.mkdtemp(prefix="memorizz-propagation-"))
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=path / "memory",
            embedding_provider=_Embedder(),
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )
    made = MemAgent(
        model=_Model(),
        background_model=_Model(),
        memory_provider=provider,
        memory_ids=["memory-1"],
        application_mode="assistant",
        auto_register=False,
        agent_id="agent-1",
    )
    try:
        yield made
    finally:
        made.close()
        shutil.rmtree(path, ignore_errors=True)


def _skill(provider, skill_id, content, agent_id="agent-1", status="active"):
    provider.store(
        data={
            "_id": skill_id,
            "skill_id": skill_id,
            "name": f"Skill {skill_id}",
            "description": "Pay an agreed claim",
            "content": content,
            "agent_id": agent_id,
            "status": status,
        },
        memory_store_type=MemoryType.SKILLBOX,
    )


def _note(provider, note_id, content, memory_id="memory-1"):
    provider.store(
        data={"_id": note_id, "memory_id": memory_id, "content": content},
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )


def test_matching_is_whole_phrase_and_needs_context_for_ordinary_values():
    terms = ["Total Payout Authority", "TPA"]
    hit = mentions("Grade 1 TPA is £2,500 today.", "£2,500", entity_terms=terms)
    assert hit and hit["confidence"] == "high" and hit["matched_on"] == "entity"
    # An amount with nothing saying what it is an amount of is not flagged.
    assert mentions("Goodwill up to £2,500.", "£2,500", entity_terms=terms) is None
    # Not inside a longer number or word.
    assert mentions("TPA is £2,500,000", "£2,500", entity_terms=terms) is None
    assert mentions("The Leedsman TPA", "Leeds", entity_terms=terms) is None
    # A name stands on its own, at low confidence, and keeps its capitals.
    name = mentions("Send it to the Leeds office.", "Leeds", entity_terms=["CFU"])
    assert name and name["confidence"] == "low" and name["matched_on"] == "value"
    assert mentions("send it to leeds", "Leeds", entity_terms=["CFU"]) is None
    # The attribute named in the text counts as context; short values never match.
    assert mentions(
        "the referral threshold is 2 days", "2 days", attribute="referral_threshold"
    )
    assert mentions("CFU is in use", "use", entity_terms=["CFU"]) is None


def test_a_superseded_value_flags_the_skills_and_notes_that_still_say_it(agent):
    provider = agent.memory_provider
    entities = EntityMemory(provider)
    scope = {"memory_id": "memory-1"}
    entities.upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "office", "value": "Leeds"}],
        **scope,
    )
    _skill(provider, "s1", "Post the Amber Pack to the Counter Fraud Unit in Leeds.")
    _skill(provider, "s2", "Look the claim up, then release the payment.")
    _skill(provider, "s3", "Counter Fraud Unit, Leeds.", agent_id="someone-else")
    _skill(provider, "s4", "Counter Fraud Unit, Leeds.", status="deprecated")
    _note(provider, "n1", "The CFU sits in the Leeds office.")
    _note(provider, "n2", "Leeds office, Counter Fraud Unit", memory_id="another")
    assert agent.stale_references() == []

    entities.upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "office", "value": "Manchester"}],
        **scope,
    )
    flags = agent.stale_references()
    assert sorted(flag["target_id"] for flag in flags) == ["n1", "s1"]
    skill = next(flag for flag in flags if flag["target_id"] == "s1")
    assert skill["target_type"] == "skill" and skill["confidence"] == "high"
    assert (skill["entity"], skill["attribute"]) == ("Counter Fraud Unit", "office")
    assert (skill["old_value"], skill["new_value"]) == ("Leeds", "Manchester")
    assert "Leeds" in skill["excerpt"] and skill["superseded_at"]
    # Nothing was rewritten.
    stored = provider.retrieve_by_id("s1", MemoryType.SKILLBOX)
    assert "Leeds" in stored["content"]

    agent.dismiss_stale_reference(skill["flag_id"], by="dev", note="historic")
    note = next(flag for flag in flags if flag["target_id"] == "n1")
    agent.resolve_stale_reference(note["flag_id"], by="dev")
    assert agent.stale_references() == []
    assert len(agent.stale_references(None)) == 2
    # A closed flag is not opened again by a sweep.
    assert agent.check_stale_references("memory-1") == []
    with pytest.raises(ValueError):
        agent.dismiss_stale_reference(skill["flag_id"], by="")


def test_a_sweep_finds_what_was_superseded_before_anyone_was_watching(agent):
    provider = agent.memory_provider
    agent.propagate_supersessions = False
    entities = EntityMemory(provider)
    for value in ("Nadia Hussain", "Sam Adeyemi"):
        entities.upsert_entity(
            name="Counter Fraud Unit",
            attributes=[{"name": "lead", "value": value}],
            memory_id="memory-1",
        )
    _note(provider, "n1", "Nadia Hussain leads the Counter Fraud Unit.")
    assert agent.stale_references() == []  # switched off: nothing as it happened
    opened = agent.check_stale_references("memory-1")
    assert [flag["old_value"] for flag in opened] == ["Nadia Hussain"]
    assert SupersessionPropagator(agent).sweep(memory_id="memory-1") == []


def test_a_value_that_only_grew_is_not_a_stale_reference(agent):
    provider = agent.memory_provider
    entities = EntityMemory(provider)
    entities.upsert_entity(
        name="CFU",
        attributes=[{"name": "office", "value": "Leeds"}],
        memory_id="memory-1",
    )
    _note(provider, "n1", "The CFU is in Leeds.")
    entities.upsert_entity(
        name="CFU",
        attributes=[{"name": "office", "value": "Leeds office"}],
        memory_id="memory-1",
    )
    assert agent.stale_references() == []


def test_a_value_consolidation_replaces_is_propagated_too(agent, monkeypatch):
    import time

    provider = agent.memory_provider
    entities = EntityMemory(provider)
    entities.upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "lead", "value": "Nadia Hussain", "source": "turn"}],
        memory_id="memory-1",
        source_level=2,
    )
    _note(provider, "n1", "Ask Nadia Hussain at the Counter Fraud Unit.")
    time.sleep(0.02)
    agent._record_interaction(
        "The Counter Fraud Unit is led by Sam Adeyemi now.", "Noted.", "memory-1", "t1"
    )
    monkeypatch.setattr(
        _Model,
        "facts",
        [
            {
                "entity": "Counter Fraud Unit",
                "attribute": "lead",
                "value": "Sam Adeyemi",
                "evidence": "led by Sam Adeyemi now",
            }
        ],
    )
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    agent.apply_memory_consolidation(plan, approved_by="operator-7")
    (flag,) = agent.stale_references()
    assert flag["target_id"] == "n1" and flag["new_value"] == "Sam Adeyemi"
