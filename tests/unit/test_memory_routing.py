"""Memory routing of an agent's own writes: remember(), the remember tool,
turns routed after they finish, and the builder argument."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.memagent import MemAgent
from memorizz.memagent.builders.agent_builder import MemAgentBuilder
from memorizz.memagent.memory_routing import MemoryRoutingConfig
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]

# What the stub router answers, by a word in the text.
ROUTES = {
    "sister": ("entity_memory", 0.97),
    "policy": ("knowledge_base", 0.96),
    "thanks": ("do_not_store", 0.99),
    "always answer": ("personas", 0.9),
    "maybe": ("knowledge_base", 0.3),
    "draft": ("short_term_memory", 0.9),
}


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-routing-"))
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
    monkeypatch.setattr(
        "memorizz.long_term.semantic.knowledge_base._embed_chunks",
        lambda chunks: [list(VECTOR) for _ in chunks],
    )
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")


@pytest.fixture()
def routed(monkeypatch):
    """The Decisions router answers from ROUTES and records what it saw."""
    seen: List[str] = []

    def decide(text: str, model: Optional[str]) -> Dict[str, Any]:
        seen.append(text)
        for word, (store, confidence) in ROUTES.items():
            if word in text.lower():
                return {
                    "memory_type": store,
                    "confidence": confidence,
                    "router": f"openai_decisions/{model or 'gpt-6-luna'}",
                }
        return {"memory_type": "conversation_memory", "confidence": 0.8}

    monkeypatch.setattr("memorizz.memagent.memory_routing._decisions", decide)
    return seen


class _Embedder:
    def get_embedding(self, text: str) -> List[float]:
        return list(VECTOR)


class _Model:
    """Chat model that answers "ok"; as the background model it extracts
    the sister's allergy, quoting the text."""

    def __init__(self) -> None:
        self.model = "stub"
        self.prompts: List[str] = []

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        self.prompts.append(prompt)
        if "durable facts the" in prompt and "sister" in prompt:
            return json.dumps(
                {
                    "facts": [
                        {
                            "entity": "Ines",
                            "entity_type": "person",
                            "attribute": "allergy",
                            "value": "peanuts",
                            "evidence": "is allergic to peanuts",
                        },
                        {
                            "entity": "Ines",
                            "attribute": "home_city",
                            "value": "Madrid",  # not in the text: rejected
                            "evidence": "lives in Madrid",
                        },
                    ]
                }
            )
        if "store this item belongs in" in prompt:
            return '{"memory_type": "knowledge_base", "confidence": 0.8}'
        return '{"facts": []}'

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "openai", "model": "stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


def _provider(work_dir: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=work_dir / "memory",
            embedding_provider=_Embedder(),
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )


def _agent(work_dir: Path, **kwargs) -> MemAgent:
    kwargs.setdefault("model", _Model())
    kwargs.setdefault("memory_provider", _provider(work_dir))
    kwargs.setdefault("memory_ids", ["memory-1"])
    kwargs.setdefault("auto_register", False)
    kwargs.setdefault("agent_id", "agent-1")
    kwargs.setdefault("application_mode", "assistant")
    return MemAgent(**kwargs)


def _facts(provider, memory_id="memory-1") -> Dict[str, str]:
    out = {}
    for record in EntityMemory(provider).list_entities(memory_id=memory_id):
        for attribute in record.get("attributes") or []:
            out[f"{record['name']}.{attribute['name']}"] = attribute["value"]
    return out


def _remembered(provider) -> List[Dict[str, Any]]:
    return [
        row
        for row in provider.list_all(MemoryType.KNOWLEDGE_BASE)
        if row.get("namespace") == "remembered"
    ]


def test_config_values():
    assert MemoryRoutingConfig.from_value(None).enabled is False
    assert MemoryRoutingConfig.from_value(True).router == "auto"
    config = MemoryRoutingConfig.from_value("openai_decisions:gpt-6-luna")
    assert (config.router, config.model, config.capture_turns) == (
        "openai_decisions",
        "gpt-6-luna",
        True,
    )
    assert MemoryRoutingConfig.from_value("finetuned:run-9").run_id == "run-9"
    assert (
        MemoryRoutingConfig.from_value(
            {"router": "llm", "capture_turns": "off"}
        ).capture_turns
        is False
    )
    with pytest.raises(ValueError):
        MemoryRoutingConfig.from_value("telepathy")
    with pytest.raises(ValueError):
        MemoryRoutingConfig(enabled=True, min_confidence=2)


def test_remember_files_each_memory_where_it_belongs(work_dir, routed):
    agent = _agent(work_dir, memory_routing="openai_decisions")
    provider = agent.memory_provider

    person = agent.remember("My sister Ines is allergic to peanuts.")
    assert person["stored"] and person["memory_type"] == "entity_memory"
    assert person["routed_to"] == "entity_memory"
    # The fact rests on the text; the invented city is not stored.
    assert _facts(provider) == {"Ines.allergy": "peanuts"}

    policy = agent.remember("Refund policy: returns within 30 days with a receipt.")
    assert policy["stored"] and policy["memory_type"] == "knowledge_base"
    rows = _remembered(provider)
    assert len(rows) == 1 and rows[0]["memory_id"] == "memory-1"
    assert rows[0]["routing"]["router"] == "openai_decisions/gpt-6-luna"

    chatter = agent.remember("thanks!")
    assert chatter["stored"] is False and chatter["memory_type"] is None

    persona = agent.remember("Always answer in Portuguese.")
    assert persona["memory_type"] == "knowledge_base" and persona["applied"] is False
    assert "update_persona" in persona["next_step"]

    unsure = agent.remember("maybe the office moves next year")
    assert unsure["memory_type"] == "knowledge_base" and unsure["applied"] is False
    assert "unsure" in unsure["reason"]

    note = agent.remember("draft: compare three hotels before Friday")
    assert note["memory_type"] == "short_term_memory" and note["stored"]
    assert provider.retrieve_by_id(note["record_id"], MemoryType.SHORT_TERM_MEMORY)
    agent.close()


def test_a_remembered_policy_is_recalled_like_the_knowledge_base(work_dir, routed):
    agent = _agent(work_dir, memory_routing="openai_decisions")
    agent.remember("Refund policy: returns within 30 days with a receipt.")
    rows = agent.memory_manager.retrieve_relevant_memories(
        query="how long do refunds take",
        memory_type=MemoryType.KNOWLEDGE_BASE,
        memory_id="memory-1",
        limit=5,
    )
    assert any("30 days" in str(row.get("content")) for row in rows)
    agent.close()


def test_a_sealed_conversation_keeps_its_words_out_of_shared_memory(work_dir, routed):
    agent = _agent(
        work_dir, memory_routing="openai_decisions", retrieval_policy="sealed"
    )
    agent._current_thread_id = "t-sealed"
    receipt = agent.remember("Refund policy: returns within 30 days.")
    assert receipt["stored"] is False and "sealed" in receipt["reason"]
    assert _remembered(agent.memory_provider) == []
    # Facts still land, in the conversation's own entity scope.
    fact = agent.remember("My sister Ines is allergic to peanuts.")
    assert fact["stored"] and fact["memory_id"] != "memory-1"
    agent.close()


def test_each_turn_is_routed_after_it_finishes(work_dir, routed):
    model = _Model()
    agent = _agent(work_dir, model=model, memory_routing="openai_decisions")
    agent.run(
        "My sister Ines is allergic to peanuts.", memory_id="memory-1", thread_id="t1"
    )
    agent.run("thanks!", memory_id="memory-1", thread_id="t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert "My sister Ines is allergic to peanuts." in routed
    assert _facts(agent.memory_provider) == {"Ines.allergy": "peanuts"}
    # Unsure or unfileable turns file nothing during capture.
    agent.run("maybe the office moves next year", memory_id="memory-1", thread_id="t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _remembered(agent.memory_provider) == []
    # The chat model is told facts are saved for it, so it skips the upserts.
    assert "saved automatically after each turn" in agent._build_system_prompt()
    agent.close()


def test_turns_are_not_routed_without_routing(work_dir, routed):
    agent = _agent(work_dir)
    agent.run(
        "My sister Ines is allergic to peanuts.", memory_id="memory-1", thread_id="t1"
    )
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert routed == []
    assert "remember" not in agent.tool_manager.list_tools()
    assert "call 'entity_memory_upsert' with" in agent._build_system_prompt()
    agent.close()


def test_the_remember_tool_and_describe_memory(work_dir, routed):
    agent = _agent(work_dir, memory_routing="openai_decisions")
    assert "remember" in agent.tool_manager.list_tools()
    assert "remember" in agent.describe_memory()["memory_tools"]
    assert agent.describe_memory()["memory_routing"]["router"] == "openai_decisions"
    agent.with_memory_routing(False)
    assert "remember" not in agent.tool_manager.list_tools()
    assert agent.describe_memory()["memory_routing"] is None
    agent.close()


def test_auto_prefers_decisions_with_an_openai_key_else_the_model(
    work_dir, routed, monkeypatch
):
    ready = {"openai": True}
    monkeypatch.setattr(
        "memorizz.memagent.memory_routing._openai_ready", lambda: ready["openai"]
    )
    agent = _agent(work_dir, memory_routing=True)
    assert agent.remember("Refund policy: 30 days.")["router"].startswith(
        "openai_decisions"
    )
    ready["openai"] = False
    receipt = agent.remember("Shipping takes five days.")
    assert (
        receipt["router"] == "llm/stub" and receipt["memory_type"] == "knowledge_base"
    )
    agent.close()


def test_routing_is_saved_and_loaded_with_the_agent(work_dir, routed):
    agent = _agent(
        work_dir,
        memory_routing={"router": "openai_decisions", "min_confidence": 0.7},
        reranker="voyage:rerank-2.5-lite",
    )
    agent.save()
    stored = agent.memory_provider.retrieve_memagent("agent-1")
    stored = stored if isinstance(stored, dict) else stored.model_dump()
    assert stored["memory_routing"]["min_confidence"] == 0.7
    assert stored["reranker"] == {"provider": "voyage", "model": "rerank-2.5-lite"}
    loaded = MemAgent.load(
        "agent-1", memory_provider=agent.memory_provider, model=_Model()
    )
    assert loaded.memory_routing.router == "openai_decisions"
    assert loaded.reranker.model == "rerank-2.5-lite"
    assert "remember" in loaded.tool_manager.list_tools()
    agent.close()
    loaded.close()


def test_routing_takes_less_code_than_filing_by_hand(work_dir, routed):
    """The same memories, filed by hand per store and by the router."""
    by_hand = _provider(work_dir / "by-hand")
    from memorizz.long_term.semantic.knowledge_base import KnowledgeBase

    # By hand: the developer picks the store and its API for each memory.
    EntityMemory(by_hand).upsert_entity(
        name="Ines",
        entity_type="person",
        attributes=[{"name": "allergy", "value": "peanuts"}],
        memory_id="memory-1",
    )
    KnowledgeBase(by_hand).ingest_knowledge(
        "Refund policy: returns within 30 days with a receipt.",
        "remembered",
        chunking_strategy="none",
        metadata={"memory_id": "memory-1"},
    )

    # Routed: one call per memory, whatever it is.
    agent = (
        MemAgentBuilder()
        .with_model(_Model())
        .with_memory_provider(_provider(work_dir / "routed"))
        .with_memory_ids(["memory-1"])
        .with_memory_routing("openai_decisions")
        .build()
    )
    for text in (
        "My sister Ines is allergic to peanuts.",
        "Refund policy: returns within 30 days with a receipt.",
        "thanks!",
    ):
        agent.remember(text)

    assert _facts(agent.memory_provider) == _facts(by_hand)
    assert [r["content"] for r in _remembered(agent.memory_provider)] == [
        r["content"] for r in _remembered(by_hand)
    ]
    agent.close()


def test_the_builder_takes_router_names_and_settings():
    builder = MemAgentBuilder().with_memory_routing(
        "openai_decisions:gpt-6-luna", capture_turns=False
    )
    assert builder._memory_routing == {
        "enabled": True,
        "router": "openai_decisions",
        "capture_turns": False,
        "min_confidence": 0.5,
        "model": "gpt-6-luna",
    }
    assert MemAgentBuilder().with_memory_routing()._memory_routing == {"router": "auto"}
    assert MemAgentBuilder().with_memory_routing(False)._memory_routing is None
    assert MemAgentBuilder().with_reranker("voyage")._reranker == "voyage"


def test_the_router_never_files_what_is_recorded_anyway(monkeypatch):
    from memorizz.memagent import memory_routing

    asked = {}

    class Decisions:
        def __init__(self, model, timeout):
            asked["model"] = model

        def evaluate(self, text, questions):
            asked["choices"] = [c["value"] for c in questions[0]["choices"]]
            return {
                "answers": [
                    {"type": "choice", "choice": "entity_memory", "confidence": 0.8}
                ]
            }

    monkeypatch.setattr("memorizz.decisions.OpenAIDecisions", Decisions)
    answer = memory_routing._decisions("My sister lives in Porto. Ideas?", None)
    assert answer["memory_type"] == "entity_memory" and asked["model"] == "gpt-6-luna"
    assert "conversation_memory" not in asked["choices"]
    assert {"entity_memory", "knowledge_base", "do_not_store"} <= set(asked["choices"])


def test_reranked_candidates_carry_their_score_in_the_ledger():
    from memorizz.memagent.utils.context_dedup import dedupe_and_select

    ledger: List[Dict[str, Any]] = []
    rows = [
        ("knowledge_base", {"_id": "a", "content": "parking permits", "score": 0.9}),
        (
            "knowledge_base",
            {"_id": "b", "content": "refunds within 30 days", "score": 0.5},
        ),
    ]
    dedupe_and_select(
        rows,
        max_items=2,
        selection_ledger=ledger,
        rerank=lambda texts: [0.1 if "parking" in t else 0.95 for t in texts],
    )
    assert sorted(entry["rerank_score"] for entry in ledger) == [0.1, 0.95]
