"""Entity capture without memory routing, and the entity lookup offered only
when the user has entities to look up."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.long_term.semantic.entity_memory import entity_memory as entity_module
from memorizz.memagent import MemAgent
from memorizz.memagent.builders.agent_builder import MemAgentBuilder
from memorizz.memagent.memory_routing import MemoryRoutingConfig
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]
FACTS_MESSAGE = (
    "Numbered facts:\n1. Ines is allergic to peanuts.\n2. Ines lives in Porto."
)


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-entity-capture-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda text: list(VECTOR))
    monkeypatch.setattr(entity_module, "get_embedding", lambda text: list(VECTOR))
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")


@pytest.fixture()
def no_router(monkeypatch):
    """Entity capture never asks a memory router."""
    asked: List[str] = []

    def decide(text, model):
        asked.append(text)
        raise AssertionError("entity capture must not call the router")

    monkeypatch.setattr("memorizz.memagent.memory_routing._decisions", decide)
    return asked


class _Embedder:
    def get_embedding(self, text: str) -> List[float]:
        return list(VECTOR)


def _tool_call(name: str, arguments: Dict[str, Any], call_id: str) -> Any:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _tool_turn(*calls: tuple) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        _tool_call(name, args, cid) for cid, name, args in calls
                    ],
                )
            )
        ]
    )


class _Chat:
    """Chat model: plays scripted replies and records the tools it was shown."""

    model = "chat-stub"

    def __init__(self, replies: Optional[List[Any]] = None) -> None:
        self.replies = list(replies or [])
        self.tool_names: List[List[str]] = []
        self.systems: List[str] = []

    def generate(self, messages, tools=None, **kwargs):
        self.tool_names.append(sorted(item["function"]["name"] for item in tools or []))
        self.systems.append(
            next(
                (str(m.get("content")) for m in messages if m.get("role") == "system"),
                "",
            )
        )
        return self.replies.pop(0) if self.replies else "ok"

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        raise AssertionError("the chat model must not do memory upkeep")

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "openai", "model": "chat-stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


class _Background:
    """Background model: extracts Ines's facts, quoting the user's words."""

    model = "background-stub"

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        self.prompts.append(prompt)
        facts = []
        if "durable facts the text" in prompt and "allergic to peanuts" in prompt:
            facts = [
                {
                    "entity": "Ines",
                    "entity_type": "person",
                    "attribute": "allergy",
                    "value": "peanuts",
                    "evidence": "Ines is allergic to peanuts",
                },
                {
                    "entity": "Ines",
                    "entity_type": "person",
                    "attribute": "home_city",
                    "value": "Porto",
                    "evidence": "Ines lives in Porto",
                },
            ]
        return json.dumps({"facts": facts})

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "openai", "model": "background-stub"}

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


def _agent(work_dir: Path, chat: Optional[_Chat] = None, **kwargs) -> MemAgent:
    return MemAgent(
        model=chat or _Chat(),
        memory_provider=kwargs.pop("memory_provider", None) or _provider(work_dir),
        memory_ids=["memory-1"],
        auto_register=False,
        agent_id="agent-1",
        application_mode="assistant",
        **kwargs,
    )


def _facts(provider, memory_id="memory-1", user_id=None) -> Dict[str, str]:
    out = {}
    for record in EntityMemory(provider).list_entities(
        memory_id=memory_id, user_id=user_id
    ):
        for attribute in record.get("attributes") or []:
            out[f"{record['name']}.{attribute['name']}"] = attribute["value"]
    return out


# -- entity capture -------------------------------------------------------------


def test_facts_are_captured_without_routing_or_tool_calls(work_dir, no_router):
    """The evaluation's setup: entity memory and a background model, no
    routing, and a chat model that makes no entity tool call."""
    chat, background = _Chat(), _Background()
    agent = _agent(work_dir, chat).with_entity_memory(True)
    agent.with_background_model(background)
    assert agent.memory_routing.enabled is False
    assert agent.describe_memory()["entity_capture"] is True

    agent.run(FACTS_MESSAGE, memory_id="memory-1", thread_id="t1", user_id="u1")
    assert agent.wait_for_memory_upkeep(timeout=10)

    assert _facts(agent.memory_provider, user_id="u1") == {
        "Ines.allergy": "peanuts",
        "Ines.home_city": "Porto",
    }
    assert no_router == []
    # The chat model is told facts are saved for it.
    assert "saved automatically after each turn" in chat.systems[-1]
    record = EntityMemory(agent.memory_provider).get_entity_by_name(
        "Ines", memory_id="memory-1", user_id="u1"
    )
    assert {a["source"] for a in record["attributes"]} == {"entity_capture"}
    agent.close()


def test_capture_skips_what_the_chat_model_wrote_in_the_same_turn(work_dir, no_router):
    chat = _Chat(
        [
            _tool_turn(
                (
                    "call-1",
                    "entity_memory_upsert",
                    {
                        "name": "Ines",
                        "entity_type": "person",
                        "attributes": [
                            {"name": "allergy", "value": "severe peanut allergy"}
                        ],
                    },
                )
            ),
            "Noted.",
        ]
    )
    agent = _agent(work_dir, chat).with_background_model(_Background())
    agent.run(FACTS_MESSAGE, memory_id="memory-1", thread_id="t1", user_id="u1")
    assert agent.wait_for_memory_upkeep(timeout=10)

    # The chat model's own value stands; the fact it did not write is added.
    assert _facts(agent.memory_provider, user_id="u1") == {
        "Ines.allergy": "severe peanut allergy",
        "Ines.home_city": "Porto",
    }
    agent.close()


def test_capture_is_off_without_a_background_model_or_when_opted_out(
    work_dir, no_router
):
    chat = _Chat()
    agent = _agent(work_dir, chat)
    assert agent._turn_capture_mode() is None
    agent.run(FACTS_MESSAGE, memory_id="memory-1", thread_id="t1", user_id="u1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _facts(agent.memory_provider, user_id="u1") == {}
    assert "call 'entity_memory_upsert' with" in chat.systems[-1]

    background = _Background()
    agent.with_background_model(background).with_entity_memory(True, capture=False)
    assert agent._turn_capture_mode() is None
    assert agent.describe_memory()["entity_capture"] is False
    agent.run(FACTS_MESSAGE, memory_id="memory-1", thread_id="t2", user_id="u1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _facts(agent.memory_provider, user_id="u1") == {}
    assert not [p for p in background.prompts if "durable facts the text" in p]
    agent.close()


def test_capture_true_uses_the_chat_model_when_there_is_no_background_model(
    work_dir,
):
    agent = _agent(work_dir, _Background())
    agent.with_entity_memory(True, capture=True)
    assert agent._turn_capture_mode() == "entities"
    agent.run(FACTS_MESSAGE, memory_id="memory-1", thread_id="t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _facts(agent.memory_provider)["Ines.home_city"] == "Porto"
    agent.close()


def test_the_opt_out_is_saved_and_survives_routing_changes(work_dir):
    agent = _agent(work_dir).with_background_model(_Background())
    agent.with_entity_memory(True, capture=False)
    agent.with_memory_routing("llm", capture_turns=False)
    assert agent.memory_routing.entity_capture is False
    agent.with_memory_routing(False)
    assert agent.memory_routing.entity_capture is False
    assert agent.memory_routing.enabled is False

    agent.save()
    stored = agent.memory_provider.retrieve_memagent("agent-1")
    stored = stored if isinstance(stored, dict) else stored.model_dump()
    assert stored["memory_routing"] == {
        "enabled": False,
        "router": "auto",
        "capture_turns": True,
        "min_confidence": 0.5,
        "entity_capture": False,
    }
    loaded = MemAgent.load(
        "agent-1", memory_provider=agent.memory_provider, model=_Chat()
    )
    assert loaded.memory_routing.enabled is False
    assert loaded.memory_routing.entity_capture is False
    assert loaded._turn_capture_mode() is None
    agent.close()
    loaded.close()


def test_capture_settings_parse():
    assert MemoryRoutingConfig().entity_capture is None
    alone = MemoryRoutingConfig.from_value({"entity_capture": "off"})
    assert alone.enabled is False and alone.entity_capture is False
    assert (
        MemoryRoutingConfig.from_value({"entity_capture": "auto"}).entity_capture
        is None
    )
    routed = MemoryRoutingConfig.from_value({"router": "llm", "entity_capture": True})
    assert routed.enabled and routed.entity_capture is True
    assert "entity_capture" not in MemoryRoutingConfig.from_value(True).to_dict()
    builder = MemAgentBuilder().with_entity_memory(True, capture=False)
    assert builder._entity_capture is False


def test_routing_that_captures_turns_is_not_doubled_by_entity_capture(work_dir):
    agent = _agent(work_dir, memory_routing="llm").with_background_model(_Background())
    assert agent._turn_capture_mode() == "route"
    # Routing set up with turn capture off keeps turns uncaptured by default.
    agent.with_memory_routing("llm", capture_turns=False)
    assert agent._turn_capture_mode() is None
    agent.close()


def test_the_upsert_tool_reports_a_partial_vector(work_dir, monkeypatch):
    agent = _agent(work_dir)
    agent._current_memory_id = "memory-1"
    agent._current_user_id = "u1"

    def embed(text):
        if len(text) > 300:
            raise ValueError("This model's maximum context length is 512 tokens")
        return list(VECTOR)

    monkeypatch.setattr(entity_module, "get_embedding", embed)
    tool = agent.tool_manager.tools["entity_memory_upsert"]["function"]
    small = tool(name="Ines", attributes={"city": "Porto"})
    assert "embedding" not in small
    big = tool(
        name="Ines",
        attributes={f"fact_{i}": "a long detail " * 4 for i in range(12)},
    )
    assert big["embedding"]["status"] == "partial"
    assert big["embedding"]["reason"] == "provider_length_error"
    stored = EntityMemory(agent.memory_provider).get_entity(
        big["entity_id"], memory_id="memory-1", user_id="u1"
    )
    assert stored["metadata"]["embedding_status"] == "partial"
    agent.close()


# -- the lookup tool is offered only when there is something to look up ---------


def _tools(agent: MemAgent, memory_id: str = "memory-1", user_id=None) -> List[str]:
    agent._current_memory_id = memory_id
    agent._current_user_id = user_id
    return sorted(item["function"]["name"] for item in agent._build_llm_tools() or [])


def test_lookup_is_left_out_until_the_user_has_an_entity(work_dir):
    agent = _agent(work_dir)
    tools = _tools(agent, user_id="u1")
    assert "entity_memory_lookup" not in tools
    assert "entity_memory_upsert" in tools  # writes stay possible
    assert "call 'entity_memory_lookup'" not in agent._build_system_prompt()

    EntityMemory(agent.memory_provider).upsert_entity(
        name="Ines",
        attributes=[{"name": "city", "value": "Porto"}],
        memory_id="memory-1",
        user_id="u1",
    )
    assert "entity_memory_lookup" in _tools(agent, user_id="u1")
    assert "call 'entity_memory_lookup'" in agent._build_system_prompt()
    agent.close()


def test_the_decision_is_per_user_and_per_memory(work_dir):
    agent = _agent(work_dir)
    EntityMemory(agent.memory_provider).upsert_entity(
        name="Ines",
        attributes=[{"name": "city", "value": "Porto"}],
        memory_id="memory-1",
        user_id="u1",
    )
    assert "entity_memory_lookup" in _tools(agent, user_id="u1")
    assert "entity_memory_lookup" not in _tools(agent, user_id="u2")
    assert "entity_memory_lookup" not in _tools(agent, user_id=None)
    assert "entity_memory_lookup" not in _tools(agent, "memory-2", user_id="u1")
    agent.close()


def test_the_check_is_cached_and_a_write_elsewhere_shows_after_the_ttl(
    work_dir, monkeypatch
):
    provider = _provider(work_dir)
    agent = _agent(work_dir, memory_provider=provider)
    queries: List[Any] = []
    original = provider.retrieve_by_query

    def counting(query, *args, **kwargs):
        if query == {}:
            queries.append(kwargs.get("user_id"))
        return original(query, *args, **kwargs)

    monkeypatch.setattr(provider, "retrieve_by_query", counting)
    clock = {"now": 1000.0}
    monkeypatch.setattr(entity_module.time, "monotonic", lambda: clock["now"])

    assert "entity_memory_lookup" not in _tools(agent, user_id="u1")
    assert "entity_memory_lookup" not in _tools(agent, user_id="u1")
    assert agent._entity_lookup_offered() is False
    assert queries == ["u1"]  # one query, then the cached answer

    # Another process writes straight to the store: seen once the "no" expires.
    provider.store(
        {
            "entity_id": "e-other",
            "name": "Bo",
            "attributes": [],
            "memory_id": "memory-1",
            "user_id": "u1",
        },
        "entity_memory",
    )
    assert "entity_memory_lookup" not in _tools(agent, user_id="u1")
    clock["now"] += entity_module.EMPTY_SCOPE_TTL_SECONDS + 1
    assert "entity_memory_lookup" in _tools(agent, user_id="u1")
    assert queries == ["u1", "u1"]
    agent.close()


def test_a_write_in_a_turn_offers_the_lookup_on_the_next_turn(work_dir):
    chat = _Chat(
        [
            _tool_turn(
                (
                    "call-1",
                    "entity_memory_upsert",
                    {
                        "name": "Ines",
                        "attributes": [{"name": "city", "value": "Porto"}],
                    },
                )
            ),
            "Noted.",
            "Ines lives in Porto.",
        ]
    )
    agent = _agent(work_dir, chat)
    agent.run(
        "Ines lives in Porto.", memory_id="memory-1", thread_id="t1", user_id="u1"
    )
    agent.run(
        "Where does Ines live?", memory_id="memory-1", thread_id="t1", user_id="u1"
    )

    first_turn, _same_turn, second_turn = chat.tool_names
    assert "entity_memory_lookup" not in first_turn
    assert "entity_memory_upsert" in first_turn
    assert "entity_memory_lookup" in second_turn
    # Another user of the same agent still has nothing to look up.
    agent.run("Hello", memory_id="memory-1", thread_id="t2", user_id="u2")
    assert "entity_memory_lookup" not in chat.tool_names[-1]
    agent.close()
