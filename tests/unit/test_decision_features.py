"""The agent features a decision model answers: tool selection, read
routing, the semantic cache verifier, the extraction gate with procedures,
write dedupe and consolidation relations, and the saved setting."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from memorizz import decision_models as dm
from memorizz.enums import MemoryType
from memorizz.learning.consolidation import (
    LINK_ROWS,
    SUPPRESS_DUPLICATE,
    ConsolidationConfig,
    MemoryConsolidator,
)
from memorizz.memagent import MemAgent
from memorizz.memagent.managers.tool_manager import ToolManager
from memorizz.memagent.persistence import _refresh_policies
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.retrieval import RetrievalPolicy
from memorizz.short_term_memory.semantic_cache import SemanticCache, SemanticCacheConfig
from memorizz.tooling import ContextPolicy, SemanticToolRouter

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-decisions-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    for target in (
        "memorizz.embeddings.get_embedding",
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        "memorizz.long_term.procedural.skillbox.skill.get_embedding",
    ):
        monkeypatch.setattr(target, lambda text: list(VECTOR))
    monkeypatch.setattr(
        "memorizz.long_term.semantic.knowledge_base._embed_chunks",
        lambda chunks: [list(VECTOR) for _ in chunks],
    )
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


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


def _answers(questions: Dict[str, Any], **values: Any) -> Dict[str, Any]:
    out = {}
    for name, question in questions.items():
        value = values[name]
        if question["type"] == "noul":
            out[name] = {"type": "noul", "noul": value}
        else:
            out[name] = {"type": "choice", "probabilities": value}
    return out


DEPLOY = "To deploy, run the tests, tag the release, then post in #releases."


class _Model:
    """Answers "ok"; as the background model it extracts the deploy
    procedure, and records every prompt."""

    model = "stub"

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        self.prompts.append(prompt)
        if "the facts the text below states" in prompt and "To deploy" in prompt:
            steps = (
                ["run the tests", "tag the release", "post in #deploys"]
                if "#deploys" in prompt
                else ["run the tests", "tag the release", "post in #releases"]
            )
            return json.dumps(
                {
                    "facts": [],
                    "procedures": [
                        {
                            "name": "Deploy a release",
                            "when": "When the user asks to deploy",
                            "steps": steps,
                            "evidence": "run the tests, tag the release",
                        },
                        {  # not in the text: dropped
                            "name": "Roll back",
                            "steps": ["revert"],
                            "evidence": "roll back the deploy",
                        },
                    ],
                }
            )
        return '{"facts": [], "procedures": []}'

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "openai", "model": "stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


def _agent(work_dir: Path, **kwargs) -> MemAgent:
    kwargs.setdefault("model", _Model())
    kwargs.setdefault("memory_provider", _provider(work_dir))
    kwargs.setdefault("memory_ids", ["memory-1"])
    kwargs.setdefault("auto_register", False)
    kwargs.setdefault("agent_id", "agent-1")
    kwargs.setdefault("application_mode", "assistant")
    return MemAgent(**kwargs)


def _knowledge(provider, namespace: str) -> List[Dict[str, Any]]:
    return [
        row
        for row in provider.list_all(MemoryType.KNOWLEDGE_BASE)
        if row.get("namespace") == namespace
    ]


# -- tool selection ----------------------------------------------------------------


def generate_report(quarter: str) -> str:
    """Create the quarterly report document."""
    return quarter


def send_email(to: str, attachment: str) -> str:
    """Send an email to a person, with optional attachments."""
    return to


def get_weather(city: str) -> str:
    """Get the weather forecast for a city."""
    return city


class _Toolbox:
    """Ranked by topic: the report tool first for the email request."""

    def get_most_similar_tools(self, query, limit=25, **kwargs):
        names = ["generate_report", "get_weather", "send_email"]
        return [
            {"name": name, "score": 0.6 - index * 0.1, "user_id": None}
            for index, name in enumerate(names)
        ]


def _router(selector=None, **options) -> SemanticToolRouter:
    manager = ToolManager()
    for tool in (generate_report, send_email, get_weather):
        manager.add_tool(tool)
    router = SemanticToolRouter(
        manager, toolbox=_Toolbox(), top_k=3, selector=selector, **options
    )
    router.begin_turn()
    return router


def test_a_decision_model_picks_the_tool_or_none():
    asked: List[Dict[str, str]] = []

    def selector(query, options):
        asked.append(dict(options))
        if "Sarah" in query:
            return {"send_email": 0.9, "generate_report": 0.05, "none": 0.05}
        return {"none": 0.8, "get_weather": 0.1, "send_email": 0.1}

    router = _router(selector)
    assert router.preview("Send the Q3 report to Sarah") == ["send_email"]
    assert "Send an email" in asked[0]["send_email"]
    assert router.preview("What is 2 + 2?") == []
    # Without a selector the closest tool by topic comes first.
    assert _router().preview("Send the Q3 report to Sarah")[0] == "generate_report"


def test_a_close_runner_up_is_offered_and_a_failure_keeps_the_closest():
    router = _router(lambda query, options: {"send_email": 0.5, "generate_report": 0.4})
    assert router.preview("Send the Q3 report") == ["send_email", "generate_report"]

    def broken(query, options):
        raise RuntimeError("down")

    assert _router(broken).preview("Send the Q3 report")[0] == "generate_report"
    assert _router(lambda q, o: None).preview("Send the Q3 report")[0] == (
        "generate_report"
    )


def test_a_selector_turns_the_whole_list_shortcut_off():
    router = _router(lambda query, options: {"none": 1.0}, stable_under_tokens=100_000)
    names = {schema["function"]["name"] for schema in router.schemas_for_turn("hi")}
    assert {"generate_report", "send_email", "get_weather"}.isdisjoint(names)
    without = _router(stable_under_tokens=100_000)
    assert "send_email" in {
        schema["function"]["name"] for schema in without.schemas_for_turn("hi")
    }


def test_the_agent_wires_tool_selection(work_dir):
    def decide(state, questions):
        options = questions["tool"]["criteria"]
        return _answers(
            questions,
            tool={
                name: (0.9 if name == "none" else 0.1 / (len(options) - 1))
                for name in options
            },
        )

    agent = _agent(work_dir, context_policy={"tool_selection": decide})
    router = agent.semantic_tool_router
    assert router.selector is not None
    assert agent._select_tools("hi", {"get_weather": "forecast"})["none"] == (
        pytest.approx(0.9)
    )
    assert agent.context_policy.to_dict()["tool_selection"] == "custom:decide"
    _refresh_policies(agent, None, None, None, {})
    assert router.selector is None
    agent.close()


# -- read routing ---------------------------------------------------------------------


def _spy_reads(agent: MemAgent) -> List[Any]:
    reads: List[Any] = []
    original = agent.memory_manager.retrieve_relevant_memories

    def spy(**kwargs):
        reads.append((kwargs["memory_type"], kwargs.get("namespace")))
        return original(**kwargs)

    agent.memory_manager.retrieve_relevant_memories = spy
    return reads


def _routes(**values):
    def decide(state, questions):
        return _answers(questions, **values)

    return decide


def test_read_routing_searches_only_the_memory_a_request_needs(work_dir):
    policy = RetrievalPolicy(
        workflow_scope="memory",
        read_routing=_routes(episodic=0.1, semantic=0.9, procedural=0.2),
    )
    agent = _agent(
        work_dir,
        memory_types=[
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.KNOWLEDGE_BASE,
            MemoryType.WORKFLOW_MEMORY,
        ],
        retrieval_policy=policy,
    )
    reads = _spy_reads(agent)
    agent._build_context("What is my dietary need?", "memory-1")
    assert reads == [(MemoryType.KNOWLEDGE_BASE, None)]
    assert agent._last_retrieval_stats["read_routes"] == {
        "episodic": 0.1,
        "semantic": 0.9,
        "procedural": 0.2,
    }
    agent.close()


def test_a_procedural_request_reads_stated_procedures(work_dir):
    policy = RetrievalPolicy(
        read_routing=_routes(episodic=0.0, semantic=0.1, procedural=0.95)
    )
    agent = _agent(
        work_dir,
        memory_types=[MemoryType.CONVERSATION_MEMORY, MemoryType.KNOWLEDGE_BASE],
        retrieval_policy=policy,
    )
    reads = _spy_reads(agent)
    agent._build_context("Deploy it the usual way", "memory-1")
    assert reads == [(MemoryType.KNOWLEDGE_BASE, "procedures")]
    agent.close()


def test_without_an_answer_every_store_is_searched(work_dir):
    def broken(state, questions):
        raise RuntimeError("down")

    for setting in (broken, None):
        agent = _agent(
            work_dir,
            memory_types=[MemoryType.CONVERSATION_MEMORY, MemoryType.KNOWLEDGE_BASE],
            retrieval_policy=RetrievalPolicy(read_routing=setting),
        )
        reads = _spy_reads(agent)
        agent._build_context("anything", "memory-1")
        assert {memory_type for memory_type, _ in reads} == {
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.KNOWLEDGE_BASE,
        }
        assert "read_routes" not in agent._last_retrieval_stats
        agent.close()


def test_read_routing_is_saved_with_the_policy():
    policy = RetrievalPolicy(read_routing="d1", read_threshold=0.3)
    assert policy.to_dict()["read_routing"] == "d1"
    assert RetrievalPolicy.from_value(policy.to_dict()) == policy
    assert "read_routing" not in RetrievalPolicy().to_dict()
    with pytest.raises(ValueError, match="read_threshold"):
        RetrievalPolicy(read_threshold=2)


# -- the semantic cache verifier ---------------------------------------------------------


class _CacheEmbedder:
    """Every Lisbon weather question is a near neighbour (0.98)."""

    def get_embedding(self, text: str) -> List[float]:
        return [1.0, 0.2, 0.0] if "tomorrow" in text else [1.0, 0.0, 0.0]


def _cache(verifier=None, **options) -> SemanticCache:
    cache = SemanticCache(
        SemanticCacheConfig(
            verifier=verifier, enable_memory_provider_sync=False, **options
        ),
        embedding_manager=_CacheEmbedder(),
        agent_id="agent-1",
    )
    cache.set("What's the weather in Lisbon today?", "Sunny, 21C.")
    return cache


def test_the_verifier_turns_away_a_near_miss_and_serves_a_paraphrase():
    seen: List[str] = []

    def verifier(state, questions):
        seen.append(state)
        serves = "tomorrow" not in state.split("Cached answer:")[0]
        return _answers(questions, q=0.95 if serves else 0.02)

    cache = _cache(verifier)
    assert cache.config.similarity_threshold == 0.7
    assert cache.get("What's the weather in Lisbon tomorrow?") is None
    assert cache.get("How is the weather in Lisbon today") == "Sunny, 21C."
    assert cache.last_hit["verified"] == 0.95
    assert seen[0].startswith("New question: What's the weather in Lisbon tomorrow?")
    # Without a verifier the near miss is served at the old 0.78 line.
    loose = _cache(similarity_threshold=0.78)
    assert loose.get("What's the weather in Lisbon tomorrow?") == "Sunny, 21C."


def test_a_repeated_question_needs_no_check_and_a_failure_serves_nothing():
    def broken(state, questions):
        raise RuntimeError("down")

    cache = _cache(broken)
    assert cache.get("what's the weather in lisbon today") == "Sunny, 21C."
    assert cache.get("How is the weather in Lisbon today") is None
    assert SemanticCacheConfig().similarity_threshold == 0.9


def test_the_agent_verifies_with_its_decision_model(work_dir):
    def decide(state, questions):
        return _answers(questions, q=0.9)

    agent = _agent(
        work_dir,
        decision_model=decide,
        semantic_cache=True,
        semantic_cache_config={"verifier": True},
    )
    agent._wire_cache_verifier()
    cache = agent.cache_manager.cache_instance
    assert cache.verify == agent._verify_cached_answer
    entry = type("Entry", (), {"response": "Sunny"})()
    assert agent._verify_cached_answer("Weather today?", entry) == 0.9
    from memorizz.memagent.persistence import _serialize_semantic_cache_config

    assert _serialize_semantic_cache_config(agent)["verifier"] is True
    agent.close()


# -- the extraction gate and procedures -----------------------------------------------------


def test_procedures_go_to_the_knowledge_base_and_a_restatement_replaces(work_dir):
    model = _Model()
    agent = _agent(work_dir, background_model=model)
    agent.run(DEPLOY, memory_id="memory-1", thread_id="t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    rows = _knowledge(agent.memory_provider, "procedures")
    assert [row["content"] for row in rows] == [
        "Procedure: Deploy a release (when the user asks to deploy). 1. run the "
        "tests. 2. tag the release. 3. post in #releases."
    ]
    agent.run(
        "To deploy, run the tests, tag the release, then post in #deploys.",
        memory_id="memory-1",
        thread_id="t1",
    )
    assert agent.wait_for_memory_upkeep(timeout=10)
    rows = _knowledge(agent.memory_provider, "procedures")
    visible = [row for row in rows if row.get("retention_state") != "suppressed"]
    assert len(rows) == 2 and len(visible) == 1
    assert "#deploys" in visible[0]["content"]
    hidden = next(row for row in rows if row.get("retention_state") == "suppressed")
    assert hidden["supersede_reason"] == "restated"
    agent.close()


def test_procedures_become_skills_when_the_agent_reads_skills(work_dir):
    agent = _agent(work_dir, background_model=_Model(), skill_retrieval=True)
    agent.run(DEPLOY, memory_id="memory-1", thread_id="t1", user_id="ada")
    agent.run(
        "To deploy, run the tests, tag the release, then post in #deploys.",
        memory_id="memory-1",
        thread_id="t1",
        user_id="ada",
    )
    assert agent.wait_for_memory_upkeep(timeout=10)
    skills = {
        (skill.version, skill.status.value): skill
        for skill in agent.skillbox.list_skills()
        if skill.name == "Deploy a release"
    }
    assert set(skills) == {(1, "deprecated"), (2, "active")}
    active = skills[(2, "active")]
    assert active.user_id == "ada" and "#deploys" in active.content
    assert active.baseline["stated_by_user"] is True
    assert _knowledge(agent.memory_provider, "procedures") == []
    agent.close()


def test_a_sealed_conversation_keeps_no_procedure(work_dir):
    agent = _agent(
        work_dir, background_model=_Model(), retrieval_policy=RetrievalPolicy.sealed()
    )
    agent.run(DEPLOY, memory_id="memory-1", thread_id="t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert _knowledge(agent.memory_provider, "procedures") == []
    agent.close()


def test_the_gate_skips_a_message_with_nothing_to_keep(work_dir):
    asked: List[str] = []

    def gate(state, questions):
        asked.append(state)
        durable = 0.9 if "deploy" in state.lower() else 0.05
        return _answers(
            questions,
            durable=durable,
            kind={"procedure": 0.9, "fact": 0.05, "preference": 0.03, "event": 0.02},
        )

    model = _Model()
    agent = _agent(
        work_dir, background_model=model, memory_routing={"extraction_gate": gate}
    )
    assert agent.memory_routing.enabled is False  # the gate alone
    agent.run("lol ok so anyway", memory_id="memory-1", thread_id="t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert asked == ["lol ok so anyway"] and model.prompts == []
    assert agent._last_capture_gate["skipped"] is True
    agent.run(DEPLOY, memory_id="memory-1", thread_id="t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert len(model.prompts) == 1
    assert len(_knowledge(agent.memory_provider, "procedures")) == 1
    agent.close()


def test_a_procedure_skips_the_router(work_dir, monkeypatch):
    routed: List[str] = []
    monkeypatch.setattr(
        "memorizz.memagent.memory_routing._decisions",
        lambda text, model: routed.append(text)
        or {"memory_type": "knowledge_base", "confidence": 0.9},
    )

    def gate(state, questions):
        return _answers(
            questions,
            durable=0.9,
            kind={"procedure": 0.85, "fact": 0.05, "preference": 0.05, "event": 0.05},
        )

    agent = _agent(
        work_dir,
        background_model=_Model(),
        memory_routing={"router": "openai_decisions", "extraction_gate": gate},
    )
    agent.run(DEPLOY, memory_id="memory-1", thread_id="t1")
    assert agent.wait_for_memory_upkeep(timeout=10)
    assert routed == []
    assert len(_knowledge(agent.memory_provider, "procedures")) == 1
    assert _knowledge(agent.memory_provider, "remembered") == []
    agent.close()


def test_the_gate_is_saved_with_routing():
    from memorizz.memagent.memory_routing import MemoryRoutingConfig

    config = MemoryRoutingConfig(extraction_gate="d1", gate_threshold=0.4)
    assert config.worth_saving and config.to_dict()["extraction_gate"] == "d1"
    assert MemoryRoutingConfig.from_value(config.to_dict()) == config


# -- write dedupe -------------------------------------------------------------------------


def _relations(state, questions):
    new, stored = state.split("\n\nStored memory: ")
    if "Leeds" in new and "London" in stored:
        relation = "contradicts"
    elif "Leeds" in new and "Leeds" in stored:
        relation = "same"
    else:
        relation = "complements"
    return _answers(
        questions,
        r={key: (0.9 if key == relation else 0.025) for key in dm.RELATIONS},
    )


def test_remembering_compares_with_what_is_stored(work_dir, monkeypatch):
    monkeypatch.setattr(
        "memorizz.memagent.memory_routing._decisions",
        lambda text, model: {"memory_type": "knowledge_base", "confidence": 0.9},
    )
    agent = _agent(
        work_dir,
        memory_routing={"router": "openai_decisions", "dedupe_writes": _relations},
    )
    first = agent.remember("Office policy: the office is in London.")
    assert first["stored"] and "relations" not in first
    moved = agent.remember("Office policy: the office moved to Leeds.")
    assert moved["stored"] and moved["relations"][0]["relation"] == "contradicts"
    rows = {
        row["content"]: row for row in _knowledge(agent.memory_provider, "remembered")
    }
    old = rows["Office policy: the office is in London."]
    assert old["retention_state"] == "suppressed"
    assert old["superseded_by"] == moved["knowledge_base_id"]
    again = agent.remember("Office policy: our office is in Leeds now.")
    assert again["stored"] is False and again["already_known"]
    assert len(_knowledge(agent.memory_provider, "remembered")) == 2
    extra = agent.remember("Office policy: the office opens at 9.")
    assert extra["stored"] and extra["related_ids"]
    agent.close()


# -- consolidation relations ------------------------------------------------------------


def _note(provider, text, *, at, vector=None, **extra):
    return provider.store(
        {
            "content": text,
            "embedding": vector or [1.0, 0.0, 0.0],
            "namespace": "remembered",
            "memory_id": "m1",
            "user_id": None,
            "knowledge_base_id": f"kb-{at}",
            "created_at": f"2026-11-0{at}T10:00:00+00:00",
            **extra,
        },
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )


def test_consolidation_relates_the_agents_own_notes(work_dir):
    provider = _provider(work_dir)
    london = _note(provider, "The office is in London.", at=1)
    hours = _note(provider, "The office opens at 9.", at=2)
    leeds = _note(provider, "The office moved to Leeds.", at=3)
    leeds_again = _note(provider, "Our office is in Leeds now.", at=4)
    far = _note(provider, "Lunch is at noon.", at=5, vector=[0.0, 1.0, 0.0])
    consolidator = MemoryConsolidator(
        provider, config=ConsolidationConfig(relations=_relations)
    )
    plan = consolidator.plan(memory_id="m1", entities=False)
    by_target = {(op.kind, op.target_id): op for op in plan.ops}
    assert (SUPPRESS_DUPLICATE, leeds_again) in by_target  # says the same
    assert by_target[(SUPPRESS_DUPLICATE, leeds_again)].details["duplicate_of"] == leeds
    assert (SUPPRESS_DUPLICATE, london) in by_target  # no longer true
    assert by_target[(SUPPRESS_DUPLICATE, london)].details["relation"] == "contradicts"
    assert (LINK_ROWS, leeds) in by_target or (LINK_ROWS, hours) in by_target
    assert all(op.target_id != far for op in plan.ops)  # too far to ask about

    result = consolidator.apply(plan, approved_by="tester")
    assert not result.errors and result.applied[SUPPRESS_DUPLICATE] == 2
    row = provider.retrieve_by_id(london, MemoryType.KNOWLEDGE_BASE)
    assert row["retention_state"] == "suppressed" and row["superseded_by"] == leeds
    counts = consolidator.restore(plan.plan_id, memory_id="m1", approved_by="tester")
    assert counts["rows"] >= 2
    row = provider.retrieve_by_id(london, MemoryType.KNOWLEDGE_BASE)
    assert row["retention_state"] == "active"


def test_an_unsure_relation_changes_nothing(work_dir):
    provider = _provider(work_dir)
    _note(provider, "The office is in London.", at=1)
    _note(provider, "The office moved to Leeds.", at=2)

    def unsure(state, questions):
        return _answers(
            questions,
            r={key: (0.4 if key == "contradicts" else 0.15) for key in dm.RELATIONS},
        )

    consolidator = MemoryConsolidator(
        provider, config=ConsolidationConfig(relations=unsure)
    )
    assert consolidator.plan(memory_id="m1", entities=False).ops == ()


def test_relations_are_off_unless_asked_and_saved_by_name():
    assert ConsolidationConfig().relations is None
    config = ConsolidationConfig.from_value({"relations": _relations})
    assert config.to_dict()["relations"] == "custom:_relations"
    assert ConsolidationConfig.from_value({"relations": True}).to_dict()["relations"]


# -- the agent's setting --------------------------------------------------------------------


def test_the_decision_model_is_saved_and_restored(work_dir):
    provider = _provider(work_dir)
    agent = _agent(work_dir, memory_provider=provider, decision_model="jev")
    assert agent.describe_memory()["decision_model"] == {"backend": "jev"}
    agent.save()
    loaded = MemAgent.load("agent-1", memory_provider=provider, model=_Model())
    assert loaded.decision_model.backend == "jev"
    # True on a feature means this model.
    assert loaded._decider(True, "test").backend == "jev"
    loaded.with_decision_model(None)
    assert loaded.decision_model is None
    agent.close()
    loaded.close()


def test_an_unusable_model_is_skipped_once(work_dir, caplog):
    agent = _agent(work_dir, retrieval_policy=RetrievalPolicy(read_routing=True))
    agent.model = None  # no OpenAI key and no model: "auto" has nothing
    with caplog.at_level("WARNING"):
        assert agent._decider(True, "read routing") is None
        assert agent._decider(True, "read routing") is None
    assert sum("read routing" in r.message for r in caplog.records) == 1
    agent.close()
