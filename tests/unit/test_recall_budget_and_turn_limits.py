"""Recall character budgets, the per-turn tool-call signal, and plain
retrieved ids for retrieval evaluation."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memagent.builders.agent_builder import MemAgentBuilder
from memorizz.memagent.managers.tool_manager import ToolManager
from memorizz.memagent.utils.context_dedup import RetrievalScoring, dedupe_and_select
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability.overview import build_operations_overview
from memorizz.retrieval import RetrievalPolicy
from memorizz.tooling import ContextPolicy, SemanticToolRouter

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-recall-budget-"))
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
    """Records every request; answers with ``replies`` in order, then "ok"."""

    model = "stub"

    def __init__(self, replies: Optional[List[Any]] = None):
        self.replies = list(replies or [])
        self.requests: List[List[Dict[str, Any]]] = []

    def generate(self, messages, tools=None, **kwargs):
        self.requests.append([dict(item) for item in messages])
        return self.replies.pop(0) if self.replies else "ok"

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        return "ok"

    def get_config(self):
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
    return MemAgent(**kwargs)


def _prompt_text(model: _Model) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for request in model.requests
        for message in request
    )


def _recall(agent: MemAgent, rows_by_type: Dict[MemoryType, List[Dict[str, Any]]]):
    agent.memory_manager.retrieve_relevant_memories = lambda **kwargs: [
        dict(row) for row in rows_by_type.get(kwargs.get("memory_type"), [])
    ]
    agent.active_memory_types = list(rows_by_type)


# -- 12. the per-item cap and the total recall budget ---------------------------


def test_the_defaults_keep_the_600_character_cap_and_the_saved_record():
    policy = RetrievalPolicy()
    assert policy.max_chars_per_item == 600
    assert policy.max_recall_chars is None
    # Older records compare equal: the defaults stay out of the saved dict.
    assert "max_chars_per_item" not in policy.to_dict()
    assert "max_recall_chars" not in policy.to_dict()


def test_the_caps_round_trip_and_none_means_whole_items():
    policy = RetrievalPolicy(max_chars_per_item=None, max_recall_chars=4000)
    saved = policy.to_dict()
    assert saved["max_chars_per_item"] is None
    assert saved["max_recall_chars"] == 4000
    assert RetrievalPolicy.from_value(saved) == policy
    assert (
        RetrievalPolicy.from_value({"max_chars_per_item": 2000}).max_chars_per_item
        == 2000
    )
    for bad in (0, -5, True):
        with pytest.raises(ValueError):
            RetrievalPolicy(max_chars_per_item=bad)
        with pytest.raises(ValueError):
            RetrievalPolicy(max_recall_chars=bad)


def _row(row_id: str, text: str, *, score: float, minutes: int) -> Dict[str, Any]:
    stamp = datetime(2026, 10, 1, 12, minutes, tzinfo=timezone.utc).isoformat()
    return {"_id": row_id, "content": text, "score": score, "timestamp": stamp}


def test_dedupe_cuts_items_to_600_characters_by_default():
    long_text = "a" * 2000
    [item] = dedupe_and_select(
        [("episodic", _row("r1", long_text, score=0.9, minutes=1))]
    )
    assert len(item["text"]) == 600 and item["text"].endswith("…")
    assert item["truncated"] is True
    [whole] = dedupe_and_select(
        [("episodic", _row("r1", long_text, score=0.9, minutes=1))],
        max_chars_per_item=None,
    )
    assert whole["text"] == long_text and "truncated" not in whole


def test_a_total_budget_serves_the_best_items_first_and_keeps_time_order():
    rows = [
        ("episodic", _row("old-best", "b" * 1000, score=0.99, minutes=1)),
        ("episodic", _row("mid", "m" * 1000, score=0.80, minutes=2)),
        ("episodic", _row("new-weak", "w" * 1000, score=0.10, minutes=3)),
    ]
    ledger: List[Dict[str, Any]] = []
    # Rank by relevance alone so the order of service is plain to read.
    by_relevance = RetrievalScoring(alpha_recency=0.0, alpha_importance=0.0)
    selected = dedupe_and_select(
        rows,
        max_items=3,
        max_chars_per_item=None,
        max_total_chars=1500,
        scoring=by_relevance,
        selection_ledger=ledger,
    )
    # The best item is whole, the next takes the room left, the weakest
    # does not fit and is dropped.
    assert [item["id"] for item in selected] == ["old-best", "mid"]
    assert selected[0]["text"] == "b" * 1000
    assert len(selected[1]["text"]) == 500 and selected[1]["truncated"] is True
    assert sum(len(item["text"]) for item in selected) <= 1500
    reasons = {entry["candidate_rank"]: entry["selection_reason"] for entry in ledger}
    assert reasons[3] == "budget_exceeded"
    # The same input renders the same block (prompt-cache safe).
    again = dedupe_and_select(
        rows,
        max_items=3,
        max_chars_per_item=None,
        max_total_chars=1500,
        scoring=by_relevance,
    )
    assert [(item["id"], item["text"]) for item in again] == [
        (item["id"], item["text"]) for item in selected
    ]


def test_the_agent_threads_the_caps_into_recall(work_dir):
    long_text = "The assistant said: " + "detail " * 300  # ~2,100 characters
    rows = {
        MemoryType.CONVERSATION_MEMORY: [
            {
                "_id": "turn-7",
                "thread_id": "session-3",
                "role": "assistant",
                "content": long_text,
                "embedding": [0.0, 1.0, 0.0],
                "score": 0.9,
            }
        ]
    }

    default_model = _Model()
    agent = _agent(work_dir, model=default_model)
    _recall(agent, rows)
    agent.run("what did you say?", memory_id="memory-1", thread_id="now")
    assert long_text.strip()[:599] in _prompt_text(default_model)
    assert long_text.strip() not in _prompt_text(default_model)
    agent.close()

    whole_model = _Model()
    agent = _agent(
        work_dir,
        model=whole_model,
        agent_id="agent-2",
        retrieval_policy=RetrievalPolicy(max_chars_per_item=None),
    )
    _recall(agent, rows)
    agent.run("what did you say?", memory_id="memory-1", thread_id="now")
    assert long_text.strip() in _prompt_text(whole_model)
    agent.close()

    budget_model = _Model()
    agent = _agent(
        work_dir,
        model=budget_model,
        agent_id="agent-3",
        retrieval_policy=RetrievalPolicy(
            max_chars_per_item=None, max_recall_chars=1000
        ),
    )
    _recall(agent, rows)
    agent.run("what did you say?", memory_id="memory-1", thread_id="now")
    [item] = agent.last_retrieval_evidence()["items"]
    assert len(item["text"]) == 1000 and item["truncated"] is True
    agent.close()


def test_the_caps_are_saved_with_the_agent(work_dir):
    provider = _provider(work_dir)
    policy = RetrievalPolicy(max_chars_per_item=None, max_recall_chars=6000)
    agent = (
        MemAgentBuilder()
        .with_name("Recall budget")
        .with_model(_Model())
        .with_memory_provider(provider)
        .with_retrieval_policy(policy)
        .build_and_save()
    )
    assert provider.retrieve_memagent(agent.agent_id).retrieval_policy == {
        **RetrievalPolicy().to_dict(),
        "max_chars_per_item": None,
        "max_recall_chars": 6000,
    }
    loaded = MemAgent.load(agent.agent_id, memory_provider=provider)
    assert loaded.retrieval_policy == policy
    agent.close()
    loaded.close()


def test_the_agent_form_keeps_the_caps_when_it_saves_other_settings():
    from memorizz.ui.routers.agents_crud import (
        _retrieval_policy_with_forgetting,
        _retrieval_policy_with_privacy,
    )

    existing = RetrievalPolicy(max_chars_per_item=1500, max_recall_chars=5000)
    merged = _retrieval_policy_with_privacy(
        _retrieval_policy_with_forgetting(existing, {}), True, None
    )
    restored = RetrievalPolicy.from_value(merged)
    assert (restored.max_chars_per_item, restored.max_recall_chars) == (1500, 5000)
    assert restored.is_sealed


# -- 13. the per-turn tool-call budget is reported --------------------------------


def test_the_router_reports_what_is_left_and_why_it_refused():
    manager = ToolManager()

    def save_fact(fact: str) -> str:
        """Save one fact."""
        return "saved"

    manager.add_tool(save_fact)
    router = SemanticToolRouter(manager, max_invocations_per_turn=3)
    router.begin_turn()
    left = [
        router.prepare_invocation("save_fact", {"fact": str(index)})["remaining"]
        for index in range(3)
    ]
    assert left == [2, 1, 0]
    refused = router.prepare_invocation("save_fact", {"fact": "late"})
    assert refused["error_code"] == "tool_budget_exhausted"
    assert refused["tool_call_limit"] == 3
    assert "did not run" in refused["error"]
    assert router.budget_note(None) == ""
    assert router.budget_note(3) == ""
    assert "2 tool calls left" in router.budget_note(2)
    assert "last of this turn's 3 tool calls" in router.budget_note(0)


def _tool_calls(count: int):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=f"call-{index}",
                            function=SimpleNamespace(
                                name="save_fact",
                                arguments=json.dumps({"fact": f"fact {index}"}),
                            ),
                        )
                        for index in range(count)
                    ],
                )
            )
        ]
    )


def test_a_turn_that_runs_out_of_tool_calls_says_so(work_dir):
    saved: List[str] = []

    def save_fact(fact: str) -> str:
        """Save one fact the user shared."""
        saved.append(fact)
        return "saved"

    model = _Model([_tool_calls(25), "Saved 20 facts; 5 are left."])
    agent = _agent(
        work_dir,
        model=model,
        context_policy=ContextPolicy(max_tool_invocations_per_turn=20),
    )
    agent.tool_manager.add_tool(save_fact)
    events: List[Dict[str, Any]] = []
    agent.set_stream_event_callback(lambda event: events.append(dict(event)))

    agent.run("remember these 25 facts", memory_id="memory-1", thread_id="t1")

    assert len(saved) == 20
    assert agent.last_turn_limits == {
        "tool_budget_exhausted": True,
        "refused_tool_calls": 5,
        "refused_tools": {"save_fact": 5},
        "tool_call_limit": 20,
    }
    refused = [
        item
        for item in agent.last_tool_outcomes
        if item.get("reason_code") == "tool_budget_exhausted"
    ]
    assert len(refused) == 5

    # The model saw the budget running out and the refusals.
    tool_messages = [
        message for message in model.requests[-1] if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 25
    assert "[Tool budget]" not in tool_messages[16]["content"]
    assert "2 tool calls left" in tool_messages[17]["content"]
    assert "last of this turn's 20 tool calls" in tool_messages[19]["content"]
    assert all("did not run" in message["content"] for message in tool_messages[20:])

    # The turn's trace says so, and the field survives persistence.
    [result] = [event for event in events if event.get("trace_kind") == "turn_result"]
    assert result["tool_budget_exhausted"] is True
    assert result["refused_tool_calls"] == 5
    assert result["tool_call_limit"] == 20
    persisted = agent._build_trace_bundle_events(
        [event for event in events if event.get("type") == "trace"]
    )
    [stored] = [event for event in persisted if event["trace_kind"] == "turn_result"]
    assert stored["refused_tool_calls"] == 5 and stored["tool_budget_exhausted"]

    # The Overview's Needs-attention item names the cause.
    overview = build_operations_overview(persisted, window="24h")
    [item] = [entry for entry in overview["attention"] if entry["kind"] == "tool"]
    assert item["reason"] == "tool_budget_exhausted"
    assert "max_tool_invocations_per_turn" in item["detail"]
    agent.close()


def test_a_turn_within_its_budget_reports_nothing(work_dir):
    def save_fact(fact: str) -> str:
        """Save one fact the user shared."""
        return "saved"

    model = _Model([_tool_calls(2), "done"])
    agent = _agent(work_dir, model=model)
    agent.tool_manager.add_tool(save_fact)
    events: List[Dict[str, Any]] = []
    agent.set_stream_event_callback(lambda event: events.append(dict(event)))
    agent.run("remember two facts", memory_id="memory-1", thread_id="t1")
    assert agent.last_turn_limits["tool_budget_exhausted"] is False
    assert agent.last_turn_limits["refused_tool_calls"] == 0
    [result] = [event for event in events if event.get("trace_kind") == "turn_result"]
    assert "tool_budget_exhausted" not in result
    tool_messages = [
        message for message in model.requests[-1] if message.get("role") == "tool"
    ]
    assert all("[Tool budget]" not in message["content"] for message in tool_messages)
    agent.close()


# -- 14. plain record ids for retrieval evaluation --------------------------------


def test_retrieved_record_ids_are_opt_in_and_never_traced(work_dir):
    agent = _agent(work_dir)
    events: List[Dict[str, Any]] = []
    agent.set_stream_event_callback(lambda event: events.append(dict(event)))
    _recall(
        agent,
        {
            MemoryType.CONVERSATION_MEMORY: [
                {
                    "_id": "record-conv-1",
                    "thread_id": "session-42",
                    "role": "user",
                    "content": "My sister lives in Lisbon",
                    "embedding": [0.0, 0.0, 1.0],
                    "score": 0.9,
                }
            ],
            MemoryType.KNOWLEDGE_BASE: [
                {
                    "_id": "record-kb-1",
                    "content": "Lisbon is the capital of Portugal",
                    "embedding": [0.0, 1.0, 0.0],
                    "score": 0.8,
                }
            ],
        },
    )
    agent.run("where does my sister live?", memory_id="memory-1", thread_id="now")

    default = agent.last_memory_context_evidence()
    assert "retrieved" not in default
    with_ids = agent.last_memory_context_evidence(include_ids=True)
    # The fingerprints are unchanged; the ids sit beside them.
    assert with_ids["semantic_refs"] == default["semantic_refs"]
    retrieved = {item["record_id"]: item for item in with_ids["retrieved"]}
    assert set(retrieved) == {"record-conv-1", "record-kb-1"}
    conversation = retrieved["record-conv-1"]
    assert conversation["memory_type"] == "conversation_memory"
    assert conversation["thread_id"] == "session-42"
    assert isinstance(conversation["score"], float)
    assert retrieved["record-kb-1"]["memory_type"] == "knowledge_base"
    assert retrieved["record-kb-1"]["thread_id"] is None
    refs = {ref["ref"] for ref in default["semantic_refs"]}
    assert {item["ref"] for item in with_ids["retrieved"]} == refs

    # Neither the traces nor the default snapshot carry the plain ids.
    serialized = json.dumps(events, default=str)
    assert "record-conv-1" not in serialized
    assert "session-42" not in serialized
    assert "record-conv-1" not in json.dumps(default)

    # The next turn starts with no ids of its own.
    agent.memory_manager.retrieve_relevant_memories = lambda **kwargs: []
    agent.run("anything else?", memory_id="memory-1", thread_id="now")
    assert agent.last_memory_context_evidence(include_ids=True)["retrieved"] == []
    agent.close()


def test_entity_and_evidence_pack_ids_join_their_refs(work_dir):
    from memorizz.learning.models import EvidenceItem, EvidencePack

    agent = _agent(work_dir)
    pack = EvidencePack(
        pack_id="pack-1",
        query="who is Ada?",
        items=(
            EvidenceItem(
                evidence_id="ev-1",
                source_type="conversation_memory",
                source_id="row-9",
                content="Ada joined in May",
                relevance=0.7,
                token_estimate=5,
                content_hash="hash-1",
                reason="semantic",
                metadata={"thread_id": "session-9"},
            ),
        ),
        created_at="2026-10-18T09:00:00+00:00",
        token_budget=1000,
        tokens_used=5,
        candidate_tokens=5,
        candidate_count=1,
        rejected_count=0,
        scope={},
        query_hash="q",
        policy_hash="p",
        latency_ms=1.0,
    )
    agent._emit_memory_context_trace(
        {
            "evidence_pack": pack,
            "entity_memory_profiles": [
                {
                    "entity_id": "entity-ada",
                    "name": "Ada",
                    "attributes": {"role": "CTO"},
                }
            ],
        }
    )
    evidence = agent.last_memory_context_evidence(include_ids=True)
    by_source = {item["source"]: item for item in evidence["retrieved"]}
    assert by_source["entity"]["record_id"] == "entity-ada"
    assert by_source["entity"]["ref"] == evidence["entity_refs"][0]["ref"]
    assert by_source["evidence_pack"]["record_id"] == "row-9"
    assert by_source["evidence_pack"]["thread_id"] == "session-9"
    assert by_source["evidence_pack"]["memory_type"] == "conversation_memory"
    assert by_source["evidence_pack"]["ref"] == evidence["evidence_refs"][0]["ref"]
    agent.close()
