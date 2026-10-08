"""MemAgent wiring of the forgetting mechanism (docs/guides/forgetting-mechanism.md).

1. Retrieval scoring: recency since last access breaks ties; per-agent
   ``scoring`` overrides on the retrieval policy are honoured.
2. Reinforcement: selected memories (and EvidencePack items) are touched
   off the user path and drained by ``close()``.
3. Importance at store time: heuristic synchronously, LLM re-rating in the
   background worker.
4. Retention SDK surface, direct and via the learning control plane.
5. Suppressed conversation rows never reach the model.
6. Reflection trigger on accumulated importance.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memagent.utils.context_dedup import RetrievalScoring
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.retrieval import RetrievalPolicy

pytestmark = pytest.mark.unit

DAY = 86_400.0
HOUR = 3600.0
QUERY_VECTOR = [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------


@pytest.fixture()
def work_dir():
    base = Path("/private/tmp")
    path = Path(
        tempfile.mkdtemp(
            prefix="memagent-forgetting-", dir=base if base.is_dir() else None
        )
    )
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _offline_embeddings(monkeypatch):
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding", lambda text: list(QUERY_VECTOR)
    )


@pytest.fixture(autouse=True)
def _heuristic_rater(monkeypatch):
    """Deterministic default; individual tests switch modes."""
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")
    monkeypatch.delenv("MEMORIZZ_REFLECTION_ENABLED", raising=False)
    monkeypatch.delenv("MEMORIZZ_REFLECTION_THRESHOLD", raising=False)


class _DummyEmbeddingProvider:
    def get_embedding(self, text: str) -> List[float]:
        return list(QUERY_VECTOR)

    def get_provider_info(self) -> str:  # pragma: no cover
        return "dummy"


def _fs_provider(work_dir: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=work_dir / "memory",
            embedding_provider=_DummyEmbeddingProvider(),
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )


class _StubModel:
    model = "stub"

    def __init__(
        self, answer: str = "ok", text_answer: str = "8", text_delay: float = 0.0
    ):
        self.answer = answer
        self.text_answer = text_answer
        self.text_delay = text_delay
        self.requests: List[List[Dict[str, Any]]] = []
        self.prompts: List[str] = []

    def generate(self, messages, tools=None, **kwargs):
        self.requests.append([dict(m) if isinstance(m, dict) else m for m in messages])
        return self.answer

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        self.prompts.append(prompt)
        if self.text_delay:
            time.sleep(self.text_delay)
        return self.text_answer

    def get_config(self):
        return {"provider": "openai", "model": "stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


def _agent(work_dir: Path, **kwargs) -> MemAgent:
    kwargs.setdefault("model", _StubModel())
    kwargs.setdefault("memory_provider", _fs_provider(work_dir))
    kwargs.setdefault("memory_ids", ["memory-1"])
    kwargs.setdefault("auto_register", False)
    kwargs.setdefault("agent_id", "agent-1")
    return MemAgent(**kwargs)


def _conversation_row(record_id: str, text: str, **fields) -> Dict[str, Any]:
    row = {
        "_id": record_id,
        "role": "user",
        "content": text,
        "memory_id": "memory-1",
        "thread_id": "thread-old",
        "user_id": None,
        "agent_id": "agent-1",
        "timestamp": time.time() - 30 * DAY,
        "embedding": list(QUERY_VECTOR),
        "score": 0.9,
    }
    row.update(fields)
    return row


def _stub_retrieval(agent: MemAgent, rows: List[Dict[str, Any]]) -> None:
    def retrieve(**kwargs):
        if kwargs.get("memory_type") == MemoryType.CONVERSATION_MEMORY:
            return [dict(row) for row in rows]
        return []

    agent.memory_manager.retrieve_relevant_memories = retrieve
    agent._current_thread_id = "thread-now"  # no history window to dedupe against


# ---------------------------------------------------------------------------
# 1. Scoring
# ---------------------------------------------------------------------------


def test_recent_access_breaks_ties_unless_the_policy_turns_recency_off(work_dir):
    agent = _agent(work_dir)
    # Equal relevance to the query (cosine 0.8 each) but not near-duplicates
    # of each other, so dedupe keeps both and scoring decides.
    rows = [
        _conversation_row("unused", "fact never recalled", embedding=[0.8, 0.6, 0.0]),
        _conversation_row(
            "used",
            "fact recalled an hour ago",
            embedding=[0.8, 0.0, 0.6],
            last_accessed_at=time.time() - HOUR,
        ),
    ]
    _stub_retrieval(agent, rows)

    agent.retrieval_policy = RetrievalPolicy(
        knowledge_base_scope="disabled", max_items=1
    )
    assert agent.retrieval_scoring == RetrievalScoring.from_env()
    context = agent._build_context("fact", "memory-1")
    selected = context["retrieved_memories"]
    assert [item["id"] for item in selected] == ["used"]
    assert selected[0]["scoring"]["anchor"] == "last_accessed"
    assert selected[0]["last_accessed_at"] is not None

    agent.retrieval_policy = RetrievalPolicy(
        knowledge_base_scope="disabled", max_items=1, scoring={"alpha_recency": 0}
    )
    assert agent.retrieval_scoring.alpha_recency == 0.0
    context = agent._build_context("fact", "memory-1")
    selected = context["retrieved_memories"]
    # Equal relevance, recency ignored: provider order decides, not the access.
    assert [item["id"] for item in selected] == ["unused"]
    assert selected[0]["scoring"]["score"] == pytest.approx(1.0)
    agent.close()


def test_retrieval_scoring_is_cached_per_policy_object(work_dir):
    agent = _agent(work_dir)
    first = agent.retrieval_scoring
    assert agent.retrieval_scoring is first
    agent.retrieval_policy = RetrievalPolicy(scoring={"alpha_importance": 2})
    assert agent.retrieval_scoring.alpha_importance == 2.0
    assert agent.retrieval_scoring is not first
    agent.close()


# ---------------------------------------------------------------------------
# 2. Reinforcement
# ---------------------------------------------------------------------------


def test_selected_memory_is_touched_after_close_drains(work_dir):
    agent = _agent(work_dir)
    provider = agent.memory_provider
    # Equal relevance to the query (cosine 0.8 each) but not near-duplicates
    # of each other, so dedupe keeps both and scoring decides.
    rows = [
        _conversation_row("unused", "fact never recalled", embedding=[0.8, 0.6, 0.0]),
        _conversation_row(
            "used",
            "fact recalled an hour ago",
            embedding=[0.8, 0.0, 0.6],
            last_accessed_at=time.time() - HOUR,
        ),
    ]
    for row in rows:
        provider.store(dict(row), MemoryType.CONVERSATION_MEMORY)
    _stub_retrieval(agent, rows)
    agent.retrieval_policy = RetrievalPolicy(
        knowledge_base_scope="disabled", max_items=1
    )
    before = provider.retrieve_by_id("used", MemoryType.CONVERSATION_MEMORY)

    context = agent._build_context("fact", "memory-1")
    assert [item["id"] for item in context["retrieved_memories"]] == ["used"]
    report = agent.close()

    assert report["closed"]["embedding_backfill"] is True
    used = provider.retrieve_by_id("used", MemoryType.CONVERSATION_MEMORY)
    assert used["access_count"] == 1
    assert (
        used["last_accessed_at"]
        and used["last_accessed_at"] != before["last_accessed_at"]
    )
    assert used["timestamp"] == before["timestamp"]
    assert used["embedding"] == before["embedding"]
    unused = provider.retrieve_by_id("unused", MemoryType.CONVERSATION_MEMORY)
    assert not unused.get("access_count") and "last_accessed_at" not in unused


def test_evidence_pack_items_are_touched_by_source_type(work_dir):
    agent = _agent(work_dir)
    provider = agent.memory_provider
    provider.store(
        _conversation_row("used", "cited row"), MemoryType.CONVERSATION_MEMORY
    )
    pack = SimpleNamespace(
        items=(
            SimpleNamespace(source_type="conversation_memory", source_id="used"),
            SimpleNamespace(source_type="skillbox", source_id="skill-1"),
        )
    )

    agent._schedule_evidence_reinforcement(pack)
    agent.close()

    assert (
        provider.retrieve_by_id("used", MemoryType.CONVERSATION_MEMORY)["access_count"]
        == 1
    )


# ---------------------------------------------------------------------------
# 3. Importance at store time
# ---------------------------------------------------------------------------


def _rows(provider: FileSystemProvider) -> Dict[str, Dict[str, Any]]:
    return {
        row["role"]: row for row in provider.list_all(MemoryType.CONVERSATION_MEMORY)
    }


def test_heuristic_importance_is_stored_on_both_rows(work_dir):
    agent = _agent(work_dir)
    agent._record_interaction(
        "We decided to ship the release on Thursday.", "Noted.", "memory-1", "thread-1"
    )
    rows = _rows(agent.memory_provider)
    assert set(rows) == {"user", "assistant"}
    for row in rows.values():
        assert 0.0 < row["importance"] <= 1.0
        assert row["importance_source"] == "heuristic"
    assert rows["user"]["importance"] > rows["assistant"]["importance"]
    assert agent.close()["ok"] is True


def test_importance_off_stores_nothing(work_dir, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "off")
    agent = _agent(work_dir)
    agent._record_interaction("hello", "hi", "memory-1", "thread-1")
    assert all("importance" not in row for row in _rows(agent.memory_provider).values())
    agent.close()


def test_llm_mode_rerates_in_the_background(work_dir, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "llm")
    model = _StubModel(text_answer="8\n8", text_delay=0.2)
    agent = _agent(work_dir, model=model)

    agent._record_interaction(
        "We decided to ship the release on Thursday.", "Noted.", "memory-1", "thread-1"
    )
    # The heuristic value is on the row before the model answers.
    assert all(
        row["importance_source"] == "heuristic"
        for row in _rows(agent.memory_provider).values()
    )

    agent.close()

    rows = _rows(agent.memory_provider)
    assert [row["importance"] for row in rows.values()] == [0.8, 0.8]
    assert all(row["importance_source"] == "llm" for row in rows.values())
    assert len(model.prompts) == 1  # one batched rating call per turn


# ---------------------------------------------------------------------------
# 4. Retention SDK surface
# ---------------------------------------------------------------------------


def _kb_row(provider, record_id, text, *, days_old, **fields):
    row = {
        "_id": record_id,
        "content": text,
        "name": record_id,
        "agent_id": "agent-1",
        "memory_id": "kb-1",
        "timestamp": time.time() - days_old * DAY,
        "embedding": [1.0, 0.0],
    }
    row.update(fields)
    provider.store(row, MemoryType.KNOWLEDGE_BASE)


def test_retention_sdk_runs_the_planner_directly_without_a_control_plane(work_dir):
    agent = _agent(work_dir, memory_ids=["kb-1"])
    provider = agent.memory_provider
    _kb_row(provider, "stale", "forgotten detail", days_old=120, importance=0.1)
    _kb_row(provider, "fresh", "new detail", days_old=2, importance=0.1)
    agent.retrieval_policy = RetrievalPolicy(
        retention={
            "enabled": True,
            "min_retention": 0.3,
            "grace_days": 30,
            "memory_types": ["knowledge_base"],
        }
    )
    assert agent.learning_control_plane is None
    assert agent.retention_config.enabled is True
    assert agent.retention_config.memory_types == ("knowledge_base",)

    report = agent.plan_memory_retention(memory_id="kb-1")
    assert report.dry_run is True
    assert [c.target_id for c in report.candidates] == ["stale"]

    applied = agent.apply_memory_retention(
        report, approved_by="operator-7", reason="tidy"
    )
    assert applied.dry_run is False and applied.tombstoned == 1
    assert (
        provider.retrieve_by_id("stale", MemoryType.KNOWLEDGE_BASE)["retention_state"]
        == "suppressed"
    )
    assert [
        item["record_id"] for item in agent.suppressed_memories(memory_id="kb-1")
    ] == ["stale"]

    assert (
        agent.unsuppress_memory("stale", "knowledge_base", approved_by="operator-7")
        is True
    )
    assert agent.suppressed_memories(memory_id="kb-1") == []
    agent.close()


def test_retention_sdk_delegates_to_the_learning_control_plane(work_dir):
    agent = _agent(
        work_dir, learning_control_plane={"enabled": True, "compile_async": False}
    )
    plane = agent.learning_control_plane
    assert plane is not None
    plane.plan_retention = MagicMock(return_value="plan")
    plane.apply_retention = MagicMock(return_value="applied")
    plane.unsuppress_memory = MagicMock(return_value=True)
    plane.suppressed_memories = MagicMock(return_value=[{"record_id": "x"}])

    assert agent.plan_memory_retention(memory_id="kb-1", user_id="alice") == "plan"
    plane.plan_retention.assert_called_once_with(
        memory_id="kb-1",
        user_id="alice",
        scoring=agent.retrieval_scoring,
        config=agent.retention_config,
    )
    assert (
        agent.apply_memory_retention("plan", approved_by="op", reason="why")
        == "applied"
    )
    plane.apply_retention.assert_called_once_with(
        "plan",
        approved_by="op",
        reason="why",
        config=agent.retention_config,
    )
    assert agent.unsuppress_memory("x", "knowledge_base", approved_by="op") is True
    plane.unsuppress_memory.assert_called_once_with(
        "x", "knowledge_base", approved_by="op", reason=None
    )
    assert agent.suppressed_memories(memory_id="kb-1") == [{"record_id": "x"}]
    plane.suppressed_memories.assert_called_once_with(memory_id="kb-1", user_id=...)
    agent.close()


# ---------------------------------------------------------------------------
# 5. Suppression on the history path
# ---------------------------------------------------------------------------


def test_suppressed_conversation_row_is_not_sent_to_the_model(work_dir):
    model = _StubModel(answer="Sure.")
    agent = _agent(work_dir, model=model)
    provider = agent.memory_provider
    provider.store(
        _conversation_row(
            "visible",
            "What is the launch budget?",
            thread_id="thread-1",
            timestamp=time.time() - 2 * HOUR,
        ),
        MemoryType.CONVERSATION_MEMORY,
    )
    provider.store(
        _conversation_row(
            "hidden",
            "SECRET-SUPPRESSED budget detail",
            role="assistant",
            thread_id="thread-1",
            timestamp=time.time() - HOUR,
            retention_state="suppressed",
        ),
        MemoryType.CONVERSATION_MEMORY,
    )

    agent.run("hello again", memory_id="memory-1", thread_id="thread-1")

    sent = json.dumps(model.requests[-1])
    assert "launch budget" in sent
    assert "SECRET-SUPPRESSED" not in sent
    history = agent._prepare_history_messages(
        [
            {
                "role": "assistant",
                "content": "SECRET-SUPPRESSED",
                "retention_state": "suppressed",
            },
            {"role": "user", "content": "kept"},
        ],
        "system",
        "q",
    )
    assert [m["content"] for m in history] == ["kept"]
    agent.close()


# ---------------------------------------------------------------------------
# 6. Reflection trigger
# ---------------------------------------------------------------------------


def test_reflection_runs_once_when_accumulated_importance_crosses_the_threshold(
    work_dir, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_REFLECTION_ENABLED", "1")
    monkeypatch.setenv("MEMORIZZ_REFLECTION_THRESHOLD", "1.0")
    monkeypatch.setattr(
        "memorizz.memagent.utils.importance.heuristic_importance", lambda *a, **k: 0.6
    )
    agent = _agent(work_dir)
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        agent, "generate_summaries", lambda **kw: calls.append(kw) or ["s1"]
    )
    events: List[Dict[str, Any]] = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    agent._record_interaction("first thing", "second thing", "memory-1", "thread-1")

    assert len(calls) == 1
    assert calls[0]["memory_id"] == "memory-1" and calls[0]["thread_id"] == "thread-1"
    assert calls[0]["summary_type"] == "compaction"
    assert agent._reflection_importance["memory-1"] == 0.0
    reflections = [e for e in events if e.get("trace_kind") == "reflection"]
    assert len(reflections) == 1
    assert reflections[0]["accumulated_importance"] == pytest.approx(1.2)
    assert reflections[0]["summary_ids"] == ["s1"]
    agent.close()


def test_reflection_is_off_by_default(work_dir, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_REFLECTION_THRESHOLD", "0.1")
    agent = _agent(work_dir)
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        agent, "generate_summaries", lambda **kw: calls.append(kw) or ["s1"]
    )
    agent._record_interaction("first thing", "second thing", "memory-1", "thread-1")
    assert calls == []
    assert agent._reflection_importance == {}
    agent.close()
