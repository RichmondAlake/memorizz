"""Regression tests for MemAgent core defect fixes.

Covers:
1. Provider failures / exhausted tool loops are returned to the caller but
   never cached, never stored as the assistant's turn, and close the turn
   trace as an error (``run`` and ``run_stream``).
2. ``_ContextLocal`` first-use is race free under concurrent runs.
3. An approval pause with parallel tool calls: safe siblings still run,
   guarded siblings get a "not executed" result, and the continuation
   request answers every tool_call_id.
4. ``close()`` waits for pending conversation-embedding backfills (which run
   in a copy of the turn's context).
5. Internet tools pause for a cool-down after repeated failures and recover.
6. ``retrieve_tool_log_entry`` is scoped to the current memory/thread.
7. Auto-compaction counts nested in-session cache rows.
8. ``knowledge_base_lookup`` scopes or widens its candidate fetch.
9. Self-aware write/delete/run tools require approval.
10. New conversations patch ``memory_ids`` instead of rewriting the agent.
11. No stray "Reset thread state" log; DeepResearchWorkflow carries user_id.
"""

from __future__ import annotations

import contextvars
import json
import logging
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock

import pytest

from memorizz.approval import ApprovalRequired, SQLiteApprovalStore
from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memagent.core import _ContextLocal
from memorizz.memagent.orchestrators.deep_research import (
    DeepResearchOrchestrator,
    DeepResearchWorkflow,
)
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.tooling import governed_tool, normalize_tool_result
from tests.mocks.mock_providers import MockMemoryProvider

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------


@pytest.fixture()
def work_dir():
    """A scratch directory under /private/tmp for filesystem providers."""
    base = Path("/private/tmp")
    if not base.is_dir():
        base = None
    path = Path(tempfile.mkdtemp(prefix="memagent-core-fixes-", dir=base))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _offline_embeddings(monkeypatch):
    """No embedding API calls from any code path these tests touch."""
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding", lambda text: [0.1, 0.2, 0.3]
    )


class _DummyEmbeddingProvider:
    def get_embedding(self, text: str) -> List[float]:
        seed = float(sum(ord(ch) for ch in (text or "")))
        return [seed, float(len(text or "")), 0.0]

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


def _tool_call(name: str, arguments: Dict[str, Any], call_id: str) -> Any:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _tool_call_response(calls: List[tuple]) -> Any:
    """An OpenAI-shaped response whose assistant message carries tool calls."""
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


class _StubModel:
    """Scripted model: each ``generate`` pops a response (or raises)."""

    model = "stub"

    def __init__(self, responses: Optional[List[Any]] = None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.requests: List[List[Dict[str, Any]]] = []

    def generate(self, messages, tools=None, **kwargs):
        self.requests.append([dict(m) if isinstance(m, dict) else m for m in messages])
        if self.error is not None:
            raise self.error
        if not self.responses:
            return "done"
        item = self.responses.pop(0)
        return item() if callable(item) else item

    def get_config(self):
        return {"provider": "openai", "model": "stub"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


class _StreamingStubModel(_StubModel):
    """Streams one scripted event list per ``generate_stream`` call."""

    def __init__(self, stream_events: List[List[Dict[str, Any]]]):
        super().__init__()
        self.stream_events = list(stream_events)

    def generate_stream(self, messages, tools=None, **kwargs):
        self.requests.append([dict(m) if isinstance(m, dict) else m for m in messages])
        events = self.stream_events.pop(0) if self.stream_events else []
        for event in events:
            yield event


def _spy_cache(agent: MemAgent) -> MagicMock:
    """Enable the semantic cache with a stub backend and spy on writes."""
    agent.cache_manager.enabled = True
    agent.cache_manager.cache_instance = Mock()
    agent.cache_manager.cache_instance.get.return_value = None
    spy = MagicMock(return_value=False)
    agent.cache_manager.cache_response = spy
    return spy


def _turn_results(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("trace_kind") == "turn_result"]


# ---------------------------------------------------------------------------
# Finding 1: provider failures are not answers
# ---------------------------------------------------------------------------


def test_provider_error_is_neither_cached_nor_recorded(work_dir):
    provider = _fs_provider(work_dir)
    agent = MemAgent(
        model=_StubModel(error=RuntimeError("429 Too Many Requests")),
        memory_provider=provider,
        memory_ids=["memory-1"],
        auto_register=False,
    )
    cache_spy = _spy_cache(agent)
    events: List[Dict[str, Any]] = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    response = agent.run("hello", memory_id="memory-1", thread_id="thread-1")

    assert "I encountered an error while processing your request" in response
    assert "429" in response
    cache_spy.assert_not_called()
    assert provider.list_all(MemoryType.CONVERSATION_MEMORY) == []
    result = _turn_results(events)[-1]
    assert result["status"] == "error"
    assert result["error_code"] == "provider_error"
    agent.close()


def test_iteration_limit_reply_is_neither_cached_nor_recorded(work_dir):
    provider = _fs_provider(work_dir)

    def ping() -> str:
        """Return pong."""
        return "pong"

    loop_forever = lambda: _tool_call_response([("call-1", "ping", {})])  # noqa: E731
    agent = MemAgent(
        model=_StubModel(responses=[loop_forever] * 10),
        tools=[ping],
        memory_provider=provider,
        memory_ids=["memory-1"],
        max_steps=2,
        auto_register=False,
    )
    cache_spy = _spy_cache(agent)
    events: List[Dict[str, Any]] = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    response = agent.run("ping forever", memory_id="memory-1", thread_id="thread-1")

    assert "maximum number of tool-call iterations" in response
    cache_spy.assert_not_called()
    assert provider.list_all(MemoryType.CONVERSATION_MEMORY) == []
    result = _turn_results(events)[-1]
    assert result["status"] == "error"
    assert result["error_code"] == "iteration_limit"
    agent.close()


def test_streamed_iteration_limit_is_neither_cached_nor_recorded(work_dir):
    provider = _fs_provider(work_dir)

    def ping() -> str:
        """Return pong."""
        return "pong"

    tool_turn = [
        {"type": "tool_calls", "response": _tool_call_response([("c1", "ping", {})])}
    ]
    agent = MemAgent(
        model=_StreamingStubModel([tool_turn] * 10),
        tools=[ping],
        memory_provider=provider,
        memory_ids=["memory-1"],
        max_steps=2,
        auto_register=False,
    )
    cache_spy = _spy_cache(agent)
    events: List[Dict[str, Any]] = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    with pytest.warns(DeprecationWarning):
        streamed = "".join(
            agent.run_stream("ping forever", memory_id="memory-1", thread_id="thread-1")
        )

    assert "maximum number of tool-call iterations" in streamed
    cache_spy.assert_not_called()
    assert provider.list_all(MemoryType.CONVERSATION_MEMORY) == []
    result = _turn_results(events)[-1]
    assert result["status"] == "error"
    assert result["error_code"] == "iteration_limit"
    end = [e for e in events if e["type"] == "stream_end"][-1]
    assert end["reason"] == "error" and end["error_code"] == "iteration_limit"
    agent.close()


def test_successful_turn_is_still_cached_and_recorded(work_dir):
    provider = _fs_provider(work_dir)
    agent = MemAgent(
        model=_StubModel(responses=["Fine, thanks."]),
        memory_provider=provider,
        memory_ids=["memory-1"],
        auto_register=False,
    )
    cache_spy = _spy_cache(agent)

    assert agent.run("hello", memory_id="memory-1", thread_id="thread-1") == (
        "Fine, thanks."
    )

    cache_spy.assert_called_once()
    roles = sorted(
        row.get("role") for row in provider.list_all(MemoryType.CONVERSATION_MEMORY)
    )
    assert roles == ["assistant", "user"]
    agent.close()


# ---------------------------------------------------------------------------
# Finding 2: _ContextLocal first-use race
# ---------------------------------------------------------------------------


def _context_local_losses(descriptor_class, rounds: int, threads: int) -> int:
    """Values lost when ``threads`` threads first touch a fresh instance."""

    class Holder:
        run_id = descriptor_class()

    losses = 0
    for _ in range(rounds):
        holder = Holder()
        start = threading.Barrier(threads)
        readback = threading.Barrier(threads)
        seen: Dict[str, Any] = {}

        def worker(label: str) -> None:
            start.wait()
            holder.run_id = label
            readback.wait()
            seen[label] = holder.run_id

        workers = [
            threading.Thread(target=worker, args=(f"turn-{index}",))
            for index in range(threads)
        ]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        losses += sum(1 for label, value in seen.items() if value != label)
    return losses


def test_context_local_first_use_loses_no_values_under_contention():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        assert _context_local_losses(_ContextLocal, rounds=1500, threads=4) == 0
    finally:
        sys.setswitchinterval(previous)


def test_context_locals_are_created_eagerly_on_init():
    agent = MemAgent(memory_provider=False, auto_register=False)
    storage_names = [
        descriptor.storage_name
        for klass in type(agent).__mro__
        for descriptor in vars(klass).values()
        if isinstance(descriptor, _ContextLocal)
    ]
    assert storage_names
    assert all(name in agent.__dict__ for name in storage_names)
    agent.close()


# ---------------------------------------------------------------------------
# Finding 3: approval pause with parallel tool calls
# ---------------------------------------------------------------------------


def _continuation_tool_ids(request: List[Dict[str, Any]]):
    assistant = [
        m for m in request if m.get("role") == "assistant" and m.get("tool_calls")
    ][-1]
    requested = {call["id"] for call in assistant["tool_calls"]}
    answered = {m.get("tool_call_id") for m in request if m.get("role") == "tool"}
    return requested, answered


def test_safe_sibling_runs_before_the_pause_and_continuation_answers_both(
    work_dir,
):
    executed: List[tuple] = []

    @governed_tool(side_effects=True, requires_approval=True)
    def create_item(name: str) -> dict:
        """Create an item (needs approval)."""
        executed.append(("create_item", name))
        return {"created": name}

    def lookup_price(sku: str) -> dict:
        """Read a price."""
        executed.append(("lookup_price", sku))
        return {"sku": sku, "price": 9}

    model = _StubModel(
        responses=[
            lambda: _tool_call_response(
                [
                    ("call_A", "create_item", {"name": "x"}),
                    ("call_B", "lookup_price", {"sku": "s1"}),
                ]
            ),
            "All done.",
        ]
    )
    agent = MemAgent(
        model=model,
        tools=[create_item, lookup_price],
        approval_store=SQLiteApprovalStore(work_dir / "approvals.sqlite3"),
        memory_provider=_fs_provider(work_dir),
        memory_ids=["memory-1"],
        auto_register=False,
    )

    paused = json.loads(agent.run("create x; price s1", memory_id="memory-1"))
    assert paused["status"] == "approval_required"
    # The safe sibling ran before the pause.
    assert executed == [("lookup_price", "s1")]

    proposal_id = paused["proposal"]["proposal_id"]
    agent.approve(proposal_id, approver_id="operator@example.com")
    resumed = agent.resume_approval(proposal_id)

    assert resumed.ok and resumed.tool_result == {"created": "x"}
    assert executed == [("lookup_price", "s1"), ("create_item", "x")]
    assert resumed.assistant_response == "All done."
    requested, answered = _continuation_tool_ids(model.requests[-1])
    assert requested == {"call_A", "call_B"}
    assert requested <= answered
    agent.close()


def test_guarded_sibling_gets_a_not_executed_result(work_dir):
    executed: List[tuple] = []

    @governed_tool(side_effects=True, requires_approval=True)
    def create_item(name: str) -> dict:
        """Create an item (needs approval)."""
        executed.append(("create_item", name))
        return {"created": name}

    @governed_tool(side_effects=True, requires_approval=True)
    def delete_item(name: str) -> dict:
        """Delete an item (needs approval)."""
        executed.append(("delete_item", name))
        return {"deleted": name}

    model = _StubModel(
        responses=[
            lambda: _tool_call_response(
                [
                    ("call_A", "create_item", {"name": "x"}),
                    ("call_C", "delete_item", {"name": "y"}),
                ]
            ),
            "Done.",
        ]
    )
    store = SQLiteApprovalStore(work_dir / "approvals.sqlite3")
    agent = MemAgent(
        model=model,
        tools=[create_item, delete_item],
        approval_store=store,
        memory_provider=_fs_provider(work_dir),
        memory_ids=["memory-1"],
        auto_register=False,
    )

    paused = json.loads(agent.run("create x, delete y", memory_id="memory-1"))
    proposal = store.get(paused["proposal"]["proposal_id"])
    assert proposal.tool_name == "create_item"
    deferred = proposal.checkpoint["deferred_tool_calls"]
    assert [item["tool_call_id"] for item in deferred] == ["call_C"]
    assert executed == []

    agent.approve(proposal.proposal_id, approver_id="operator@example.com")
    resumed = agent.resume_approval(proposal.proposal_id)

    assert resumed.ok and executed == [("create_item", "x")]
    request = model.requests[-1]
    requested, answered = _continuation_tool_ids(request)
    assert requested == {"call_A", "call_C"} and requested <= answered
    sibling = [m for m in request if m.get("tool_call_id") == "call_C"][0]
    assert json.loads(sibling["content"])["error_code"] == "sibling_awaiting_approval"
    agent.close()


# ---------------------------------------------------------------------------
# Finding 4: embedding backfill shutdown and context propagation
# ---------------------------------------------------------------------------


def test_close_waits_for_pending_embedding_backfill(work_dir, monkeypatch):
    provider = _fs_provider(work_dir)
    actor_var: contextvars.ContextVar = contextvars.ContextVar("test_actor")
    seen_actor: List[Any] = []

    def slow_embedding(text: str) -> List[float]:
        seen_actor.append(actor_var.get(None))
        time.sleep(0.2)
        return [0.5, 0.25, 0.125]

    monkeypatch.setattr("memorizz.embeddings.get_embedding", slow_embedding)
    agent = MemAgent(
        memory_provider=provider, memory_ids=["memory-1"], auto_register=False
    )
    agent._conversation_embedding_enabled = True

    token = actor_var.set("agent-turn")
    try:
        agent._record_interaction("hello", "fine", "memory-1", "thread-1")
    finally:
        actor_var.reset(token)

    report = agent.close()

    assert report["closed"]["embedding_backfill"] is True
    rows = provider.list_all(MemoryType.CONVERSATION_MEMORY)
    assert len(rows) == 2
    assert all(row.get("embedding") == [0.5, 0.25, 0.125] for row in rows)
    # The worker ran in a copy of the turn's context.
    assert seen_actor == ["agent-turn", "agent-turn"]


# ---------------------------------------------------------------------------
# Finding 5: internet access circuit breaker
# ---------------------------------------------------------------------------


def test_internet_tools_retry_after_the_cooldown(monkeypatch):
    provider = MagicMock()
    provider.get_provider_name.return_value = "dummy"
    provider.get_config.return_value = {}
    provider.search.side_effect = RuntimeError("upstream down")
    agent = MemAgent(
        instruction="net",
        internet_access_provider=provider,
        memory_provider=False,
        auto_register=False,
    )
    now = [1000.0]
    monkeypatch.setattr(agent, "_internet_access_clock", lambda: now[0])
    agent.internet_access_cooldown_seconds = 120

    for _ in range(3):
        raw, _ = agent.tool_manager.execute_tool("internet_search", {"query": "q"})
    assert raw["ok"] is False and raw["retryable"] is False
    assert "retrying in 120s" in raw["error"]
    assert agent.has_internet_access() is False

    provider.search.side_effect = None
    provider.search.return_value = [{"url": "https://example.com", "title": "t"}]

    now[0] += 60
    raw, _ = agent.tool_manager.execute_tool("internet_search", {"query": "q"})
    payload, outcome = normalize_tool_result(raw)
    assert outcome.ok is False and "retrying in 60s" in payload["error"]
    assert provider.search.call_count == 3

    now[0] += 61
    raw, _ = agent.tool_manager.execute_tool("internet_search", {"query": "q"})
    payload, outcome = normalize_tool_result(raw)
    assert outcome.ok is True
    assert payload["results"][0]["url"] == "https://example.com"
    assert provider.search.call_count == 4
    assert agent.has_internet_access() is True
    assert agent._internet_access_failure_count == 0
    agent.close()


# ---------------------------------------------------------------------------
# Finding 6: tool log reads are conversation scoped
# ---------------------------------------------------------------------------


def test_tool_log_from_another_conversation_is_not_found(work_dir):
    provider = _fs_provider(work_dir)
    agent = MemAgent(
        memory_provider=provider, memory_ids=["memory-1"], auto_register=False
    )
    other = agent.memory_manager.store_tool_log(
        tool_name="lookup",
        arguments={"q": 1},
        result="secret from another conversation",
        memory_id="memory-other",
        agent_id=agent.agent_id,
        thread_id="thread-other",
        user_id=None,
    )
    mine = agent.memory_manager.store_tool_log(
        tool_name="lookup",
        arguments={"q": 2},
        result="my own result",
        memory_id="memory-1",
        agent_id=agent.agent_id,
        thread_id="thread-1",
        user_id=None,
    )
    agent._current_user_id = None
    agent._current_memory_id = "memory-1"
    agent._current_thread_id = "thread-1"

    foreign, _ = agent.tool_manager.execute_tool(
        "retrieve_tool_log_entry", {"tool_log_id": other}
    )
    assert foreign["ok"] is False
    assert foreign["error_code"] == "tool_log_not_found"

    own, _ = agent.tool_manager.execute_tool(
        "retrieve_tool_log_entry", {"tool_log_id": mine}
    )
    assert own["ok"] is True
    assert own["tool_log"]["result"] == "my own result"
    agent.close()


# ---------------------------------------------------------------------------
# Finding 7: compaction sees nested in-session rows
# ---------------------------------------------------------------------------


def _flat_history(count: int, words: int = 60) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": " ".join(f"word{index}" for _ in range(words)),
        }
        for index in range(count)
    ]


def _nested_history(count: int, words: int = 60) -> List[Dict[str, Any]]:
    """Rows as the in-session conversation cache stores them."""
    return [
        {
            "id": f"row-{index}",
            "memory_id": "memory-1",
            "content": {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": " ".join(f"nested{index}" for _ in range(words)),
                "thread_id": "thread-1",
            },
        }
        for index in range(count)
    ]


def test_compaction_counts_and_keeps_nested_in_session_rows(monkeypatch):
    agent = MemAgent(
        memory_provider=False,
        auto_register=False,
        context_window_tokens=2000,
        context_policy={"compact_at": 50, "keep_recent_messages": 4},
    )
    agent.memory_manager = SimpleNamespace(load_summaries_for_thread=lambda **kw: [])
    if MemoryType.SUMMARIES not in agent.active_memory_types:
        agent.active_memory_types.append(MemoryType.SUMMARIES)
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        agent, "generate_summaries", lambda **kw: calls.append(kw) or ["s1"]
    )
    flat = _flat_history(8)
    nested = _nested_history(4)
    history = flat + nested

    assert agent._compaction_estimate(history, "system", "q") > (
        agent._compaction_estimate(flat, "system", "q")
    )
    assert agent._compaction_estimate(nested, "system", "q") == (
        agent._compaction_estimate(
            [dict(row["content"]) for row in nested], "system", "q"
        )
    )

    kept = agent._auto_compact_history(history, "system", "q", {})

    assert calls and calls[0]["keep_recent"] == 4
    assert [m["content"] for m in kept] == [row["content"]["content"] for row in nested]
    assert [m["role"] for m in kept] == ["user", "assistant", "user", "assistant"]
    agent.close()


# ---------------------------------------------------------------------------
# Finding 8: knowledge base lookup scoping
# ---------------------------------------------------------------------------


_KB_STORE = [
    {"knowledge_base_id": "kb-other", "content": f"other {i}", "namespace": "n"}
    for i in range(40)
] + [
    {"knowledge_base_id": "kb-small", "content": f"mine {i}", "namespace": "n"}
    for i in range(2)
]


class _ScopedKB:
    instances = 0
    calls: List[Dict[str, Any]] = []

    def __init__(self, memory_provider):
        type(self).instances += 1
        self.memory_provider = memory_provider

    def retrieve_knowledge_by_query(
        self, query, namespace=None, limit=5, knowledge_base_ids=None, user_id=None
    ):
        type(self).calls.append(
            {
                "limit": limit,
                "knowledge_base_ids": knowledge_base_ids,
                "user_id": user_id,
            }
        )
        rows = _KB_STORE
        if knowledge_base_ids is not None:
            rows = [r for r in rows if r["knowledge_base_id"] in knowledge_base_ids]
        return rows[:limit]


class _LegacyKB:
    calls: List[int] = []

    def __init__(self, memory_provider):
        self.memory_provider = memory_provider

    def retrieve_knowledge_by_query(self, query, namespace=None, limit=5):
        type(self).calls.append(limit)
        return _KB_STORE[:limit]


def _kb_agent(monkeypatch, kb_class) -> MemAgent:
    kb_class.calls = []
    monkeypatch.setattr(
        "memorizz.long_term.semantic.knowledge_base.KnowledgeBase", kb_class
    )
    agent = MemAgent(memory_provider=MockMemoryProvider(), auto_register=False)
    agent.knowledge_base_ids = ["kb-small"]
    agent._current_user_id = "alice"
    return agent


def test_knowledge_lookup_scopes_to_the_agents_kb_ids(monkeypatch):
    _ScopedKB.instances = 0
    agent = _kb_agent(monkeypatch, _ScopedKB)

    result, _ = agent.tool_manager.execute_tool(
        "knowledge_base_lookup", {"query": "budget", "limit": 2}
    )

    assert [m["content"] for m in result["matches"]] == ["mine 0", "mine 1"]
    assert _ScopedKB.calls == [
        {"limit": 2, "knowledge_base_ids": ["kb-small"], "user_id": "alice"}
    ]
    agent.tool_manager.execute_tool("knowledge_base_lookup", {"query": "x"})
    assert _ScopedKB.instances == 1  # one KnowledgeBase per agent, not per call
    agent.close()


def test_knowledge_lookup_widens_the_fetch_for_legacy_stores(monkeypatch):
    agent = _kb_agent(monkeypatch, _LegacyKB)

    result, _ = agent.tool_manager.execute_tool(
        "knowledge_base_lookup", {"query": "budget", "limit": 2}
    )

    assert [m["content"] for m in result["matches"]] == ["mine 0", "mine 1"]
    assert _LegacyKB.calls == [8, 40, 200]
    agent.close()


# ---------------------------------------------------------------------------
# Finding 9: self-aware write/delete/run need approval
# ---------------------------------------------------------------------------


def test_self_aware_dangerous_tools_pause_for_approval(work_dir):
    root = work_dir / "repo"
    root.mkdir()
    agent = MemAgent(
        memory_provider=False,
        auto_register=False,
        approval_store=SQLiteApprovalStore(work_dir / "approvals.sqlite3"),
        self_aware=True,
        self_aware_config={
            "root_paths": [str(root)],
            "allow_writes": True,
            "allow_deletes": True,
        },
    )
    for name in (
        "self_aware_write_file",
        "self_aware_delete_path",
        "self_aware_run_command",
    ):
        assert agent.tool_manager.get_tool_policy(name)["requires_approval"] is True
    assert (
        agent.tool_manager.get_tool_policy("self_aware_read_file")["requires_approval"]
        is False
    )

    agent.semantic_tool_router.begin_turn(user_id=None)
    agent._build_llm_tools("write a note", user_id=None)
    agent._current_memory_id = "memory-1"
    target = root / "note.txt"
    messages: List[Dict[str, Any]] = []

    with pytest.raises(ApprovalRequired) as raised:
        agent._execute_and_record_tool_call(
            _tool_call(
                "self_aware_write_file",
                {"path": str(target), "content": "hi"},
                "call-write",
            ),
            messages,
            None,
            None,
            query="write a note",
        )
    assert not target.exists()

    proposal_id = raised.value.proposal.proposal_id
    agent.approve(proposal_id, approver_id="operator@example.com")
    resumed = agent.resume_approval(proposal_id, continue_model=False)

    assert resumed.ok is True
    assert target.read_text() == "hi"
    agent.close()


# ---------------------------------------------------------------------------
# Finding 10: memory id registration without a full agent rewrite
# ---------------------------------------------------------------------------


def test_second_conversation_patches_memory_ids(work_dir, monkeypatch):
    provider = _fs_provider(work_dir)
    store_spy = MagicMock(wraps=provider.store_memagent)
    update_spy = MagicMock(wraps=provider.update_memagent_memory_ids)
    monkeypatch.setattr(provider, "store_memagent", store_spy)
    monkeypatch.setattr(provider, "update_memagent_memory_ids", update_spy)
    agent = MemAgent(memory_provider=provider, agent_id="agent-1", auto_register=True)

    agent._resolve_execution_state("memory-1", None)
    assert store_spy.call_count == 1 and update_spy.call_count == 0

    # Another turn of the same conversation changes nothing.
    agent._resolve_execution_state("memory-1", "thread-2")
    assert store_spy.call_count == 1 and update_spy.call_count == 0

    agent._resolve_execution_state("memory-2", None)
    assert store_spy.call_count == 1
    update_spy.assert_called_once_with("agent-1", ["memory-1", "memory-2"])
    assert provider.retrieve_memagent("agent-1").memory_ids == [
        "memory-1",
        "memory-2",
    ]
    agent.close()


# ---------------------------------------------------------------------------
# Finding 11: stray log line and DeepResearchWorkflow user_id
# ---------------------------------------------------------------------------


def test_remember_thread_does_not_log_a_reset(caplog):
    agent = MemAgent(memory_provider=False, auto_register=False)
    with caplog.at_level(logging.INFO, logger="memorizz.memagent.core"):
        agent._remember_thread("memory-1", "thread-1")
    assert "Reset thread state" not in caplog.text
    agent.close()


def test_deep_research_workflow_carries_user_id():
    captured: Dict[str, Any] = {}
    orchestrator = object.__new__(DeepResearchOrchestrator)

    def fake_workflow(user_query, memory_id=None, thread_id=None, **kwargs):
        captured.update(query=user_query, memory_id=memory_id, thread_id=thread_id)
        captured.update(kwargs)
        return "report"

    orchestrator.execute_multi_agent_workflow = fake_workflow
    workflow = DeepResearchWorkflow(orchestrator)

    assert workflow.run("q", memory_id="m-1", user_id="alice") == "report"
    assert captured == {
        "query": "q",
        "memory_id": "m-1",
        "thread_id": None,
        "user_id": "alice",
    }
