"""Auto-compaction: summarize older messages before a request passes compact_at."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memorizz.tooling import ContextPolicy

pytestmark = pytest.mark.unit


def test_threshold_is_bounded_and_saved_with_the_policy():
    assert ContextPolicy().compact_at == 80
    assert ContextPolicy(compact_at=95).compact_at == 80  # requests cap at 80%
    assert ContextPolicy(compact_at=10).compact_at == 30
    assert ContextPolicy(compact_at=0).compact_at == 0  # off
    assert ContextPolicy(keep_recent_messages=1).keep_recent_messages == 2
    saved = ContextPolicy(compact_at=60).to_dict()
    assert saved["compact_at"] == 60 and saved["keep_recent_messages"] == 6
    assert ContextPolicy.from_value(saved).compact_at == 60


def test_summaries_can_leave_the_newest_messages_alone(tmp_path, monkeypatch):
    from memorizz.enums import MemoryType
    from memorizz.memagent.managers.memory_manager import MemoryManager
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    for index in range(10):
        provider.store(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"message {index}",
                "memory_id": "thread-1",
                "thread_id": "thread-1",
                "user_id": None,
                "timestamp": f"2026-09-29T10:00:{index:02d}",
            },
            MemoryType.CONVERSATION_MEMORY,
        )
    manager = MemoryManager(provider)
    captured = []
    monkeypatch.setattr(
        manager,
        "compress_memories_with_llm",
        lambda memories, **kwargs: captured.append(memories) or "Older talk.",
    )
    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda text: [0.0] * 8)

    ids = manager.generate_summaries(
        model=None,
        agent_id="agent-1",
        memory_ids=["thread-1"],
        current_memory_id="thread-1",
        days_back=36500,
        max_memories_per_summary=100,
        keep_recent=4,
        summary_type="compaction",
    )

    assert len(ids) == 1
    assert [row["content"] for row in captured[0]] == [f"message {i}" for i in range(6)]
    summaries = manager.load_summaries_for_thread(
        memory_id="thread-1", agent_id="agent-1"
    )
    assert summaries[0]["summary_type"] == "compaction"
    assert summaries[0]["content"] == "Older talk."


def _history(count, words=60):
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": " ".join(f"word{index}" for _ in range(words)),
        }
        for index in range(count)
    ]


def _agent(monkeypatch, **policy):
    from memorizz import MemAgent
    from memorizz.enums import MemoryType

    agent = MemAgent(
        memory_provider=False,
        auto_register=False,
        context_window_tokens=2000,
        context_policy=policy or None,
    )
    agent.memory_manager = SimpleNamespace(
        load_summaries_for_thread=lambda **kwargs: [
            {
                "summary_id": "s1",
                "summary_type": "compaction",
                "content": "They agreed to ship on Thursday.",
            }
        ]
    )
    if MemoryType.SUMMARIES not in agent.active_memory_types:
        agent.active_memory_types.append(MemoryType.SUMMARIES)
    return agent


def test_compaction_keeps_recent_messages_and_reports_it(monkeypatch):
    from memorizz.memagent import core

    agent = _agent(monkeypatch, compact_at=50, keep_recent_messages=4)
    calls = []
    monkeypatch.setattr(
        agent, "generate_summaries", lambda **kwargs: calls.append(kwargs) or ["s1"]
    )
    events = []
    session = SimpleNamespace(
        emit=lambda kind, **payload: events.append((kind, payload))
    )
    monkeypatch.setattr(core, "session_for", lambda _agent: session)
    try:
        history = _history(12)
        context = {"conversation_history": history}
        kept = agent._auto_compact_history(history, "system", "question", context)

        assert kept == history[-4:]
        assert calls[0]["keep_recent"] == 4
        assert calls[0]["summary_type"] == "compaction"
        assert context["summaries"][0]["summary_id"] == "s1"
        kinds = [kind for kind, _ in events]
        assert kinds == ["status", "context.compacted"]
        compacted = events[-1][1]
        assert compacted["messages"] == 8 and compacted["threshold"] == 50
        assert compacted["tokens_after"] < compacted["tokens_before"]

        messages = agent._build_prompt_messages(
            "system",
            "question",
            {"conversation_history": kept, "summaries": context["summaries"]},
        )
        assert (
            "Earlier in this conversation (compacted summary):"
            in messages[-1]["content"]
        )
        assert "ship on Thursday" in messages[-1]["content"]
    finally:
        agent.close()


def test_only_this_conversations_compactions_enter_the_prompt(monkeypatch):
    agent = _agent(monkeypatch)
    try:
        agent._current_memory_id = "thread-b"
        summaries = [
            {
                "summary_type": "compaction",
                "memory_id": "thread-a",
                "content": "Alpha.",
            },
            {"summary_type": "compaction", "memory_id": "thread-b", "content": "B."},
        ]
        messages = agent._build_prompt_messages(
            "system", "question", {"conversation_history": [], "summaries": summaries}
        )
        prompt = messages[-1]["content"]
        assert "compacted summary):\nB." in prompt
        assert "Alpha." not in prompt
    finally:
        agent.close()


def test_no_compaction_below_threshold_or_when_off(monkeypatch):
    cases = (
        ({"compact_at": 80}, 4),
        ({"compact_at": 0}, 12),
        # Over the threshold but too few older messages for a batch.
        ({"compact_at": 30, "keep_recent_messages": 6}, 11),
    )
    for policy, count in cases:
        agent = _agent(monkeypatch, **policy)
        monkeypatch.setattr(
            agent, "generate_summaries", lambda **kwargs: pytest.fail("compacted")
        )
        try:
            history = _history(count)
            assert agent._auto_compact_history(history, "s", "q", {}) is history
        finally:
            agent.close()


# --------------------------------------------------------------------- UI


pytest.importorskip("fastapi")


def test_panel_counts_what_is_sent_not_traces_or_summarized_rows():
    import json

    from memorizz import MemAgent
    from memorizz.ui.routers.playground import _build_token_stats

    agent = MemAgent(
        memory_provider=False, auto_register=False, context_window_tokens=16384
    )
    try:
        trace = {
            "role": "tool",
            "content": json.dumps({"type": "trace_bundle", "events": ["x"] * 500}),
        }
        rows = _history(4, words=10) + [
            trace,
            {"role": "user", "content": "old", "summary_id": "s1"},
        ]
        stats = _build_token_stats(agent, rows, toolbox_memory=[{"name": "dup"}] * 50)
        assert stats["thread_message_count"] == 4
        assert stats["message_count"] == 4
        assert set(stats["composition"]) == {
            "system_prompt",
            "tools",
            "memory",
            "history",
        }
        assert stats["composition"]["history"] < 200
        assert stats["context_window_tokens"] == 16384
        assert stats["compact_at"] == 80
    finally:
        agent.close()


class _Provider:
    def __init__(self):
        self.agent = SimpleNamespace(
            agent_id="agent-1",
            name="Assistant",
            persona=None,
            context_policy={"tool_top_k": 3},
            memory_types=["conversation_memory", "entity_memory"],
        )

    def list_memagents(self):
        return [self.agent]

    def retrieve_memagent(self, agent_id):
        return self.agent if agent_id == "agent-1" else None

    def store_memagent(self, agent):
        self.agent = agent
        return agent.agent_id


def test_threshold_is_changed_from_the_ui(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _Provider()
    client = TestClient(create_app())
    with monkeypatch.context() as patcher:
        patcher.setitem(state._state, "provider", provider)
        patcher.setitem(state._state, "provider_type", "filesystem")
        patcher.setitem(state._state, "connection_info", {})

        saved = client.post(
            "/api/agents/agent-1/context-policy", json={"compact_at": 60}
        )
        assert saved.status_code == 200, saved.text
        assert provider.agent.context_policy["compact_at"] == 60
        assert provider.agent.context_policy["tool_top_k"] == 3  # kept
        assert "summaries" in provider.agent.memory_types
        assert saved.json()["summary_memory"] is True

        off = client.post("/api/agents/agent-1/context-policy", json={"compact_at": 0})
        assert off.json()["context_policy"]["compact_at"] == 0
        assert (
            client.post("/api/agents/agent-1/context-policy", json={}).status_code
            == 400
        )


def test_summaries_are_kept_when_embedding_fails(tmp_path, monkeypatch):
    """No embedding provider (no key) must not silently lose the summary."""
    from memorizz.enums import MemoryType
    from memorizz.memagent.managers.memory_manager import MemoryManager
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    for index in range(4):
        provider.store(
            {
                "role": "user",
                "content": f"m{index}",
                "memory_id": "t",
                "user_id": None,
                "timestamp": f"2026-09-29T10:00:0{index}",
            },
            MemoryType.CONVERSATION_MEMORY,
        )
    manager = MemoryManager(provider)
    monkeypatch.setattr(manager, "compress_memories_with_llm", lambda *a, **k: "Gist.")

    def no_key(text):
        raise RuntimeError("Missing credentials")

    monkeypatch.setattr("memorizz.embeddings.get_embedding", no_key)
    ids = manager.generate_summaries(
        model=None,
        agent_id="a",
        memory_ids=["t"],
        current_memory_id="t",
        days_back=36500,
        keep_recent=2,
        summary_type="compaction",
    )
    assert len(ids) == 1
    assert provider.retrieve_by_id(ids[0], MemoryType.SUMMARIES)["content"] == "Gist."
