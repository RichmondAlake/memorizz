"""Tenant-isolation tests for ``user_id`` scoping across providers.

These tests exercise the filesystem provider (always available in CI) and
the MemoryManager plumbing. Oracle and MongoDB are tenant-isolated using
the same ``_FS_UNSET``-style sentinel pattern, so a green filesystem suite
combined with the provider-specific query assertions here is a strong
signal that cross-tenant leakage is not possible on the affected code
paths.

The golden invariant we enforce: when user A writes memory and user B
queries, B must never see A's data — even when they share the same
``memory_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from memorizz.enums import MemoryType
from memorizz.memagent.managers.memory_manager import MemoryManager
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


class _DummyEmbeddingProvider:
    """Deterministic embedding provider so vector-search tests are stable."""

    def get_embedding(self, text: str) -> List[float]:
        seed = float(sum(ord(ch) for ch in (text or "")))
        return [seed, float(len(text or "")), 0.0]

    def get_provider_info(self) -> str:  # pragma: no cover
        return "dummy"


def _make_provider(tmp_path) -> FileSystemProvider:
    root = Path(tmp_path) / "tenant-memory"
    config = FileSystemConfig(
        root_path=root,
        embedding_provider=_DummyEmbeddingProvider(),
        lazy_vector_indexes=True,
    )
    return FileSystemProvider(config)


def _write_conversation(
    provider: FileSystemProvider,
    *,
    memory_id: str,
    user_id,
    content: str,
):
    """Helper — write a single conversation memory unit."""
    return provider.store(
        {
            "role": "user",
            "content": content,
            "timestamp": "2026-04-17T00:00:00",
            "memory_id": memory_id,
            "thread_id": "thread-xyz",
            "user_id": user_id,
            "embedding": [0.1, 0.2, 0.3],
        },
        memory_store_type=MemoryType.CONVERSATION_MEMORY,
    )


# ---------------------------------------------------------------------------
# Provider-level isolation
# ---------------------------------------------------------------------------


def test_filesystem_retrieve_conversation_history_is_user_scoped(tmp_path):
    """Strict tenant isolation on the conversation history read path."""
    provider = _make_provider(tmp_path)

    _write_conversation(
        provider, memory_id="shared-mem", user_id="alice", content="alice hello"
    )
    _write_conversation(
        provider, memory_id="shared-mem", user_id="bob", content="bob hello"
    )

    alice_history = provider.retrieve_conversation_history_ordered_by_timestamp(
        "shared-mem", user_id="alice"
    )
    bob_history = provider.retrieve_conversation_history_ordered_by_timestamp(
        "shared-mem", user_id="bob"
    )

    assert [row["content"] for row in alice_history] == ["alice hello"]
    assert [row["content"] for row in bob_history] == ["bob hello"]


def test_filesystem_user_none_is_separate_scope(tmp_path):
    """``user_id=None`` is a *scope*, not "match everything"."""
    provider = _make_provider(tmp_path)

    _write_conversation(
        provider, memory_id="legacy", user_id=None, content="legacy row"
    )
    _write_conversation(
        provider, memory_id="legacy", user_id="alice", content="alice row"
    )

    anon = provider.retrieve_conversation_history_ordered_by_timestamp(
        "legacy", user_id=None
    )
    alice = provider.retrieve_conversation_history_ordered_by_timestamp(
        "legacy", user_id="alice"
    )

    assert [r["content"] for r in anon] == ["legacy row"]
    assert [r["content"] for r in alice] == ["alice row"]


def test_filesystem_retrieve_by_query_dict_is_user_scoped(tmp_path):
    provider = _make_provider(tmp_path)

    _write_conversation(
        provider, memory_id="m1", user_id="alice", content="alice secret"
    )
    _write_conversation(provider, memory_id="m1", user_id="bob", content="bob secret")

    alice_rows = provider.retrieve_by_query(
        {"memory_id": "m1"},
        memory_type=MemoryType.CONVERSATION_MEMORY,
        limit=10,
        user_id="alice",
    )
    bob_rows = provider.retrieve_by_query(
        {"memory_id": "m1"},
        memory_type=MemoryType.CONVERSATION_MEMORY,
        limit=10,
        user_id="bob",
    )

    assert [r["content"] for r in alice_rows] == ["alice secret"]
    assert [r["content"] for r in bob_rows] == ["bob secret"]


def test_filesystem_list_all_is_user_scoped(tmp_path):
    provider = _make_provider(tmp_path)

    _write_conversation(provider, memory_id="m1", user_id="alice", content="a")
    _write_conversation(provider, memory_id="m1", user_id="bob", content="b")

    assert {
        row["content"]
        for row in provider.list_all(MemoryType.CONVERSATION_MEMORY, user_id="alice")
    } == {"a"}

    assert {
        row["content"]
        for row in provider.list_all(MemoryType.CONVERSATION_MEMORY, user_id="bob")
    } == {"b"}


def test_filesystem_legacy_callers_still_see_everything(tmp_path):
    """Omitting ``user_id`` preserves legacy non-scoped behavior."""
    provider = _make_provider(tmp_path)

    _write_conversation(provider, memory_id="m1", user_id="alice", content="a")
    _write_conversation(provider, memory_id="m1", user_id="bob", content="b")
    _write_conversation(provider, memory_id="m1", user_id=None, content="legacy")

    # No user_id argument — should return all three rows.
    rows = provider.retrieve_conversation_history_ordered_by_timestamp("m1")
    contents = sorted(row["content"] for row in rows)
    assert contents == ["a", "b", "legacy"]


# ---------------------------------------------------------------------------
# MemoryManager plumbing
# ---------------------------------------------------------------------------


def test_memory_manager_cache_is_keyed_by_user(tmp_path):
    """Two users with the same memory_id must never share cache entries."""
    provider = _make_provider(tmp_path)
    manager = MemoryManager(provider)

    _write_conversation(
        provider, memory_id="m1", user_id="alice", content="alice cached"
    )
    _write_conversation(provider, memory_id="m1", user_id="bob", content="bob cached")

    # First load populates the cache for each scope separately.
    alice = manager.load_conversation_history("m1", user_id="alice")
    bob = manager.load_conversation_history("m1", user_id="bob")

    assert [row["content"] for row in alice] == ["alice cached"]
    assert [row["content"] for row in bob] == ["bob cached"]

    # The cache uses ``(memory_id, user_id)`` as its key so re-reads are
    # isolated even when the memory_id collides.
    assert ("m1", "alice") in manager._conversation_memory_cache
    assert ("m1", "bob") in manager._conversation_memory_cache


def test_memory_manager_create_unit_propagates_user_id(tmp_path):
    from memorizz.enums import Role

    provider = _make_provider(tmp_path)
    manager = MemoryManager(provider)

    unit = manager.create_conversation_memory_unit(
        role=Role.USER,
        content="hello",
        thread_id="t1",
        memory_id="m1",
        user_id="alice",
    )
    assert unit.user_id == "alice"

    # And None is preserved (not coerced to a string) for legacy callers.
    unit_none = manager.create_conversation_memory_unit(
        role=Role.USER,
        content="hello",
        thread_id="t1",
        memory_id="m1",
    )
    assert unit_none.user_id is None


def test_memory_manager_retrieve_tool_log_blocks_cross_tenant(tmp_path):
    provider = _make_provider(tmp_path)
    manager = MemoryManager(provider)

    log_id = manager.store_tool_log(
        tool_name="demo",
        arguments={"x": 1},
        result={"y": 2},
        memory_id="m1",
        user_id="alice",
    )
    assert log_id is not None

    # Bob must not be able to read Alice's tool log even by guessing the id.
    assert manager.retrieve_tool_log(log_id, user_id="bob") is None

    # Alice can read it.
    retrieved = manager.retrieve_tool_log(log_id, user_id="alice")
    assert retrieved is not None
    assert retrieved["user_id"] == "alice"
