"""Tests for the filesystem memory provider."""

from pathlib import Path
from typing import List

import pytest

from memorizz.enums import MemoryType
from memorizz.memagent import MemAgentModel
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


class DummyEmbeddingProvider:
    """Minimal embedding provider used to avoid network calls."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def get_embedding(self, text: str) -> List[float]:
        self.calls.append(text)
        seed = float(sum(ord(ch) for ch in text))
        return [seed, float(len(text) or 1), 0.0]

    def get_provider_info(self) -> str:
        return "dummy"


def _make_provider(tmp_path, embedding_provider=None) -> FileSystemProvider:
    root = Path(tmp_path) / "fs-memory"
    config = FileSystemConfig(
        root_path=root, embedding_provider=embedding_provider, lazy_vector_indexes=True
    )
    return FileSystemProvider(config)


def test_store_and_query_documents(tmp_path):
    provider = _make_provider(tmp_path)

    doc_id = provider.store(
        {
            "name": "demo",
            "content": "hello filesystem memory",
            "memory_id": "memory-123",
        },
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )

    retrieved = provider.retrieve_by_id(doc_id, MemoryType.KNOWLEDGE_BASE)
    assert retrieved["content"] == "hello filesystem memory"

    results = provider.retrieve_by_query(
        {"memory_id": "memory-123"},
        memory_type=MemoryType.KNOWLEDGE_BASE,
        limit=1,
    )
    assert results and results[0]["id"] == doc_id

    provider.delete_by_id(doc_id, MemoryType.KNOWLEDGE_BASE)
    assert provider.list_all(MemoryType.KNOWLEDGE_BASE) == []


def test_memagent_round_trip(tmp_path):
    provider = _make_provider(tmp_path)

    agent = MemAgentModel(
        instruction="test agent",
        memory_ids=["mem-1"],
        application_mode="assistant",
        self_aware=True,
        self_aware_config={
            "root_paths": ["."],
            "allow_writes": True,
            "allow_deletes": False,
            "policy_version": "v1",
            "timeout_seconds": 30,
            "max_output_chars": 50000,
            "max_file_read_bytes": 250000,
            "max_file_write_bytes": 250000,
        },
        skill_paths=["skills/alpha.skills.md"],
        mcp_servers=[
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
            }
        ],
    )
    agent_id = provider.store_memagent(agent)

    loaded = provider.retrieve_memagent(agent_id)
    assert loaded is not None
    assert loaded.memory_ids == ["mem-1"]
    assert loaded.skill_paths == ["skills/alpha.skills.md"]
    assert loaded.mcp_servers[0]["name"] == "filesystem"
    assert loaded.self_aware is True
    assert loaded.self_aware_config["allow_writes"] is True

    listed_agents = provider.list_memagents()
    assert len(listed_agents) == 1
    assert listed_agents[0].self_aware is True
    assert listed_agents[0].self_aware_config["allow_writes"] is True

    provider.delete_memagent(agent_id)
    assert provider.retrieve_memagent(agent_id) is None


def test_semantic_query_uses_embedding_provider(tmp_path):
    dummy = DummyEmbeddingProvider()
    provider = _make_provider(tmp_path, embedding_provider=dummy)

    provider.store(
        {
            "content": "alpha memory block",
            "memory_id": "alpha",
            "embedding": dummy.get_embedding("alpha memory block"),
        },
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )
    provider.store(
        {
            "content": "beta unrelated record",
            "memory_id": "beta",
            "embedding": dummy.get_embedding("beta unrelated record"),
        },
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )

    results = provider.retrieve_by_query(
        "alpha memory block",
        memory_type=MemoryType.KNOWLEDGE_BASE,
        limit=1,
        memory_id="alpha",
    )
    assert results and results[0]["memory_id"] == "alpha"
    assert "alpha memory block" in dummy.calls


def test_keyword_search_without_embeddings(tmp_path):
    provider = _make_provider(tmp_path)
    provider.store(
        {"content": "remember keyword fallback", "memory_id": "k1"},
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )

    # Force keyword path by disabling embedding lookups
    provider._embedding_provider = None
    provider._get_embedding_provider = lambda: None

    results = provider.retrieve_by_query(
        "keyword fallback", memory_type=MemoryType.KNOWLEDGE_BASE, limit=1
    )
    assert results and results[0]["memory_id"] == "k1"


def test_conversation_history_can_be_scoped_to_one_thread(tmp_path):
    provider = _make_provider(tmp_path)
    for thread_id, content in (
        ("thread-a", "message from a"),
        ("thread-b", "message from b"),
    ):
        provider.store(
            {
                "memory_id": "shared-memory",
                "thread_id": thread_id,
                "role": "user",
                "content": content,
                "timestamp": "2026-01-01T12:00:00+00:00",
                "user_id": None,
            },
            memory_store_type=MemoryType.CONVERSATION_MEMORY,
        )

    rows = provider.retrieve_conversation_history_ordered_by_timestamp(
        "shared-memory",
        memory_type=MemoryType.CONVERSATION_MEMORY,
        user_id=None,
        thread_id="thread-b",
    )

    assert [row["content"] for row in rows] == ["message from b"]


def test_delete_memagent_cascade_removes_memories(tmp_path):
    provider = _make_provider(tmp_path)

    memory_id = "shared-memory"
    provider.store(
        {"content": "greeting", "memory_id": memory_id},
        memory_store_type=MemoryType.CONVERSATION_MEMORY,
    )

    agent = MemAgentModel(
        instruction="cascade",
        memory_ids=[memory_id],
        application_mode="assistant",
    )
    agent_id = provider.store_memagent(agent)

    provider.delete_memagent(agent_id, cascade=True)
    assert provider.list_all(MemoryType.CONVERSATION_MEMORY) == []
