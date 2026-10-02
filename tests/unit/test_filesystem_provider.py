"""Tests for the filesystem memory provider."""

from pathlib import Path
from typing import List

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.semantic.entity_memory import EntityMemory
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


def test_tool_log_search_recovers_old_output_with_bounded_scoped_excerpt(tmp_path):
    from memorizz.memagent.managers.memory_manager import MemoryManager

    provider = _make_provider(tmp_path)
    manager = MemoryManager(provider)
    original = (
        "setup\n" * 300 + "FAILED test_clock_skew expected 7 got 9\n" + "tail\n" * 300
    )
    source_id = manager.store_tool_log(
        "pytest",
        {"path": "tests/test_clock.py"},
        original,
        memory_id="task",
        user_id="alice",
        thread_id="window-1",
    )
    for i in range(30):
        manager.store_tool_log(
            "pytest",
            {},
            f"new passing test {i}",
            memory_id="task",
            user_id="alice",
            thread_id="window-2",
        )
    for task, user in [("other-task", "alice"), ("task", "bob"), ("task", None)]:
        manager.store_tool_log(
            "pytest",
            {},
            "test_clock_skew PRIVATE",
            memory_id=task,
            user_id=user,
            thread_id="window-1",
        )
    hits = manager.search_tool_logs(
        "test_clock_skew",
        memory_id="task",
        user_id="alice",
        excerpt_chars=100,
    )
    assert len(hits) == 1
    assert hits[0]["tool_log_id"] == source_id
    assert "test_clock_skew" in hits[0]["excerpt"]
    assert len(hits[0]["excerpt"]) <= 100
    assert hits[0]["excerpt_start"] > 0
    assert manager.retrieve_tool_log(source_id, user_id="alice")["result"] == original
    assert (
        manager.search_tool_logs(
            "test_clock_skew", memory_id="task", user_id="alice", thread_id="window-2"
        )
        == []
    )
    assert manager.search_tool_logs("  ", memory_id="task", user_id="alice") == []
    assert (
        manager.search_tool_logs(
            "tests/test_clock.py", memory_id="task", user_id="alice"
        )[0]["matched_field"]
        == "arguments"
    )

    # The raw filesystem query also searches the actual result field.
    hits = provider.retrieve_by_query(
        "test_clock_skew", MemoryType.TOOL_LOG, memory_id="task", user_id="alice"
    )
    assert len(hits) == 1
    assert hits[0]["result"] == original


@pytest.mark.parametrize(
    "kwargs", [{"limit": 0}, {"excerpt_chars": 0}, {"limit": True}]
)
def test_tool_log_search_rejects_unbounded_output(tmp_path, kwargs):
    from memorizz.memagent.managers.memory_manager import MemoryManager

    manager = MemoryManager(_make_provider(tmp_path))
    with pytest.raises(ValueError):
        manager.search_tool_logs("failure", memory_id="task", user_id="alice", **kwargs)


def _make_provider(tmp_path, embedding_provider=None) -> FileSystemProvider:
    root = Path(tmp_path) / "fs-memory"
    config = FileSystemConfig(
        root_path=root, embedding_provider=embedding_provider, lazy_vector_indexes=True
    )
    return FileSystemProvider(config)


def test_exact_search_mode_does_not_load_faiss(tmp_path, monkeypatch):
    from memorizz.memory_provider.filesystem import provider as provider_module

    def unexpected_import(name):
        if name == "faiss":
            raise AssertionError("FAISS must not load in exact-search mode")
        return __import__(name)

    monkeypatch.setattr(provider_module.importlib, "import_module", unexpected_import)
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=tmp_path / "exact-memory",
            embedding_provider=DummyEmbeddingProvider(),
            use_faiss=False,
        )
    )

    provider.store(
        {"content": "alpha memory", "embedding": [1.0, 0.0, 0.0]},
        MemoryType.KNOWLEDGE_BASE,
    )
    assert provider.retrieve_by_query("alpha", MemoryType.KNOWLEDGE_BASE, limit=1)


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


def test_entity_legacy_scope_migration_is_explicit_and_strict(tmp_path):
    provider = _make_provider(tmp_path)
    provider.store(
        {
            "entity_id": "legacy-entity",
            "name": "user",
            "memory_id": "shared-memory",
            "user_id": None,
        },
        MemoryType.ENTITY_MEMORY,
    )
    entities = EntityMemory(provider)

    assert (
        entities.get_entity_by_name("user", memory_id="shared-memory", user_id="user-a")
        is None
    )
    assert (
        entities.migrate_legacy_scope(memory_id="shared-memory", user_id="user-a") == 1
    )
    assert (
        entities.get_entity_by_name(
            "user", memory_id="shared-memory", user_id="user-a"
        )["entity_id"]
        == "legacy-entity"
    )
    assert (
        entities.get_entity_by_name("user", memory_id="shared-memory", user_id=None)
        is None
    )


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


def test_memagent_stores_a_persona_given_as_a_dict(tmp_path):
    """The UI passes personas as dicts when the PERSONAS store is unavailable."""
    provider = _make_provider(tmp_path)
    persona = {"name": "Helper", "role": "assistant", "goals": "g", "background": "b"}

    agent_id = provider.store_memagent(MemAgentModel(instruction="x", persona=persona))

    assert provider.retrieve_memagent(agent_id).persona.name == "Helper"


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


def test_keyword_search_recalls_scoped_memory_for_delegated_query_suffix(tmp_path):
    provider = _make_provider(tmp_path)
    base_query = (
        "Review access_policy.py and verify.py for security and correctness "
        "against the retrieved export policy requirements"
    )
    expected_id = provider.store(
        {
            "content": f"Applicable request: {base_query}. Expired grants deny export.",
            "memory_id": "panel-memory",
            "user_id": "alice",
            "thread_id": "repeat-1",
        },
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )
    provider.store(
        {
            "content": "Unrelated deployment notes for a different tenant",
            "memory_id": "other-memory",
            "user_id": "bob",
            "thread_id": "repeat-1",
        },
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )
    provider._embedding_provider = None
    provider._get_embedding_provider = lambda: None

    results = provider.retrieve_by_query(
        base_query + ". Focus on datetime behavior, typing, and verifier strength.",
        memory_type=MemoryType.KNOWLEDGE_BASE,
        memory_id="panel-memory",
        user_id="alice",
        thread_id="repeat-1",
        limit=2,
    )

    assert [row["_id"] for row in results] == [expected_id]
    assert results[0]["score"] > 0.8


def test_semantic_search_falls_back_without_scoped_document_vectors(tmp_path):
    embeddings = DummyEmbeddingProvider()
    provider = _make_provider(tmp_path, embedding_provider=embeddings)
    provider.store(
        {
            "content": "exact scoped grounding requirement",
            "memory_id": "memory-a",
            "user_id": "alice",
            "thread_id": "thread-a",
        },
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )

    results = provider.retrieve_by_query(
        "exact scoped grounding requirement",
        memory_type=MemoryType.KNOWLEDGE_BASE,
        memory_id="memory-a",
        user_id="alice",
        thread_id="thread-a",
        limit=2,
    )

    assert results and results[0]["memory_id"] == "memory-a"
    assert embeddings.calls == []


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


def test_tool_logs_resolve_by_tool_log_id(tmp_path):
    """The recent-logs hint shows the model tool_log_id; it must resolve."""
    from memorizz.memagent.managers.memory_manager import MemoryManager

    provider = _make_provider(tmp_path)
    manager = MemoryManager(provider)
    stored = manager.store_tool_log(
        "web_search", {"q": "x"}, "output", memory_id="mem-1", user_id="u1"
    )
    row = manager.list_tool_logs("mem-1", user_id="u1")[0]
    assert row["tool_log_id"] == stored
    assert manager.retrieve_tool_log(row["tool_log_id"], user_id="u1")["result"] == (
        "output"
    )

    # Logs stored before the IDs matched are still found by tool_log_id.
    legacy = provider.store(
        {"tool_log_id": "legacy-uuid", "result": "old", "_id": "legacy-record"},
        memory_store_type=MemoryType.TOOL_LOG,
    )
    assert legacy == "legacy-record"
    assert (
        provider.retrieve_by_id("legacy-uuid", MemoryType.TOOL_LOG)["result"] == "old"
    )
    assert provider.retrieve_by_id("missing", MemoryType.TOOL_LOG) is None


def test_memagents_keep_their_creation_time_across_saves(tmp_path):
    """The agents list sorts by created_at, so saves must not reset it."""
    import json

    provider = _make_provider(tmp_path)
    first = provider.store_memagent(MemAgentModel(instruction="first"))
    created = provider.list_all(MemoryType.MEMAGENT)[0]["created_at"]

    agent = provider.retrieve_memagent(first)
    agent.instruction = "edited"
    provider.store_memagent(agent)
    second = provider.store_memagent(MemAgentModel(instruction="second"))

    docs = {doc["agent_id"]: doc for doc in provider.list_all(MemoryType.MEMAGENT)}
    assert docs[first]["created_at"] == created
    assert docs[second]["created_at"] >= created

    # Agents written before created_at existed get their file's time.
    path = next(tmp_path.rglob(f"{second}.json"))
    legacy = json.loads(path.read_text())
    legacy.pop("created_at")
    path.write_text(json.dumps(legacy))
    assert provider.retrieve_by_id(second, MemoryType.MEMAGENT)["created_at"]
