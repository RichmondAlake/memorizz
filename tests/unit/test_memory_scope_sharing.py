"""Memory scope, deletion and agent-wide sharing.

* Deleting an agent's memory removes every row of its memory_ids, and a
  cascade agent delete keeps the rows of a memory_id another agent still uses.
* Clearing one user's (or session's) semantic cache leaves other users' rows.
* Knowledge bases, skills and tools without a user belong to the whole agent:
  every user of the agent sees them, while rows written for a user stay
  private. Personal stores (conversations, ...) keep exact isolation.
* Learned skills are promoted and stamped per user scope.
* Evidence packs see attached knowledge bases and agent-owned skills.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from memorizz.enums import MemoryType
from memorizz.memagent.managers.memory_manager import MemoryManager
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


class _Embeddings:
    def get_embedding(self, text):
        value = str(text).lower()
        return [
            float("release" in value),
            float("tuesday" in value),
            float("london" in value) + 0.01,
        ]

    def get_provider_info(self):
        return "unit-test"


@pytest.fixture()
def provider(tmp_path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "scope-memory",
            embedding_provider=_Embeddings(),
            lazy_vector_indexes=True,
        )
    )


def _store_agent(provider, agent_id, memory_ids, knowledge_base_ids=None):
    provider.store_memagent(
        MemAgentModel(
            agent_id=agent_id,
            name=agent_id,
            instruction="hi",
            memory_ids=memory_ids,
            knowledge_base_ids=knowledge_base_ids,
        )
    )


_MEMORY_STORES = (
    MemoryType.CONVERSATION_MEMORY,
    MemoryType.SUMMARIES,
    MemoryType.TOOL_LOG,
    MemoryType.WORKFLOW_MEMORY,
    MemoryType.ENTITY_MEMORY,
)


def _ids(provider, memory_type) -> List[str]:
    return sorted(str(row["_id"]) for row in provider.list_all(memory_type))


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_memory_manager_delete_memory_removes_every_store_row(provider):
    for memory_type in _MEMORY_STORES:
        for memory_id in ("mem-1", "mem-2"):
            provider.store(
                {"_id": f"{memory_type.value}-{memory_id}", "memory_id": memory_id},
                memory_type,
            )

    assert MemoryManager(provider).delete_memory("mem-1")

    for memory_type in _MEMORY_STORES:
        assert _ids(provider, memory_type) == [f"{memory_type.value}-mem-2"]


@pytest.mark.unit
def test_memagent_delete_memory_reports_failure_and_keeps_ids():
    from memorizz.memagent import MemAgent

    class _FailingManager:
        def delete_memory(self, memory_id, agent_id=None):
            return memory_id != "mem-broken"

    agent = SimpleNamespace(
        agent_id="agent-1",
        memory_ids=["mem-1", "mem-broken"],
        memory_manager=_FailingManager(),
        memory_provider=SimpleNamespace(delete_memagent_memory_ids=lambda _id: True),
        _current_memory_id="mem-1",
        _current_thread_id="thread-1",
        _thread_ids_by_memory={},
        _thread_map_lock=threading.Lock(),
        _last_thread_by_memory={},
    )

    assert MemAgent.delete_memory(agent) is False
    assert agent.memory_ids == ["mem-1", "mem-broken"]


@pytest.mark.unit
def test_filesystem_cascade_keeps_memory_another_agent_uses(provider):
    _store_agent(provider, "root", ["shared"])
    _store_agent(provider, "delegate", ["shared", "own"])
    for memory_id in ("shared", "own"):
        provider.store(
            {"_id": f"turn-{memory_id}", "memory_id": memory_id},
            MemoryType.CONVERSATION_MEMORY,
        )

    assert provider.delete_memagent("delegate", cascade=True)

    assert _ids(provider, MemoryType.CONVERSATION_MEMORY) == ["turn-shared"]
    assert provider.retrieve_memagent("root") is not None


@pytest.mark.unit
def test_filesystem_cascade_deletes_memory_once_no_agent_uses_it(provider):
    _store_agent(provider, "solo", ["only"])
    provider.store({"_id": "turn", "memory_id": "only"}, MemoryType.SUMMARIES)

    assert provider.delete_memagent("solo", cascade=True)

    assert _ids(provider, MemoryType.SUMMARIES) == []


# ---------------------------------------------------------------------------
# Semantic cache clearing
# ---------------------------------------------------------------------------


class _CacheEmbeddings:
    def get_embedding(self, text):
        return [1.0, float(len(str(text)) % 7)]


@pytest.mark.unit
def test_semantic_cache_user_clear_keeps_other_users_stored_rows(provider):
    from memorizz.short_term_memory.semantic_cache import (
        SemanticCache,
        SemanticCacheConfig,
    )

    def cache():
        return SemanticCache(
            config=SemanticCacheConfig(enable_memory_provider_sync=True),
            memory_provider=provider,
            embedding_manager=_CacheEmbeddings(),
            agent_id="agent-1",
            memory_id="memory-1",
        )

    writer = cache()
    assert writer.set("what is the plan", "alice's answer", user_id="alice")
    assert writer.set("what is the plan", "bob's answer", user_id="bob")

    cache().clear(user_id="alice")

    owners = [
        row.get("user_id") for row in provider.list_all(MemoryType.SEMANTIC_CACHE)
    ]
    assert owners == ["bob"]


@pytest.mark.unit
def test_clear_semantic_cache_scopes_by_user_and_session(provider):
    for user_id, session_id in (("alice", "s1"), ("alice", "s2"), ("bob", "s1")):
        provider.store(
            {
                "_id": f"{user_id}-{session_id}",
                "agent_id": "agent-1",
                "user_id": user_id,
                "session_id": session_id,
            },
            MemoryType.SEMANTIC_CACHE,
        )

    assert provider.clear_semantic_cache("agent-1", user_id="alice", session_id="s1")
    assert _ids(provider, MemoryType.SEMANTIC_CACHE) == ["alice-s2", "bob-s1"]
    assert provider.clear_semantic_cache("agent-1", session_id="s1") == 1
    assert _ids(provider, MemoryType.SEMANTIC_CACHE) == ["alice-s2"]


# ---------------------------------------------------------------------------
# Agent-wide sharing
# ---------------------------------------------------------------------------


def _store_owned(provider, memory_type, content, **fields):
    for user_id in (None, "alice", "bob"):
        provider.store(
            {
                "_id": f"{memory_type.value}-{user_id or 'agent'}",
                "memory_id": "memory-1",
                "user_id": user_id,
                "content": content,
                "embedding": _Embeddings().get_embedding(content),
                **fields,
            },
            memory_type,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "query",
    [
        "When is the London release?",
        {"memory_id": "memory-1"},
    ],
)
def test_agent_wide_knowledge_is_visible_to_each_user_but_not_others(provider, query):
    _store_owned(provider, MemoryType.KNOWLEDGE_BASE, "London release on Tuesday")

    rows = provider.retrieve_by_query(
        query,
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
        memory_id="memory-1",
        user_id="alice",
        limit=10,
    )

    assert sorted(row["user_id"] or "agent" for row in rows) == ["agent", "alice"]
    listed = provider.list_all(MemoryType.KNOWLEDGE_BASE, user_id="alice")
    assert sorted(row["user_id"] or "agent" for row in listed) == ["agent", "alice"]


@pytest.mark.unit
def test_personal_stores_keep_exact_user_isolation(provider):
    _store_owned(provider, MemoryType.CONVERSATION_MEMORY, "London release on Tuesday")

    rows = provider.retrieve_by_query(
        "When is the London release?",
        memory_store_type=MemoryType.CONVERSATION_MEMORY,
        memory_id="memory-1",
        user_id="alice",
        limit=10,
    )

    assert [row["user_id"] for row in rows] == ["alice"]
    listed = provider.list_all(MemoryType.CONVERSATION_MEMORY, user_id="alice")
    assert [row["user_id"] for row in listed] == ["alice"]


@pytest.mark.unit
def test_anonymous_reads_do_not_see_user_knowledge(provider):
    _store_owned(provider, MemoryType.KNOWLEDGE_BASE, "London release on Tuesday")

    listed = provider.list_all(MemoryType.KNOWLEDGE_BASE, user_id=None)

    assert [row["user_id"] for row in listed] == [None]


@pytest.mark.unit
def test_memory_manager_knowledge_recall_includes_agent_wide_rows(provider):
    _store_owned(provider, MemoryType.KNOWLEDGE_BASE, "London release on Tuesday")

    rows = MemoryManager(provider).retrieve_relevant_memories(
        "When is the London release?",
        MemoryType.KNOWLEDGE_BASE,
        "memory-1",
        limit=10,
        user_id="bob",
    )

    assert sorted(row["user_id"] or "agent" for row in rows) == ["agent", "bob"]


@pytest.mark.unit
def test_knowledge_base_reads_include_agent_wide_chunks(provider):
    from memorizz.long_term.semantic.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(provider)
    kb_id = kb.ingest_knowledge(
        "The London release train runs on Tuesday.", "docs", embeddings="off"
    )

    assert kb.retrieve_knowledge(kb_id, user_id="alice")
    agent = SimpleNamespace(knowledge_base_ids=[], save=lambda: None)
    assert kb.attach_to_agent(agent, kb_id, user_id="alice")
    assert agent.knowledge_base_ids == [kb_id]


@pytest.mark.unit
def test_skill_lookup_includes_agent_wide_skills(provider, monkeypatch):
    from memorizz.long_term.procedural.skillbox import Skill, Skillbox, SkillStatus
    from memorizz.long_term.procedural.skillbox import skillbox as skillbox_module

    monkeypatch.setattr(skillbox_module, "get_embedding", lambda text: [1.0, 0.0, 0.0])
    box = Skillbox(provider, agent_id="agent-1")
    for user_id in (None, "alice", "bob"):
        box.add_skill(
            Skill(
                name=f"refunds {user_id or 'agent'}",
                description="Use when the user asks for a refund",
                content="Look up the order, then refund it.",
                agent_id="agent-1",
                user_id=user_id,
                source_canonical_hash="hash-1",
                status=SkillStatus.ACTIVE,
                embedding=[1.0, 0.0, 0.0],
            )
        )

    found = box.retrieve_skills_by_query(
        "refund my order", limit=5, min_similarity=0.0, user_id="alice"
    )

    assert sorted(item.skill.user_id or "agent" for item in found) == [
        "agent",
        "alice",
    ]
    # A user without a skill of their own is covered by the agent-wide one.
    covering = box.get_active_skill_for_hash("hash-1", user_id="carol")
    assert covering is not None and covering.user_id is None


@pytest.mark.unit
def test_toolbox_and_router_include_agent_wide_tools():
    from memorizz.long_term.procedural.toolbox.toolbox import Toolbox
    from memorizz.tooling import SemanticToolRouter

    rows = [
        {"name": "shared_tool", "user_id": None, "agent_id": "agent-1"},
        {"name": "alice_tool", "user_id": "alice", "agent_id": "agent-1"},
        {"name": "bob_tool", "user_id": "bob", "agent_id": "agent-1"},
    ]
    toolbox = Toolbox.__new__(Toolbox)
    toolbox.agent_id = "agent-1"
    toolbox.memory_provider = SimpleNamespace(
        retrieve_by_query=lambda *args, **kwargs: list(rows)
    )

    found = toolbox.get_most_similar_tools("q", limit=5, user_id="alice")
    assert [row["name"] for row in found] == ["shared_tool", "alice_tool"]

    manager = SimpleNamespace(
        list_tools=lambda: ["shared_tool", "alice_tool", "bob_tool"]
    )
    router = SemanticToolRouter.__new__(SemanticToolRouter)
    router.toolbox = SimpleNamespace(
        get_most_similar_tools=lambda *args, **kwargs: list(rows)
    )
    router.tool_manager = manager
    router.agent_id = "agent-1"
    router._resolve_name = lambda name: name

    names = router._semantic_candidates("q", user_id="alice", limit=5)
    assert names == ["shared_tool", "alice_tool"]


# ---------------------------------------------------------------------------
# Skill promotion per user scope
# ---------------------------------------------------------------------------


NOW = datetime.now()


class _WorkflowProvider:
    """Workflow docs for aggregation plus recording updates."""

    def __init__(self, docs):
        self.docs = docs
        self.updates = []

    def list_all(self, memory_store_type=None):
        return self.docs

    def retrieve_by_id(self, id, memory_store_type=None):
        return next((d for d in self.docs if str(d.get("_id")) == str(id)), None)

    def update_by_id(self, id, data, memory_store_type=None):
        self.updates.append((id, data))
        return True

    def retrieve_by_name(self, name, memory_store_type=None):
        return {"name": name}


def _runs(user_id, count, *, hours_ago=0):
    return [
        {
            "_id": f"{user_id or 'anon'}-{i}",
            "workflow_id": f"wf-{user_id or 'anon'}-{i}",
            "agent_id": "agent-1",
            "user_id": user_id,
            "canonical_hash": "class-1",
            "outcome": "success",
            "user_query": f"refund variant {i}",
            "created_at": (NOW - timedelta(hours=hours_ago + i)).isoformat(),
            "steps": {},
        }
        for i in range(count)
    ]


@pytest.mark.unit
def test_trajectory_classes_never_mix_users():
    from memorizz.long_term.procedural.workflow.canonicalization import (
        aggregate_trajectory_stats,
    )

    docs = _runs("alice", 3) + _runs(None, 2) + _runs("bob", 4)
    provider = _WorkflowProvider(docs)

    (alice,) = aggregate_trajectory_stats(provider, "agent-1", user_id="alice")
    assert (alice.user_id, alice.executions) == ("alice", 3)
    by_user = {
        stat.user_id: stat.executions
        for stat in aggregate_trajectory_stats(provider, "agent-1")
    }
    assert by_user == {"alice": 3, None: 2, "bob": 4}


def _promotion_engine(provider, box):
    from memorizz.long_term.procedural.skillbox import PromotionConfig, PromotionEngine
    from tests.unit.test_promotion import VALID_SKILL_MD

    class _LLM:
        def generate_text(self, prompt, instructions=None):
            if instructions and "YES or NO" in instructions:
                return "YES"
            return VALID_SKILL_MD

    return PromotionEngine(
        memory_provider=provider,
        skillbox=box,
        llm_provider=_LLM(),
        config=PromotionConfig(min_executions=5, min_distinct_queries=2),
        agent_id="agent-1",
        resolve_tool=lambda name: True,
    )


@pytest.mark.unit
def test_promoting_one_users_class_stamps_only_their_runs():
    from tests.unit.test_promotion import _StubSkillbox

    # Alice's runs are the most recent, so they fill the distillation sample.
    provider = _WorkflowProvider(_runs("alice", 5) + _runs("bob", 2, hours_ago=10))
    box = _StubSkillbox()

    report = _promotion_engine(provider, box).run_promotion_cycle()

    assert len(report.promoted) == 1
    assert box.skills[0].user_id == "alice"
    stamped = {record_id for record_id, _ in provider.updates}
    assert stamped == {f"alice-{i}" for i in range(5)}


@pytest.mark.unit
def test_skill_from_anonymous_runs_is_agent_wide_and_covers_every_user():
    from tests.unit.test_promotion import _StubSkillbox

    provider = _WorkflowProvider(_runs(None, 5) + _runs("bob", 2, hours_ago=10))
    box = _StubSkillbox()

    report = _promotion_engine(provider, box).run_promotion_cycle()

    assert len(report.promoted) == 1
    assert box.skills[0].user_id is None
    # Bob sees the agent-wide skill, so his runs of the class are covered.
    stamped = {record_id for record_id, _ in provider.updates}
    assert stamped == {f"anon-{i}" for i in range(5)} | {"bob-0", "bob-1"}


# ---------------------------------------------------------------------------
# Evidence packs
# ---------------------------------------------------------------------------


def _evidence_plane(provider, sources):
    from memorizz import LearningControlPlane

    return LearningControlPlane(
        provider,
        agent_id="agent-1",
        config={
            "enabled": True,
            "compile_async": False,
            "compile_every_n_events": 0,
            "evidence_sources": sources,
            "evidence_min_relevance": 0.0,
        },
    )


def _store_attached_chunk(provider):
    content = "The London release train runs on Tuesday."
    provider.store(
        {
            "_id": "chunk-1",
            "knowledge_base_id": "kb-1",
            "user_id": None,
            "content": content,
            "embedding": _Embeddings().get_embedding(content),
        },
        MemoryType.KNOWLEDGE_BASE,
    )


@pytest.mark.unit
def test_evidence_includes_attached_knowledge_base_chunks(provider):
    _store_attached_chunk(provider)

    pack = _evidence_plane(provider, ["knowledge_base"]).retrieve_evidence(
        "When is the London release?",
        memory_id="memory-1",
        user_id="user-1",
        knowledge_base_ids=["kb-1"],
    )

    assert [item.source_id for item in pack.items] == ["chunk-1"]


@pytest.mark.unit
def test_evidence_reads_attached_knowledge_bases_from_the_agent_record(provider):
    _store_attached_chunk(provider)
    _store_agent(provider, "agent-1", ["memory-1"], knowledge_base_ids=["kb-1"])

    pack = _evidence_plane(provider, ["knowledge_base"]).retrieve_evidence(
        "When is the London release?", memory_id="memory-1", user_id="user-1"
    )

    assert [item.source_id for item in pack.items] == ["chunk-1"]


@pytest.mark.unit
def test_evidence_includes_agent_skills_without_a_memory_id(provider):
    content = "When the London release slips, notify the release channel."
    for agent_id in ("agent-1", "agent-2"):
        provider.store(
            {
                "_id": f"skill-{agent_id}",
                "skill_id": f"skill-{agent_id}",
                "name": "London release",
                "description": content,
                "content": content,
                "agent_id": agent_id,
                "user_id": None,
                "status": "active",
                "embedding": _Embeddings().get_embedding(content),
            },
            MemoryType.SKILLBOX,
        )

    pack = _evidence_plane(provider, ["skillbox"]).retrieve_evidence(
        "When is the London release?", memory_id="memory-1", user_id="user-1"
    )

    assert [item.source_id for item in pack.items] == ["skill-agent-1"]


# ---------------------------------------------------------------------------
# Writers keep nested metadata
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_knowledge_chunks_keep_metadata_flat_and_nested(provider):
    from memorizz.long_term.semantic.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(provider)
    metadata = {"source_path": "/docs/a.md", "source_sha256": "abc"}
    kb_id = kb.ingest_knowledge(
        "Some text.", "docs", metadata=metadata, embeddings="off"
    )

    (chunk,) = kb.retrieve_knowledge(kb_id)
    assert chunk["source_path"] == "/docs/a.md"
    assert chunk["metadata"] == metadata


@pytest.mark.unit
def test_mcp_remember_keeps_metadata_flat_and_nested(tmp_path):
    from memorizz.approval import SQLiteApprovalStore
    from memorizz.mcp_server import MemorizzMCPServerConfig, MemorizzRuntime
    from memorizz.mcp_server.auth import RequestIdentity
    from memorizz.mcp_server.config import ALL_SCOPES

    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "mcp-memory", lazy_vector_indexes=True)
    )
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(allow_writes=True),
        provider=store,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    alice = RequestIdentity(
        principal="alice", scopes=frozenset(ALL_SCOPES), authenticated=True
    )

    stored = runtime.remember(
        "Launch is on Tuesday", "knowledge_base", alice, metadata={"source": "notes"}
    )

    row = store.retrieve_by_id(stored["record_id"], MemoryType.KNOWLEDGE_BASE)
    assert row["source"] == "notes"
    assert row["metadata"] == {"source": "notes"}


# ---------------------------------------------------------------------------
# CLI /ingest
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_ingest_attaches_the_knowledge_base_to_the_agent(
    provider, tmp_path, monkeypatch
):
    from rich.console import Console

    from memorizz.cli import commands
    from memorizz.long_term.semantic import knowledge_base as kb_module

    monkeypatch.setattr(
        kb_module, "_embed_chunks", lambda chunks: [[1.0, 0.0, 0.0] for _ in chunks]
    )
    saved = []
    agent = SimpleNamespace(knowledge_base_ids=[], save=lambda: saved.append(True))
    session = SimpleNamespace(
        provider=provider, agent=agent, console=Console(file=open("/dev/null", "w"))
    )
    document = tmp_path / "notes.md"
    document.write_text("The London release train runs on Tuesday.")

    commands.cmd_ingest(session, str(document))

    assert len(agent.knowledge_base_ids) == 1
    assert saved
