"""Regression tests for peripheral provider defects.

Each test fails on the code before its fix:

1. Internet-access provider configs leaked the API key into the saved agent.
2. A model-supplied ``metadata.identity_key`` poisoned an entity scope.
3. Name-only entity upserts were check-then-insert, so racing writers
   duplicated an entity.
4. ``KnowledgeBase`` retrieval and deletion had no knowledge-base or tenant
   scope.
5. The Daytona provider created a throwaway sandbox per call, so files
   never persisted and ``timeout`` was ignored.
6. GraalPy raised out of ``execute_code`` for a non-allowlisted environment
   variable and could not say whether it is a security sandbox.
7. Nothing wrote the WhatsApp active-agent setting that the worker reads.

No external service is contacted: providers are stubbed and the filesystem
memory provider runs on a temp dir under ``/private/tmp``.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.internet_access.providers.firecrawl import FirecrawlProvider
from memorizz.internet_access.providers.tavily import TavilyProvider
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.long_term.semantic.knowledge_base import KnowledgeBase
from memorizz.memagent.core import MemAgent
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.sandbox.providers import graalpy_provider
from memorizz.sandbox.providers.daytona_provider import DaytonaSandboxProvider
from memorizz.sandbox.providers.graalpy_provider import GraalPySandboxProvider

pytestmark = pytest.mark.unit

_UNSET = object()


# macOS resolves /tmp through /private/tmp; the sandbox profile allow-lists the
# resolved path, so use it where it exists and the platform temp dir elsewhere.
_SCRATCH_PARENT = "/private/tmp" if Path("/private/tmp").is_dir() else None


@pytest.fixture()
def scratch_dir():
    root = Path(tempfile.mkdtemp(prefix="memorizz-peripheral-", dir=_SCRATCH_PARENT))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def filesystem_memory(scratch_dir):
    return FileSystemProvider(
        FileSystemConfig(root_path=scratch_dir / "memory", use_faiss=False)
    )


# ---------------------------------------------------------------------------
# 1. Internet-access credentials never reach a saved record
# ---------------------------------------------------------------------------


_TAVILY_SAVED_CONFIG = {
    "base_url": "https://api.tavily.com",
    "timeout": 30,
    "search_depth": "basic",
    "default_max_results": 5,
    "max_content_chars": 12000,
    "include_raw_results": False,
    "include_raw_page": False,
    "api_key_set": True,
}


@pytest.mark.parametrize(
    "provider_cls, env_name",
    [(TavilyProvider, "TAVILY_API_KEY"), (FirecrawlProvider, "FIRECRAWL_API_KEY")],
)
def test_internet_provider_config_never_contains_the_api_key(
    provider_cls, env_name, monkeypatch
):
    monkeypatch.delenv(env_name, raising=False)
    provider = provider_cls(api_key="sk-private-internet-key")
    try:
        config = provider.get_config()
    finally:
        provider.close()

    assert "api_key" not in config
    assert config["api_key_set"] is True
    assert "sk-private-internet-key" not in json.dumps(config)
    # The provider itself still holds the key for requests.
    assert provider.api_key == "sk-private-internet-key"


def test_saved_agent_record_contains_no_internet_api_key(filesystem_memory):
    provider = TavilyProvider(api_key="tvly-private-key")
    agent = MemAgent(
        memory_provider=filesystem_memory,
        internet_access_provider=provider,
        auto_register=False,
    )
    try:
        agent.save()
        saved = filesystem_memory.retrieve_memagent(agent.agent_id)
        raw_document = filesystem_memory.retrieve_by_id(
            agent.agent_id, MemoryType.MEMAGENT
        )
    finally:
        agent.close()

    assert saved.internet_access_provider == "tavily"
    assert saved.internet_access_config["api_key_set"] is True
    assert "tvly-private-key" not in json.dumps(saved.internet_access_config)
    assert "tvly-private-key" not in json.dumps(raw_document, default=str)


def test_save_scrubs_api_key_reported_by_a_third_party_provider(filesystem_memory):
    provider = MagicMock()
    provider.get_provider_name.return_value = "dummy-provider"
    provider.get_config.return_value = {"api_key": "third-party-secret", "region": "eu"}
    provider.search.return_value = []
    provider.fetch_url.return_value = {}

    agent = MemAgent(
        memory_provider=filesystem_memory,
        internet_access_provider=provider,
        auto_register=False,
    )
    try:
        agent.save()
        saved = filesystem_memory.retrieve_memagent(agent.agent_id)
    finally:
        agent.close()

    assert saved.internet_access_config == {"region": "eu", "api_key_set": True}


def test_restored_provider_resolves_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-env")
    memory_provider = MagicMock()
    memory_provider.retrieve_memagent.return_value = MemAgentModel(
        instruction="Load",
        internet_access_provider="tavily",
        internet_access_config=dict(_TAVILY_SAVED_CONFIG),
    )

    agent = MemAgent.load(agent_id="agent-net", memory_provider=memory_provider)

    provider = agent.internet_access_manager.provider
    assert isinstance(provider, TavilyProvider)
    assert provider.api_key == "tvly-from-env"
    assert provider.search_depth == "basic"
    assert agent.has_internet_access() is True


def test_restore_never_uses_a_key_saved_by_an_older_release(monkeypatch):
    stale_config = dict(_TAVILY_SAVED_CONFIG, api_key="stale-saved-key")
    memory_provider = MagicMock()
    memory_provider.retrieve_memagent.return_value = MemAgentModel(
        instruction="Load",
        internet_access_provider="tavily",
        internet_access_config=stale_config,
    )

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-env")
    agent = MemAgent.load(agent_id="agent-net", memory_provider=memory_provider)
    assert agent.internet_access_manager.provider.api_key == "tvly-from-env"
    assert "stale-saved-key" not in json.dumps(
        agent.internet_access_manager.get_provider_config()
    )

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    agent = MemAgent.load(agent_id="agent-net", memory_provider=memory_provider)
    assert agent.has_internet_access() is False


# ---------------------------------------------------------------------------
# 2 + 3. Entity memory: identity poisoning and name-only convergence
# ---------------------------------------------------------------------------


class InMemoryEntityProvider:
    """Minimal entity-memory provider keyed by ``entity_id``."""

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}
        self.vector_available = True

    def supports_entity_memory(self) -> bool:
        return True

    def store(self, data: Dict[str, Any], memory_store_type: MemoryType, **_) -> str:
        assert memory_store_type == MemoryType.ENTITY_MEMORY
        record = dict(data)
        record.setdefault("_id", record.get("entity_id", str(uuid.uuid4())))
        self.records[record["entity_id"]] = record
        return record["_id"]

    def retrieve_by_query(
        self,
        query: Any,
        memory_type: MemoryType,
        limit: int = 5,
        memory_id: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        user_id = kwargs.get("user_id", _UNSET)
        if isinstance(query, dict):
            candidates = [
                rec
                for rec in self.records.values()
                if all(rec.get(key) == value for key, value in query.items())
                and (memory_id is None or rec.get("memory_id") == memory_id)
            ]
        else:
            if not self.vector_available:
                return []
            candidates = [
                rec
                for rec in self.records.values()
                if memory_id is None or rec.get("memory_id") == memory_id
            ]
        if user_id is not _UNSET:
            candidates = [rec for rec in candidates if rec.get("user_id") == user_id]
        return candidates[:limit]

    def list_all(
        self, memory_store_type: MemoryType, user_id: Any = _UNSET
    ) -> List[Dict[str, Any]]:
        records = [dict(rec) for rec in self.records.values()]
        if user_id is not _UNSET:
            records = [rec for rec in records if rec.get("user_id") == user_id]
        return records

    def get_vector_search_status(self, memory_store_type: MemoryType):
        if self.vector_available:
            return {"available": True, "queryable": True, "status": "READY"}
        return {
            "available": False,
            "queryable": False,
            "reason": "vector_index_missing",
        }


@pytest.fixture(autouse=True)
def _stub_entity_embeddings(monkeypatch):
    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        lambda text: [float(len(text or ""))],
    )


@pytest.fixture()
def entity_provider() -> InMemoryEntityProvider:
    return InMemoryEntityProvider()


@pytest.fixture()
def entity_store(entity_provider) -> EntityMemory:
    return EntityMemory(entity_provider)


def _legacy_row(**overrides) -> Dict[str, Any]:
    row = {
        "_id": "legacy-row",
        "entity_id": "legacy-row",
        "name": "user",
        "entity_type": "person",
        "attributes": [{"name": "role", "value": "engineer", "confidence": 0.9}],
        "relations": [],
        "metadata": {"identity_key": "!!!"},
        "memory_id": "tenant-1",
        "user_id": "user-a",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }
    row.update(overrides)
    return row


def test_model_supplied_identity_key_cannot_poison_the_scope(
    entity_provider, entity_store
):
    first_id = entity_store.upsert_entity(
        name="user",
        entity_type="person",
        attributes=[{"name": "role", "value": "engineer"}],
        metadata={"identity_key": "!!!", "source": "chat"},
        memory_id="tenant-1",
        user_id="user-a",
    )
    stored = entity_provider.records[first_id]
    assert "identity_key" not in stored["metadata"]
    assert stored["metadata"]["source"] == "chat"

    # The host binds the canonical identity afterwards, lands on the same
    # record, and scoped search still works.
    bound_id = entity_store.upsert_entity(
        name="user",
        entity_type="person",
        identity_key="authenticated_user",
        attributes=[{"name": "timezone", "value": "UTC"}],
        memory_id="tenant-1",
        user_id="user-a",
    )
    assert bound_id == first_id
    assert len(entity_provider.records) == 1
    assert entity_provider.records[first_id]["metadata"]["identity_key"] == (
        "authenticated_user"
    )

    matches, diagnostics = entity_store.search_entities_with_diagnostics(
        "user profile", memory_id="tenant-1", user_id="user-a"
    )
    assert [match["entity_id"] for match in matches] == [first_id]
    assert diagnostics["match_count"] == 1


def test_invalid_stored_identity_key_is_treated_as_absent(
    entity_provider, entity_store, caplog, monkeypatch
):
    monkeypatch.setattr(EntityMemory, "_invalid_identity_key_logged", False)
    entity_provider.records["legacy-row"] = _legacy_row()
    entity_provider.vector_available = False  # force the exact fallback ranker
    caplog.set_level("WARNING", logger="memorizz.long_term.semantic")

    entity_id = entity_store.upsert_entity(
        name="user",
        identity_key="authenticated_user",
        attributes=[{"name": "timezone", "value": "UTC"}],
        memory_id="tenant-1",
        user_id="user-a",
    )
    assert entity_id == "legacy-row"
    assert entity_provider.records["legacy-row"]["metadata"]["identity_key"] == (
        "authenticated_user"
    )

    # A second poisoned row: reads and consolidation skip the bad key.
    entity_provider.records["legacy-dup"] = _legacy_row(
        _id="legacy-dup", entity_id="legacy-dup"
    )
    matches, _ = entity_store.search_entities_with_diagnostics(
        "user profile", memory_id="tenant-1", user_id="user-a"
    )
    assert {match["entity_id"] for match in matches} == {"legacy-row", "legacy-dup"}

    plan = entity_store.consolidate_duplicate_entities(
        memory_id="tenant-1", user_id="user-a"
    )
    assert plan["duplicate_group_count"] >= 0

    warnings = [
        record
        for record in caplog.records
        if "identity_key" in record.getMessage() and record.levelname == "WARNING"
    ]
    assert len(warnings) == 1, "an invalid stored identity key is logged once"


def test_concurrent_name_only_upserts_converge_on_one_record(
    entity_provider, entity_store, monkeypatch
):
    # Two racing first writes both miss the existence check.
    monkeypatch.setattr(entity_store, "_fetch_one", lambda *args, **kwargs: None)

    first = entity_store.upsert_entity(
        name="Avery Chen",
        entity_type="customer",
        attributes=[{"name": "language", "value": "English"}],
        memory_id="tenant-1",
        user_id="user-a",
    )
    second = entity_store.upsert_entity(
        name="avery   chen",
        entity_type="Customer",
        attributes=[{"name": "timezone", "value": "PST"}],
        memory_id="tenant-1",
        user_id="user-a",
    )

    assert first == second
    assert len(entity_provider.records) == 1
    assert entity_provider.records[first]["_id"] == first

    other_scope = entity_store.upsert_entity(
        name="Avery Chen",
        entity_type="customer",
        memory_id="tenant-2",
        user_id="user-a",
    )
    assert other_scope != first


def test_name_variants_merge_sequentially_instead_of_overwriting(
    entity_provider, entity_store
):
    entity_store.upsert_entity(
        name="Avery",
        entity_type="customer",
        attributes=[{"name": "language", "value": "English"}],
        memory_id="tenant-1",
        user_id="user-a",
    )
    entity_store.upsert_entity(
        name="avery",
        entity_type="customer",
        attributes=[{"name": "timezone", "value": "PST"}],
        memory_id="tenant-1",
        user_id="user-a",
    )

    assert len(entity_provider.records) == 1
    record = next(iter(entity_provider.records.values()))
    stored = {attr["name"]: attr["value"] for attr in record["attributes"]}
    assert stored == {"language": "English", "timezone": "PST"}


# ---------------------------------------------------------------------------
# 4. KnowledgeBase scoping, delete guard and batch embedding
# ---------------------------------------------------------------------------


class InMemoryKnowledgeProvider:
    """Knowledge-base store that returns rows in insertion order."""

    def __init__(self, accept_user_id: bool = True):
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.accept_user_id = accept_user_id
        self.query_log: List[Dict[str, Any]] = []

    def store(self, data, memory_store_type=None, memory_id=None, memory_unit=None):
        row = dict(data)
        row_id = str(row.get("_id") or uuid.uuid4())
        row["_id"] = row_id
        self.rows[row_id] = row
        return row_id

    def retrieve_by_query(
        self,
        query,
        memory_store_type=None,
        limit: int = 1,
        memory_id=None,
        memory_type=None,
        **kwargs,
    ):
        if not self.accept_user_id and "user_id" in kwargs:
            raise TypeError("retrieve_by_query() got an unexpected keyword 'user_id'")
        self.query_log.append({"limit": limit, **kwargs})
        rows = list(self.rows.values())
        if "user_id" in kwargs:
            rows = [row for row in rows if row.get("user_id") == kwargs["user_id"]]
        namespace = kwargs.get("namespace")
        if namespace:
            rows = [row for row in rows if row.get("namespace") == namespace]
        return rows[:limit]

    def list_all(self, memory_store_type, user_id: Any = _UNSET):
        rows = [dict(row) for row in self.rows.values()]
        if user_id is not _UNSET:
            rows = [row for row in rows if row.get("user_id") == user_id]
        return rows

    def delete_by_id(self, id, memory_store_type=None):
        return self.rows.pop(id, None) is not None


def _ingest(kb: KnowledgeBase, text: str, **kwargs) -> str:
    kwargs.setdefault("chunking_strategy", "none")
    kwargs.setdefault("embeddings", "off")
    return kb.ingest_knowledge(text, kwargs.pop("namespace", "docs"), **kwargs)


def test_retrieval_scoped_to_knowledge_base_ids_returns_the_small_kb():
    provider = InMemoryKnowledgeProvider()
    kb = KnowledgeBase(provider)
    big = kb.ingest_knowledge(
        "big one. Big two. Big three. Big four. Big five. Big six.",
        "docs",
        chunking_strategy="sentence",
        chunk_size=8,
        embeddings="off",
    )
    small = _ingest(kb, "small alpha")
    assert len(kb.retrieve_knowledge(big)) >= 4

    rows = kb.retrieve_knowledge_by_query("alpha", knowledge_base_ids=[small], limit=2)

    assert rows
    assert {row["knowledge_base_id"] for row in rows} == {small}
    assert provider.query_log[-1]["limit"] > 2, "candidate window grows with scoping"
    assert kb.retrieve_knowledge_by_query("alpha", knowledge_base_ids=[]) == []


@pytest.mark.parametrize("provider_accepts_user_id", [True, False])
def test_retrieval_scoped_to_a_tenant(provider_accepts_user_id):
    provider = InMemoryKnowledgeProvider(accept_user_id=provider_accepts_user_id)
    kb = KnowledgeBase(provider)
    _ingest(kb, "alice secret", namespace="secrets", user_id="alice")
    _ingest(kb, "bob secret", namespace="secrets", user_id="bob")
    _ingest(kb, "public note", namespace="secrets")

    alice_rows = kb.retrieve_knowledge_by_query("secret", user_id="alice", limit=5)
    assert {row["content"] for row in alice_rows} == {"alice secret"}
    if provider_accepts_user_id:
        assert provider.query_log[-1]["user_id"] == "alice"
    else:
        assert "user_id" not in provider.query_log[-1]

    anonymous_rows = kb.retrieve_knowledge_by_query("secret", user_id=None, limit=5)
    assert {row["content"] for row in anonymous_rows} == {"public note"}

    unscoped = kb.retrieve_knowledge_by_query("secret", limit=5)
    assert len(unscoped) == 3


def test_delete_knowledge_refuses_a_mismatched_owner():
    provider = InMemoryKnowledgeProvider()
    kb = KnowledgeBase(provider)
    kb_id = _ingest(kb, "alice secret", user_id="alice")

    assert kb.delete_knowledge(kb_id, user_id="bob") is False
    assert kb.retrieve_knowledge(kb_id)
    assert kb.retrieve_knowledge(kb_id, user_id="bob") == []

    assert kb.delete_knowledge(kb_id, user_id="alice") is True
    assert kb.retrieve_knowledge(kb_id) == []


def test_ingest_embeds_chunks_in_one_batch(monkeypatch):
    calls: List[List[str]] = []

    class _BatchManager:
        def get_embeddings(self, texts, **kwargs):
            calls.append(list(texts))
            return [[float(len(text)), 1.0] for text in texts]

        def get_embedding(self, text, **kwargs):  # pragma: no cover - not used
            raise AssertionError("single-item embedder must not be used")

    monkeypatch.setattr("memorizz.embeddings.get_embedding_manager", _BatchManager)
    provider = InMemoryKnowledgeProvider()
    kb = KnowledgeBase(provider)

    kb_id = kb.ingest_knowledge(
        "alpha. Beta. Gamma.", "docs", chunking_strategy="sentence", chunk_size=6
    )

    chunks = kb.retrieve_knowledge(kb_id)
    assert len(chunks) == 3
    assert calls == [[chunk["content"] for chunk in chunks]]
    assert all(chunk["embedding"] for chunk in chunks)
    assert kb.last_ingest == {"knowledge_base_id": kb_id, "chunks": 3, "embedded": 3}


# ---------------------------------------------------------------------------
# 5. Daytona keeps one sandbox per provider instance
# ---------------------------------------------------------------------------


def _install_fake_daytona(monkeypatch) -> Dict[str, Any]:
    calls: Dict[str, Any] = {"created": 0, "removed": 0, "code_runs": []}

    class _FakeFS:
        def __init__(self):
            self.files: Dict[str, bytes] = {}

        def upload_file(self, path, content):
            self.files[path] = content

        def download_file(self, path):
            return self.files[path]

    class _FakeProcess:
        def code_run(self, code, **kwargs):
            calls["code_runs"].append({"code": code, **kwargs})
            return types.SimpleNamespace(result="ok", exit_code=0)

    class _FakeSandbox:
        def __init__(self):
            self.fs = _FakeFS()
            self.process = _FakeProcess()

    class _FakeDaytona:
        def __init__(self, config):
            calls["config"] = config

        def create(self, **kwargs):
            calls["created"] += 1
            calls["create_kwargs"] = kwargs
            return _FakeSandbox()

        def remove(self, sandbox):
            calls["removed"] += 1

    class _FakeDaytonaConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "daytona",
        types.SimpleNamespace(Daytona=_FakeDaytona, DaytonaConfig=_FakeDaytonaConfig),
    )
    return calls


def test_daytona_shares_one_sandbox_and_passes_timeout(monkeypatch):
    calls = _install_fake_daytona(monkeypatch)
    provider = DaytonaSandboxProvider(api_key="dt-private-key")

    assert provider.write_file("/home/daytona/notes.txt", "hello") is True
    assert provider.read_file("/home/daytona/notes.txt") == "hello"
    result = provider.execute_code("print('x')", timeout=7, envs={"MODE": "test"})

    assert result.exit_code == 0
    assert result.stdout == ["ok"]
    assert calls["code_runs"][0]["timeout"] == 7
    assert calls["created"] == 1
    assert calls["removed"] == 0

    provider.close()
    assert calls["removed"] == 1
    assert provider.read_file("/home/daytona/notes.txt") is None
    assert calls["created"] == 2, "a closed provider starts a fresh session"

    config = provider.get_config()
    assert config["api_key_set"] is True
    assert "dt-private-key" not in json.dumps(config)
    provider.close()


def test_daytona_reports_a_missing_sdk_as_a_result(monkeypatch):
    monkeypatch.setitem(sys.modules, "daytona", None)
    provider = DaytonaSandboxProvider(api_key="dt-private-key")

    result = provider.execute_code("print(1)")

    assert result.exit_code == 1
    assert "pip install daytona" in (result.error or "")
    assert provider.write_file("/tmp/x", "y") is False
    assert provider.read_file("/tmp/x") is None


# ---------------------------------------------------------------------------
# 6. GraalPy: environment refusal is a result; config is honest
# ---------------------------------------------------------------------------


def test_graalpy_refuses_a_non_allowlisted_env_var_as_a_result(scratch_dir):
    provider = GraalPySandboxProvider(
        graalpy_path="/bin/echo", working_dir=str(scratch_dir)
    )
    try:
        result = provider.execute_code("print(1)", envs={"OPENAI_API_KEY": "leak"})
    finally:
        provider.close()

    assert result.exit_code == 2
    assert "env_allowlist" in (result.error or "")
    assert result.metadata["provider"] == "graalpy"


def test_graalpy_config_says_whether_it_is_a_security_sandbox(scratch_dir):
    subprocess_provider = GraalPySandboxProvider(
        graalpy_path="/bin/echo", working_dir=str(scratch_dir / "a")
    )
    wrapper_provider = GraalPySandboxProvider(
        graalpy_path="/bin/echo",
        mode="java_wrapper",
        java_wrapper_jar=str(scratch_dir / "missing.jar"),
        working_dir=str(scratch_dir / "b"),
    )
    try:
        assert subprocess_provider.get_config()["is_security_sandbox"] is False
        assert wrapper_provider.get_config()["is_security_sandbox"] is True
    finally:
        subprocess_provider.close()
        wrapper_provider.close()


def test_graalpy_darwin_profile_denies_home_reads_but_allows_the_workdir(
    scratch_dir, monkeypatch
):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(graalpy_provider, "_sandbox_exec_available", lambda: True)
    monkeypatch.setenv("HOME", "/Users/example")
    monkeypatch.setenv("MEMORIZZ_HOME", "/srv/memorizz-home")
    provider = GraalPySandboxProvider(
        graalpy_path="/bin/echo", working_dir=str(scratch_dir)
    )
    try:
        command = provider._subprocess_command("print(1)")
        workdir = Path(provider.working_dir).resolve()
    finally:
        provider.close()

    assert command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert command[-3:] == ["/bin/echo", "-c", "print(1)"]
    profile = command[2]
    assert "(deny network*)" in profile
    home_rule = '(deny file-read* (subpath "/Users/example"))'
    assert home_rule in profile
    assert '(deny file-read* (subpath "/srv/memorizz-home"))' in profile
    workdir_rule = f'(allow file-read* (subpath "{workdir}"))'
    assert workdir_rule in profile
    assert profile.index(workdir_rule) > profile.index(
        home_rule
    ), "the working-directory allow must come after the deny so it wins"
    assert provider.get_config()["host_read_deny_roots"] == [
        "/Users/example",
        "/srv/memorizz-home",
    ]


# ---------------------------------------------------------------------------
# 7. WhatsApp active-agent setting has a writer
# ---------------------------------------------------------------------------


def test_whatsapp_active_agent_setting_roundtrip(filesystem_memory):
    from memorizz.channels.whatsapp.settings import (
        WhatsAppSettings,
        clear_active_agent_id,
        get_active_agent_id,
        set_active_agent_id,
    )

    settings = WhatsAppSettings(filesystem_memory)
    assert settings.get_active_agent_id() is None
    assert get_active_agent_id(filesystem_memory) is None

    assert set_active_agent_id(filesystem_memory, "agent-1") is True
    assert settings.get_active_agent_id() == "agent-1"

    assert settings.set_active_agent_id("agent-2") is True
    assert get_active_agent_id(filesystem_memory) == "agent-2"
    rows = filesystem_memory.retrieve_conversation_history_ordered_by_timestamp(
        memory_id=settings.settings_memory_id,
        memory_type=MemoryType.CONVERSATION_MEMORY,
    )
    assert len(rows) == 1, "re-pointing the setting replaces the previous row"

    assert clear_active_agent_id(filesystem_memory) is True
    assert settings.get_active_agent_id() is None
    assert clear_active_agent_id(filesystem_memory) is True

    with pytest.raises(ValueError):
        settings.set_active_agent_id("   ")
