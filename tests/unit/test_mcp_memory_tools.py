"""The MCP server's ingest, entity, update, status and keyword-search tools."""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import pytest

from memorizz import embeddings
from memorizz.approval import SQLiteApprovalStore
from memorizz.long_term.semantic import knowledge_base as knowledge_base_module
from memorizz.long_term.semantic.entity_memory import (
    entity_memory as entity_memory_module,
)
from memorizz.mcp_server import (
    MemorizzMCPServerConfig,
    MemorizzRuntime,
    create_memorizz_mcp_server,
)
from memorizz.mcp_server import runtime as runtime_module
from memorizz.mcp_server.auth import RequestIdentity
from memorizz.mcp_server.config import ALL_SCOPES, READ_SCOPE
from memorizz.mcp_server.runtime import MemorizzServerError
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


class _WordEmbedder:
    """Bag-of-words vectors, so texts sharing words are similar."""

    def get_embedding(self, text, **kwargs):
        vector = [0.0] * 64
        for word in re.findall(r"[a-z]+", str(text).lower()):
            vector[zlib.crc32(word.encode()) % 64] += 1.0
        vector[0] += 0.01
        return vector

    def get_dimensions(self):
        return 64

    def get_default_model(self):
        return "words-64"

    def get_provider_info(self):
        return {"provider": "test-words", "model": "words-64", "dimensions": 64}


class _NoEmbedder:
    """What a fresh install looks like: OpenAI by default, with no key."""

    def get_embedding(self, text, **kwargs):
        raise RuntimeError("Missing credentials: set OPENAI_API_KEY")

    def get_dimensions(self):
        return 1536

    def get_default_model(self):
        return "text-embedding-3-small"

    def get_provider_info(self):
        return {"provider": "openai", "model": "text-embedding-3-small"}


@pytest.fixture
def use_embedder(monkeypatch):
    def install(manager):
        embed = manager.get_embedding
        monkeypatch.setattr(embeddings, "get_embedding_manager", lambda: manager)
        monkeypatch.setattr(embeddings, "get_embedding", embed)
        monkeypatch.setattr(knowledge_base_module, "get_embedding", embed)
        monkeypatch.setattr(entity_memory_module, "get_embedding", embed)
        return manager

    return install


def _identity(principal="alice", *scopes):
    return RequestIdentity(
        principal=principal,
        scopes=frozenset(scopes or ALL_SCOPES),
        authenticated=True,
    )


def test_timeline_reads_are_exactly_tenant_scoped(tmp_path):
    from memorizz import MemoryHistory, MemoryType

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    runtime = _runtime(tmp_path, provider=provider)
    history = MemoryHistory(provider)
    for tenant in ("alice", "bob", None):
        with history.recording(actor=tenant or "anonymous", user_id=tenant):
            provider.store(
                {"content": "private", "user_id": tenant}, MemoryType.KNOWLEDGE_BASE
            )
    result = runtime.memory_timeline(_identity("alice", READ_SCOPE))
    assert len(result["events"]) == 1
    assert result["events"][0]["actor"] == "alice"
    assert "private" not in str(result)


def _runtime(tmp_path, *, provider=None, **config):
    config.setdefault("allow_writes", True)
    config.setdefault("ingest_roots", {str(tmp_path / "project")})
    return MemorizzRuntime(
        MemorizzMCPServerConfig(**config),
        provider=provider
        or FileSystemProvider(
            FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
        ),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )


def _project(tmp_path) -> Path:
    root = tmp_path / "project"
    files = {
        "README.md": "# Tests\nRun the tests with pytest -n 8.\n",
        "docs/deploy.md": "Deploys go through make release on Fridays.\n",
        "docs/notes.txt": "The billing service owns invoices.\n",
        ".env": "OPENAI_API_KEY=sk-should-never-be-read\n",
        ".env.local": "TOKEN=never\n",
        "server.pem": "-----BEGIN PRIVATE KEY-----\n",
        "deploy.key": "never\n",
        "id_rsa": "never\n",
        "credentials.json": '{"password": "never"}\n',
        "secrets.yaml": "token: never\n",
        "bundle.p12": "never\n",
        ".hidden.md": "hidden file\n",
        ".config/settings.md": "hidden folder\n",
        ".git/HEAD": "ref: refs/heads/main\n",
        "node_modules/pkg/README.md": "vendored\n",
        ".venv/lib/site.py": "vendored\n",
        "logo.png": "not text\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _knowledge_rows(runtime, identity, memory_id="proj", **kwargs):
    return runtime.list_memories(
        "knowledge_base", identity, memory_id=memory_id, limit=100, **kwargs
    )["memories"]


@pytest.mark.unit
def test_ingest_stores_allowed_files_and_skips_secrets_and_vendor_folders(
    tmp_path, use_embedder
):
    use_embedder(_WordEmbedder())
    root = _project(tmp_path)
    runtime = _runtime(tmp_path)
    alice = _identity()

    result = runtime.ingest([str(root)], "proj", alice)

    assert result["ok"] is True
    assert result["files_ingested"] == 3
    assert result["chunks_stored"] == 3
    assert result["chunks_embedded"] == 3
    assert "note" not in result
    reasons = {Path(item["path"]).name: item["reason"] for item in result["skipped"]}
    for secret in (
        ".env",
        ".env.local",
        "server.pem",
        "deploy.key",
        "id_rsa",
        "credentials.json",
        "secrets.yaml",
        "bundle.p12",
    ):
        assert reasons[secret] == "looks like a secret"
    assert reasons[".hidden.md"] == "hidden file"
    assert reasons[".config"] == "hidden folder"
    assert reasons[".git"] == reasons["node_modules"] == reasons[".venv"]
    assert reasons[".git"] == "dependency or version-control folder"
    assert reasons["logo.png"] == "unsupported file type"
    assert result["files_skipped"] == len(result["skipped"]) == 14

    rows = _knowledge_rows(runtime, alice)
    assert {Path(row["source_path"]).name for row in rows} == {
        "README.md",
        "deploy.md",
        "notes.txt",
    }
    assert all(row["memory_id"] == "proj" for row in rows)
    assert all(row["chunk_index"] == 0 and row["chunk_count"] == 1 for row in rows)
    stored = " ".join(row["content"] for row in rows)
    assert "never" not in stored and "sk-should" not in stored
    # Another tenant sees none of it.
    assert _knowledge_rows(runtime, _identity("bob")) == []


@pytest.mark.unit
def test_ingest_include_patterns_recursion_and_chunking(tmp_path, use_embedder):
    use_embedder(_NoEmbedder())
    root = _project(tmp_path)
    (root / "docs" / "long.md").write_text("word " * 600, encoding="utf-8")
    runtime = _runtime(tmp_path)
    alice = _identity()

    flat = runtime.ingest([str(root)], "flat", alice, recursive=False)
    assert flat["files_ingested"] == 1  # README.md only

    only_md = runtime.ingest(
        [str(root / "docs")],
        "md",
        alice,
        include=["*.md"],
        chunk_size=500,
        chunk_overlap=50,
    )
    assert [Path(item["path"]).name for item in only_md["files"]] == [
        "deploy.md",
        "long.md",
    ]
    assert only_md["skipped_by_reason"] == {"not matched by include": 1}
    long_chunks = [
        row
        for row in _knowledge_rows(runtime, alice, memory_id="md")
        if row["source_name"] == "long.md"
    ]
    assert len(long_chunks) == only_md["files"][1]["chunks"] > 1
    assert sorted(row["chunk_index"] for row in long_chunks) == list(
        range(len(long_chunks))
    )
    # Stored without embeddings, and the result says how to turn them on.
    assert only_md["chunks_embedded"] == 0
    assert "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER" in only_md["note"]

    with pytest.raises(MemorizzServerError) as error:
        runtime.ingest([str(root)], "proj", alice, chunk_size=10)
    assert error.value.code == "invalid_chunking"


@pytest.mark.unit
def test_ingest_refuses_paths_outside_roots_and_unsafe_requests(
    tmp_path, use_embedder, monkeypatch
):
    use_embedder(_NoEmbedder())
    root = _project(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "notes.md").write_text("outside\n", encoding="utf-8")
    (root / "link.md").symlink_to(outside / "notes.md")
    runtime = _runtime(tmp_path)
    alice = _identity()

    def code(*args, identity=alice, target=runtime, **kwargs):
        with pytest.raises(MemorizzServerError) as error:
            target.ingest(*args, identity, **kwargs)
        return error.value.code

    assert code([str(outside)], "proj") == "path_not_allowed"
    assert code([str(root / ".." / "elsewhere")], "proj") == "path_not_allowed"
    assert code(["project/README.md"], "proj") == "invalid_paths"
    assert code([], "proj") == "invalid_paths"
    assert code([str(root)], "  ") == "invalid_memory_id"
    assert (
        code([str(root)], "proj", identity=_identity("reader", READ_SCOPE))
        == "insufficient_scope"
    )
    read_only = _runtime(tmp_path, allow_writes=False)
    assert code([str(root)], "proj", target=read_only) == "writes_disabled"
    remote = _runtime(
        tmp_path,
        transport="streamable-http",
        allow_anonymous_http=True,
        ingest_roots=None,
    )
    assert code([str(root)], "proj", target=remote) == "ingest_disabled"

    # A symlink inside the root that points outside it is never read: refused
    # when named, skipped when found in a folder.
    assert code([str(root / "link.md")], "proj") == "path_not_allowed"
    result = runtime.ingest([str(root)], "proj", alice, include=["link.md"])
    assert result["files_ingested"] == 0
    assert {
        "path": str(root / "link.md"),
        "reason": "links outside the allowed folders",
    } in result["skipped"]
    # MemoRizz's own data folder is never ingested, even inside a root.
    home = root / "memorizz-home"
    home.mkdir()
    (home / "notes.md").write_text("internal\n", encoding="utf-8")
    monkeypatch.setenv("MEMORIZZ_HOME", str(home))
    own = runtime.ingest([str(home)], "proj", alice)
    assert own["files_ingested"] == 0
    assert own["skipped"][0]["reason"] == "MemoRizz's own data folder"


@pytest.mark.unit
def test_ingest_cap_refuses_before_storing_anything(
    tmp_path, use_embedder, monkeypatch
):
    use_embedder(_NoEmbedder())
    root = _project(tmp_path)
    runtime = _runtime(tmp_path)
    alice = _identity()
    monkeypatch.setattr(runtime_module, "_INGEST_MAX_FILES", 2)

    with pytest.raises(MemorizzServerError) as error:
        runtime.ingest([str(root)], "proj", alice)
    assert error.value.code == "ingest_too_large"
    assert "at most 2 files" in error.value.message
    assert _knowledge_rows(runtime, alice) == []

    monkeypatch.setattr(runtime_module, "_INGEST_MAX_FILES", 500)
    monkeypatch.setattr(runtime_module, "_INGEST_MAX_BYTES", 20)
    with pytest.raises(MemorizzServerError, match="20 MB|0 MB"):
        runtime.ingest([str(root)], "proj", alice)
    assert _knowledge_rows(runtime, alice) == []


@pytest.mark.unit
def test_reingest_skips_unchanged_files_and_supersedes_changed_ones(
    tmp_path, use_embedder
):
    use_embedder(_NoEmbedder())
    root = _project(tmp_path)
    runtime = _runtime(tmp_path)
    alice = _identity()
    runtime.ingest([str(root / "docs")], "proj", alice)

    again = runtime.ingest([str(root / "docs")], "proj", alice)
    assert again["files_ingested"] == 0
    assert again["skipped_by_reason"] == {"unchanged since the last ingest": 2}

    (root / "docs" / "deploy.md").write_text(
        "Deploys go through make ship on Mondays.\n", encoding="utf-8"
    )
    changed = runtime.ingest([str(root / "docs")], "proj", alice)
    assert changed["files_ingested"] == 1
    assert changed["old_chunks_superseded"] == 1

    current = [row["content"] for row in _knowledge_rows(runtime, alice)]
    assert any("Mondays" in text for text in current)
    assert not any("Fridays" in text for text in current)
    history = _knowledge_rows(runtime, alice, include_superseded=True)
    old = next(row for row in history if "Fridays" in row["content"])
    assert old["status"] == "superseded"
    assert old["superseded_by"] == changed["files"][0]["knowledge_base_id"]


@pytest.mark.unit
def test_update_memory_supersedes_without_deleting(tmp_path, use_embedder):
    use_embedder(_NoEmbedder())
    runtime = _runtime(tmp_path)
    alice = _identity()
    first = runtime.remember(
        "Tests run with pytest -q",
        "knowledge_base",
        alice,
        memory_id="proj",
        metadata={"source": "chat"},
    )

    updated = runtime.update_memory(
        first["record_id"],
        "Tests run with pytest -n 8",
        alice,
        reason="The suite runs in parallel now",
    )

    assert updated["ok"] is True
    assert updated["supersedes"] == first["record_id"]
    assert updated["memory_id"] == "proj"
    new = runtime.get_memory(updated["record_id"], "knowledge_base", alice)["memory"]
    assert new["content"] == "Tests run with pytest -n 8"
    assert new["supersedes"] == first["record_id"]
    assert new["supersede_reason"] == "The suite runs in parallel now"
    assert new["source"] == "chat"
    assert "status" not in new
    old = runtime.get_memory(first["record_id"], "knowledge_base", alice)["memory"]
    assert old["content"] == "Tests run with pytest -q"
    assert old["status"] == "superseded"
    assert old["superseded_by"] == updated["record_id"]
    assert old["superseded_at"] == updated["superseded"]["superseded_at"]

    listed = _knowledge_rows(runtime, alice)
    assert [row["content"] for row in listed] == ["Tests run with pytest -n 8"]
    assert len(_knowledge_rows(runtime, alice, include_superseded=True)) == 2
    search = runtime.search_memories("pytest", "knowledge_base", alice)
    assert [row["content"] for row in search["memories"]] == [
        "Tests run with pytest -n 8"
    ]
    everything = runtime.search_memories(
        "pytest", "knowledge_base", alice, include_superseded=True
    )
    assert everything["count"] == 2

    with pytest.raises(MemorizzServerError) as error:
        runtime.update_memory(first["record_id"], "again", alice)
    assert error.value.code == "memory_superseded"
    assert updated["record_id"] in error.value.message
    with pytest.raises(MemorizzServerError) as error:
        runtime.update_memory(updated["record_id"], "stolen", _identity("bob"))
    assert error.value.code == "memory_not_found"
    with pytest.raises(MemorizzServerError) as error:
        runtime.update_memory(
            updated["record_id"], "x", alice, memory_type="entity_memory"
        )
    assert error.value.code == "memory_type_not_writable"
    with pytest.raises(MemorizzServerError) as error:
        runtime.update_memory(updated["record_id"], "  ", alice)
    assert error.value.code == "invalid_update"

    note = runtime.remember("Standup at 10", "short_term_memory", alice)
    moved = runtime.update_memory(
        note["record_id"], "Standup at 9:30", alice, memory_type="short_term_memory"
    )
    assert moved["memory_type"] == "short_term_memory"


@pytest.mark.unit
def test_a_duplicate_is_retired_in_favour_of_the_memory_kept(tmp_path, use_embedder):
    use_embedder(_NoEmbedder())
    runtime = _runtime(tmp_path)
    alice = _identity()
    kept = runtime.remember(
        "Migration 0042 renames users to accounts",
        "knowledge_base",
        alice,
        memory_id="proj",
    )
    dup = runtime.remember(
        "0042 renames the users table", "knowledge_base", alice, memory_id="proj"
    )
    other = runtime.remember("Elsewhere", "knowledge_base", alice, memory_id="other")

    retired = runtime.update_memory(
        dup["record_id"], "", alice, duplicate_of=kept["record_id"], reason="duplicate"
    )

    assert retired["record_id"] == kept["record_id"]
    old = runtime.get_memory(dup["record_id"], "knowledge_base", alice)["memory"]
    assert old["status"] == "superseded" and old["superseded_by"] == kept["record_id"]
    assert old["supersede_reason"] == "duplicate"
    rows = _knowledge_rows(runtime, alice, include_superseded=True)
    assert len(rows) == 2  # nothing new stored
    assert [row["content"] for row in _knowledge_rows(runtime, alice)] == [
        "Migration 0042 renames users to accounts"
    ]

    for args, code in (
        ((kept["record_id"], "new text"), "invalid_update"),  # content and duplicate_of
        ((kept["record_id"], ""), "invalid_update"),  # itself
        ((other["record_id"], ""), "invalid_update"),  # another memory_id
    ):
        with pytest.raises(MemorizzServerError) as error:
            runtime.update_memory(*args, alice, duplicate_of=kept["record_id"])
        assert error.value.code == code
    with pytest.raises(MemorizzServerError) as error:
        runtime.update_memory(
            kept["record_id"], "", _identity("bob"), duplicate_of=dup["record_id"]
        )
    assert error.value.code == "memory_not_found"


@pytest.mark.unit
def test_update_memory_changes_nothing_when_the_backend_cannot_mark(
    tmp_path, use_embedder
):
    use_embedder(_NoEmbedder())
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    provider.update_by_id = lambda *args, **kwargs: False
    runtime = _runtime(tmp_path, provider=provider)
    alice = _identity()
    first = runtime.remember("Keep me", "knowledge_base", alice, memory_id="m")

    with pytest.raises(MemorizzServerError) as error:
        runtime.update_memory(first["record_id"], "Replacement", alice)

    assert error.value.code == "supersede_unsupported"
    rows = _knowledge_rows(runtime, alice, memory_id="m", include_superseded=True)
    assert [row["content"] for row in rows] == ["Keep me"]


@pytest.mark.unit
def test_search_without_embeddings_falls_back_to_keywords(tmp_path, use_embedder):
    use_embedder(_NoEmbedder())
    runtime = _runtime(tmp_path)
    alice = _identity()
    for text in (
        "Deploys go through make release on Fridays",
        "Run the tests with pytest before every deploy",
        "The billing service owns invoices",
    ):
        runtime.remember(text, "knowledge_base", alice, memory_id="proj")
    runtime.remember("Deploy notes for another project", "knowledge_base", alice)
    runtime.remember(
        "Bob deploys on Mondays", "knowledge_base", _identity("bob"), memory_id="proj"
    )

    result = runtime.search_memories(
        "How do we DEPLOY a release?", "knowledge_base", alice, memory_id="proj"
    )

    assert result["search_mode"] == "keyword"
    assert "memorizz config set MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER ollama" in (
        result["note"]
    )
    contents = [row["content"] for row in result["memories"]]
    # Best overlap first; other memory IDs, other tenants and non-matches out.
    assert contents == [
        "Deploys go through make release on Fridays",
        "Run the tests with pytest before every deploy",
    ]
    assert result["memories"][0]["score"] > result["memories"][1]["score"]
    nothing = runtime.search_memories("kubernetes", "knowledge_base", alice)
    assert nothing["count"] == 0 and nothing["search_mode"] == "keyword"
    with pytest.raises(MemorizzServerError) as error:
        runtime.search_memories("   ", "knowledge_base", alice)
    assert error.value.code == "invalid_query"


@pytest.mark.unit
def test_search_is_semantic_with_embeddings(tmp_path, use_embedder):
    use_embedder(_WordEmbedder())
    runtime = _runtime(tmp_path)
    alice = _identity()
    runtime.remember("Deploys go through make release", "knowledge_base", alice)
    runtime.remember("The billing service owns invoices", "knowledge_base", alice)

    result = runtime.search_memories("make release", "knowledge_base", alice, limit=1)

    assert result["search_mode"] == "semantic"
    assert "note" not in result
    assert result["memories"][0]["content"] == "Deploys go through make release"


@pytest.mark.unit
def test_search_falls_back_when_semantic_search_raises(tmp_path, use_embedder):
    use_embedder(_WordEmbedder())
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    runtime = _runtime(tmp_path, provider=provider)
    alice = _identity()
    runtime.remember("Deploys go through make release", "knowledge_base", alice)

    def broken(*args, **kwargs):
        raise RuntimeError("vector index offline")

    provider.retrieve_by_query = broken
    result = runtime.search_memories("deploy", "knowledge_base", alice)

    assert result["search_mode"] == "keyword"
    assert result["note"] == "Semantic search failed, so these are keyword matches."
    assert result["count"] == 1


@pytest.mark.unit
def test_entities_upsert_merge_and_lookup(tmp_path, use_embedder):
    use_embedder(_NoEmbedder())
    runtime = _runtime(tmp_path)
    alice = _identity()

    created = runtime.upsert_entity(
        "Priya",
        alice,
        entity_type="person",
        attributes={"role": "tech lead", "team": "payments"},
        relations=[{"target": "Billing service", "relation_type": "owns"}],
        memory_id="proj",
    )
    assert created["created"] is True
    assert created["created_related_entities"] == ["Billing service"]
    entity = created["entity"]
    assert {(item["name"], item["value"]) for item in entity["attributes"]} == {
        ("role", "tech lead"),
        ("team", "payments"),
    }
    billing = runtime.lookup_entities(alice, name="billing SERVICE", memory_id="proj")
    assert billing["count"] == 1
    assert entity["relations"][0] == {
        **entity["relations"][0],
        "entity_id": billing["entities"][0]["entity_id"],
        "relation_type": "owns",
    }

    merged = runtime.upsert_entity(
        "Priya",
        alice,
        attributes={"role": "engineering manager"},
        relations=[
            {"target": billing["entities"][0]["entity_id"], "relation_type": "owns"}
        ],
        memory_id="proj",
    )
    assert merged["created"] is False
    assert merged["entity_id"] == created["entity_id"]
    assert merged["created_related_entities"] == []
    facts = {item["name"]: item["value"] for item in merged["entity"]["attributes"]}
    assert facts == {"role": "engineering manager", "team": "payments"}

    by_name = runtime.lookup_entities(alice, name="priya", memory_id="proj")
    assert by_name["search_mode"] == "name"
    assert [item["name"] for item in by_name["entities"]] == ["Priya"]
    by_query = runtime.lookup_entities(alice, query="who manages engineering?")
    assert by_query["search_mode"] == "keyword"
    assert by_query["entities"][0]["name"] == "Priya"
    assert "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER" in by_query["note"]
    assert all("embedding" not in item for item in by_query["entities"])

    assert runtime.lookup_entities(_identity("bob"), name="Priya")["count"] == 0
    assert runtime.lookup_entities(alice, name="Priya", memory_id="other")["count"] == 0
    with pytest.raises(MemorizzServerError) as error:
        runtime.lookup_entities(alice)
    assert error.value.code == "invalid_query"
    with pytest.raises(MemorizzServerError) as error:
        runtime.upsert_entity("", alice)
    assert error.value.code == "invalid_entity"
    with pytest.raises(MemorizzServerError) as error:
        runtime.upsert_entity(
            "X", alice, relations=[{"target": "", "relation_type": "owns"}]
        )
    assert error.value.code == "invalid_entity"
    with pytest.raises(MemorizzServerError) as error:
        runtime.upsert_entity("X", _identity("reader", READ_SCOPE))
    assert error.value.code == "insufficient_scope"


@pytest.mark.unit
def test_entity_lookup_by_query_is_semantic_with_embeddings(tmp_path, use_embedder):
    use_embedder(_WordEmbedder())
    runtime = _runtime(tmp_path)
    alice = _identity()
    runtime.upsert_entity(
        "Priya", alice, entity_type="person", attributes={"role": "tech lead"}
    )
    runtime.upsert_entity(
        "Billing service", alice, entity_type="service", attributes={"owner": "ops"}
    )

    result = runtime.lookup_entities(alice, query="tech lead", limit=1)

    assert result["search_mode"] == "semantic"
    assert result["entities"][0]["name"] == "Priya"
    assert "note" not in result


@pytest.mark.unit
def test_memory_status_reports_backend_policy_embeddings_and_counts(
    tmp_path, use_embedder
):
    use_embedder(_NoEmbedder())
    runtime = _runtime(tmp_path, allow_agent_execution=False)
    alice = _identity()
    first = runtime.remember("A", "knowledge_base", alice, memory_id="proj")
    runtime.update_memory(first["record_id"], "B", alice)
    runtime.upsert_entity("Priya", alice, memory_id="proj")

    status = runtime.memory_status(_identity("alice", READ_SCOPE), memory_id="proj")

    assert status["ok"] is True
    assert status["transport"] == "stdio"
    assert status["version"]
    assert status["backend"] == {
        "name": "FileSystemProvider",
        "location": str((tmp_path / "memory").resolve()),
    }
    assert status["policy"]["allow_writes"] is True
    assert status["policy"]["writes_available"] is False  # read-only caller
    assert status["policy"]["allow_agent_execution"] is False
    assert status["policy"]["allow_harness_execution"] is False
    assert status["policy"]["allow_trace_queries"] is False
    assert status["directly_writable_memory_types"] == [
        "knowledge_base",
        "short_term_memory",
    ]
    assert status["embeddings"]["ready"] is False
    assert status["embeddings"]["provider"] == "openai"
    assert "OPENAI_API_KEY" in status["embeddings"]["error"]
    assert status["counts"]["knowledge_base"] == {"active": 1, "superseded": 1}
    assert status["counts"]["entity_memory"] == {"active": 1, "superseded": 0}
    assert status["ingest"]["roots"] == [str((tmp_path / "project").resolve())]
    assert status["ingest"]["available"] is False  # read-only caller

    use_embedder(_WordEmbedder())
    ready = runtime.memory_status(alice)
    assert ready["embeddings"]["ready"] is True
    assert ready["embeddings"]["model"] == "words-64"
    assert ready["ingest"]["available"] is True

    class _BrokenList(FileSystemProvider):
        def list_all(self, *args, **kwargs):
            raise RuntimeError("database offline")

    broken = _runtime(
        tmp_path,
        provider=_BrokenList(
            FileSystemConfig(root_path=tmp_path / "broken", lazy_vector_indexes=True)
        ),
    )
    degraded = broken.memory_status(alice)
    assert degraded["ok"] is True
    assert degraded["counts"]["knowledge_base"] == {"error": "RuntimeError"}


@pytest.mark.unit
def test_embedding_probe_never_raises_or_hangs(tmp_path, use_embedder, monkeypatch):
    class _Hanging(_NoEmbedder):
        def get_embedding(self, text, **kwargs):
            import time

            time.sleep(5)

    use_embedder(_Hanging())
    monkeypatch.setattr(runtime_module, "_EMBEDDING_PROBE_TIMEOUT_SECONDS", 0.2)
    runtime = _runtime(tmp_path)

    status = runtime.memory_status(_identity())

    assert status["embeddings"]["ready"] is False
    assert "didn't answer" in status["embeddings"]["error"]


@pytest.mark.unit
def test_server_declares_the_new_tools_with_policy_annotations(tmp_path):
    config = MemorizzMCPServerConfig(allow_writes=True)
    server = create_memorizz_mcp_server(config, runtime=_runtime(tmp_path))
    tools = server._tool_manager._tools

    for name in ("memorizz_lookup_entities", "memorizz_memory_status"):
        assert tools[name].annotations.read_only_hint is True
    for name in (
        "memorizz_ingest",
        "memorizz_upsert_entity",
        "memorizz_update_memory",
    ):
        annotations = tools[name].annotations
        assert annotations.read_only_hint is False
        assert annotations.destructive_hint is False
    for name in (
        "memorizz_ingest",
        "memorizz_lookup_entities",
        "memorizz_upsert_entity",
        "memorizz_update_memory",
        "memorizz_memory_status",
        "memorizz_list_memories",
        "memorizz_search_memories",
    ):
        assert tools[name].parameters["additionalProperties"] is False
    relation = tools["memorizz_upsert_entity"].parameters["$defs"][
        "EntityRelationArgument"
    ]
    assert relation["additionalProperties"] is False
    assert set(relation["required"]) == {"target", "relation_type"}
    assert set(tools["memorizz_ingest"].parameters["required"]) == {
        "paths",
        "memory_id",
    }
    # New content, or duplicate_of: the runtime requires exactly one.
    assert set(tools["memorizz_update_memory"].parameters["required"]) == {"record_id"}
    assert "duplicate_of" in tools["memorizz_update_memory"].parameters["properties"]
    for name in ("memorizz_list_memories", "memorizz_search_memories"):
        flag = tools[name].parameters["properties"]["include_superseded"]
        assert flag["default"] is False


def test_turns_are_recorded_and_summarized_for_the_caller_only(
    tmp_path, use_embedder, monkeypatch
):
    use_embedder(_WordEmbedder())
    runtime = _runtime(tmp_path)
    alice, bob = _identity("alice"), _identity("bob")

    class _Model:
        def generate(self, messages, **kwargs):
            assert "coding session" in messages[0]["content"]
            assert "Move deploys" in messages[1]["content"]
            return "- Deploys moved to Thursdays."

    recorded = runtime.record_turn(
        "proj",
        "codex-s1",
        "Move deploys to Thursday; token sk-abcdefghijklmnopqrstuvwxyz0123",
        "Done.",
        alice,
        agent_id="codex",
    )
    assert recorded["ok"] and len(recorded["record_ids"]) == 2
    rows = runtime.list_memories(
        "conversation_memory", alice, memory_id="proj", limit=10
    )["memories"]
    assert len(rows) == 2 and all("sk-abcdef" not in row["content"] for row in rows)
    assert all(
        row["thread_id"] == "codex-s1" and row["agent_id"] == "codex" for row in rows
    )
    assert (
        runtime.list_memories("conversation_memory", bob, memory_id="proj", limit=10)[
            "memories"
        ]
        == []
    )

    monkeypatch.setattr("memorizz.episodic_capture.default_model", lambda: None)
    assert runtime.summarize_session("proj", "codex-s1", alice)["reason"] == "no_model"
    monkeypatch.setattr("memorizz.episodic_capture.default_model", lambda: _Model())
    made = runtime.summarize_session("proj", "codex-s1", alice, agent_id="codex")
    assert made["ok"] and len(made["summary_ids"]) == 1
    summaries = runtime.list_memories("summaries", alice, memory_id="proj", limit=10)[
        "memories"
    ]
    assert summaries[0]["content"] == "- Deploys moved to Thursdays."
    # Bob has no turns in that thread, so nothing is made for him.
    assert runtime.summarize_session("proj", "codex-s1", bob)["summary_ids"] == []

    with pytest.raises(MemorizzServerError):
        runtime.record_turn("proj", "t", "x", "y", _identity("carol", READ_SCOPE))
    with pytest.raises(MemorizzServerError, match="memory_id and thread_id"):
        runtime.record_turn("", "t", "x", "y", alice)
