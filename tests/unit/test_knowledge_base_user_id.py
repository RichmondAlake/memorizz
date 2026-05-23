"""Tenant-isolation tests for ``KnowledgeBase`` ingestion (M1).

These tests verify that ``user_id`` flows from every ``ingest_*`` entry point
on :class:`memorizz.long_term.semantic.knowledge_base.KnowledgeBase` all the
way down to every chunk written to the memory provider. Without M1, the
chunk dicts contained no ``user_id`` field and were therefore stored in the
anonymous bucket regardless of who called ``ingest_*`` — which made
``_MONGO_USER_SCOPED_TYPES`` and the equivalent Oracle predicate useless for
the KNOWLEDGE_BASE memory type.

The filesystem provider is the always-available backend in CI, and shares
the same user-scoping contract as the Mongo and Oracle providers, so a
green filesystem suite combined with the explicit chunk-dict assertions
below is a strong signal that the M1 behaviour is correct across all three
providers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.semantic.knowledge_base import KnowledgeBase
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


class _DummyEmbeddingProvider:
    """Deterministic embedding provider so vector-search tests are stable."""

    def get_embedding(self, text: str) -> List[float]:
        seed = float(sum(ord(ch) for ch in (text or "")))
        return [seed, float(len(text or "")), 0.0]

    def get_provider_info(self) -> str:  # pragma: no cover
        return "dummy"


@pytest.fixture()
def kb(tmp_path):
    """A fresh KnowledgeBase backed by an isolated filesystem provider."""
    root = Path(tmp_path) / "kb-memory"
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=root,
            embedding_provider=_DummyEmbeddingProvider(),
            lazy_vector_indexes=True,
        )
    )
    return KnowledgeBase(memory_provider=provider), provider


# ---------------------------------------------------------------------------
# ingest_knowledge
# ---------------------------------------------------------------------------


def test_ingest_knowledge_writes_user_id_to_every_chunk(kb):
    """Every chunk produced by ingest_knowledge carries the caller's user_id."""
    knowledge_base, provider = kb
    corpus = "alpha. beta. gamma. delta. epsilon."  # forces multiple chunks

    kb_id = knowledge_base.ingest_knowledge(
        corpus=corpus,
        namespace="alice-notes",
        chunking_strategy="fixed",
        chunk_size=12,
        chunk_overlap=0,
        user_id="alice",
    )

    chunks = knowledge_base.retrieve_knowledge(kb_id)
    assert len(chunks) >= 2, "fixture should produce multiple chunks"
    assert all(c.get("user_id") == "alice" for c in chunks)


def test_ingest_knowledge_without_user_id_preserves_legacy_none(kb):
    """Omitting user_id keeps the anonymous (None) bucket — no behaviour change."""
    knowledge_base, _ = kb

    kb_id = knowledge_base.ingest_knowledge(
        corpus="legacy content body",
        namespace="legacy",
        chunking_strategy="none",
    )

    chunks = knowledge_base.retrieve_knowledge(kb_id)
    assert len(chunks) == 1
    # The field must EXIST (so the provider can index it) and be None.
    assert "user_id" in chunks[0]
    assert chunks[0]["user_id"] is None


def test_ingest_knowledge_isolates_users_when_provider_filters(kb):
    """The provider filter must isolate per-user chunks when user_id is set."""
    knowledge_base, provider = kb

    alice_kb = knowledge_base.ingest_knowledge(
        corpus="alice secret",
        namespace="secrets",
        chunking_strategy="none",
        user_id="alice",
    )
    bob_kb = knowledge_base.ingest_knowledge(
        corpus="bob secret",
        namespace="secrets",
        chunking_strategy="none",
        user_id="bob",
    )

    # Provider-level list_all with user_id must isolate.
    alice_rows = provider.list_all(MemoryType.KNOWLEDGE_BASE, user_id="alice")
    bob_rows = provider.list_all(MemoryType.KNOWLEDGE_BASE, user_id="bob")

    alice_contents = {r["content"] for r in alice_rows}
    bob_contents = {r["content"] for r in bob_rows}

    assert "alice secret" in alice_contents
    assert "bob secret" not in alice_contents
    assert "bob secret" in bob_contents
    assert "alice secret" not in bob_contents

    # And the KB-level retrieval still works (kb_id is the source of truth there).
    assert any(
        c["content"] == "alice secret"
        for c in knowledge_base.retrieve_knowledge(alice_kb)
    )
    assert any(
        c["content"] == "bob secret" for c in knowledge_base.retrieve_knowledge(bob_kb)
    )


def test_ingest_knowledge_user_id_none_is_separate_scope_from_alice(kb):
    """``user_id=None`` is a scope (anonymous bucket), not match-everything."""
    knowledge_base, provider = kb

    knowledge_base.ingest_knowledge(
        corpus="public content",
        namespace="public",
        chunking_strategy="none",
        user_id=None,
    )
    knowledge_base.ingest_knowledge(
        corpus="alice content",
        namespace="alice",
        chunking_strategy="none",
        user_id="alice",
    )

    anon_rows = provider.list_all(MemoryType.KNOWLEDGE_BASE, user_id=None)
    alice_rows = provider.list_all(MemoryType.KNOWLEDGE_BASE, user_id="alice")

    anon_contents = {r["content"] for r in anon_rows}
    alice_contents = {r["content"] for r in alice_rows}

    assert anon_contents == {"public content"}
    assert alice_contents == {"alice content"}


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------


def test_ingest_file_forwards_user_id(kb, tmp_path):
    """ingest_file forwards user_id through to ingest_knowledge."""
    knowledge_base, _ = kb
    source = tmp_path / "notes.txt"
    source.write_text("hello from alice\nsecond line\nthird line")

    kb_id = knowledge_base.ingest_file(
        source=source,
        chunking_strategy="paragraph",
        user_id="alice",
    )

    chunks = knowledge_base.retrieve_knowledge(kb_id)
    assert chunks, "file should have produced at least one chunk"
    assert all(c.get("user_id") == "alice" for c in chunks)


def test_ingest_file_without_user_id_is_anonymous(kb, tmp_path):
    knowledge_base, _ = kb
    source = tmp_path / "legacy.txt"
    source.write_text("content without a tenant")

    kb_id = knowledge_base.ingest_file(source=source, chunking_strategy="none")

    chunks = knowledge_base.retrieve_knowledge(kb_id)
    assert chunks
    assert all(c.get("user_id") is None for c in chunks)


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------


def test_ingest_directory_forwards_user_id_to_every_file(kb, tmp_path):
    """Every file ingested by ingest_directory carries the caller's user_id."""
    knowledge_base, _ = kb

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.txt").write_text("first document body content")
    (docs / "two.txt").write_text("second document body content")
    (docs / "three.md").write_text("# heading\n\nmarkdown body content")

    result = knowledge_base.ingest_directory(
        root=docs,
        chunking_strategy="none",
        user_id="alice",
    )

    assert result["total"] == 3
    assert result["ingested"] == 3

    for entry in result["results"]:
        kb_id = entry["knowledge_base_id"]
        assert kb_id, f"expected kb_id for {entry['path']}"
        chunks = knowledge_base.retrieve_knowledge(kb_id)
        assert chunks
        assert all(
            c.get("user_id") == "alice" for c in chunks
        ), f"chunks for {entry['path']} missing user_id alice"


def test_ingest_directory_without_user_id_is_anonymous(kb, tmp_path):
    knowledge_base, provider = kb
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "only.txt").write_text("body content")

    result = knowledge_base.ingest_directory(root=docs, chunking_strategy="none")
    assert result["ingested"] == 1

    kb_id = result["results"][0]["knowledge_base_id"]
    chunks = knowledge_base.retrieve_knowledge(kb_id)
    assert all(c.get("user_id") is None for c in chunks)


# ---------------------------------------------------------------------------
# Backwards-compatibility surface
# ---------------------------------------------------------------------------


def test_ingest_knowledge_call_without_kwarg_still_works(kb):
    """Positional-only existing callers must not break (default user_id=None)."""
    knowledge_base, _ = kb

    # Mimic an old caller that only passes the documented positional args.
    kb_id = knowledge_base.ingest_knowledge("some content", "some-namespace", "none")

    chunks = knowledge_base.retrieve_knowledge(kb_id)
    assert chunks
    assert all(c.get("user_id") is None for c in chunks)
