"""Entities past the embedding model's input limit get a bounded vector of
their name, type and newest facts, and the record says so."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List

import pytest

from memorizz.long_term.semantic.entity_memory import (
    EntityMemory,
    embedding_input_tokens,
)
from memorizz.long_term.semantic.entity_memory import entity_memory as module
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]
SCOPE = {"memory_id": "memory-1", "user_id": "user-a"}


class LengthError(Exception):
    """Shaped like OpenAI's BadRequestError for an over-long input."""

    code = "context_length_exceeded"

    def __init__(self) -> None:
        super().__init__(
            "Error code: 400 - This model's maximum context length is 8192 "
            "tokens, however you requested 9300 tokens. Please reduce the length."
        )


class Embedder:
    """Rejects any text over ``max_chars`` the way a hosted model does."""

    def __init__(self, max_chars: int = 10**9, error: Exception = None) -> None:
        self.max_chars = max_chars
        self.error = error
        self.texts: List[str] = []

    def __call__(self, text: str) -> List[float]:
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        if len(text) > self.max_chars:
            raise LengthError()
        return list(VECTOR)

    def get_embedding(self, text: str) -> List[float]:
        return list(VECTOR)


@pytest.fixture()
def provider():
    path = Path(tempfile.mkdtemp(prefix="memorizz-entity-vectors-"))
    try:
        yield FileSystemProvider(
            FileSystemConfig(
                root_path=path / "memory",
                embedding_provider=Embedder(),
                use_faiss=False,
                lazy_vector_indexes=True,
            )
        )
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _use(monkeypatch, embedder: Embedder) -> Embedder:
    monkeypatch.setattr(module, "get_embedding", embedder)
    return embedder


def _facts(count: int, start: int = 0) -> List[dict]:
    return [
        {"name": f"fact_{index:03d}", "value": f"value number {index} " + "x" * 80}
        for index in range(start, start + count)
    ]


def test_an_entity_under_the_limit_gets_a_full_vector(provider, monkeypatch):
    embedder = _use(monkeypatch, Embedder())
    entities = EntityMemory(provider)
    report = entities.upsert_entity_with_report(
        name="Ines", entity_type="person", attributes=_facts(3), **SCOPE
    )
    assert report["embedding"]["status"] == "full"
    record = entities.get_entity(report["entity_id"], **SCOPE)
    assert record["embedding"] == VECTOR
    assert "embedding_status" not in (record.get("metadata") or {})
    assert "fact_000" in embedder.texts[-1]


def test_text_over_the_estimated_limit_is_cut_to_the_newest_facts(
    provider, monkeypatch
):
    embedder = _use(monkeypatch, Embedder())
    entities = EntityMemory(provider, max_embedding_tokens=400)
    # Older facts first, newer ones in a later write.
    entities.upsert_entity(
        name="Ines", entity_type="person", attributes=_facts(30), **SCOPE
    )
    entity_id = entities.upsert_entity(
        name="Ines", entity_type="person", attributes=_facts(30, 30), **SCOPE
    )

    record = entities.get_entity(entity_id, **SCOPE)
    text = embedder.texts[-1]
    # The vector is stored, and it rests on a bounded text: name, type and
    # the newest facts, not the oldest.
    assert record["embedding"] == VECTOR
    assert len(text) <= int(400 * 0.9) * 3
    assert text.startswith("Name: Ines\nType: person")
    assert "fact_059" in text and "fact_000" not in text
    metadata = record["metadata"]
    assert metadata["embedding_status"] == "partial"
    assert metadata["embedding_reason"] == "over_token_limit"
    covered = metadata["embedding_attributes_covered"]
    assert 0 < covered < 60
    assert EntityMemory.embedding_report(record) == {
        "status": "partial",
        "reason": "over_token_limit",
        "attributes_covered": covered,
        "attributes_total": 60,
    }
    # Every fact is still stored, and found by name.
    assert len(entities.get_entity_by_name("Ines", **SCOPE)["attributes"]) == 60


def test_a_provider_length_error_retries_with_a_shorter_text(provider, monkeypatch):
    # The estimate says it fits; the provider counts tokens its own way.
    embedder = _use(monkeypatch, Embedder(max_chars=2000))
    entities = EntityMemory(provider)
    report = entities.upsert_entity_with_report(
        name="Ines", entity_type="person", attributes=_facts(40), **SCOPE
    )

    embedding = report["embedding"]
    assert embedding["status"] == "partial"
    assert embedding["reason"] == "provider_length_error"
    assert 0 < embedding["attributes_covered"] < 40
    assert len(embedder.texts) >= 2 and len(embedder.texts[-1]) <= 2000
    record = entities.get_entity(report["entity_id"], **SCOPE)
    assert record["embedding"] == VECTOR
    assert record["metadata"]["embedding_status"] == "partial"
    # Status fields never leak into the embedded text itself.
    assert "embedding_status" not in embedder.texts[-1]


def test_a_failed_embed_is_recorded_as_missing(provider, monkeypatch):
    _use(monkeypatch, Embedder(error=RuntimeError("embedder offline")))
    entities = EntityMemory(provider)
    report = entities.upsert_entity_with_report(
        name="Ines", attributes=[{"name": "city", "value": "Porto"}], **SCOPE
    )
    assert report["embedding"] == {
        "status": "missing",
        "reason": "embedding_failed",
        "attributes_covered": 0,
        "attributes_total": 1,
    }
    record = entities.get_entity(report["entity_id"], **SCOPE)
    assert record.get("embedding") is None
    assert record["metadata"]["embedding_status"] == "missing"


def test_a_failed_reembed_keeps_the_previous_vector_and_says_so(provider, monkeypatch):
    embedder = _use(monkeypatch, Embedder())
    entities = EntityMemory(provider)
    entity_id = entities.upsert_entity(
        name="Ines", attributes=[{"name": "city", "value": "Porto"}], **SCOPE
    )
    embedder.error = RuntimeError("embedder offline")
    report = entities.upsert_entity_with_report(
        name="Ines", attributes=[{"name": "pet", "value": "a cat"}], **SCOPE
    )
    assert report["embedding"]["status"] == "partial"
    assert report["embedding"]["reason"] == "previous_vector_kept"
    assert entities.get_entity(entity_id, **SCOPE)["embedding"] == VECTOR

    # Once an embed succeeds again the note goes away.
    embedder.error = None
    entities.upsert_entity(
        name="Ines", attributes=[{"name": "job", "value": "nurse"}], **SCOPE
    )
    metadata = entities.get_entity(entity_id, **SCOPE).get("metadata") or {}
    assert not set(metadata) & {
        "embedding_status",
        "embedding_reason",
        "embedding_attributes_covered",
    }


def _duplicate(provider, entity_id: str, name: str, facts: List[dict]) -> None:
    provider.store(
        {
            "_id": entity_id,
            "entity_id": entity_id,
            "name": name,
            "entity_type": "person",
            "attributes": facts,
            "relations": [],
            "metadata": {},
            "created_at": "2026-10-01T00:00:00Z",
            "updated_at": "2026-10-01T00:00:00Z",
            **SCOPE,
        },
        "entity_memory",
    )


def test_merged_duplicates_past_the_limit_get_a_bounded_vector(provider, monkeypatch):
    embedder = _use(monkeypatch, Embedder(max_chars=600))
    _duplicate(provider, "a", "Ines", _facts(10))
    _duplicate(provider, "b", "Ines", _facts(10, 10))
    entities = EntityMemory(provider)
    plan = entities.consolidate_duplicate_entities(apply=True, **SCOPE)
    group = plan["groups"][0]
    assert group["applied"]
    merged = entities.get_entity(group["canonical_entity_id"], **SCOPE)
    assert merged["embedding"] == VECTOR
    assert merged["metadata"]["embedding_status"] == "partial"

    # An embedder outage still stops a merge, and both rows are kept.
    _duplicate(provider, "c", "Bo", _facts(1))
    _duplicate(provider, "d", "Bo", _facts(1, 1))
    embedder.error = RuntimeError("embedder offline")
    with pytest.raises(RuntimeError):
        entities.consolidate_duplicate_entities(apply=True, **SCOPE)
    assert {row["entity_id"] for row in entities.list_entities(**SCOPE)} >= {"c", "d"}


def test_the_input_limit_follows_the_model_and_the_override(monkeypatch):
    from memorizz import embeddings

    monkeypatch.setattr(embeddings, "_global_embedding_manager", None)
    monkeypatch.delenv("MEMORIZZ_ENTITY_EMBEDDING_MAX_TOKENS", raising=False)
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
    assert embedding_input_tokens() == 8191
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "voyageai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "voyage-3.5")
    assert embedding_input_tokens() == 32000
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "mxbai-embed-large")
    assert embedding_input_tokens() == 512
    # Ollama runs nomic-embed-text with a 2,048-token context, not 8,192.
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")
    assert embedding_input_tokens() == 2048
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text-v1.5")
    assert embedding_input_tokens() == 8192
    monkeypatch.setenv("MEMORIZZ_ENTITY_EMBEDDING_MAX_TOKENS", "1500")
    assert embedding_input_tokens() == 1500


def test_length_errors_are_told_apart_from_other_errors():
    assert module._is_length_error(LengthError())
    assert module._is_length_error(
        ValueError("input length exceeds the context window of 512 tokens")
    )
    assert not module._is_length_error(RuntimeError("rate limit reached"))
    assert not module._is_length_error(ConnectionError("connection reset"))


def test_the_memory_pages_note_a_partial_or_missing_vector():
    from memorizz.ui.memory_view import shape_record
    from memorizz.ui.routers.playground import _serialize_entity_memory_item

    partial = {
        "entity_id": "e1",
        "name": "Ines",
        "attributes": [{"name": "city", "value": "Porto"}],
        "metadata": {
            "embedding_status": "partial",
            "embedding_reason": "over_token_limit",
        },
    }
    row = shape_record(partial)
    assert row["vector_note"]["label"] == "partial vector"
    assert "over token limit" in row["vector_note"]["title"]
    missing = {**partial, "metadata": {"embedding_status": "missing"}}
    assert shape_record(missing)["vector_note"]["label"] == "no vector"
    assert shape_record({"name": "Ines"})["vector_note"]["label"] == ""
    item = _serialize_entity_memory_item(partial)
    assert item["vector_note"] == "partial vector"
    assert _serialize_entity_memory_item({"name": "Ines"})["vector_note"] == ""
