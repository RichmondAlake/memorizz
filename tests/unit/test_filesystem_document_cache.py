"""The file store parses an unchanged document once and hands out copies."""

from __future__ import annotations

import json

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


def _provider(tmp_path, **env):
    return FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )


@pytest.mark.unit
def test_an_unchanged_document_is_parsed_once_and_each_reader_gets_its_own_copy(
    tmp_path,
):
    provider = _provider(tmp_path)
    try:
        kind = MemoryType.CONVERSATION_MEMORY
        doc_id = provider.store({"role": "user", "content": {"text": "hello"}}, kind)
        cache = provider._document_cache
        cache.clear()
        cache.hits = cache.misses = 0

        first = provider.list_all(kind)
        second = provider.list_all(kind)

        assert (cache.misses, cache.hits) == (1, 1)
        assert first == second and first[0] is not second[0]
        # What one reader does to its copy never reaches the next.
        first[0]["content"]["text"] = "changed by a caller"
        first[0]["extra"] = True
        again = provider.retrieve_by_id(doc_id, kind)
        assert again["content"]["text"] == "hello" and "extra" not in again
    finally:
        provider.close()


@pytest.mark.unit
def test_a_rewritten_or_removed_document_is_read_afresh(tmp_path):
    provider = _provider(tmp_path)
    other = _provider(tmp_path)  # another process on the same store
    try:
        kind = MemoryType.CONVERSATION_MEMORY
        doc_id = provider.store({"role": "user", "content": "one"}, kind)
        assert provider.retrieve_by_id(doc_id, kind)["content"] == "one"

        provider.update_by_id(doc_id, {"content": "two"}, kind)
        assert provider.retrieve_by_id(doc_id, kind)["content"] == "two"

        # A write made elsewhere, straight to the file.
        path = other._document_path(kind, doc_id)
        changed = json.loads(path.read_text())
        changed["content"] = "three, written by another process"
        replacement = path.with_suffix(".swap")
        replacement.write_text(json.dumps(changed))
        replacement.replace(path)
        assert provider.retrieve_by_id(doc_id, kind)["content"].startswith("three")

        path.unlink()
        assert provider._read_document(kind, doc_id) is None
        assert (kind, doc_id) not in provider._document_cache._entries
    finally:
        provider.close()
        other.close()


@pytest.mark.unit
def test_the_cache_can_be_turned_off_and_stays_within_its_size(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_FS_DOCUMENT_CACHE_MB", "0")
    off = _provider(tmp_path)
    try:
        kind = MemoryType.CONVERSATION_MEMORY
        off.store({"role": "user", "content": "x"}, kind)
        off.list_all(kind)
        off.list_all(kind)
        assert off._document_cache.hits == 0 and not off._document_cache._entries
    finally:
        off.close()

    from memorizz.memory_provider.filesystem.provider import _DocumentCache

    small = _DocumentCache(4_000)
    for number in range(40):
        small.put("kind", f"doc-{number}", 1, {"text": "x" * 300})
    assert 0 < small._bytes <= 4_000
    assert small.get("kind", "doc-0", 1) is None
    assert small.get("kind", "doc-39", 1) == {"text": "x" * 300}
    assert small.get("kind", "doc-39", 2) is None
