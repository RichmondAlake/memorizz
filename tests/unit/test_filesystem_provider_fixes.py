"""Regression tests for filesystem-provider defects.

Covers record-id validation (path traversal), patch timestamps on
``update_by_id``, index merging between provider instances that share one
root, vector-index staleness after a write during a rebuild, and scoped FAISS
queries that a larger tenant used to crowd out.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


class MappedEmbeddingProvider:
    """Embedding stub with explicit vectors so cosine ranking is predictable."""

    def __init__(self, vectors: Dict[str, List[float]], default: List[float]):
        self.vectors = vectors
        self.default = default

    def get_embedding(self, text: str) -> List[float]:
        return list(self.vectors.get(text, self.default))

    def get_provider_info(self) -> str:
        return "mapped"


class _FakeIndexFlatIP:
    """Inner-product flat index with FAISS's (distances, indices) contract."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors = np.zeros((0, dimension), dtype="float32")

    def add(self, vectors) -> None:
        self._vectors = np.vstack([self._vectors, np.asarray(vectors, "float32")])

    def search(self, queries, k):
        queries = np.asarray(queries, dtype="float32")
        scores = queries @ self._vectors.T
        total = self._vectors.shape[0]
        k = int(k)
        order = np.argsort(-scores, axis=1)[:, : min(k, total)]
        distances = np.take_along_axis(scores, order, axis=1)
        if k > total:  # FAISS pads missing slots with -1 / -inf.
            pad = k - total
            order = np.hstack([order, -np.ones((order.shape[0], pad), dtype=int)])
            distances = np.hstack(
                [distances, np.full((distances.shape[0], pad), -np.inf, "float32")]
            )
        return distances, order


class _FakeFaiss:
    IndexFlatIP = _FakeIndexFlatIP


def _make_provider(root: Path, embedding_provider=None) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=root,
            embedding_provider=embedding_provider,
            lazy_vector_indexes=True,
        )
    )


def _written_documents(provider: FileSystemProvider) -> List[Path]:
    return [
        path for path in provider.root_path.rglob("*.json") if path.name != "index.json"
    ]


# ---------------------------------------------------------------------------
# 1. Record ids are validated before they become filenames
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_id",
    [
        "../../escaped",
        "..",
        "nested/child",
        "nested\\child",
        ".hidden",
        "nul\x00byte",
        "x" * 256,
    ],
)
def test_store_rejects_ids_that_escape_or_hide_inside_the_store(tmp_path, bad_id):
    provider = _make_provider(tmp_path / "fs-memory")

    with pytest.raises(ValueError):
        provider.store({"_id": bad_id, "content": "x"}, MemoryType.CONVERSATION_MEMORY)
    with pytest.raises(ValueError):
        provider.store({"id": bad_id, "content": "x"}, MemoryType.CONVERSATION_MEMORY)

    assert not (tmp_path / "escaped.json").exists()
    assert _written_documents(provider) == []
    assert provider.list_all(MemoryType.CONVERSATION_MEMORY) == []
    # Reads with a bad id are a plain miss, never a filesystem escape.
    assert provider.retrieve_by_id(bad_id, MemoryType.CONVERSATION_MEMORY) is None
    assert provider.delete_by_id(bad_id, MemoryType.CONVERSATION_MEMORY) is False
    assert (
        provider.update_by_id(bad_id, {"content": "y"}, MemoryType.CONVERSATION_MEMORY)
        is False
    )


def test_store_archive_record_rejects_traversal_ids(tmp_path):
    provider = _make_provider(tmp_path / "fs-memory")
    with pytest.raises(ValueError):
        provider.store_archive_record(
            MemoryType.KNOWLEDGE_BASE, "../../escaped", {"content": "x"}
        )
    assert not (tmp_path / "escaped.json").exists()


@pytest.mark.parametrize(
    "good_id",
    ["agent:tool", "web-search_v2.beta", "user@example.com", "a b", "ümlaut"],
)
def test_safe_ids_round_trip_through_store_update_and_delete(tmp_path, good_id):
    provider = _make_provider(tmp_path / "fs-memory")

    assert provider.store({"_id": good_id, "content": "v1"}, MemoryType.TOOLBOX) == (
        good_id
    )
    assert provider.retrieve_by_id(good_id, MemoryType.TOOLBOX)["content"] == "v1"
    assert provider.update_by_id(good_id, {"content": "v2"}, MemoryType.TOOLBOX)
    assert provider.retrieve_by_id(good_id, MemoryType.TOOLBOX)["content"] == "v2"
    assert provider.delete_by_id(good_id, MemoryType.TOOLBOX) is True
    assert provider.retrieve_by_id(good_id, MemoryType.TOOLBOX) is None


def test_existing_raw_id_files_stay_readable(tmp_path):
    """Stores written before validation used the raw id as the filename."""
    provider = _make_provider(tmp_path / "fs-memory")
    store_path = provider._store_paths[MemoryType.TOOLBOX]
    (store_path / "agent:tool.json").write_text(
        json.dumps({"_id": "agent:tool", "id": "agent:tool", "name": "legacy"}),
        encoding="utf-8",
    )
    (store_path / "index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "items": {"agent:tool": {"id": "agent:tool", "name": "legacy"}},
            }
        ),
        encoding="utf-8",
    )

    reopened = _make_provider(tmp_path / "fs-memory")
    assert reopened.retrieve_by_id("agent:tool", MemoryType.TOOLBOX)["name"] == "legacy"
    assert (
        reopened.retrieve_by_name("legacy", MemoryType.TOOLBOX)["_id"] == "agent:tool"
    )


# ---------------------------------------------------------------------------
# 2. update_by_id patches do not get a default timestamp
# ---------------------------------------------------------------------------
def test_embedding_only_update_keeps_conversation_order_and_timestamp(tmp_path):
    provider = _make_provider(tmp_path / "fs-memory")
    base = 1_700_000_000.0
    ids = [
        provider.store(
            {
                "content": f"msg{i}",
                "role": "user",
                "memory_id": "memory-1",
                "timestamp": base + i,
            },
            MemoryType.CONVERSATION_MEMORY,
        )
        for i in range(3)
    ]

    assert provider.update_by_id(
        ids[0], {"embedding": [1.0, 0.0, 0.0]}, MemoryType.CONVERSATION_MEMORY
    )

    history = provider.retrieve_conversation_history_ordered_by_timestamp("memory-1")
    assert [row["content"] for row in history] == ["msg0", "msg1", "msg2"]
    first = provider.retrieve_by_id(ids[0], MemoryType.CONVERSATION_MEMORY)
    assert first["timestamp"] == base
    assert first["embedding"] == [1.0, 0.0, 0.0]
    assert first["updated_at"]


def test_update_patch_can_still_set_an_explicit_timestamp(tmp_path):
    provider = _make_provider(tmp_path / "fs-memory")
    doc_id = provider.store(
        {"content": "msg", "memory_id": "m", "timestamp": 10.0},
        MemoryType.CONVERSATION_MEMORY,
    )
    assert provider.update_by_id(
        doc_id, {"timestamp": 20.0}, MemoryType.CONVERSATION_MEMORY
    )
    assert (
        provider.retrieve_by_id(doc_id, MemoryType.CONVERSATION_MEMORY)["timestamp"]
        == 20.0
    )


# ---------------------------------------------------------------------------
# 3. Two provider instances on one root merge the index instead of
#    overwriting each other
# ---------------------------------------------------------------------------
def test_two_instances_on_one_root_keep_each_others_records(tmp_path):
    root = tmp_path / "shared"
    first = _make_provider(root)
    second = _make_provider(root)

    first.store({"_id": "A", "content": "from first"}, MemoryType.KNOWLEDGE_BASE)
    second.store({"_id": "B", "content": "from second"}, MemoryType.KNOWLEDGE_BASE)

    fresh = _make_provider(root)
    assert {row["_id"] for row in fresh.list_all(MemoryType.KNOWLEDGE_BASE)} == {
        "A",
        "B",
    }
    assert fresh.retrieve_by_id("A", MemoryType.KNOWLEDGE_BASE)["content"] == (
        "from first"
    )
    assert fresh.delete_by_id("A", MemoryType.KNOWLEDGE_BASE) is True
    assert not (fresh._store_paths[MemoryType.KNOWLEDGE_BASE] / "A.json").exists()

    # A later save from an instance that never saw A must not resurrect it,
    # and must keep B which it also never wrote.
    first.store({"_id": "C", "content": "from first again"}, MemoryType.KNOWLEDGE_BASE)
    again = _make_provider(root)
    assert {row["_id"] for row in again.list_all(MemoryType.KNOWLEDGE_BASE)} == {
        "B",
        "C",
    }


def test_stale_instance_reads_and_deletes_records_written_by_another(tmp_path):
    root = tmp_path / "shared"
    writer = _make_provider(root)
    stale = _make_provider(root)  # index loaded before A exists

    writer.store({"_id": "A", "content": "late arrival"}, MemoryType.KNOWLEDGE_BASE)

    assert stale.retrieve_by_id("A", MemoryType.KNOWLEDGE_BASE)["content"] == (
        "late arrival"
    )
    assert stale.delete_by_id("A", MemoryType.KNOWLEDGE_BASE) is True
    assert stale.retrieve_by_id("A", MemoryType.KNOWLEDGE_BASE) is None
    assert writer.retrieve_by_id("A", MemoryType.KNOWLEDGE_BASE) is None
    assert _make_provider(root).list_all(MemoryType.KNOWLEDGE_BASE) == []


def test_index_is_rebuilt_from_directory_when_missing_or_corrupt(tmp_path):
    root = tmp_path / "shared"
    provider = _make_provider(root)
    provider.store({"_id": "A", "content": "kept"}, MemoryType.KNOWLEDGE_BASE)
    index_path = provider._store_paths[MemoryType.KNOWLEDGE_BASE] / "index.json"

    index_path.write_text("{not json", encoding="utf-8")
    assert [
        row["_id"] for row in _make_provider(root).list_all(MemoryType.KNOWLEDGE_BASE)
    ] == ["A"]

    index_path.unlink()
    rebuilt = _make_provider(root)
    assert [row["_id"] for row in rebuilt.list_all(MemoryType.KNOWLEDGE_BASE)] == ["A"]
    rebuilt.store({"_id": "B", "content": "new"}, MemoryType.KNOWLEDGE_BASE)
    assert json.loads(index_path.read_text(encoding="utf-8"))["items"].keys() == {
        "A",
        "B",
    }


# ---------------------------------------------------------------------------
# 4. A write that lands during a vector-index rebuild is not hidden
# ---------------------------------------------------------------------------
def test_write_during_vector_rebuild_leaves_index_dirty_and_searchable(tmp_path):
    embeddings = MappedEmbeddingProvider(
        {"first": [1.0, 0.0, 0.0], "second": [0.0, 1.0, 0.0]}, default=[1.0, 0.0, 0.0]
    )
    provider = _make_provider(tmp_path / "fs-memory", embedding_provider=embeddings)
    provider.store(
        {"content": "first", "memory_id": "m", "embedding": [1.0, 0.0, 0.0]},
        MemoryType.KNOWLEDGE_BASE,
    )

    fired = {"count": 0}

    class _HookedFaiss:
        """Stores a new document after the rebuild released the store lock."""

        @staticmethod
        def IndexFlatIP(dimension):
            if fired["count"] == 0:
                fired["count"] += 1
                provider.store(
                    {
                        "content": "second",
                        "memory_id": "m",
                        "embedding": [0.0, 1.0, 0.0],
                    },
                    MemoryType.KNOWLEDGE_BASE,
                )
            return _FakeIndexFlatIP(dimension)

    provider._faiss = _HookedFaiss()

    provider.retrieve_by_query("second", MemoryType.KNOWLEDGE_BASE, limit=1)
    assert fired["count"] == 1
    assert provider._vector_state[MemoryType.KNOWLEDGE_BASE]["dirty"] is True

    results = provider.retrieve_by_query("second", MemoryType.KNOWLEDGE_BASE, limit=1)
    assert results and results[0]["content"] == "second"


# ---------------------------------------------------------------------------
# 5. user_id / memory_id scoped FAISS queries are not crowded out
# ---------------------------------------------------------------------------
def _seed_crowded_store(provider: FileSystemProvider) -> None:
    for i in range(12):
        provider.store(
            {
                "content": f"row {i}",
                "memory_id": "big",
                "user_id": "A",
                "embedding": [1.0, 0.01 * i, 0.0],
            },
            MemoryType.KNOWLEDGE_BASE,
        )
    provider.store(
        {
            "content": "lonely row",
            "memory_id": "small",
            "user_id": "B",
            "embedding": [0.5, 0.5, 0.0],
        },
        MemoryType.KNOWLEDGE_BASE,
    )


def test_user_scoped_query_returns_the_small_tenants_row(tmp_path):
    embeddings = MappedEmbeddingProvider({}, default=[1.0, 0.0, 0.0])
    provider = _make_provider(tmp_path / "fs-memory", embedding_provider=embeddings)
    provider._faiss = _FakeFaiss()
    _seed_crowded_store(provider)

    results = provider.retrieve_by_query(
        "probe", MemoryType.KNOWLEDGE_BASE, limit=1, user_id="B"
    )
    assert len(results) == 1
    assert results[0]["user_id"] == "B"
    assert results[0]["content"] == "lonely row"


def test_memory_scoped_query_returns_the_small_memorys_row(tmp_path):
    embeddings = MappedEmbeddingProvider({}, default=[1.0, 0.0, 0.0])
    provider = _make_provider(tmp_path / "fs-memory", embedding_provider=embeddings)
    provider._faiss = _FakeFaiss()
    _seed_crowded_store(provider)

    results = provider.retrieve_by_query(
        "probe", MemoryType.KNOWLEDGE_BASE, limit=1, memory_id="small"
    )
    assert len(results) == 1
    assert results[0]["memory_id"] == "small"


def test_unscoped_query_still_ranks_the_closest_row_first(tmp_path):
    embeddings = MappedEmbeddingProvider({}, default=[1.0, 0.0, 0.0])
    provider = _make_provider(tmp_path / "fs-memory", embedding_provider=embeddings)
    provider._faiss = _FakeFaiss()
    _seed_crowded_store(provider)

    results = provider.retrieve_by_query("probe", MemoryType.KNOWLEDGE_BASE, limit=2)
    assert [row["content"] for row in results] == ["row 0", "row 1"]


@pytest.mark.unit
def test_read_paths_see_records_written_by_another_instance_without_restart(tmp_path):
    from memorizz.enums.memory_type import MemoryType
    from memorizz.memory_provider.filesystem.provider import (
        FileSystemConfig,
        FileSystemProvider,
    )

    root = tmp_path / "shared"
    first = FileSystemProvider(
        FileSystemConfig(root_path=root, use_faiss=False, lazy_vector_indexes=True)
    )
    second = FileSystemProvider(
        FileSystemConfig(root_path=root, use_faiss=False, lazy_vector_indexes=True)
    )
    first.store(
        {
            "_id": "a",
            "name": "a",
            "content": "written by first",
            "memory_id": "kb",
            "embedding": [1.0, 0.0],
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    # The second instance was created before the write and never wrote itself.
    assert {row["_id"] for row in second.list_all(MemoryType.KNOWLEDGE_BASE)} == {"a"}
    second.store(
        {
            "_id": "b",
            "name": "b",
            "content": "written by second",
            "memory_id": "kb",
            "embedding": [0.0, 1.0],
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    assert {row["_id"] for row in first.list_all(MemoryType.KNOWLEDGE_BASE)} == {
        "a",
        "b",
    }
    assert first.retrieve_by_name("b", MemoryType.KNOWLEDGE_BASE) is not None
    assert second.delete_by_id("a", MemoryType.KNOWLEDGE_BASE) is True
    assert {row["_id"] for row in first.list_all(MemoryType.KNOWLEDGE_BASE)} == {"b"}
    # Keyword retrieval also sees the other instance's row.
    hits = first.retrieve_by_query(
        "written by second", MemoryType.KNOWLEDGE_BASE, limit=5
    )
    assert any(row.get("_id") == "b" for row in hits)
