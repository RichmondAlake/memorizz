import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.memory_provider.vectors import (
    scope_hash,
    validate_vector,
    vector_document,
    vector_id,
)


@pytest.mark.parametrize(
    "invalid", [[], [0, 0], [float("nan")], [float("inf")], [True], ["1"]]
)
def test_invalid_vectors_rejected(invalid):
    with pytest.raises(ValueError):
        validate_vector(invalid)


def test_filesystem_vector_contract_update_scope_and_namespace(tmp_path):
    provider = FileSystemProvider(FileSystemConfig(tmp_path, use_faiss=False))
    provider.upsert_vector(
        "notion:a", "one", [1, 0], metadata={"page_id": "page"}, scope={"user_id": None}
    )
    provider.upsert_vector(
        "notion:a", "two", [1, 0], metadata={}, scope={"user_id": ""}
    )
    provider.upsert_vector(
        "notion:b", "one", [1, 0], metadata={}, scope={"user_id": None}
    )
    for _ in range(2):
        provider.upsert_vector(
            "notion:a",
            "one",
            [0, 1],
            metadata={"page_id": "page"},
            scope={"user_id": None},
        )
    assert len(provider.list_all(MemoryType.KNOWLEDGE_BASE)) == 3
    hits = provider.query_vectors(
        "notion:a", [0, 1], scope={"user_id": None}, include_embedding=True
    )
    assert [hit["source_id"] for hit in hits] == ["one"]
    assert hits[0]["embedding"] == [0, 1]
    assert provider.delete_vector("notion:a", "one")
    assert not provider.delete_vector("notion:a", "one")
    assert len(provider.query_vectors("notion:b", [1, 0])) == 1
    with pytest.raises(ValueError, match="dimensions"):
        provider.query_vectors("notion:b", [1, 0, 0])
    with pytest.raises(ValueError, match="scope"):
        provider.query_vectors("notion:b", [1, 0], scope={"untrusted_sql": "value"})


def test_mongodb_vector_crud_uses_scoped_string_ids_and_native_ranking():
    mongomock = pytest.importorskip("mongomock")
    from memorizz.memory_provider.mongodb import MongoDBProvider

    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.config = SimpleNamespace(read_only=False)
    collection = mongomock.MongoClient().db.knowledge_base
    provider.knowledge_base_collection = collection
    identifier = provider.upsert_vector(
        "ns",
        "source",
        [1, 0],
        metadata={"page_id": "page"},
        scope={"user_id": None, "thread_id": "t"},
    )
    provider.upsert_vector(
        "ns",
        "source",
        [0, 1],
        metadata={"page_id": "page"},
        scope={"user_id": None, "thread_id": "t"},
    )
    assert collection.count_documents({}) == 1
    assert identifier == vector_id("ns", "source")
    row = collection.find_one({"_id": identifier})
    provider._composed_vector_index_ready = True
    spy = MagicMock()
    spy.aggregate.return_value = [{**row, "score": 0.75}]
    provider.knowledge_base_collection = spy
    hits = provider.query_vectors(
        "ns", [0, 1], scope={"user_id": None, "thread_id": "t"}
    )
    predicate = spy.aggregate.call_args.args[0][0]["$vectorSearch"]["filter"]
    assert predicate["namespace"] == "ns"
    assert predicate["metadata.scope_hashes.user_id"] == scope_hash(None)
    assert predicate["metadata.scope_hashes.thread_id"] == scope_hash("t")
    assert hits[0]["source_id"] == "source"
    assert (
        hits[0]["score"] == 0.5
    ), "Atlas normalized scores become portable cosine scores"
    spy.aggregate.side_effect = RuntimeError("vector search unavailable")
    with pytest.raises(RuntimeError):
        provider.query_vectors("ns", [0, 1])
    provider.knowledge_base_collection = collection
    assert provider.delete_vector("ns", "source")
    assert collection.count_documents({}) == 0


def test_filesystem_warm_queries_do_not_reread_every_file_and_writes_invalidate(
    tmp_path, monkeypatch
):
    provider = FileSystemProvider(FileSystemConfig(tmp_path, use_faiss=False))
    for index in range(20):
        provider.upsert_vector("ns", str(index), [1, index], metadata={})
    provider.query_vectors("ns", [1, 0])
    read = MagicMock(wraps=provider._read_document)
    monkeypatch.setattr(provider, "_read_document", read)
    for _ in range(10):
        assert provider.query_vectors("ns", [1, 0], limit=1)[0]["source_id"] == "0"
    assert read.call_count == 0
    provider.delete_vector("ns", "0")
    assert provider.query_vectors("ns", [1, 0], limit=1)[0]["source_id"] == "1"


def test_filesystem_vector_reader_observes_separate_sync_process_writes(tmp_path):
    writer = FileSystemProvider(FileSystemConfig(tmp_path, use_faiss=False))
    writer.upsert_vector("ns", "old", [1, 0], metadata={})
    reader = FileSystemProvider(FileSystemConfig(tmp_path, use_faiss=False))
    assert reader.query_vectors("ns", [1, 0], limit=1)[0]["source_id"] == "old"
    writer.upsert_vector("ns", "old", [0, 1], metadata={})
    writer.upsert_vector("ns", "new", [1, 0], metadata={})
    assert reader.query_vectors("ns", [1, 0], limit=1)[0]["source_id"] == "new"


def test_oracle_native_vector_sql_binds_scope_and_has_no_unscoped_fallback():
    from memorizz.memory_provider.oracle import OracleProvider

    provider = OracleProvider.__new__(OracleProvider)
    provider._get_table_name = lambda _: "TEST.KNOWLEDGE_BASE"
    cursor = MagicMock()
    cursor.rowcount = 1
    connection = MagicMock()
    connection.cursor.return_value = cursor

    @contextmanager
    def connect():
        yield connection

    provider._get_connection = connect
    provider._set_vector_input_size = lambda *args: None
    row = vector_document(
        "ns",
        "source",
        [1, 0],
        {"page_id": "page"},
        {"user_id": "", "thread_id": "t' OR 1=1"},
    )
    cursor.__iter__.return_value = iter([(json.dumps(row["metadata"]), 1.0, [1, 0])])
    hits = provider.query_vectors(
        "ns", [1, 0], scope={"user_id": "", "thread_id": "t' OR 1=1"}
    )
    sql, params = cursor.execute.call_args.args
    assert "t' OR 1=1" not in sql
    assert scope_hash("") in params.values()
    assert scope_hash(None) not in params.values()
    assert "namespace = :namespace" in sql
    assert "FETCH FIRST :limit ROWS ONLY" in sql
    assert hits[0]["source_id"] == "source"
    identifier = provider.upsert_vector(
        "ns", "source", [0, 1], metadata={}, scope={"user_id": None}
    )
    assert identifier == row["id"]
    assert "MERGE INTO" in cursor.execute.call_args.args[0]
    assert provider.delete_vector("ns", "source")
    assert "namespace = :namespace" in cursor.execute.call_args.args[0]
    cursor.execute.side_effect = RuntimeError("missing vector column")
    with pytest.raises(RuntimeError):
        provider.query_vectors("ns", [1, 0], scope={"user_id": "alice"})
