"""Provider-side pre-top-k filtering for passive SHADOW evaluation."""

from types import SimpleNamespace

import pytest

from memorizz.enums import MemoryType


def test_mongodb_shadow_retrieval_pushes_status_and_tenant_filters(monkeypatch):
    from memorizz.memory_provider.mongodb import provider as mongo_module

    provider = mongo_module.MongoDBProvider.__new__(mongo_module.MongoDBProvider)
    provider.skillbox_collection = object()
    provider.lazy_vector_indexes = False
    captured = {}

    monkeypatch.setattr(mongo_module, "get_embedding", lambda _query: [1.0, 0.0])

    def run_search(collection, pipeline, *, label, raise_on_error=False):
        captured.update(
            {
                "collection": collection,
                "pipeline": pipeline,
                "label": label,
                "raise_on_error": raise_on_error,
            }
        )
        return []

    provider._run_vector_search = run_search
    provider.retrieve_skillbox_candidates(
        "refund order",
        limit=3,
        statuses=["shadow"],
        agent_id="agent-a",
        user_id="user-a",
    )

    vector_stage = captured["pipeline"][0]["$vectorSearch"]
    assert vector_stage["limit"] == 3
    assert vector_stage["filter"] == {
        "$and": [
            {"status": {"$in": ["shadow"]}},
            {"agent_id": {"$eq": "agent-a"}},
            {"user_id": {"$eq": "user-a"}},
        ]
    }
    assert captured["label"] == "skillbox"


def test_mongodb_skillbox_index_reconciles_filter_fields():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    class Collection:
        name = "skillbox"

        def __init__(self):
            self.updated = None

        def list_search_indexes(self):
            return [
                {
                    "name": "vector_index",
                    "type": "vectorSearch",
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 2,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "memory_id"},
                        ]
                    },
                }
            ]

        def update_search_index(self, name, definition):
            self.updated = (name, definition)

    collection = Collection()
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider._ensure_vector_index(
        collection,
        memory_store=True,
        filter_fields=["status", "agent_id", "user_id"],
    )
    assert collection.updated is not None
    paths = {
        field["path"]
        for field in collection.updated[1]["fields"]
        if field.get("type") == "filter"
    }
    assert {"memory_id", "status", "agent_id", "user_id"} <= paths


def test_mongodb_lazy_skillbox_index_reconciles_existing_definition():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    class Collection:
        name = "skillbox"

        def __init__(self):
            self.updated = None

        def list_search_indexes(self):
            return [
                {
                    "name": "vector_index",
                    "type": "vectorSearch",
                    "latestDefinition": {
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 2,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "memory_id"},
                        ]
                    },
                }
            ]

        def update_search_index(self, name, definition):
            self.updated = (name, definition)

    collection = Collection()
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider._vector_indexes_created = set()
    provider._ensure_vector_index_for_collection(
        collection,
        MemoryType.SKILLBOX.value,
        memory_store=True,
    )
    assert collection.updated is not None
    paths = {
        field["path"]
        for field in collection.updated[1]["fields"]
        if field.get("type") == "filter"
    }
    assert {"memory_id", "status", "agent_id", "user_id"} <= paths


def test_mongodb_lazy_entity_index_reconciles_user_filter():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    class Collection:
        name = "entity_memory"

        def __init__(self):
            self.updated = None

        def list_search_indexes(self):
            return [
                {
                    "name": "vector_index",
                    "type": "vectorSearch",
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 2,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "memory_id"},
                        ]
                    },
                }
            ]

        def update_search_index(self, name, definition):
            self.updated = (name, definition)

    collection = Collection()
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider._vector_indexes_created = set()
    provider._ensure_vector_index_for_collection(
        collection,
        MemoryType.ENTITY_MEMORY.value,
        memory_store=True,
    )

    assert collection.updated is not None
    paths = {
        field["path"]
        for field in collection.updated[1]["fields"]
        if field.get("type") == "filter"
    }
    assert {"memory_id", "user_id"} <= paths


def test_mongodb_entity_retrieval_prefilters_memory_and_user(monkeypatch):
    from memorizz.memory_provider.mongodb import provider as mongo_module

    provider = mongo_module.MongoDBProvider.__new__(mongo_module.MongoDBProvider)
    provider.entity_memory_collection = object()
    provider.lazy_vector_indexes = False
    captured = {}
    monkeypatch.setattr(mongo_module, "get_embedding", lambda _query: [1.0, 0.0])

    def run_search(collection, pipeline, *, label, raise_on_error=False):
        captured.update(
            {
                "collection": collection,
                "pipeline": pipeline,
                "label": label,
                "raise_on_error": raise_on_error,
            }
        )
        return []

    provider._run_vector_search = run_search
    provider.retrieve_entity_memory_records(
        "user profile",
        limit=3,
        memory_id="primary-user-a",
        user_id="user-a",
    )

    vector_stage = captured["pipeline"][0]["$vectorSearch"]
    assert vector_stage["filter"] == {
        "memory_id": "primary-user-a",
        "user_id": "user-a",
    }
    assert captured["label"] == "entity_memory"
    assert captured["raise_on_error"] is True


def test_mongodb_lazy_entity_retrieval_reconciles_index_before_query(monkeypatch):
    from memorizz.memory_provider.mongodb import provider as mongo_module

    class Collection:
        name = MemoryType.ENTITY_MEMORY.value

    provider = mongo_module.MongoDBProvider.__new__(mongo_module.MongoDBProvider)
    provider.entity_memory_collection = Collection()
    provider.lazy_vector_indexes = True
    provider._vector_indexes_unavailable = set()
    ensured = []
    provider._ensure_vector_index_for_collection = lambda *args, **kwargs: (
        ensured.append((args, kwargs))
    )
    provider._run_vector_search = lambda *_args, **_kwargs: []
    monkeypatch.setattr(mongo_module, "get_embedding", lambda _query: [1.0, 0.0])

    provider.retrieve_entity_memory_records(
        "user profile",
        memory_id="primary-user-a",
        user_id="user-a",
    )

    assert ensured == [
        (
            (provider.entity_memory_collection, MemoryType.ENTITY_MEMORY.value),
            {"memory_store": True},
        )
    ]


def test_mongodb_vector_status_reports_missing_index_and_caches():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    class Collection:
        name = MemoryType.ENTITY_MEMORY.value

        def __init__(self):
            self.calls = 0

        def list_search_indexes(self):
            self.calls += 1
            return []

    collection = Collection()
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.db = {MemoryType.ENTITY_MEMORY.value: collection}
    provider._vector_indexes_unavailable = set()
    provider._vector_index_unavailable_root_causes = set()
    provider._vector_search_status_cache = {}

    first = provider.get_vector_search_status(MemoryType.ENTITY_MEMORY)
    second = provider.get_vector_search_status(MemoryType.ENTITY_MEMORY)

    assert (
        first
        == second
        == {
            "available": False,
            "queryable": False,
            "reason": "vector_index_missing",
        }
    )
    assert collection.calls == 1


def test_mongodb_lazy_missing_index_is_eligible_for_on_demand_creation():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    class Collection:
        name = MemoryType.ENTITY_MEMORY.value

        def list_search_indexes(self):
            return []

    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.db = {MemoryType.ENTITY_MEMORY.value: Collection()}
    provider.config = SimpleNamespace(read_only=False)
    provider.lazy_vector_indexes = True
    provider._vector_indexes_unavailable = set()
    provider._vector_index_unavailable_root_causes = set()
    provider._vector_search_status_cache = {}

    status = provider.get_vector_search_status(MemoryType.ENTITY_MEMORY)

    assert status == {
        "available": None,
        "queryable": False,
        "reason": "vector_index_pending_lazy_creation",
    }


def test_mongodb_failed_filter_reconciliation_remains_retryable():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    class Collection:
        name = MemoryType.ENTITY_MEMORY.value

        def list_search_indexes(self):
            return [
                {
                    "name": "vector_index",
                    "type": "vectorSearch",
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 2,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "memory_id"},
                        ]
                    },
                }
            ]

        def update_search_index(self, _name, _definition):
            raise RuntimeError("reconciliation denied")

    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider._vector_indexes_created = set()
    provider._vector_indexes_unavailable = set()
    provider._vector_index_unavailable_root_causes = set()
    provider._vector_search_status_cache = {}

    provider._ensure_vector_index_for_collection(
        Collection(), MemoryType.ENTITY_MEMORY.value, memory_store=True
    )

    assert (
        f"{MemoryType.ENTITY_MEMORY.value}_vector_index"
        not in provider._vector_indexes_created
    )


def test_mongodb_entity_vector_failure_is_typed_and_marks_diagnostics_state():
    from memorizz.memory_provider.mongodb.provider import (
        MongoDBProvider,
        VectorSearchExecutionError,
    )

    class Collection:
        name = MemoryType.ENTITY_MEMORY.value

        def aggregate(self, _pipeline):
            raise RuntimeError("query failed")

    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider._vector_indexes_unavailable = set()
    provider._vector_index_unavailable_root_causes = set()
    provider._vector_search_status_cache = {}

    with pytest.raises(VectorSearchExecutionError):
        provider._run_vector_search(
            Collection(), [], label="entity_memory", raise_on_error=True
        )

    status = provider._vector_search_status_cache[MemoryType.ENTITY_MEMORY.value][1]
    assert status == {"available": None, "reason": "vector_query_failed"}


def test_mongodb_legacy_entity_migration_is_explicitly_scoped():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    class Result:
        modified_count = 2

    class Collection:
        def __init__(self):
            self.call = None

        def update_many(self, query, update):
            self.call = (query, update)
            return Result()

    collection = Collection()
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.entity_memory_collection = collection

    migrated = provider.migrate_entity_memory_user_scope(
        memory_id="shared-memory", user_id="user-a"
    )

    assert migrated == 2
    assert collection.call == (
        {"memory_id": "shared-memory", "user_id": {"$in": [None]}},
        {"$set": {"user_id": "user-a"}},
    )


def test_mongodb_vector_status_treats_unsupported_search_as_unavailable():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    class UnsupportedSearch(Exception):
        code = 59
        details = {"codeName": "CommandNotFound"}

    class Collection:
        name = MemoryType.ENTITY_MEMORY.value

        def list_search_indexes(self):
            raise UnsupportedSearch("no such command: 'listSearchIndexes'")

    collection = Collection()
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.db = {MemoryType.ENTITY_MEMORY.value: collection}
    provider._vector_indexes_unavailable = set()
    provider._vector_index_unavailable_root_causes = set()
    provider._vector_search_status_cache = {}

    status = provider.get_vector_search_status(MemoryType.ENTITY_MEMORY)

    assert status == {
        "available": False,
        "queryable": False,
        "reason": "vector_search_unsupported",
    }
    assert MemoryType.ENTITY_MEMORY.value in provider._vector_indexes_unavailable


def test_mongodb_vector_status_recognizes_community_server_error():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    class CommunitySearchError(Exception):
        code = 6047401
        details = {"codeName": "Location6047401"}

    class Collection:
        name = MemoryType.ENTITY_MEMORY.value

        def list_search_indexes(self):
            raise CommunitySearchError(
                "$listSearchIndexes stage is only allowed on MongoDB Atlas"
            )

    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.db = {MemoryType.ENTITY_MEMORY.value: Collection()}
    provider._vector_indexes_unavailable = set()
    provider._vector_index_unavailable_root_causes = set()
    provider._vector_search_status_cache = {}

    status = provider.get_vector_search_status(MemoryType.ENTITY_MEMORY)

    assert status["available"] is False
    assert status["reason"] == "vector_search_unsupported"


def test_oracle_shadow_retrieval_pushes_status_and_tenant_filters(monkeypatch):
    import memorizz.embeddings as embedding_module
    from memorizz.memory_provider.oracle.provider import OracleProvider

    provider = OracleProvider.__new__(OracleProvider)
    captured = {}
    monkeypatch.setattr(embedding_module, "get_embedding", lambda _query: [1.0, 0.0])

    def vector_search(memory_type, embedding, **kwargs):
        captured.update(
            {
                "memory_type": memory_type,
                "embedding": embedding,
                **kwargs,
            }
        )
        return []

    provider._vector_search = vector_search
    provider.retrieve_skillbox_candidates(
        "refund order",
        limit=3,
        statuses=["shadow"],
        agent_id="agent-a",
        user_id="user-a",
    )

    assert captured == {
        "memory_type": MemoryType.SKILLBOX,
        "embedding": [1.0, 0.0],
        "limit": 3,
        "filters": {"status": "shadow", "agent_id": "agent-a"},
        "user_id": "user-a",
    }
