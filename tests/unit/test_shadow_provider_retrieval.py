"""Provider-side pre-top-k filtering for passive SHADOW evaluation."""

from memorizz.enums import MemoryType


def test_mongodb_shadow_retrieval_pushes_status_and_tenant_filters(monkeypatch):
    from memorizz.memory_provider.mongodb import provider as mongo_module

    provider = mongo_module.MongoDBProvider.__new__(mongo_module.MongoDBProvider)
    provider.skillbox_collection = object()
    provider.lazy_vector_indexes = False
    captured = {}

    monkeypatch.setattr(mongo_module, "get_embedding", lambda _query: [1.0, 0.0])

    def run_search(collection, pipeline, *, label):
        captured.update(
            {"collection": collection, "pipeline": pipeline, "label": label}
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
