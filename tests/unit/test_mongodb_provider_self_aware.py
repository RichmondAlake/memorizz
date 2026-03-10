"""Unit tests for MongoDB provider self-aware field mapping."""

from bson import ObjectId

from memorizz.memory_provider.mongodb.provider import MongoDBProvider


class _FakeMemagentCollection:
    def __init__(self, doc):
        self._doc = doc

    def find_one(self, _query, _projection):
        return self._doc

    def find(self, _query, _projection):
        return [self._doc]


def test_retrieve_memagent_maps_self_aware_fields():
    doc_id = ObjectId()
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.memagent_collection = _FakeMemagentCollection(
        {
            "_id": doc_id,
            "instruction": "mongo",
            "application_mode": "assistant",
            "max_steps": 20,
            "memory_ids": ["mem-1"],
            "self_aware": True,
            "self_aware_config": {
                "root_paths": ["."],
                "allow_writes": True,
                "allow_deletes": False,
            },
        }
    )

    loaded = provider.retrieve_memagent(str(doc_id))
    assert loaded is not None
    assert loaded.self_aware is True
    assert loaded.self_aware_config["allow_writes"] is True


def test_list_memagents_maps_self_aware_fields():
    doc_id = ObjectId()
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.memagent_collection = _FakeMemagentCollection(
        {
            "_id": doc_id,
            "instruction": "mongo",
            "application_mode": "assistant",
            "max_steps": 20,
            "memory_ids": ["mem-1"],
            "self_aware": True,
            "self_aware_config": {
                "root_paths": ["."],
                "allow_writes": False,
                "allow_deletes": False,
            },
        }
    )

    agents = provider.list_memagents()
    assert len(agents) == 1
    assert agents[0].self_aware is True
    assert agents[0].self_aware_config["allow_writes"] is False
