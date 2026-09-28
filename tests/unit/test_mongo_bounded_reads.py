# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Console pages read production MongoDB stores in bounded pages."""

import pytest

from memorizz.enums.memory_type import MemoryType

mongomock = pytest.importorskip("mongomock")


@pytest.fixture()
def provider():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    p = MongoDBProvider.__new__(MongoDBProvider)
    p.db = mongomock.MongoClient()["memorizz-bounded"]
    collection = p.db[MemoryType.CONVERSATION_MEMORY.value]
    collection.insert_many(
        [{"content": f"turn {i}", "embedding": [0.1] * 32} for i in range(250)]
    )
    return p


@pytest.mark.unit
def test_list_recent_returns_newest_page_without_embeddings(provider):
    rows = provider.list_recent(MemoryType.CONVERSATION_MEMORY, 101)
    assert len(rows) == 101
    assert rows[0]["content"] == "turn 249", "newest first by insertion"
    assert "embedding" not in rows[0]


@pytest.mark.unit
def test_list_recent_orders_by_timestamp_when_id_types_are_mixed(provider):
    shared = provider.db[MemoryType.SHARED_MEMORY.value]
    shared.insert_one({"_id": "obs-trace-new", "timestamp": "2026-09-28T01:00:00Z"})
    shared.insert_one({"timestamp": "2026-09-05T01:00:00Z"})  # ObjectId, older
    rows = provider.list_recent(MemoryType.SHARED_MEMORY, 2)
    assert rows[0]["_id"] == "obs-trace-new"


@pytest.mark.unit
def test_estimate_count_reads_collection_size(provider):
    assert provider.estimate_count(MemoryType.CONVERSATION_MEMORY) == 250
    assert provider.estimate_count(MemoryType.PERSONAS) == 0


@pytest.mark.unit
def test_read_only_proxy_allows_bounded_reads(provider):
    from memorizz.ui.security import ReadOnlyProviderProxy

    proxy = ReadOnlyProviderProxy(provider)
    assert len(proxy.list_recent(MemoryType.CONVERSATION_MEMORY, 5)) == 5
    assert proxy.estimate_count(MemoryType.CONVERSATION_MEMORY) == 250


@pytest.mark.unit
def test_observability_paging_crosses_objectid_and_string_ids(provider):
    from bson import ObjectId

    shared = provider.db[MemoryType.SHARED_MEMORY.value]
    for i in range(3):
        shared.insert_one({"_id": ObjectId(), "record_type": "bundle", "n": f"o{i}"})
    for i in range(4):
        shared.insert_one(
            {"_id": f"obs-trace-{i}", "record_type": "bundle", "n": f"s{i}"}
        )
    seen, cursor = [], None
    while True:
        page = provider.query_observability_records(
            MemoryType.SHARED_MEMORY, limit=2, cursor=cursor, record_type="bundle"
        )
        seen += [row["n"] for row in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert sorted(seen) == ["o0", "o1", "o2", "s0", "s1", "s2", "s3"]


@pytest.mark.unit
def test_legacy_untagged_cursors_still_decode(provider):
    import base64

    from bson import ObjectId

    oid = ObjectId()
    legacy = base64.urlsafe_b64encode(str(oid).encode()).decode().rstrip("=")
    assert provider._decode_observability_cursor(legacy) == oid
    tagged = provider._encode_observability_cursor("0123456789abcdef01234567")
    assert provider._decode_observability_cursor(tagged) == "0123456789abcdef01234567"


@pytest.mark.unit
def test_agent_saves_upsert_one_record_and_index_blocks_copies(provider, caplog):
    from pymongo.errors import DuplicateKeyError

    provider.memagent_collection = provider.db["agents"]
    provider._sync_agent_tools_to_toolbox = lambda *args, **kwargs: None
    provider._prepare_memagent_payload = lambda agent: (dict(agent), agent["agent_id"])
    provider._ensure_unique_agent_ids()

    provider.store_memagent({"agent_id": "assistant", "llm_config": {"model": "a"}})
    stored = provider.store_memagent(
        {"agent_id": "assistant", "llm_config": {"model": "b"}}
    )
    rows = list(provider.memagent_collection.find({"agent_id": "assistant"}))
    assert len(rows) == 1 and rows[0]["llm_config"]["model"] == "b"
    assert stored["_id"] == rows[0]["_id"]
    with pytest.raises(DuplicateKeyError):
        provider.memagent_collection.insert_one({"agent_id": "assistant"})


@pytest.mark.unit
def test_unique_agent_index_warns_while_duplicates_exist(provider, caplog):
    provider.memagent_collection = provider.db["agents_dupes"]
    provider.memagent_collection.insert_many([{"agent_id": "x"}, {"agent_id": "x"}])
    with caplog.at_level("WARNING"):
        provider._ensure_unique_agent_ids()
    assert "Remove duplicate agent records" in caplog.text
