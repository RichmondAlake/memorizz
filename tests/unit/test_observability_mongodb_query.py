import pytest

mongomock = pytest.importorskip("mongomock")

from memorizz.enums import MemoryType  # noqa: E402
from memorizz.memory_provider.mongodb.provider import MongoDBProvider  # noqa: E402


@pytest.mark.unit
def test_mongodb_observability_query_is_filtered_bounded_and_cursor_paginated():
    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.db = mongomock.MongoClient()["trace-query"]
    collection = provider.db[MemoryType.CONVERSATION_MEMORY.value]
    for index in range(6):
        collection.insert_one(
            {
                "agent_id": "agent-a" if index < 5 else "agent-b",
                "memory_id": "memory-a",
                "thread_id": "thread-a" if index % 2 == 0 else "thread-b",
                "timestamp": index,
                "content": str(index),
            }
        )

    first = provider.query_observability_records(
        MemoryType.CONVERSATION_MEMORY,
        agent_ids=["agent-a"],
        limit=2,
    )
    second = provider.query_observability_records(
        MemoryType.CONVERSATION_MEMORY,
        agent_ids=["agent-a"],
        limit=2,
        cursor=first["next_cursor"],
    )

    assert first["provider_native"] is True
    assert first["truncated"] is True
    assert len(first["items"]) == 2
    assert len(second["items"]) == 2
    assert {row["_id"] for row in first["items"]}.isdisjoint(
        {row["_id"] for row in second["items"]}
    )
    assert all(
        row["agent_id"] == "agent-a" for row in [*first["items"], *second["items"]]
    )

    thread_page = provider.query_observability_records(
        MemoryType.CONVERSATION_MEMORY,
        agent_ids=["agent-a"],
        thread_id="thread-a",
        limit=10,
    )
    assert all(row["thread_id"] == "thread-a" for row in thread_page["items"])
    with pytest.raises(ValueError):
        provider.query_observability_records(
            MemoryType.CONVERSATION_MEMORY,
            cursor="not-a-valid-cursor!",
        )

    shared = provider.db[MemoryType.SHARED_MEMORY.value]
    shared.insert_many(
        [
            {
                "record_type": "observability_trace_bundle",
                "agent_id": "agent-a",
                "trace_memory_id": "memory-a",
                "thread_id": "thread-a",
                "timestamp": index,
                "content": "{}",
            }
            for index in range(3)
        ]
        + [
            {
                "record_type": "observability_feedback",
                "agent_id": "agent-a",
                "trace_memory_id": "memory-a",
                "content": "{}",
            }
        ]
    )
    trace_page = provider.query_observability_records(
        MemoryType.SHARED_MEMORY,
        agent_ids=["agent-a"],
        memory_ids=["memory-a"],
        record_type="observability_trace_bundle",
        limit=10,
    )
    assert len(trace_page["items"]) == 3
    assert all(
        row["record_type"] == "observability_trace_bundle"
        for row in trace_page["items"]
    )
