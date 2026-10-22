# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""delete_scope(keep_traces=...) on the filesystem (portable), MongoDB
(mongomock) and Oracle (fake cursor) providers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.memory_provider.base import delete_scope_matches, is_trace_record


class _Model:
    def get_config(self):
        return {"model": "gpt-4o-mini", "provider": "openai"}

    def get_last_usage(self):
        return {"prompt_tokens": 100, "completion_tokens": 10}

    def generate(self, messages, tools=None):
        return "answer"


@pytest.fixture()
def provider(tmp_path):
    instance = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield instance
    instance.close()


def _evaluation_run(provider, memory_id, *, user_id="grader", name="eval"):
    """One evaluation run: an agent, a tagged turn, and a failing verdict."""
    agent = MemAgent(
        model=_Model(),
        memory_provider=provider,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
        name=name,
        context_policy={"progressive_tool_disclosure": False},
    )
    agent._schedule_conversation_embedding_backfill = lambda rows: None
    agent.run(
        "question",
        memory_id=memory_id,
        thread_id=f"thread-{memory_id}",
        user_id=user_id,
        trace_tags={"experiment": "lme-v2"},
    )
    agent.save()
    agent.record_task_outcome(
        turn_id=agent.get_trace_context()["turn_id"],
        verdict="fail",
        grader="llm-judge",
        reason_code="wrong_answer",
    )
    return agent


def _record_types(provider):
    rows = provider.list_all(MemoryType.SHARED_MEMORY)
    return sorted(row.get("record_type") or "" for row in rows)


@pytest.mark.unit
def test_delete_scope_requires_a_scope(provider):
    with pytest.raises(ValueError, match="requires"):
        provider.delete_scope()
    with pytest.raises(ValueError, match="requires"):
        provider.delete_scope(agent_ids=["  "], keep_traces=True)


@pytest.mark.unit
def test_keep_traces_deletes_memories_and_keeps_the_run_on_traces(provider, tmp_path):
    agent = _evaluation_run(provider, "run-1")
    other = _evaluation_run(provider, "run-2", name="other")
    assert _record_types(provider) == [
        "observability_outcome",
        "observability_outcome",
        "observability_trace_bundle",
        "observability_trace_bundle",
    ]

    result = provider.delete_scope(
        memory_id="run-1",
        user_id="grader",
        agent_ids=[agent.agent_id],
        keep_traces=True,
    )

    assert result["ok"] is True
    assert result["scope"]["keep_traces"] is True
    assert result["counts"]["conversation_memory"] == 2
    assert result["counts"]["shared_memory"] == 0
    assert result["counts"]["agents"] == 1
    assert result["total_deleted"] == sum(result["counts"].values())
    conversations = provider.list_all(MemoryType.CONVERSATION_MEMORY)
    assert {row["memory_id"] for row in conversations} == {"run-2"}
    assert provider.retrieve_memagent(agent.agent_id) is None
    assert provider.retrieve_memagent(other.agent_id) is not None
    # Both runs' trace bundles and verdicts are still there.
    assert len(_record_types(provider)) == 4

    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    ), patch.dict(
        "os.environ",
        {
            "MEMORIZZ_UI_AUTH_TOKEN": "",
            "MEMORIZZ_UI_AUTH_ACCOUNTS": "{}",
            "MEMORIZZ_UI_AUDIT_LOG": str(tmp_path / "audit.jsonl"),
        },
    ):
        client = TestClient(create_app(), follow_redirects=False)
        # The deleted agent's run still opens from its link.
        page = client.get(agent.trace_url("http://testserver"))
        assert page.status_code == 200, page.text
        assert ">Turn Result<" in page.text and ">Model Call<" in page.text
        assert "Verdict: fail · llm-judge" in page.text
        listed = client.get("/traces", params={"tag": "experiment=lme-v2"})
        assert "thread-run-1" in listed.text and "thread-run-2" in listed.text


@pytest.mark.unit
def test_without_keep_traces_the_scope_trace_records_go_too(provider):
    agent = _evaluation_run(provider, "run-1")
    _evaluation_run(provider, "run-2", name="other")

    result = provider.delete_scope(
        memory_id="run-1", user_id="grader", agent_ids=[agent.agent_id]
    )

    assert result["scope"]["keep_traces"] is False
    assert result["counts"]["shared_memory"] == 2  # its bundle and verdict
    remaining = provider.list_all(MemoryType.SHARED_MEMORY)
    assert {row.get("trace_memory_id") for row in remaining} == {"run-2"}


@pytest.mark.unit
def test_memory_scope_alone_reaches_traces_and_agent_memory_lists(provider):
    agent = _evaluation_run(provider, "run-1")
    provider.update_memagent_memory_ids(agent.agent_id, ["run-1", "keep"])

    result = provider.delete_scope(memory_id="run-1", keep_traces=True)

    assert result["counts"]["agent_memories"] == 1
    assert provider.retrieve_memagent(agent.agent_id).memory_ids == ["keep"]
    assert len(_record_types(provider)) == 2
    assert provider.list_all(MemoryType.CONVERSATION_MEMORY) == []

    provider.delete_scope(memory_id="run-1")
    assert _record_types(provider) == []


@pytest.mark.unit
def test_keep_traces_spares_legacy_bundles_in_conversation_memory(provider):
    legacy = {
        "_id": "legacy-bundle",
        "role": "tool",
        "memory_id": "m1",
        "content": json.dumps({"type": "trace_bundle", "events": []}),
    }
    message = {"_id": "message", "role": "user", "memory_id": "m1", "content": "hi"}
    for row in (legacy, message):
        provider.store(dict(row), MemoryType.CONVERSATION_MEMORY)
    assert is_trace_record(legacy, MemoryType.CONVERSATION_MEMORY)
    assert not is_trace_record(message, MemoryType.CONVERSATION_MEMORY)

    provider.delete_scope(memory_id="m1", keep_traces=True)
    rows = provider.list_all(MemoryType.CONVERSATION_MEMORY)
    assert [row["_id"] for row in rows] == ["legacy-bundle"]


@pytest.mark.unit
def test_scope_matching_rules():
    trace = {
        "memory_id": "obs-trace-1",
        "record_type": "observability_trace_bundle",
        "trace_memory_id": "m1",
        "user_id": "u1",
        "agent_id": "a1",
        "owner_agent_id": "a1",
    }
    shared = {"memory_id": "board", "owner_agent_id": "a1"}
    conversation = {"memory_id": "m1", "user_id": None, "agent_id": "a1"}
    shared_type = MemoryType.SHARED_MEMORY
    # A trace record is scoped by the turn it traced, not its own id.
    assert delete_scope_matches(trace, shared_type, memory_id="m1")
    assert not delete_scope_matches(trace, shared_type, memory_id="obs-trace-1")
    assert delete_scope_matches(trace, shared_type, user_id="u1", agent_ids=["a1"])
    assert not delete_scope_matches(trace, shared_type, user_id="u2")
    # Ordinary shared memory has no user field: a user-only scope skips it.
    assert not delete_scope_matches(shared, shared_type, user_id="u1")
    assert delete_scope_matches(shared, shared_type, user_id="u1", agent_ids=["a1"])
    # user_id=None selects rows without a user.
    assert delete_scope_matches(
        conversation, MemoryType.CONVERSATION_MEMORY, user_id=None
    )
    assert not delete_scope_matches(
        conversation, MemoryType.CONVERSATION_MEMORY, memory_id="m1", agent_ids=["a2"]
    )


@pytest.mark.unit
def test_agent_lifecycle_cleanup_scope_forwards_keep_traces(provider):
    agent = _evaluation_run(provider, "run-1")
    report = agent.close(
        cleanup_scope={
            "memory_id": "run-1",
            "agent_ids": [agent.agent_id],
            "keep_traces": True,
        },
        close_memory_provider=False,
    )
    assert report["ok"] is True, report
    assert report["cleanup"]["scope"]["keep_traces"] is True
    assert len(_record_types(provider)) == 2


# --- MongoDB (mongomock) ------------------------------------------------------


@pytest.fixture()
def mongo():
    mongomock = pytest.importorskip("mongomock")
    pytest.importorskip("pymongo")
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    instance = MongoDBProvider.__new__(MongoDBProvider)
    instance.db = mongomock.MongoClient()["delete-scope-test"]
    for memory_type in MemoryType:
        attribute = (
            "persona_collection"
            if memory_type == MemoryType.PERSONAS
            else "memagent_collection"
            if memory_type == MemoryType.MEMAGENT
            else f"{memory_type.value}_collection"
        )
        setattr(instance, attribute, instance.db[memory_type.value])
    instance._vector_indexes_unavailable = set()
    instance._vector_index_unavailable_root_causes = set()
    instance._vector_search_status_cache = {}
    return instance


def _seed_mongo(mongo):
    from memorizz.observability import ObservabilityStore

    store = ObservabilityStore(mongo)
    for memory_id, agent_id in (("m1", "a1"), ("m2", "a2")):
        context = {
            "agent_id": agent_id,
            "memory_id": memory_id,
            "thread_id": f"t-{memory_id}",
            "user_id": "u1",
            "turn_id": f"turn-{memory_id}",
            "root_trace_id": f"root-{memory_id}",
        }
        store.record_trace_bundle(
            trace_context=context,
            events=[{"trace_kind": "turn_result", "status": "success"}],
        )
        store.record_outcome(trace_context=context, status="failure", verified=True)
        mongo.conversation_memory_collection.insert_many(
            [
                {
                    "memory_id": memory_id,
                    "user_id": "u1",
                    "agent_id": agent_id,
                    "role": "user",
                    "content": "q",
                },
                {
                    "memory_id": memory_id,
                    "user_id": "u1",
                    "agent_id": agent_id,
                    "role": "tool",
                    "content": json.dumps({"type": "trace_bundle", "events": []}),
                },
            ]
        )
        mongo.tool_log_collection.insert_one(
            {
                "memory_id": memory_id,
                "user_id": "u1",
                "agent_id": agent_id,
                "tool_name": "x",
            }
        )
        mongo.memagent_collection.insert_one(
            {"agent_id": agent_id, "memory_ids": [memory_id, "shared"]}
        )
    mongo.shared_memory_collection.insert_one(
        {"memory_id": "blackboard", "owner_agent_id": "a1", "content": "{}"}
    )


@pytest.mark.unit
def test_mongodb_keep_traces_keeps_observability_and_legacy_bundles(mongo):
    _seed_mongo(mongo)
    with patch(
        "memorizz.automation.store.factory.get_automation_store", return_value=None
    ):
        result = mongo.delete_scope(
            memory_id="m1", user_id="u1", agent_ids=["a1"], keep_traces=True
        )
    counts = result["counts"]
    assert counts["conversation_memory"] == 1  # the message, not the legacy bundle
    assert counts["tool_log"] == 1
    # Trace records stay; the blackboard's memory_id is its own, not m1.
    assert counts["shared_memory"] == 0
    assert counts["agents"] == 1 and counts["agent_memories"] == 1
    kinds = sorted(
        doc.get("record_type") or "" for doc in mongo.shared_memory_collection.find()
    )
    assert (
        kinds
        == [""] + ["observability_outcome"] * 2 + ["observability_trace_bundle"] * 2
    )
    assert (
        mongo.conversation_memory_collection.count_documents({"memory_id": "m1"}) == 1
    )


@pytest.mark.unit
def test_mongodb_without_keep_traces_deletes_the_scope_traces(mongo):
    _seed_mongo(mongo)
    with patch(
        "memorizz.automation.store.factory.get_automation_store", return_value=None
    ):
        result = mongo.delete_scope(memory_id="m1", user_id="u1", agent_ids=["a1"])
        assert result["counts"]["shared_memory"] == 2
        assert result["counts"]["conversation_memory"] == 2
        traced = {
            doc.get("trace_memory_id")
            for doc in mongo.shared_memory_collection.find(
                {"record_type": {"$regex": "^observability_"}}
            )
        }
        assert traced == {"m2"}
        # An agent scope alone also reaches ordinary shared memory it owns.
        result = mongo.delete_scope(agent_ids=["a1"])
        assert result["counts"]["shared_memory"] == 1
        assert (
            mongo.shared_memory_collection.count_documents({"memory_id": "blackboard"})
            == 0
        )


@pytest.mark.unit
def test_mongodb_filters_by_collection():
    pytest.importorskip("pymongo")
    from memorizz.memory_provider.base import _UNSET
    from memorizz.memory_provider.mongodb.provider import _delete_scope_filter

    assert _delete_scope_filter("personas", None, "u1", [], False) is None
    assert _delete_scope_filter("skillbox", "m1", _UNSET, [], False) is None
    assert _delete_scope_filter("tool_log", "m1", None, ["a"], False) == {
        "$and": [
            {"memory_id": "m1"},
            {"user_id": {"$in": [None]}},
            {"agent_id": {"$in": ["a"]}},
        ]
    }
    kept = _delete_scope_filter("shared_memory", None, "u1", [], True)
    assert kept is None, "a user scope never reaches ordinary shared memory"


# --- Oracle (fake cursor) -----------------------------------------------------


class _Cursor:
    def __init__(self):
        self.calls = []
        self.rowcount = 1

    def execute(self, sql, params=None):
        self.calls.append((" ".join(str(sql).split()), dict(params or {})))

    def fetchone(self):
        return (1,)


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _oracle(cursor):
    from memorizz.memory_provider.oracle.provider import OracleProvider

    instance = OracleProvider.__new__(OracleProvider)
    instance.config = SimpleNamespace(schema="MEMORIZZ", user="APP_USER")
    instance._table_columns_cache = {}
    connection = _Connection(cursor)
    instance._get_connection = lambda: connection
    return instance


def _delete_sql(cursor, table):
    return [
        (sql, params)
        for sql, params in cursor.calls
        if sql.startswith(f"DELETE FROM MEMORIZZ.{table.upper()} ")
        or sql.startswith(f"DELETE FROM MEMORIZZ.{table} ")
    ]


@pytest.mark.unit
def test_oracle_keep_traces_leaves_observability_records_alone():
    pytest.importorskip("oracledb")
    cursor = _Cursor()
    provider = _oracle(cursor)
    result = provider.delete_scope(
        memory_id="m1", user_id="u1", agent_ids=["a1"], keep_traces=True
    )
    assert result["scope"]["keep_traces"] is True
    ((shared, shared_params),) = _delete_sql(cursor, "shared_memory")
    assert "NOT LIKE 'observability\\_%' ESCAPE '\\'" in shared
    assert ") OR (" not in shared, "no branch reaches trace records"
    assert set(shared_params) == {"scope_memory_id", "scope_agent_0"}
    ((conversation, conversation_params),) = _delete_sql(cursor, "conversation_memory")
    assert "role <> 'tool' OR COALESCE(JSON_VALUE(content, '$.type'" in conversation
    assert set(conversation_params) == {
        "scope_memory_id",
        "scope_user_id",
        "scope_agent_0",
    }


@pytest.mark.unit
def test_oracle_default_scopes_trace_records_by_their_json():
    pytest.importorskip("oracledb")
    cursor = _Cursor()
    provider = _oracle(cursor)
    provider.delete_scope(memory_id="m1", user_id="u1", agent_ids=["a1"])
    ((shared, params),) = _delete_sql(cursor, "shared_memory")
    where = shared.split("WHERE", 1)[1]
    ordinary, traced = where.split(") OR (")
    assert "memory_id = :scope_memory_id" in ordinary
    assert "owner_agent_id IN (:scope_agent_0)" in ordinary
    assert "LIKE 'observability\\_%' ESCAPE '\\'" in traced
    assert "COALESCE(JSON_VALUE(content, '$.trace_memory_id'" in traced
    assert (
        "JSON_VALUE(content, '$.user_id' RETURNING VARCHAR2(512) NULL ON ERROR) = :scope_user_id"
        in traced
    )
    assert "COALESCE(JSON_VALUE(content, '$.agent_id'" in traced
    assert set(params) == {"scope_memory_id", "scope_user_id", "scope_agent_0"}
    ((conversation, _params),) = _delete_sql(cursor, "conversation_memory")
    assert "trace_bundle" not in conversation

    # A user-only scope reaches trace records but not ordinary shared memory.
    cursor.calls.clear()
    provider.delete_scope(user_id=None)
    ((shared, params),) = _delete_sql(cursor, "shared_memory")
    assert "IS NULL" in shared and "owner_agent_id IN" not in shared
    assert params == {}
    cursor.calls.clear()
    provider.delete_scope(user_id="u1", keep_traces=True)
    assert _delete_sql(cursor, "shared_memory") == []
