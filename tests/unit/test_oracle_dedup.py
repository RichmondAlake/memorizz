"""Pinned SQL / row-mapping behaviour for the Oracle provider de-duplication.

Every test here runs against fake connections (no database). The SQL text,
bind values and output dicts were captured from the provider *before* the
persona/toolbox store methods were folded into the ``_upsert_*_row``
helpers and the hand-written row mappings were routed through the
``_apply_row_fields`` registry, so they guard the refactor.
"""

from __future__ import annotations

import array
import json
import uuid
from datetime import datetime

import oracledb
import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.procedural.toolbox.toolbox import Toolbox
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider.oracle.provider import OracleProvider
from tests.unit.test_oracle_provider_fixes import _Cursor, _Lob, _provider


class _RecordingCursor(_Cursor):
    """``_Cursor`` that also records ``setinputsizes`` calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_sizes = []

    def setinputsizes(self, **kwargs):
        self.input_sizes.append(kwargs)


def _columns_of(insert_sql: str) -> list:
    return insert_sql[insert_sql.index("(") + 1 : insert_sql.index(")")].split(", ")


def _binds_without_id(params: dict) -> dict:
    assert isinstance(params["id"], bytes) and len(params["id"]) == 16
    return {key: value for key, value in params.items() if key != "id"}


# ---------------------------------------------------------------------------
# 1. _store_persona / _store_toolbox vs the _upsert_*_row helpers
# ---------------------------------------------------------------------------

_PERSONA_SELECT = "SELECT id FROM MEMORIZZ.personas WHERE persona_id = :persona_id"
_PERSONA_COLUMNS = [
    "id",
    "persona_id",
    "name",
    "role_type",
    "background",
    "memory_id",
    "agent_id",
    "traits",
    "expertise",
]
_TOOLBOX_SELECT = "SELECT id FROM MEMORIZZ.toolbox WHERE tool_id = :tool_id"
_TOOLBOX_COLUMNS = [
    "id",
    "tool_id",
    "name",
    "description",
    "signature",
    "docstring",
    "tool_type",
    "parameters",
    "input_schema",
    "tool_policy",
    "aliases",
    "deprecated_arguments",
    "queries",
    "import_reference",
    "user_id",
    "memory_id",
    "agent_id",
]


@pytest.mark.unit
def test_store_persona_insert_sql_and_binds_pinned():
    cursor = _Cursor()
    provider, connection = _provider(cursor)
    embedding = [0.1, 0.2]
    provider._generate_embedding_if_needed = lambda *_a, **_k: embedding

    persona_id = provider._store_persona(
        {
            "persona_id": "persona-1",
            "name": "Researcher",
            "role": "general",
            "background": "Keeps provenance",
            "goals": "Cite sources",
            "traits": ["careful"],
            "expertise": {"domain": "ml"},
            "memory_id": "memory-1",
            "agent_id": "agent-1",
        }
    )

    assert persona_id == "persona-1"
    assert cursor.calls[0] == (_PERSONA_SELECT, {"persona_id": "persona-1"})
    insert_sql, params = cursor.calls[1]
    columns = _PERSONA_COLUMNS + ["embedding"]
    assert insert_sql == (
        f"INSERT INTO MEMORIZZ.personas ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + c for c in columns)})"
    )
    assert _binds_without_id(params) == {
        "persona_id": "persona-1",
        "name": "Researcher",
        "role_type": "general",
        "background": "Keeps provenance",
        "memory_id": "memory-1",
        "agent_id": "agent-1",
        "traits": '["careful"]',
        "expertise": '{"domain": "ml"}',
        "embedding": array.array("f", embedding),
    }
    assert len(cursor.calls) == 2
    assert connection.commits == 1


@pytest.mark.unit
def test_store_persona_update_sql_and_binds_pinned():
    cursor = _Cursor(routes=[(_PERSONA_SELECT, (uuid.uuid4().bytes,))])
    provider, connection = _provider(cursor)

    class _Role:
        value = "assistant"

    persona_id = provider._store_persona(
        {"name": "Planner", "role_type": _Role(), "agentId": "agent-2"}
    )

    assert persona_id == "Planner"  # falls back to the name
    assert cursor.calls[0] == (_PERSONA_SELECT, {"persona_id": "Planner"})
    assert cursor.calls[1] == (
        "UPDATE MEMORIZZ.personas SET name = :name, role_type = :role_type, "
        "background = :background, memory_id = :memory_id, agent_id = :agent_id, "
        "traits = :traits, expertise = :expertise, updated_at = CURRENT_TIMESTAMP "
        "WHERE persona_id = :persona_id",
        {
            "persona_id": "Planner",
            "name": "Planner",
            "role_type": "assistant",
            "background": "",
            "memory_id": None,
            "agent_id": "agent-2",
            "traits": None,
            "expertise": None,
        },
    )
    assert connection.commits == 1


@pytest.mark.unit
def test_store_toolbox_insert_sql_and_binds_pinned():
    cursor = _Cursor()
    provider, connection = _provider(cursor)
    embedding = [0.3, 0.4]
    provider._generate_embedding_if_needed = lambda *_a, **_k: embedding
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
        "additionalProperties": False,
    }
    tool_uuid = uuid.uuid4()

    tool_id = provider._store_toolbox(
        {
            "_id": tool_uuid,
            "name": "lookup",
            "description": "Look up",
            "signature": "(q)",
            "docstring": "Look up a thing",
            "parameters": schema["properties"],
            "required": ["q"],
            "input_schema": schema,
            "tool_policy": {"side_effects": False},
            "aliases": ["find"],
            "deprecated_arguments": {"query": "q"},
            "queries": ["find things"],
            "import_reference": "tools:lookup",
            "user_id": "alice",
            "memory_id": "memory-1",
            "agent_id": "agent-1",
        }
    )

    assert tool_id == str(tool_uuid)
    assert cursor.calls[0] == (_TOOLBOX_SELECT, {"tool_id": str(tool_uuid)})
    insert_sql, params = cursor.calls[1]
    columns = _TOOLBOX_COLUMNS + ["embedding"]
    assert insert_sql == (
        f"INSERT INTO MEMORIZZ.toolbox ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + c for c in columns)})"
    )
    assert _binds_without_id(params) == {
        "tool_id": str(tool_uuid),
        "name": "lookup",
        "description": "Look up",
        "signature": "(q)",
        "docstring": "Look up a thing",
        "tool_type": "function",
        "parameters": json.dumps(schema["properties"]),
        "input_schema": json.dumps(schema),
        "tool_policy": '{"side_effects": false}',
        "aliases": '["find"]',
        "deprecated_arguments": '{"query": "q"}',
        "queries": '["find things"]',
        "import_reference": "tools:lookup",
        "user_id": "alice",
        "memory_id": "memory-1",
        "agent_id": "agent-1",
        "embedding": array.array("f", embedding),
    }
    assert len(cursor.calls) == 2
    assert connection.commits == 1


@pytest.mark.unit
def test_store_toolbox_update_sql_and_binds_pinned():
    cursor = _Cursor(routes=[(_TOOLBOX_SELECT, (uuid.uuid4().bytes,))])
    provider, connection = _provider(cursor)

    tool_id = provider._store_toolbox(
        {"name": "lookup", "type": "mcp", "parameters": {"q": {"type": "string"}}}
    )

    assert tool_id == "lookup"
    assert cursor.calls[1] == (
        "UPDATE MEMORIZZ.toolbox SET name = :name, description = :description, "
        "signature = :signature, docstring = :docstring, tool_type = :tool_type, "
        "parameters = :parameters, input_schema = :input_schema, "
        "tool_policy = :tool_policy, aliases = :aliases, "
        "deprecated_arguments = :deprecated_arguments, queries = :queries, "
        "import_reference = :import_reference, user_id = :user_id, "
        "memory_id = :memory_id, agent_id = :agent_id, "
        "updated_at = CURRENT_TIMESTAMP WHERE tool_id = :tool_id",
        {
            "tool_id": "lookup",
            "name": "lookup",
            "description": "",
            "signature": "",
            "docstring": "",
            "tool_type": "mcp",
            "parameters": '{"q": {"type": "string"}}',
            "input_schema": json.dumps(
                {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                }
            ),
            "tool_policy": "{}",
            "aliases": "[]",
            "deprecated_arguments": "{}",
            "queries": "[]",
            "import_reference": None,
            "user_id": None,
            "memory_id": None,
            "agent_id": None,
        },
    )
    assert connection.commits == 1


_OBJECT_PARAMETERS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_PROMOTED_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


@pytest.mark.unit
def test_store_toolbox_promotes_object_parameters_into_input_schema():
    cursor = _Cursor()
    provider, _connection = _provider(cursor)

    provider._store_toolbox({"name": "weather", "parameters": _OBJECT_PARAMETERS})

    _sql, params = cursor.calls[1]
    assert json.loads(params["input_schema"]) == _PROMOTED_SCHEMA
    assert json.loads(params["parameters"]) == _OBJECT_PARAMETERS["properties"]


def _toolbox_binds(cursor: _Cursor) -> dict:
    inserts = [
        params for sql, params in cursor.calls if "INSERT INTO MEMORIZZ.toolbox" in sql
    ]
    assert len(inserts) == 1
    return {
        "input_schema": json.loads(inserts[0]["input_schema"]),
        "parameters": json.loads(inserts[0]["parameters"]),
    }


def _store_via_register_tool(tool_meta: dict) -> dict:
    """``Toolbox.register_tool`` persists through ``store(..., TOOLBOX)``."""
    cursor = _Cursor()
    provider, _connection = _provider(cursor)
    provider.store(dict(tool_meta, _id=str(uuid.uuid4())), MemoryType.TOOLBOX)
    return _toolbox_binds(cursor)


def _store_via_memagent_save(tool_meta: dict) -> dict:
    """``memagent.save()`` persists tools through ``store_memagent``."""
    cursor = _Cursor(
        routes=[
            ("SELECT id FROM MEMORIZZ.agents", (uuid.uuid4().bytes,)),
            (_TOOLBOX_SELECT, None),
        ]
    )
    provider, _connection = _provider(cursor)
    provider._table_has_column = lambda *_args: True
    agent = MemAgentModel(agent_id="team", name="Team", tools=[dict(tool_meta)])
    provider.store_memagent(agent)
    return _toolbox_binds(cursor)


@pytest.mark.unit
def test_tool_schema_identical_via_memagent_save_and_register_tool():
    # Behaviour change of the de-duplication: a full object schema handed
    # over as ``parameters`` is promoted into ``input_schema`` on both
    # write paths (previously only ``_store_toolbox`` did so).
    bare = {"name": "weather", "description": "Forecast", "signature": "(city)"}
    promoted = dict(bare, parameters=_OBJECT_PARAMETERS)
    assert _store_via_register_tool(promoted) == _store_via_memagent_save(promoted)
    assert _store_via_register_tool(promoted)["input_schema"] == _PROMOTED_SCHEMA

    def forecast(city: str, days: int = 1) -> str:
        """Forecast the weather."""
        return city

    real = Toolbox._metadata_from_callable(forecast)
    assert _store_via_register_tool(real) == _store_via_memagent_save(real)
    assert _store_via_register_tool(real)["input_schema"]["required"] == ["city"]


# ---------------------------------------------------------------------------
# 2. Hand-written row mappings routed through the registry
# ---------------------------------------------------------------------------

_CREATED = datetime(2026, 1, 2, 3, 4, 5)
_UPDATED = datetime(2026, 1, 2, 4, 0, 0)


@pytest.mark.unit
def test_retrieve_by_filter_semantic_cache_mapping_pinned():
    row_id = uuid.uuid4()
    expires = datetime(2026, 2, 1)
    row = (
        row_id.bytes,
        "cache-1",
        _Lob("stock levels"),
        _Lob("12 units"),
        "local",
        0.9,
        2,
        "agent-1",
        "memory-1",
        "session-1",
        "alice",
        _Lob('{"domain": "inventory"}'),
        [0.1, 0.2],
        _CREATED,
        expires,
    )
    cursor = _Cursor(rows=[row])
    provider, _connection = _provider(cursor)

    with_vectors = provider._retrieve_by_filter(
        {"agent_id": "agent-1"}, MemoryType.SEMANTIC_CACHE, 5, include_embedding=True
    )
    without = provider._retrieve_by_filter(
        {"agent_id": "agent-1"}, MemoryType.SEMANTIC_CACHE, 5
    )

    assert cursor.calls[0] == (
        "SELECT id, cache_key, query_text, response, scope, similarity_threshold, "
        "hit_count, agent_id, memory_id, session_id, user_id, metadata, embedding, "
        "created_at, expires_at FROM MEMORIZZ.semantic_cache WHERE "
        "agent_id = :agent_id AND (expires_at IS NULL OR expires_at > SYSTIMESTAMP) "
        "FETCH FIRST :limit ROWS ONLY",
        {"agent_id": "agent-1", "limit": 5},
    )
    expected = {
        "_id": str(row_id),
        "cache_key": "cache-1",
        "query_text": "stock levels",
        "response": "12 units",
        "scope": "local",
        "similarity_threshold": 0.9,
        "hit_count": 2,
        "usage_count": 2,
        "agent_id": "agent-1",
        "memory_id": "memory-1",
        "session_id": "session-1",
        "user_id": "alice",
        "metadata": {"domain": "inventory"},
        "timestamp": _CREATED.timestamp(),
        "created_at": _CREATED,
        "expires_at": expires,
    }
    assert without == [expected]
    assert with_vectors == [dict(expected, embedding=[0.1, 0.2])]


@pytest.mark.unit
def test_retrieve_by_filter_semantic_cache_defaults_pinned():
    row = (uuid.uuid4().bytes,) + (None,) * 14
    cursor = _Cursor(rows=[row])
    provider, _connection = _provider(cursor)

    (record,) = provider._retrieve_by_filter(
        {}, MemoryType.SEMANTIC_CACHE, 1, include_embedding=True
    )

    assert record["similarity_threshold"] == 0.85
    assert record["hit_count"] == 0 and record["usage_count"] == 0
    assert record["metadata"] == {}
    assert isinstance(record["timestamp"], float)
    assert record["created_at"] is None and record["expires_at"] is None
    assert "embedding" not in record


_RETENTION_ROW = (0.75, _UPDATED, 3, "active", _Lob('{"suppressed_by": "ops"}'))
_RETENTION_DOC = {
    "importance": 0.75,
    "last_accessed_at": _UPDATED.isoformat(),
    "access_count": 3,
    "retention_state": "active",
    "suppressed_by": "ops",
}


@pytest.mark.unit
def test_retrieve_by_id_conversation_memory_mapping_pinned():
    row_id = uuid.uuid4()
    row = (
        row_id.bytes,
        "memory-1",
        "thread-1",
        "user",
        _Lob("hello"),
        _CREATED,
        "agent-1",
        "alice",
        "summary-1",
    ) + _RETENTION_ROW
    cursor = _Cursor(rows=[row])
    provider, _connection = _provider(cursor)

    record = provider.retrieve_by_id(str(row_id), MemoryType.CONVERSATION_MEMORY)

    assert cursor.calls == [
        (
            "SELECT id, memory_id, thread_id, role, content, timestamp, agent_id, "
            "user_id, summary_id, importance, last_accessed_at, access_count, "
            "retention_state, retention_meta FROM MEMORIZZ.conversation_memory "
            "WHERE id = :id",
            {"id": row_id.bytes},
        )
    ]
    assert record == dict(
        {
            "_id": str(row_id),
            "memory_id": "memory-1",
            "thread_id": "thread-1",
            "role": "user",
            "content": "hello",
            "timestamp": _CREATED.isoformat(),
            "agent_id": "agent-1",
            "user_id": "alice",
            "summary_id": "summary-1",
        },
        **_RETENTION_DOC,
    )
    assert provider.retrieve_by_id("not-a-uuid", MemoryType.CONVERSATION_MEMORY) is None


# Columns the retention migration adds; until it has run the registry
# projects NULLs (``_retention_fields``) so the row shape stays stable, which
# is what ``list_all`` already did and what the by-id / history reads do now.
_UNMIGRATED_RETENTION_SQL = "NULL, NULL, NULL, NULL, NULL"
_UNMIGRATED_RETENTION_DOC = {
    "importance": None,
    "last_accessed_at": None,
    "access_count": 0,
    "retention_state": "active",
}


def _without_scope_columns(_type, column):
    return column not in ("user_id", "summary_id", "retention_meta")


@pytest.mark.unit
def test_retrieve_by_id_conversation_memory_skips_unmigrated_scope_columns():
    row_id = uuid.uuid4()
    row = (row_id.bytes, "memory-1", "thread-1", "user", "hi", None, None) + (None,) * 5
    cursor = _Cursor(rows=[row])
    provider, _connection = _provider(cursor)
    provider._memory_type_has_column = _without_scope_columns

    record = provider.retrieve_by_id(str(row_id), MemoryType.CONVERSATION_MEMORY)

    assert cursor.calls[0][0] == (
        "SELECT id, memory_id, thread_id, role, content, timestamp, agent_id, "
        f"{_UNMIGRATED_RETENTION_SQL} FROM MEMORIZZ.conversation_memory "
        "WHERE id = :id"
    )
    assert record == dict(
        {
            "_id": str(row_id),
            "memory_id": "memory-1",
            "thread_id": "thread-1",
            "role": "user",
            "content": "hi",
            "timestamp": None,
            "agent_id": None,
        },
        **_UNMIGRATED_RETENTION_DOC,
    )


@pytest.mark.unit
def test_retrieve_by_id_shared_memory_mapping_pinned():
    row_id = uuid.uuid4()
    content = '{ "blackboard": [] }'
    row = (
        row_id.bytes,
        "memory-1",
        _Lob(content),
        "shared_memory",
        "global",
        "agent-1",
        [0.5, 0.6],
        _CREATED,
        _UPDATED,
        _Lob('["agent-1", "agent-2"]'),
    )
    cursor = _Cursor(rows=[row])
    provider, _connection = _provider(cursor)

    record = provider.retrieve_by_id("memory-1", MemoryType.SHARED_MEMORY)

    assert cursor.calls == [
        (
            "SELECT id, memory_id, content, memory_type, scope, owner_agent_id, "
            "embedding, created_at, updated_at, access_list "
            "FROM MEMORIZZ.shared_memory WHERE memory_id = :memory_id",
            {"memory_id": "memory-1"},
        )
    ]
    assert record == {
        "_id": str(row_id),
        "memory_id": "memory-1",
        "content": content,  # exact serialized snapshot (CAS)
        "memory_type": "shared_memory",
        "scope": "global",
        "owner_agent_id": "agent-1",
        "created_at": _CREATED.isoformat(),
        "updated_at": _UPDATED.isoformat(),
        "embedding": [0.5, 0.6],
        "access_list": ["agent-1", "agent-2"],
    }


@pytest.mark.unit
def test_retrieve_by_id_shared_memory_omits_empty_optional_keys_pinned():
    row_id = uuid.uuid4()
    row = (row_id.bytes, "memory-1", "{}", None, None, None, None, None, None, "[]")
    cursor = _Cursor(rows=[row])
    provider, _connection = _provider(cursor)

    record = provider.retrieve_by_id("memory-1", MemoryType.SHARED_MEMORY)

    assert record == {
        "_id": str(row_id),
        "memory_id": "memory-1",
        "content": "{}",
        "memory_type": None,
        "scope": None,
        "owner_agent_id": None,
        "created_at": None,
        "updated_at": None,
    }
    cursor_missing = _Cursor()
    provider, _connection = _provider(cursor_missing)
    assert provider.retrieve_by_id("missing", MemoryType.SHARED_MEMORY) is None


_HISTORY_COLUMNS = (
    "id",
    "memory_id",
    "thread_id",
    "role",
    "content",
    "timestamp",
    "agent_id",
    "user_id",
    "summary_id",
    "importance",
    "last_accessed_at",
    "access_count",
    "retention_state",
    "retention_meta",
)


def _history_cursor(rows, columns):
    return _Cursor(rows=rows, description=[(name.upper(),) for name in columns])


@pytest.mark.unit
def test_retrieve_conversation_history_mapping_pinned():
    row_id = uuid.uuid4()
    base = (
        row_id.bytes,
        "memory-1",
        "thread-1",
        "assistant",
        _Lob("reply"),
        _CREATED,
        "agent-1",
        "alice",
        None,
    ) + _RETENTION_ROW
    expected = dict(
        {
            "_id": str(row_id),
            "memory_id": "memory-1",
            "thread_id": "thread-1",
            "role": "assistant",
            "content": "reply",
            "timestamp": _CREATED.isoformat(),
            "agent_id": "agent-1",
            "user_id": "alice",
            "summary_id": None,
        },
        **_RETENTION_DOC,
    )

    cursor = _history_cursor([base], _HISTORY_COLUMNS)
    provider, _connection = _provider(cursor)
    rows = provider.retrieve_conversation_history_ordered_by_timestamp(
        "memory-1", user_id="alice", thread_id="thread-1", limit=2
    )
    assert rows == [expected]
    assert cursor.calls == [
        (
            "SELECT * FROM ( SELECT id, memory_id, thread_id, role, content, "
            "timestamp, agent_id, user_id, summary_id, importance, "
            "last_accessed_at, access_count, retention_state, retention_meta "
            "FROM MEMORIZZ.conversation_memory WHERE memory_id = :memory_id AND "
            "user_id = :user_id_scope AND thread_id = :thread_id "
            "ORDER BY timestamp DESC FETCH FIRST :limit ROWS ONLY ) ORDER BY timestamp",
            {
                "memory_id": "memory-1",
                "user_id_scope": "alice",
                "thread_id": "thread-1",
                "limit": 2,
            },
        )
    ]

    cursor = _history_cursor(
        [base + ([0.7, 0.8],), base + (None,)], _HISTORY_COLUMNS + ("embedding",)
    )
    provider, _connection = _provider(cursor)
    rows = provider.retrieve_conversation_history_ordered_by_timestamp(
        "memory-1", include_embedding=True
    )
    assert rows == [dict(expected, embedding=[0.7, 0.8]), expected]
    assert cursor.calls == [
        (
            "SELECT id, memory_id, thread_id, role, content, timestamp, agent_id, "
            "user_id, summary_id, importance, last_accessed_at, access_count, "
            "retention_state, retention_meta, embedding "
            "FROM MEMORIZZ.conversation_memory WHERE memory_id = :memory_id "
            "ORDER BY timestamp",
            {"memory_id": "memory-1"},
        )
    ]


@pytest.mark.unit
def test_retrieve_conversation_history_legacy_conversation_id_column_pinned():
    row_id = uuid.uuid4()
    legacy = _RecordingCursor(
        rows=[
            (row_id.bytes, "memory-1", "thread-9", "user", "hi", "2026-01-01", None)
            + (None,) * 5
        ]
    )
    legacy.fail_on = "memory_id, thread_id, role"
    provider, _connection = _provider(legacy)
    provider._memory_type_has_column = _without_scope_columns

    rows = provider.retrieve_conversation_history_ordered_by_timestamp(
        "memory-1", thread_id="thread-9"
    )

    assert [sql for sql, _params in legacy.calls] == [
        "SELECT id, memory_id, thread_id, role, content, timestamp, agent_id, "
        f"{_UNMIGRATED_RETENTION_SQL} FROM MEMORIZZ.conversation_memory "
        "WHERE memory_id = :memory_id AND thread_id = :thread_id ORDER BY timestamp",
        "SELECT id, memory_id, conversation_id, role, content, timestamp, agent_id, "
        f"{_UNMIGRATED_RETENTION_SQL} FROM MEMORIZZ.conversation_memory "
        "WHERE memory_id = :memory_id AND conversation_id = :thread_id "
        "ORDER BY timestamp",
    ]
    assert rows == [
        dict(
            {
                "_id": str(row_id),
                "memory_id": "memory-1",
                "thread_id": "thread-9",
                "role": "user",
                "content": "hi",
                "timestamp": "2026-01-01",
                "agent_id": None,
            },
            **_UNMIGRATED_RETENTION_DOC,
        )
    ]


_SHARED_SELECT = "SELECT id FROM MEMORIZZ.shared_memory WHERE memory_id = :memory_id"


@pytest.mark.unit
def test_store_shared_memory_insert_binds_vector_pinned():
    cursor = _RecordingCursor()
    provider, connection = _provider(cursor)

    memory_id = provider._store_shared_memory(
        {
            "memory_id": "memory-1",
            "content": {"blackboard": []},
            "embedding": [0.1, 0.2],
            "access_list": ["agent-1"],
            "owner_agent_id": "agent-1",
        }
    )

    assert memory_id == "memory-1"
    assert cursor.calls[0] == (_SHARED_SELECT, {"memory_id": "memory-1"})
    insert_sql, params = cursor.calls[1]
    assert insert_sql == (
        "INSERT INTO MEMORIZZ.shared_memory (id, memory_id, content, memory_type, "
        "scope, owner_agent_id, created_at, updated_at, embedding, access_list) "
        "VALUES (:id, :memory_id, :content, :memory_type, :scope, :owner_agent_id, "
        "SYSTIMESTAMP, SYSTIMESTAMP, :embedding, :access_list)"
    )
    assert _binds_without_id(params) == {
        "memory_id": "memory-1",
        "content": '{"blackboard": []}',
        "memory_type": "shared_memory",
        "scope": "global",
        "owner_agent_id": "agent-1",
        "embedding": array.array("f", [0.1, 0.2]),
        "access_list": '["agent-1"]',
    }
    assert cursor.input_sizes == [{"embedding": oracledb.DB_TYPE_VECTOR}]
    assert connection.commits == 1


@pytest.mark.unit
def test_store_shared_memory_update_binds_vector_pinned():
    cursor = _RecordingCursor(routes=[(_SHARED_SELECT, (uuid.uuid4().bytes,))])
    provider, connection = _provider(cursor)
    existing_vector = array.array("f", [0.9])

    provider._store_shared_memory(
        {"memory_id": "memory-1", "content": "raw text", "embedding": existing_vector}
    )

    assert cursor.calls[1] == (
        "UPDATE MEMORIZZ.shared_memory SET content = :content, "
        "updated_at = SYSTIMESTAMP, embedding = :embedding "
        "WHERE memory_id = :memory_id",
        {"memory_id": "memory-1", "content": "raw text", "embedding": existing_vector},
    )
    assert cursor.calls[1][1]["embedding"] is existing_vector
    assert cursor.input_sizes == [{"embedding": oracledb.DB_TYPE_VECTOR}]
    assert connection.commits == 1

    cursor = _RecordingCursor(routes=[(_SHARED_SELECT, (uuid.uuid4().bytes,))])
    provider, _connection = _provider(cursor)
    provider._store_shared_memory({"memory_id": "memory-1", "content": "x"})
    assert cursor.input_sizes == []
    assert cursor.calls[1][0] == (
        "UPDATE MEMORIZZ.shared_memory SET content = :content, "
        "updated_at = SYSTIMESTAMP WHERE memory_id = :memory_id"
    )


# ---------------------------------------------------------------------------
# 3. Startup DDL: standard and scope indexes
# ---------------------------------------------------------------------------


def _catalog(columns, indexes=(), indexed_columns=()):
    return {
        "columns": {table: set(cols) for table, cols in columns.items()},
        "indexes": set(indexes),
        "indexed_columns": set(indexed_columns),
    }


@pytest.mark.unit
def test_create_standard_indexes_ddl_pinned():
    cursor = _Cursor()
    provider, connection = _provider(cursor)
    catalog = _catalog(
        {
            "PERSONAS": {"NAME", "MEMORY_ID", "AGENT_ID", "CREATED_AT"},
            "TOOLBOX": {"NAME", "AGENT_ID"},
        },
        indexes={"IDX_PERSONAS_NAME"},
        indexed_columns={("TOOLBOX", "AGENT_ID")},
    )

    provider._create_standard_indexes(cursor, connection, catalog)

    assert [sql for sql, _params in cursor.calls] == [
        "CREATE INDEX idx_personas_memory_id ON MEMORIZZ.personas (memory_id)",
        "CREATE INDEX idx_personas_agent_id ON MEMORIZZ.personas (agent_id)",
        "CREATE INDEX idx_personas_created_at ON MEMORIZZ.personas (created_at)",
        "CREATE INDEX idx_toolbox_name ON MEMORIZZ.toolbox (name)",
    ]
    assert connection.commits == 4
    assert {"IDX_PERSONAS_MEMORY_ID", "IDX_TOOLBOX_NAME"} <= catalog["indexes"]
    assert ("TOOLBOX", "NAME") in catalog["indexed_columns"]


@pytest.mark.unit
def test_create_standard_indexes_rolls_back_failed_ddl_pinned():
    cursor = _Cursor()
    cursor.fail_on = "idx_personas_name"
    provider, connection = _provider(cursor)
    catalog = _catalog({"PERSONAS": {"NAME"}})

    provider._create_standard_indexes(cursor, connection, catalog)

    assert connection.rollbacks == 1 and connection.commits == 0
    assert "IDX_PERSONAS_NAME" not in catalog["indexes"]


@pytest.mark.unit
def test_migrate_table_schemas_scope_index_ddl_pinned():
    required = OracleProvider._REQUIRED_COLUMNS
    cursor = _Cursor()
    provider, connection = _provider(cursor)
    catalog = _catalog(
        {
            "SUMMARIES": {"MEMORY_ID", "AGENT_ID"}
            | {col.upper() for col, _t in required[MemoryType.SUMMARIES]},
            "KNOWLEDGE_BASE": {"MEMORY_ID", "AGENT_ID"}
            | {col.upper() for col, _t in required[MemoryType.KNOWLEDGE_BASE]},
            "SUMMARY_MESSAGE_LINKS": {"SUMMARY_ID"},
        }
    )

    provider._migrate_table_schemas(catalog)

    assert [sql for sql, _params in cursor.calls] == [
        "CREATE INDEX idx_summaries_memory_thread ON MEMORIZZ.summaries "
        "(memory_id, thread_id)",
        "CREATE INDEX idx_kb_namespace ON MEMORIZZ.knowledge_base (namespace)",
    ]
    assert connection.commits == 2
    assert {"IDX_SUMMARIES_MEMORY_THREAD", "IDX_KB_NAMESPACE"} <= catalog["indexes"]
    assert provider._table_columns_cache["SUMMARIES"] == catalog["columns"]["SUMMARIES"]

    # An already-present index and an absent table issue no DDL.
    cursor = _Cursor()
    provider, connection = _provider(cursor)
    catalog = _catalog(
        {
            "SUMMARIES": {"MEMORY_ID"}
            | {col.upper() for col, _t in required[MemoryType.SUMMARIES]},
            "SUMMARY_MESSAGE_LINKS": set(),
        },
        indexes={"IDX_SUMMARIES_MEMORY_THREAD"},
    )
    provider._migrate_table_schemas(catalog)
    assert cursor.calls == [] and connection.commits == 0
