"""Regression tests for Oracle provider defects (fake connections, no DB).

Covers: semantic-cache listing/purge/expiry, logical-id deletes, single
transaction workflow writes, schema-qualified SQL and ALL_* dictionary
introspection, filter-key validation, LOB reads on generic projections, and
the SQL push-down / startup-DDL performance fixes.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest

from memorizz.enums import MemoryType
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider.oracle.provider import (
    _BY_ID_SPECS,
    _LIST_SPECS,
    OracleProvider,
)


class _Lob:
    """Stand-in for an ``oracledb.LOB`` locator."""

    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text


class _Cursor:
    """Records executed SQL and serves canned rows.

    ``routes`` maps a SQL substring to the ``fetchone`` value (or list of
    rows for ``fetchall``/iteration, or a callable producing either)
    returned after a matching ``execute``. ``rows`` is the default row list
    when nothing routes.
    """

    def __init__(self, *, rows=None, routes=None, description=None, rowcount=1):
        self.calls = []
        self.rowcount = rowcount
        self.description = description
        self._default_rows = list(rows or [])
        self._routes = list(routes or [])
        self._pending = None
        self.fail_on = None

    def execute(self, sql, params=None):
        text = " ".join(str(sql).split())
        self.calls.append((text, dict(params or {})))
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("ORA-00001: simulated failure")
        self._pending = None
        for needle, value in self._routes:
            if needle in text:
                self._pending = value
                return
        self._pending = list(self._default_rows)

    def _rows(self):
        pending = self._pending
        self._pending = None
        if callable(pending):
            pending = pending()
        if pending is None:
            return []
        if isinstance(pending, list):
            return pending
        return [pending]

    def fetchone(self):
        rows = self._rows()
        return rows[0] if rows else None

    def fetchall(self):
        return self._rows()

    def __iter__(self):
        return iter(self.fetchall())

    def setinputsizes(self, **_kwargs):
        return None


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _provider(cursor, schema="MEMORIZZ"):
    provider = OracleProvider.__new__(OracleProvider)
    provider.config = SimpleNamespace(
        schema=schema, user="APP_USER", index_policy="none", vector_search_mode="exact"
    )
    provider._table_columns_cache = {}
    provider._archive_types = set()
    provider._embedding_provider = None
    provider._vector_search_disabled_for = set()
    provider._vector_dimension_mismatch_warning_emitted_for = set()
    provider._generate_embedding_if_needed = lambda *_args, **_kwargs: None
    provider._memory_type_has_column = lambda *_args: True
    connection = _Connection(cursor)
    provider._get_connection = lambda: connection
    return provider, connection


def _row_for(fields, values):
    return tuple(values.get(field.col) for field in fields)


_SQL_TABLE_REF = re.compile(
    r"\b(?:FROM|INTO|UPDATE|JOIN)\s+(agents|agent_\w+|personas|toolbox|"
    r"semantic_cache|summaries|entity_memory|workflow_memory|tool_log)\b"
)


# ---------------------------------------------------------------------------
# Finding 1: semantic cache listing, purge, expiry, embeddings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_specs_cover_semantic_cache_and_personas():
    cache_fields = {field.col for field in _LIST_SPECS[MemoryType.SEMANTIC_CACHE][0]}
    assert {"cache_key", "query_text", "response", "metadata", "expires_at"} <= (
        cache_fields
    )
    persona_fields = {field.col for field in _LIST_SPECS[MemoryType.PERSONAS][0]}
    assert {"persona_id", "name", "background", "traits"} <= persona_fields
    assert _BY_ID_SPECS[MemoryType.SEMANTIC_CACHE][0] == "cache_key"


@pytest.mark.unit
def test_list_all_semantic_cache_rows_and_honours_include_embedding():
    fields, _order_by = _LIST_SPECS[MemoryType.SEMANTIC_CACHE]
    row_id = uuid.uuid4()
    values = {
        "id": row_id.bytes,
        "cache_key": "cache-1",
        "query_text": _Lob("stock levels"),
        "response": _Lob("12 units"),
        "scope": "local",
        "similarity_threshold": 0.9,
        "hit_count": 2,
        "agent_id": "agent-1",
        "memory_id": "memory-1",
        "metadata": '{"domain": "inventory"}',
        "embedding": [0.1, 0.2],
        "created_at": datetime(2026, 1, 1, 12, 0),
    }
    cursor = _Cursor(rows=[_row_for(fields, values)])
    provider, _connection = _provider(cursor)

    with_vectors = provider.list_all(MemoryType.SEMANTIC_CACHE, include_embedding=True)
    without = provider.list_all(MemoryType.SEMANTIC_CACHE)

    assert "FROM MEMORIZZ.semantic_cache" in cursor.calls[0][0]
    assert with_vectors[0]["cache_key"] == "cache-1"
    assert with_vectors[0]["query_text"] == "stock levels"
    assert with_vectors[0]["metadata"] == {"domain": "inventory"}
    assert with_vectors[0]["usage_count"] == 2
    assert with_vectors[0]["embedding"] == [0.1, 0.2]
    assert "embedding" not in without[0]


@pytest.mark.unit
def test_list_all_personas_returns_rows():
    fields, _order_by = _LIST_SPECS[MemoryType.PERSONAS]
    values = {
        "persona_id": "persona-1",
        "name": "Researcher",
        "role_type": "general",
        "background": _Lob("Keeps provenance"),
        "traits": '["careful"]',
        "created_at": datetime(2026, 1, 1),
    }
    cursor = _Cursor(rows=[_row_for(fields, values)])
    provider, _connection = _provider(cursor)

    rows = provider.list_all(MemoryType.PERSONAS)

    assert rows[0]["_id"] == "persona-1"
    assert rows[0]["background"] == "Keeps provenance"
    assert rows[0]["traits"] == ["careful"]


@pytest.mark.unit
def test_base_clear_semantic_cache_now_deletes_oracle_rows():
    fields, _order_by = _LIST_SPECS[MemoryType.SEMANTIC_CACHE]
    row_id = uuid.uuid4()
    cursor = _Cursor(
        routes=[
            (
                "SELECT",
                [_row_for(fields, {"id": row_id.bytes, "cache_key": "cache-1"})],
            )
        ]
    )
    provider, connection = _provider(cursor)

    deleted = provider.clear_semantic_cache()

    assert deleted == 1
    delete_sql, params = cursor.calls[-1]
    assert delete_sql.startswith("DELETE FROM MEMORIZZ.semantic_cache")
    assert "cache_key = :cache_key OR id = :row_id" in delete_sql
    assert params["row_id"] == row_id.bytes
    assert connection.commits == 1


@pytest.mark.unit
def test_purge_expired_semantic_cache_defaults_to_database_clock():
    cursor = _Cursor(rowcount=3)
    provider, connection = _provider(cursor)

    assert provider.purge_expired_semantic_cache() == 3

    sql, params = cursor.calls[0]
    assert sql == (
        "DELETE FROM MEMORIZZ.semantic_cache "
        "WHERE expires_at IS NOT NULL AND expires_at < SYSTIMESTAMP"
    )
    assert params == {}
    assert connection.commits == 1


@pytest.mark.unit
def test_purge_expired_semantic_cache_binds_explicit_now():
    cursor = _Cursor(rowcount=0)
    provider, _connection = _provider(cursor)
    now = datetime(2026, 2, 1, 9, 30)

    assert provider.purge_expired_semantic_cache(now=now) == 0
    assert provider.purge_expired_semantic_cache(now=now.isoformat()) == 0

    for sql, params in cursor.calls:
        assert sql.endswith("expires_at < :now")
        assert params == {"now": now}


@pytest.mark.unit
def test_semantic_cache_filter_and_vector_search_exclude_expired_rows():
    cursor = _Cursor()
    provider, _connection = _provider(cursor)

    provider._retrieve_by_filter({"agent_id": "agent-1"}, MemoryType.SEMANTIC_CACHE, 5)
    provider._vector_search(MemoryType.SEMANTIC_CACHE, [0.1, 0.2], limit=3)

    for sql, _params in cursor.calls:
        assert "(expires_at IS NULL OR expires_at > SYSTIMESTAMP)" in sql


@pytest.mark.unit
def test_retrieve_by_id_semantic_cache_by_cache_key_then_row_id():
    id_column, fields = _BY_ID_SPECS[MemoryType.SEMANTIC_CACHE]
    row_id = uuid.uuid4()
    values = {
        "id": row_id.bytes,
        "cache_key": str(row_id),
        "query_text": _Lob("q"),
        "response": _Lob("r"),
        "embedding": [0.5],
        "created_at": datetime(2026, 1, 1),
    }
    cursor = _Cursor(routes=[("WHERE id = :row_id", _row_for(fields, values))])
    provider, _connection = _provider(cursor)

    record = provider.retrieve_by_id(str(row_id), MemoryType.SEMANTIC_CACHE)

    assert id_column == "cache_key"
    assert record["cache_key"] == str(row_id)
    assert record["response"] == "r"
    assert record["embedding"] == [0.5]
    assert "WHERE cache_key = :cache_key" in cursor.calls[0][0]
    assert cursor.calls[1][1] == {"row_id": row_id.bytes}


# ---------------------------------------------------------------------------
# Finding 2: delete_by_id keyed on logical ids
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("memory_type", "column"),
    [
        (MemoryType.SUMMARIES, "summary_id"),
        (MemoryType.ENTITY_MEMORY, "entity_id"),
        (MemoryType.SEMANTIC_CACHE, "cache_key"),
        (MemoryType.TOOL_LOG, "tool_log_id"),
    ],
)
def test_delete_by_logical_id_and_raw_fallback(memory_type, column):
    cursor = _Cursor(rowcount=1)
    provider, connection = _provider(cursor)

    assert provider.delete_by_id("logical-id-1", memory_type) is True
    sql, params = cursor.calls[-1]
    assert sql == (
        f"DELETE FROM MEMORIZZ.{memory_type.value} WHERE {column} = :{column}"
    )
    assert params == {column: "logical-id-1"}

    row_id = uuid.uuid4()
    assert provider.delete_by_id(str(row_id), memory_type) is True
    sql, params = cursor.calls[-1]
    assert f"WHERE {column} = :{column} OR id = :row_id" in sql
    assert params == {column: str(row_id), "row_id": row_id.bytes}

    cursor.rowcount = 0
    assert provider.delete_by_id("missing", memory_type) is False
    assert connection.commits == 3


# ---------------------------------------------------------------------------
# Finding 3: workflow rows are written in one INSERT / one transaction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_store_workflow_memory_single_insert_carries_json_columns():
    cursor = _Cursor()
    provider, connection = _provider(cursor)
    steps = [{"tool": "search", "status": "ok"}]

    workflow_id = provider._store_workflow_memory(
        {
            "workflow_id": "workflow-1",
            "name": "Refund",
            "description": "Refund flow",
            "steps": steps,
            "outcome": {"status": "completed"},
            "canonical_signature": ["search"],
            "skills_activated": ["skill-1"],
            "user_id": "alice",
        }
    )

    assert workflow_id == "workflow-1"
    assert len(cursor.calls) == 1
    sql, params = cursor.calls[0]
    assert sql.startswith("INSERT INTO MEMORIZZ.workflow_memory (")
    columns = sql[sql.index("(") + 1 : sql.index(")")].split(", ")
    assert {"steps", "outcome", "canonical_signature", "skills_activated"} <= set(
        columns
    )
    assert json.loads(params["steps"]) == steps
    assert json.loads(params["outcome"]) == {"status": "completed"}
    assert params["shadow_evaluations"] is None
    assert connection.commits == 1
    assert not any(call[0].startswith("UPDATE") for call in cursor.calls)


@pytest.mark.unit
def test_store_workflow_memory_failure_commits_nothing():
    cursor = _Cursor()
    cursor.fail_on = "INSERT INTO MEMORIZZ.workflow_memory"
    provider, connection = _provider(cursor)

    with pytest.raises(RuntimeError):
        provider._store_workflow_memory(
            {"workflow_id": "workflow-2", "name": "Broken", "steps": [{"a": 1}]}
        )

    assert connection.commits == 0


# ---------------------------------------------------------------------------
# Finding 4: schema-qualified SQL and owner-scoped dictionary reads
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_store_memagent_qualifies_every_table_with_configured_schema():
    agent_uuid = uuid.uuid4().bytes
    cursor = _Cursor(
        routes=[
            ("SELECT id FROM MEMORIZZ.agents", (agent_uuid,)),
            ("SELECT id FROM MEMORIZZ.toolbox", None),
            ("SELECT id FROM MEMORIZZ.personas", None),
        ]
    )
    provider, connection = _provider(cursor)
    provider._table_has_column = lambda *_args: True
    agent = MemAgentModel(
        agent_id="team",
        name="Team",
        tools=[{"name": "lookup", "description": "Look up", "signature": "(q)"}],
    )

    assert provider.store_memagent(agent) == "team"

    statements = [sql for sql, _params in cursor.calls]
    assert statements[0].startswith("SELECT id FROM MEMORIZZ.agents")
    assert any(s.startswith("UPDATE MEMORIZZ.agents SET") for s in statements)
    assert any("INSERT INTO MEMORIZZ.agent_llm_configs" in s for s in statements)
    assert any("DELETE FROM MEMORIZZ.agent_delegates" in s for s in statements)
    assert any("INSERT INTO MEMORIZZ.toolbox (" in s for s in statements)
    for sql in statements:
        assert _SQL_TABLE_REF.search(sql) is None, sql
    assert connection.commits == 1


@pytest.mark.unit
def test_memory_type_has_column_reads_all_tab_columns_for_owner():
    cursor = _Cursor(routes=[("all_tab_columns", [("USER_ID",), ("CONTENT",)])])
    provider, _connection = _provider(cursor, schema="memorizz")
    provider._memory_type_has_column = OracleProvider._memory_type_has_column.__get__(
        provider
    )

    assert provider._memory_type_has_column(MemoryType.SUMMARIES, "user_id") is True
    assert provider._memory_type_has_column(MemoryType.SUMMARIES, "missing") is False

    sql, params = cursor.calls[0]
    assert "FROM all_tab_columns" in sql
    assert "owner = :owner" in sql
    assert params == {"owner": "MEMORIZZ", "table_name": "SUMMARIES"}
    assert len(cursor.calls) == 1  # second lookup served from the cache


@pytest.mark.unit
def test_table_has_column_and_delete_scope_bind_owner():
    cursor = _Cursor(routes=[("all_tab_columns", (1,)), ("all_tables", (0,))])
    provider, _connection = _provider(cursor)

    assert provider._table_has_column(cursor, "agents", "is_favorite") is True
    sql, params = cursor.calls[0]
    assert "FROM all_tab_columns" in sql
    assert params["owner"] == "MEMORIZZ"

    provider.delete_scope(memory_id="memory-1")
    table_checks = [call for call in cursor.calls if "all_tables" in call[0]]
    assert table_checks
    assert all(call[1]["owner"] == "MEMORIZZ" for call in table_checks)
    assert not any("user_tables" in call[0] for call in cursor.calls)


# ---------------------------------------------------------------------------
# Finding 5: dict keys are validated before they become SQL identifiers
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_query",
    [{"nonexistent": 1}, {"agent_id = '' OR 1=1 --": "x"}, {"embedding": [0.1]}],
)
def test_retrieve_by_filter_rejects_unknown_columns_before_sql(bad_query):
    cursor = _Cursor()
    provider, _connection = _provider(cursor)

    with pytest.raises(ValueError, match="Unsupported filter column"):
        provider._retrieve_by_filter(bad_query, MemoryType.ENTITY_MEMORY, 5)

    assert cursor.calls == []


@pytest.mark.unit
def test_retrieve_by_filter_accepts_known_columns():
    cursor = _Cursor()
    provider, _connection = _provider(cursor)

    provider._retrieve_by_filter(
        {"entity_id": "e-1", "user_id": None}, MemoryType.ENTITY_MEMORY, 5
    )

    sql, params = cursor.calls[0]
    assert "WHERE entity_id = :entity_id AND user_id IS NULL" in sql
    assert params == {"entity_id": "e-1", "limit": 5}


@pytest.mark.unit
def test_update_shared_memory_rejects_unknown_columns_before_sql():
    cursor = _Cursor()
    provider, connection = _provider(cursor)

    with pytest.raises(ValueError, match="Unsupported shared_memory update column"):
        provider._update_shared_memory_by_id("memory-1", {"scope); DROP": "x"})

    assert cursor.calls == []
    assert connection.commits == 0

    assert provider._update_shared_memory_by_id("memory-1", {"scope": "team"}) is True
    assert "SET updated_at = SYSTIMESTAMP, scope = :scope" in cursor.calls[0][0]


# ---------------------------------------------------------------------------
# Finding 6: generic projections read LOB locators
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_retrieve_by_name_reads_lob_columns_as_text():
    row_id = uuid.uuid4().bytes
    cursor = _Cursor(
        rows=[(row_id, "lookup", _Lob("Look things up"), [0.1])],
        description=[("ID",), ("NAME",), ("DESCRIPTION",), ("EMBEDDING",)],
    )
    provider, _connection = _provider(cursor)

    record = provider.retrieve_by_name("lookup", MemoryType.TOOLBOX)

    assert record["description"] == "Look things up"
    assert record["id"] == row_id
    assert "embedding" not in record
    assert provider.retrieve_by_name(
        "lookup", MemoryType.TOOLBOX, include_embedding=True
    )["embedding"] == [0.1]


@pytest.mark.unit
def test_generic_filter_and_entity_by_id_read_lob_columns():
    cursor = _Cursor(
        rows=[("ent-1", _Lob('{"city": "London"}'), _Lob("[]"), None)],
        description=[("ENTITY_ID",), ("ATTRIBUTES",), ("RELATIONS",), ("METADATA",)],
    )
    provider, _connection = _provider(cursor)

    by_filter = provider._retrieve_by_filter(
        {"entity_id": "ent-1"}, MemoryType.ENTITY_MEMORY, 1
    )
    by_id = provider.retrieve_by_id("ent-1", MemoryType.ENTITY_MEMORY)

    assert by_filter[0]["attributes"] == {"city": "London"}
    assert by_id["attributes"] == {"city": "London"}
    assert by_id["relations"] == []
    assert not any(hasattr(value, "read") for value in by_id.values())


# ---------------------------------------------------------------------------
# Finding 7: SQL push-down and startup DDL
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_conversation_history_limits_in_sql_and_projects_embedding_on_request():
    row_id = uuid.uuid4()
    stamp = datetime(2026, 3, 1, 10, 0)
    cursor = _Cursor(
        rows=[
            (
                row_id.bytes,
                "memory-1",
                "thread-1",
                "user",
                _Lob("hello"),
                stamp,
                "agent-1",
                "alice",
                None,
            )
        ],
        description=[
            ("ID",),
            ("MEMORY_ID",),
            ("THREAD_ID",),
            ("ROLE",),
            ("CONTENT",),
            ("TIMESTAMP",),
            ("AGENT_ID",),
            ("USER_ID",),
            ("SUMMARY_ID",),
        ],
    )
    provider, _connection = _provider(cursor)

    rows = provider.retrieve_conversation_history_ordered_by_timestamp(
        "memory-1", limit=2, thread_id="thread-1"
    )

    sql, params = cursor.calls[0]
    assert "ORDER BY timestamp DESC FETCH FIRST :limit ROWS ONLY" in sql
    assert sql.rstrip().endswith("ORDER BY timestamp")
    assert "embedding" not in sql
    assert params == {"memory_id": "memory-1", "thread_id": "thread-1", "limit": 2}
    assert rows[0]["content"] == "hello"
    assert rows[0]["_id"] == str(row_id)

    provider.retrieve_conversation_history_ordered_by_timestamp(
        "memory-1", include_embedding=True
    )
    sql, params = cursor.calls[-1]
    assert ", embedding" in sql
    assert "FETCH FIRST" not in sql
    assert "limit" not in params


@pytest.mark.unit
def test_list_tool_logs_pushes_scope_and_limit_into_sql():
    fields, _order_by = _LIST_SPECS[MemoryType.TOOL_LOG]
    values = {
        "id": uuid.uuid4().bytes,
        "tool_log_id": "log-1",
        "tool_name": "search",
        "arguments": _Lob("{}"),
        "result": _Lob("ok"),
        "success": 1,
        "timestamp": datetime(2026, 3, 1, 10, 0),
        "thread_id": "thread-1",
        "memory_id": "memory-1",
        "user_id": "alice",
    }
    cursor = _Cursor(rows=[_row_for(fields, values)])
    provider, _connection = _provider(cursor)

    rows = provider.list_tool_logs(
        memory_id="memory-1", user_id="alice", thread_id="thread-1", limit=5
    )

    sql, params = cursor.calls[0]
    assert sql.startswith("SELECT id, tool_log_id")
    assert sql.endswith(
        "FROM MEMORIZZ.tool_log WHERE memory_id = :memory_id AND user_id = :user_id "
        "AND thread_id = :thread_id ORDER BY timestamp DESC "
        "FETCH FIRST :limit ROWS ONLY"
    )
    assert params == {
        "memory_id": "memory-1",
        "user_id": "alice",
        "thread_id": "thread-1",
        "limit": 5,
    }
    assert rows[0]["tool_log_id"] == "log-1"
    assert rows[0]["result"] == "ok"

    provider.list_tool_logs(user_id=None, thread_id="", limit=0)
    sql, params = cursor.calls[-1]
    assert "WHERE user_id IS NULL AND thread_id IS NULL ORDER BY timestamp DESC" in sql
    assert "FETCH FIRST" not in sql
    assert params == {}


@pytest.mark.unit
def test_startup_ddl_reads_dictionary_once_and_only_touches_missing_objects():
    required = OracleProvider._REQUIRED_COLUMNS
    summaries = {col.upper() for col, _type in required[MemoryType.SUMMARIES]} | {
        "ID",
        "MEMORY_ID",
        "AGENT_ID",
        "USER_ID",
        "CREATED_AT",
        "EMBEDDING",
    }
    tool_log = {col.upper() for col, _type in required[MemoryType.TOOL_LOG]} | {
        "ID",
        "MEMORY_ID",
        "AGENT_ID",
        "USER_ID",
    }
    tool_log.discard("OUTCOME_DETAILS")
    columns = [("SUMMARIES", col) for col in sorted(summaries)]
    columns += [("TOOL_LOG", col) for col in sorted(tool_log)]
    columns += [("SUMMARY_MESSAGE_LINKS", "SUMMARY_ID")]
    indexes = [
        ("IDX_SUMMARIES_MEMORY_THREAD", "SUMMARIES", "MEMORY_ID"),
        ("IDX_SUMMARIES_MEMORY_THREAD", "SUMMARIES", "THREAD_ID"),
        ("IDX_SUMMARIES_MEMORY_ID", "SUMMARIES", "MEMORY_ID"),
        ("IDX_SUMMARIES_AGENT_ID", "SUMMARIES", "AGENT_ID"),
        ("IDX_TOOL_LOG_MEMORY_ID", "TOOL_LOG", "MEMORY_ID"),
        ("IDX_TOOL_LOG_AGENT_ID", "TOOL_LOG", "AGENT_ID"),
    ]
    cursor = _Cursor(
        routes=[("all_tab_columns", columns), ("all_ind_columns", indexes)]
    )
    provider, connection = _provider(cursor)

    catalog = provider._load_schema_catalog(cursor)
    provider._create_standard_indexes(cursor, connection, catalog)
    provider._migrate_table_schemas(catalog)

    dictionary_reads = [
        c
        for c in cursor.calls
        if "all_tab_columns" in c[0] or "all_ind_columns" in c[0]
    ]
    assert len(dictionary_reads) == 2
    assert all(call[1] == {"owner": "MEMORIZZ"} for call in dictionary_reads)
    ddl = [sql for sql, _params in cursor.calls if sql.startswith(("CREATE", "ALTER"))]
    assert ddl == [
        "CREATE INDEX idx_summaries_created_at ON MEMORIZZ.summaries (created_at)",
        "ALTER TABLE MEMORIZZ.tool_log ADD (outcome_details CLOB)",
    ]
    assert not any(
        "user_tab_columns" in sql or "user_tables" in sql for sql, _ in cursor.calls
    )
    assert "OUTCOME_DETAILS" in provider._table_columns_cache["TOOL_LOG"]
    assert "IDX_SUMMARIES_CREATED_AT" in catalog["indexes"]


@pytest.mark.unit
def test_store_memagent_skips_reembedding_unchanged_tools_and_personas():
    agent_uuid = uuid.uuid4().bytes
    embed_calls = []

    class _Embedder:
        def get_embedding(self, text):
            embed_calls.append(text)
            return [0.1, 0.2]

    stored_tool = ["lookup", _Lob("Look up"), "(q)", 1]
    stored_persona = ["Researcher", _Lob("Keeps provenance"), 1]
    cursor = _Cursor(
        routes=[
            ("SELECT id FROM MEMORIZZ.agents", (agent_uuid,)),
            ("SELECT name, description, signature, CASE", lambda: tuple(stored_tool)),
            ("SELECT name, background, CASE", lambda: tuple(stored_persona)),
            ("SELECT id FROM MEMORIZZ.toolbox", (b"\x01" * 16,)),
            ("SELECT id FROM MEMORIZZ.personas", (b"\x02" * 16,)),
        ]
    )
    provider, _connection = _provider(cursor)
    provider._table_has_column = lambda *_args: True
    provider._embedding_provider = _Embedder()
    agent = MemAgentModel(
        agent_id="team",
        name="Team",
        tools=[{"name": "lookup", "description": "Look up", "signature": "(q)"}],
        persona={
            "name": "Researcher",
            "background": "Keeps provenance",
            "role": "general",
        },
    )

    provider.store_memagent(agent)

    assert embed_calls == []
    toolbox_updates = [
        s for s, _ in cursor.calls if s.startswith("UPDATE MEMORIZZ.toolbox")
    ]
    persona_updates = [
        s for s, _ in cursor.calls if s.startswith("UPDATE MEMORIZZ.personas")
    ]
    assert len(toolbox_updates) == 1 and "embedding" not in toolbox_updates[0]
    assert len(persona_updates) == 1 and "embedding" not in persona_updates[0]

    # A changed description invalidates the stored vector for that tool only.
    cursor.calls.clear()
    stored_tool[1] = _Lob("Old text")
    provider.store_memagent(agent)
    assert embed_calls == ["lookup: Look up (q)"]
    toolbox_updates = [
        s for s, _ in cursor.calls if s.startswith("UPDATE MEMORIZZ.toolbox")
    ]
    assert "embedding = :embedding" in toolbox_updates[0]


# ---------------------------------------------------------------------------
# Live-DB follow-up: cache rows addressed by cache_key on archive-extended
# schemas, and Generative-Agents forgetting-field parity
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from memorizz.memory_provider.oracle.provider import _RETENTION_FIELDS  # noqa: E402


@pytest.mark.unit
def test_retrieve_by_id_semantic_cache_falls_through_when_archive_misses(monkeypatch):
    from memorizz.memory_provider.oracle import archive as archive_module

    _id_column, fields = _BY_ID_SPECS[MemoryType.SEMANTIC_CACHE]
    values = {
        "id": uuid.uuid4().bytes,
        "cache_key": "cache-live-1",
        "query_text": _Lob("q"),
        "response": _Lob("r"),
        "embedding": [0.1],
        "created_at": datetime(2026, 1, 1),
    }
    cursor = _Cursor(
        routes=[("WHERE cache_key = :cache_key", _row_for(fields, values))]
    )
    provider, _connection = _provider(cursor)
    provider._archive_types = {"semantic_cache"}
    archive_calls = []
    monkeypatch.setattr(
        archive_module, "get_record", lambda *args: archive_calls.append(args) or None
    )

    record = provider.retrieve_by_id("cache-live-1", MemoryType.SEMANTIC_CACHE)

    assert len(archive_calls) == 1
    assert record["cache_key"] == "cache-live-1"
    assert record["response"] == "r"


@pytest.mark.unit
def test_required_columns_and_migration_cover_forgetting_fields():
    required = OracleProvider._REQUIRED_COLUMNS
    full = {
        "importance",
        "last_accessed_at",
        "access_count",
        "retention_state",
        "retention_meta",
    }
    for memory_type in (
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.ENTITY_MEMORY,
        MemoryType.SUMMARIES,
        MemoryType.WORKFLOW_MEMORY,
    ):
        assert full <= {col for col, _type in required[memory_type]}
    assert {"last_accessed_at", "retention_state", "retention_meta"} <= {
        col for col, _type in required[MemoryType.KNOWLEDGE_BASE]
    }
    migration = (
        Path(__file__).parents[2]
        / "src/memorizz/memory_provider/oracle/migrations/009_forgetting_fields.sql"
    ).read_text(encoding="utf-8")
    assert "add_column_if_missing" in migration
    for table in (
        "conversation_memory",
        "knowledge_base",
        "entity_memory",
        "summaries",
        "workflow_memory",
    ):
        assert f"'{table}', 'retention_meta'" in migration
        assert f"'{table}', 'last_accessed_at'" in migration


@pytest.mark.unit
def test_store_paths_bind_forgetting_fields_from_epoch_and_iso():
    provider = OracleProvider.__new__(OracleProvider)
    provider._generate_embedding_if_needed = lambda *_args, **_kwargs: None
    captured = {}
    provider._insert_base_row = lambda *_args, **kwargs: captured.update(kwargs)
    accessed = datetime(2026, 4, 1, 9, 0)

    provider._store_knowledge_base(
        {
            "memory_id": "m",
            "content": "c",
            "last_accessed_at": accessed.timestamp(),
            "retention_state": "Suppressed",
            "suppressed_by": "plan-x",
            "suppression_reason": "stale",
        }
    )
    optional = captured["optional_columns"]
    assert optional["last_accessed_at"] == accessed
    assert optional["retention_state"] == "suppressed"
    assert json.loads(optional["retention_meta"]) == {
        "suppressed_by": "plan-x",
        "suppression_reason": "stale",
    }
    assert "importance" not in optional  # required column on knowledge_base

    captured.clear()
    provider._store_workflow_memory(
        {"workflow_id": "w", "name": "n", "importance": 7, "access_count": "3"}
    )
    assert captured["optional_columns"]["importance"] == 0.7
    assert captured["optional_columns"]["access_count"] == 3
    assert "retention_meta" not in captured["optional_columns"]


@pytest.mark.unit
def test_summary_conversation_and_entity_inserts_carry_forgetting_columns():
    cursor = _Cursor(rowcount=1)
    provider, _connection = _provider(cursor)

    provider._store_summary(
        {
            "summary_id": "s",
            "content": "c",
            "memory_id": "m",
            "importance": 0.4,
            "last_accessed_at": "2026-04-01T09:00:00",
        }
    )
    sql, params = cursor.calls[0]
    assert sql.startswith("INSERT INTO MEMORIZZ.summaries (")
    assert "importance, last_accessed_at" in sql
    assert params["importance"] == 0.4
    assert params["last_accessed_at"] == datetime(2026, 4, 1, 9, 0)

    cursor.calls.clear()
    provider._store_conversation_memory_impl(
        {
            "memory_id": "m",
            "role": "user",
            "content": "hi",
            "retention_state": "active",
            "access_count": 2,
        }
    )
    sql, params = cursor.calls[0]
    assert "access_count, retention_state" in sql
    assert params["access_count"] == 2 and params["retention_state"] == "active"

    cursor.calls.clear()
    provider._store_entity_memory(
        {
            "entity_id": "e",
            "name": "n",
            "memory_id": "m",
            "importance": 0.9,
            "suppressed_by": "plan-y",
        }
    )
    merge_sql, merge_params = next(c for c in cursor.calls if c[0].startswith("MERGE"))
    assert "importance = :importance" in merge_sql
    assert "retention_meta = :retention_meta" in merge_sql
    assert ", importance, retention_meta, embedding" in merge_sql
    assert json.loads(merge_params["retention_meta"]) == {"suppressed_by": "plan-y"}


@pytest.mark.unit
def test_list_rows_carry_forgetting_fields_and_flatten_retention_meta():
    fields, _order_by = _LIST_SPECS[MemoryType.SUMMARIES]
    values = {
        "id": uuid.uuid4().bytes,
        "summary_id": "s",
        "content": _Lob("c"),
        "importance": 0.25,
        "last_accessed_at": datetime(2026, 4, 1, 9, 0),
        "access_count": 3,
        "retention_state": "suppressed",
        "retention_meta": _Lob(
            '{"suppressed_by": "plan-x", "suppressed_at": "2026-04-01T10:00:00"}'
        ),
    }
    cursor = _Cursor(rows=[_row_for(fields + _RETENTION_FIELDS, values)])
    provider, _connection = _provider(cursor)

    row = provider.list_all(MemoryType.SUMMARIES)[0]

    assert (
        ", retention_state, retention_meta FROM MEMORIZZ.summaries"
        in cursor.calls[0][0]
    )
    assert row["importance"] == 0.25
    assert row["last_accessed_at"] == "2026-04-01T09:00:00"
    assert row["access_count"] == 3
    assert row["retention_state"] == "suppressed"
    assert row["suppressed_by"] == "plan-x"
    assert row["suppressed_at"] == "2026-04-01T10:00:00"
    assert "retention_meta" not in row


@pytest.mark.unit
def test_forgetting_fields_project_null_until_migrated():
    fields, _order_by = _LIST_SPECS[MemoryType.WORKFLOW_MEMORY]
    row = _row_for(fields, {"id": uuid.uuid4().bytes, "workflow_id": "w"}) + (None,) * 5
    cursor = _Cursor(rows=[row])
    provider, _connection = _provider(cursor)
    provider._memory_type_has_column = lambda _type, column: column != "retention_meta"

    listed = provider.list_all(MemoryType.WORKFLOW_MEMORY)[0]

    assert (
        "NULL, NULL, NULL, NULL, NULL FROM MEMORIZZ.workflow_memory"
        in cursor.calls[0][0]
    )
    assert listed["importance"] is None
    assert listed["access_count"] == 0
    assert listed["retention_state"] == "active"
    assert listed["last_accessed_at"] is None


@pytest.mark.unit
def test_conversation_by_id_and_history_carry_forgetting_fields():
    row_id = uuid.uuid4()
    stamp = datetime(2026, 3, 1, 10, 0)
    base = (row_id.bytes, "m", "t", "user", _Lob("hi"), stamp, "agent-1", "alice", None)
    retention = (
        0.5,
        datetime(2026, 3, 2, 10, 0),
        4,
        None,
        _Lob('{"suppressed_by": "plan-z"}'),
    )
    columns = (
        "ID",
        "MEMORY_ID",
        "THREAD_ID",
        "ROLE",
        "CONTENT",
        "TIMESTAMP",
        "AGENT_ID",
        "USER_ID",
        "SUMMARY_ID",
        "IMPORTANCE",
        "LAST_ACCESSED_AT",
        "ACCESS_COUNT",
        "RETENTION_STATE",
        "RETENTION_META",
    )
    cursor = _Cursor(rows=[base + retention], description=[(c,) for c in columns])
    provider, _connection = _provider(cursor)

    by_id = provider.retrieve_by_id(str(row_id), MemoryType.CONVERSATION_MEMORY)
    history = provider.retrieve_conversation_history_ordered_by_timestamp("m")

    assert "retention_meta FROM MEMORIZZ.conversation_memory" in cursor.calls[0][0]
    for record in (by_id, history[0]):
        assert record["importance"] == 0.5
        assert record["last_accessed_at"] == "2026-03-02T10:00:00"
        assert record["access_count"] == 4
        assert record["retention_state"] == "active"
        assert record["suppressed_by"] == "plan-z"
        assert record["content"] == "hi"


@pytest.mark.unit
def test_update_by_id_persists_suppression_and_ignores_unknown_keys():
    cursor = _Cursor(rowcount=1)
    provider, connection = _provider(cursor)
    row_id = uuid.uuid4()

    assert (
        provider.update_by_id(
            str(row_id),
            {
                "retention_state": "suppressed",
                "suppressed_by": "plan-x",
                "suppressed_at": datetime(2026, 4, 1, 10, 0),
                "bogus": 1,
            },
            MemoryType.KNOWLEDGE_BASE,
        )
        is True
    )
    sql, params = cursor.calls[0]
    assert "retention_state = :retention_state" in sql
    assert (
        "retention_meta = JSON_MERGEPATCH(COALESCE(retention_meta, TO_CLOB('{}')), "
        ":retention_meta RETURNING CLOB)"
    ) in sql
    assert params["retention_state"] == "suppressed"
    assert json.loads(params["retention_meta"]) == {
        "suppressed_by": "plan-x",
        "suppressed_at": "2026-04-01T10:00:00",
    }
    assert "bogus" not in params and params["id"] == row_id.bytes
    assert connection.commits == 1

    cursor.calls.clear()
    assert (
        provider.update_by_id(str(row_id), {"bogus": 1}, MemoryType.KNOWLEDGE_BASE)
        is False
    )
    assert cursor.calls == []

    provider.update_by_id(
        "summary-1",
        {
            "retention_state": "active",
            "unsuppressed_by": "alice",
            "last_accessed_at": 1_800_000_000,
            "access_count": "4",
        },
        MemoryType.SUMMARIES,
    )
    sql, params = cursor.calls[-1]
    assert "WHERE summary_id = :summary_id" in sql
    assert params["last_accessed_at"] == datetime.fromtimestamp(1_800_000_000)
    assert params["access_count"] == 4
    assert json.loads(params["retention_meta"]) == {"unsuppressed_by": "alice"}


@pytest.mark.unit
def test_touch_many_updates_rows_in_one_statement():
    cursor = _Cursor(rowcount=2)
    provider, connection = _provider(cursor)
    first, second = uuid.uuid4(), uuid.uuid4()

    touched = provider.touch_many(
        [str(first), "not-a-uuid", str(second), "", str(first)],
        MemoryType.KNOWLEDGE_BASE,
    )

    assert touched == 2
    sql, params = cursor.calls[0]
    assert sql == (
        "UPDATE MEMORIZZ.knowledge_base SET last_accessed_at = CURRENT_TIMESTAMP, "
        "access_count = NVL(access_count, 0) + 1 WHERE id IN (:row_0, :row_2)"
    )
    assert params == {"row_0": first.bytes, "row_2": second.bytes}
    assert connection.commits == 1

    cursor.calls.clear()
    now = datetime(2026, 4, 1, 9, 0)
    provider.touch_many(["summary-1", str(first)], "summaries", now=now)
    sql, params = cursor.calls[0]
    assert "SET last_accessed_at = :now, access_count = NVL(access_count, 0) + 1" in sql
    assert sql.endswith(
        "WHERE summary_id IN (:logical_0, :logical_1) OR id IN (:row_1)"
    )
    assert params == {
        "now": now,
        "logical_0": "summary-1",
        "logical_1": str(first),
        "row_1": first.bytes,
    }

    assert provider.touch_many([], MemoryType.SUMMARIES) == 0
    assert provider.touch_many(["tool"], MemoryType.TOOLBOX) == 0


@pytest.mark.unit
def test_touch_many_failure_is_logged_not_raised():
    cursor = _Cursor()
    cursor.fail_on = "UPDATE MEMORIZZ.summaries"
    provider, connection = _provider(cursor)

    assert provider.touch_many(["summary-1"], MemoryType.SUMMARIES) == 0
    assert connection.rollbacks == 1 and connection.commits == 0


def _observability_row(number, timestamp, fields):
    # The page query leaves the VECTOR column out of its projection.
    fields = tuple(field for field in fields if field.col != "embedding")
    row_id = uuid.UUID(int=number)
    payload = {
        "record_type": "learning_record",
        "agent_id": "agent-1",
        "memory_id": f"rec-{number}",
        "timestamp": timestamp,
    }
    values = {
        "id": row_id.bytes,
        "memory_id": f"rec-{number}",
        "content": _Lob(json.dumps(payload)),
        "created_at": datetime(2026, 4, 1),
    }
    return _row_for(fields, values) + (timestamp, row_id.hex.upper())


@pytest.mark.unit
def test_query_observability_records_runs_indexed_sql_with_keyset_cursor():
    fields, _order_by = _LIST_SPECS[MemoryType.SHARED_MEMORY]
    rows = [
        _observability_row(3, "2026-04-03T00:00:00", fields),
        _observability_row(2, "2026-04-02T00:00:00", fields),
        _observability_row(1, "2026-04-01T00:00:00", fields),
    ]
    cursor = _Cursor(rows=rows)
    provider, _connection = _provider(cursor)

    page = provider.query_observability_records(
        MemoryType.SHARED_MEMORY,
        agent_ids=["agent-1"],
        memory_ids=["rec-9"],
        record_type="learning_record",
        user_id=None,
        application_id="app",
        limit=2,
    )

    sql, binds = cursor.calls[0]
    json_value = "JSON_VALUE(content, '$.{}' RETURNING VARCHAR2(512) NULL ON ERROR)"
    assert f"{json_value.format('record_type')} = :record_type" in sql
    assert f"{json_value.format('agent_id')} IN (:agent_0)" in sql
    assert "memory_id) IN (:scope_memory_0)" in sql
    assert f"{json_value.format('user_id')} IS NULL" in sql
    assert f"{json_value.format('application_id')} = :application_id" in sql
    assert sql.endswith(
        "ORDER BY obs_timestamp DESC, obs_row_key DESC FETCH FIRST :page_size ROWS ONLY"
    )
    assert binds["page_size"] == 3 and binds["record_type"] == "learning_record"
    assert binds["agent_0"] == "agent-1" and binds["scope_memory_0"] == "rec-9"
    assert [item["memory_id"] for item in page["items"]] == ["rec-3", "rec-2"]
    assert page["items"][0]["record_type"] == "learning_record"
    assert page["items"][0]["agent_id"] == "agent-1"
    assert page["items"][0]["timestamp"] == "2026-04-03T00:00:00"
    assert page["has_more"] is True and page["next_cursor"]
    assert provider.observability_row_cursor(page["items"][1]) == page["next_cursor"]

    cursor._default_rows = [rows[2]]
    second = provider.query_observability_records(
        MemoryType.SHARED_MEMORY,
        record_type="learning_record",
        limit=2,
        cursor=page["next_cursor"],
    )
    sql, binds = cursor.calls[-1]
    assert (
        "WHERE (obs_timestamp < :last_timestamp OR "
        "(obs_timestamp = :last_timestamp AND obs_row_key < :last_row_key))"
    ) in sql
    assert binds["last_timestamp"] == "2026-04-02T00:00:00"
    assert binds["last_row_key"] == uuid.UUID(int=2).hex.upper()
    assert [item["memory_id"] for item in second["items"]] == ["rec-1"]
    assert second["next_cursor"] is None and second["has_more"] is False


@pytest.mark.unit
def test_query_observability_records_falls_back_for_unsupported_filters():
    cursor = _Cursor()
    provider, _connection = _provider(cursor)
    provider.list_all = lambda *_args, **_kwargs: []

    page = provider.query_observability_records(
        MemoryType.SHARED_MEMORY, tool_name="search", limit=5
    )

    assert page["items"] == [] and cursor.calls == []
    with pytest.raises(ValueError, match="Invalid observability cursor"):
        provider.query_observability_records(
            MemoryType.SHARED_MEMORY, cursor="@@not-a-cursor@@"
        )
