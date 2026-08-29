"""Oracle schema/persistence parity regressions for MemoRizz 0.5."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memory_provider.oracle.provider import (
    _BY_ID_SPECS,
    _LIST_SPECS,
    _VECTOR_SPECS,
    OracleConfig,
    OracleProvider,
)
from tests.mocks.mock_providers import MockMemoryProvider


class _Cursor:
    def __init__(self, *, update_rowcounts=None, fetchone_value=None):
        self.calls = []
        self.rowcount = 0
        self._update_rowcounts = iter(update_rowcounts or [])
        self._fetchone_value = fetchone_value

    def execute(self, sql, params=None):
        self.calls.append((" ".join(str(sql).split()), dict(params or {})))
        if str(sql).lstrip().upper().startswith("UPDATE"):
            self.rowcount = next(self._update_rowcounts, 1)
        else:
            self.rowcount = 1

    def fetchone(self):
        return self._fetchone_value


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _ConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return None


def _bare_provider(connection):
    provider = OracleProvider.__new__(OracleProvider)
    provider.config = SimpleNamespace(schema="MEMORIZZ", user="MEMORIZZ")
    provider._get_connection = lambda: _ConnectionContext(connection)
    provider._get_table_name = lambda memory_type: (f"MEMORIZZ.{memory_type.value}")
    provider._generate_embedding_if_needed = lambda **_kwargs: None
    return provider


@pytest.mark.unit
def test_oracle_knowledge_base_honors_store_memory_id_argument():
    provider = OracleProvider.__new__(OracleProvider)
    captured = {}

    def store_knowledge_base(data):
        captured.update(data)
        return "row-1"

    provider._store_knowledge_base = store_knowledge_base
    result = provider.store(
        {"content": "Scoped policy", "user_id": "alice"},
        MemoryType.KNOWLEDGE_BASE,
        memory_id="memory-argument",
    )

    assert result == "row-1"
    assert captured["memory_id"] == "memory-argument"


@pytest.mark.unit
def test_oracle_summary_schema_and_by_id_projection_are_complete():
    oracle_root = (
        Path(__file__).parents[2] / "src" / "memorizz" / "memory_provider" / "oracle"
    )
    schema = (oracle_root / "schema_relational.sql").read_text(encoding="utf-8")
    migration = (
        oracle_root / "migrations" / "004_production_governance_050.sql"
    ).read_text(encoding="utf-8")
    for required in (
        "source_message_ids",
        "period_start",
        "period_end",
        "memory_units_count",
        "summary_id VARCHAR2(255)",
        "CREATE TABLE summary_message_links",
    ):
        assert required in schema
        assert required in migration
    assert MemoryType.SUMMARIES in _BY_ID_SPECS
    fields = {field.col for field in _BY_ID_SPECS[MemoryType.SUMMARIES][1]}
    assert {
        "summary_id",
        "source_message_ids",
        "period_start",
        "period_end",
        "memory_units_count",
        "user_id",
    }.issubset(fields)
    tool_log_fields = {field.col for field in _BY_ID_SPECS[MemoryType.TOOL_LOG][1]}
    assert "user_id" in tool_log_fields
    assert {"outcome", "outcome_details"}.issubset(tool_log_fields)


@pytest.mark.unit
def test_oracle_tool_outcome_migration_is_rerunnable_and_backfills_legacy_rows():
    migration = (
        Path(__file__).parents[2]
        / "src"
        / "memorizz"
        / "memory_provider"
        / "oracle"
        / "migrations"
        / "007_structured_tool_outcomes.sql"
    ).read_text(encoding="utf-8")

    assert "user_tab_columns" in migration
    assert "add_column_if_missing" in migration
    assert "'tool_log', 'outcome'" in migration
    assert "'tool_log', 'outcome_details'" in migration
    assert "CASE WHEN success = 1 THEN 'success' ELSE 'error' END" in migration


@pytest.mark.unit
@pytest.mark.parametrize(
    "memory_type",
    [
        MemoryType.KNOWLEDGE_BASE,
        MemoryType.SHORT_TERM_MEMORY,
        MemoryType.ENTITY_MEMORY,
    ],
)
def test_oracle_tenant_scoped_list_and_vector_projections_include_user_id(
    memory_type,
):
    list_fields = {field.col for field in _LIST_SPECS[memory_type][0]}
    vector_fields = {field.col for field in _VECTOR_SPECS[memory_type]}
    assert "user_id" in list_fields
    assert "user_id" in vector_fields


@pytest.mark.unit
@pytest.mark.parametrize(
    "memory_type",
    [MemoryType.KNOWLEDGE_BASE, MemoryType.SHORT_TERM_MEMORY],
)
def test_oracle_record_memory_types_have_raw_by_id_projections(memory_type):
    id_column, fields = _BY_ID_SPECS[memory_type]
    assert id_column == "id"
    assert {"id", "memory_id", "content", "user_id"}.issubset(
        {field.col for field in fields}
    )


@pytest.mark.unit
def test_oracle_shared_memory_has_a_complete_list_projection():
    fields, order_by = _LIST_SPECS[MemoryType.SHARED_MEMORY]
    assert {
        "id",
        "memory_id",
        "content",
        "memory_type",
        "scope",
        "owner_agent_id",
        "access_list",
        "created_at",
        "updated_at",
    }.issubset({field.col for field in fields})
    assert order_by == "created_at"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("memory_type", "store_method"),
    [
        (MemoryType.KNOWLEDGE_BASE, "_store_knowledge_base"),
        (MemoryType.SHORT_TERM_MEMORY, "_store_short_term_memory"),
    ],
)
def test_oracle_record_store_returns_physical_row_id(memory_type, store_method):
    provider = OracleProvider.__new__(OracleProvider)
    provider._generate_embedding_if_needed = lambda *args, **kwargs: None
    captured = {}

    def insert_base_row(memory_store_type, required_columns, **kwargs):
        captured.update(
            memory_store_type=memory_store_type,
            required_columns=required_columns,
            kwargs=kwargs,
        )

    provider._insert_base_row = insert_base_row
    record_id = str(uuid.uuid4())
    result = getattr(provider, store_method)(
        {
            "id": record_id,
            "memory_id": "shared-group-id",
            "content": "one physical row",
            "user_id": "alice",
        }
    )

    assert result == record_id
    assert captured["memory_store_type"] == memory_type
    assert captured["required_columns"]["id"] == uuid.UUID(record_id).bytes
    assert result != "shared-group-id"


@pytest.mark.unit
def test_oracle_knowledge_base_preserves_source_provenance():
    provider = OracleProvider.__new__(OracleProvider)
    provider._generate_embedding_if_needed = lambda *args, **kwargs: None
    captured = {}
    provider._insert_base_row = lambda *args, **kwargs: captured.update(kwargs)

    provider._store_knowledge_base(
        {
            "memory_id": "evaluation",
            "content": "A source-linked semantic event",
            "source_id": "derived",
            "parent_source_id": "cue:1",
            "linked_source_ids": ["cue:1", "cue:2"],
            "metadata": {"event_time": "2026-08-23T12:00:00"},
        }
    )

    optional = captured["optional_columns"]
    assert optional["source_id"] == "derived"
    assert optional["parent_source_id"] == "cue:1"
    assert json.loads(optional["linked_source_ids"]) == ["cue:1", "cue:2"]
    assert json.loads(optional["metadata"])["event_time"].startswith("2026-08-23")

    fields = {field.col for field in _VECTOR_SPECS[MemoryType.KNOWLEDGE_BASE]}
    assert {"source_id", "parent_source_id", "linked_source_ids", "metadata"}.issubset(
        fields
    )


@pytest.mark.unit
def test_oracle_entity_upsert_rejects_cross_tenant_entity_id():
    cursor = _Cursor(fetchone_value=("shared-memory", "user-b"))
    connection = _Connection(cursor)
    provider = _bare_provider(connection)
    provider._memory_type_has_column = lambda *_args: True

    with pytest.raises(PermissionError, match="ownership mismatch"):
        provider._store_entity_memory(
            {
                "entity_id": "entity-shared-id",
                "name": "user",
                "memory_id": "shared-memory",
                "user_id": "user-a",
            }
        )

    assert len(cursor.calls) == 1
    assert "FOR UPDATE" in cursor.calls[0][0]
    assert connection.commits == 0


@pytest.mark.unit
def test_oracle_entity_upsert_allows_existing_owner_scope():
    cursor = _Cursor(fetchone_value=("shared-memory", "user-a"))
    connection = _Connection(cursor)
    provider = _bare_provider(connection)
    provider._memory_type_has_column = lambda *_args: True

    entity_id = provider._store_entity_memory(
        {
            "entity_id": "entity-shared-id",
            "name": "user",
            "memory_id": "shared-memory",
            "user_id": "user-a",
        }
    )

    assert entity_id == "entity-shared-id"
    assert len(cursor.calls) == 2
    assert "MERGE INTO MEMORIZZ.entity_memory" in cursor.calls[1][0]
    assert connection.commits == 1


@pytest.mark.unit
def test_oracle_entity_upsert_recovers_same_scope_concurrent_insert():
    class ConcurrentCursor(_Cursor):
        def __init__(self):
            super().__init__()
            self._scopes = iter([None, ("shared-memory", "user-a")])
            self._merge_count = 0

        def execute(self, sql, params=None):
            super().execute(sql, params)
            if "MERGE INTO" in str(sql).upper():
                self._merge_count += 1
                if self._merge_count == 1:
                    raise RuntimeError("ORA-00001: unique constraint violated")

        def fetchone(self):
            return next(self._scopes)

    cursor = ConcurrentCursor()
    connection = _Connection(cursor)
    provider = _bare_provider(connection)
    provider._memory_type_has_column = lambda *_args: True

    entity_id = provider._store_entity_memory(
        {
            "entity_id": "entity-shared-id",
            "name": "user",
            "memory_id": "shared-memory",
            "user_id": "user-a",
        }
    )

    assert entity_id == "entity-shared-id"
    assert ["FOR UPDATE" in sql for sql, _params in cursor.calls].count(True) == 2
    assert ["MERGE INTO" in sql for sql, _params in cursor.calls].count(True) == 2
    assert connection.commits == 1


@pytest.mark.unit
def test_oracle_summary_creation_marks_and_links_messages_atomically():
    cursor = _Cursor(update_rowcounts=[1, 1])
    connection = _Connection(cursor)
    provider = _bare_provider(connection)
    message_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    summary_id = provider.store_summary_with_links(
        {
            "summary_id": "summary-1",
            "content": "A lossless compacted interval",
            "source_message_ids": message_ids,
            "period_start": 10.0,
            "period_end": 20.0,
            "memory_units_count": 2,
            "memory_id": "memory-1",
            "agent_id": "agent-1",
            "user_id": "alice",
        }
    )

    assert summary_id == "summary-1"
    assert connection.commits == 1
    assert connection.rollbacks == 0
    summary_insert = cursor.calls[0]
    assert "INSERT INTO MEMORIZZ.summaries" in summary_insert[0]
    assert json.loads(summary_insert[1]["source_message_ids"]) == message_ids
    updates = [call for call in cursor.calls if call[0].startswith("UPDATE")]
    links = [
        call
        for call in cursor.calls
        if "INSERT INTO MEMORIZZ.summary_message_links" in call[0]
    ]
    assert len(updates) == 2
    assert len(links) == 2
    assert all("summary_id IS NULL" in sql for sql, _ in updates)
    assert all(params["summary_user_id"] == "alice" for _, params in updates)


@pytest.mark.unit
def test_oracle_summary_creation_rolls_back_if_a_source_is_out_of_scope():
    cursor = _Cursor(update_rowcounts=[0])
    connection = _Connection(cursor)
    provider = _bare_provider(connection)
    with pytest.raises(ValueError, match="out of scope"):
        provider.store_summary_with_links(
            {
                "content": "must roll back",
                "source_message_ids": [str(uuid.uuid4())],
                "memory_id": "memory-1",
                "user_id": "alice",
            }
        )
    assert connection.commits == 0
    assert connection.rollbacks == 1


@pytest.mark.unit
def test_oracle_memagent_tool_upsert_preserves_complete_json_schema():
    cursor = _Cursor()
    connection = _Connection(cursor)
    provider = _bare_provider(connection)
    nested_schema = {
        "type": "object",
        "properties": {
            "calendar": {
                "type": "object",
                "properties": {
                    "visibility": {
                        "type": "string",
                        "enum": ["private", "public"],
                        "default": "private",
                    }
                },
                "required": ["visibility"],
                "additionalProperties": False,
            }
        },
        "required": ["calendar"],
        "additionalProperties": False,
    }
    provider._upsert_toolbox_row(
        cursor,
        tool_id="agent:create_event",
        name="create_event",
        description="Create an event",
        signature="(calendar)",
        docstring="Create an event",
        tool_type="function",
        memory_id=None,
        agent_id="agent-1",
        embedding=None,
        parameters=nested_schema["properties"],
        required=nested_schema["required"],
        input_schema=nested_schema,
        tool_policy={"side_effects": True, "requires_approval": True},
        aliases=["new_event"],
        deprecated_arguments={"cal": "calendar"},
        queries=["schedule a meeting"],
        import_reference="trusted_tools:create_event",
        user_id="alice",
    )
    insert = cursor.calls[-1]
    persisted_schema = json.loads(insert[1]["input_schema"])
    assert persisted_schema == nested_schema
    assert json.loads(insert[1]["tool_policy"])["requires_approval"] is True
    assert json.loads(insert[1]["aliases"]) == ["new_event"]
    assert insert[1]["import_reference"] == "trusted_tools:create_event"
    assert insert[1]["user_id"] == "alice"


class _SummaryProvider(MockMemoryProvider):
    def __init__(self, documents):
        super().__init__()
        self.documents = documents

    def retrieve_by_id(self, unit_id, memory_store_type=None):
        return self.documents.get((memory_store_type, unit_id))


@pytest.mark.unit
def test_expand_summary_reconstructs_original_messages_in_order():
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    provider = _SummaryProvider(
        {
            (MemoryType.SUMMARIES, "summary-1"): {
                "summary_id": "summary-1",
                "content": "User asked, assistant answered.",
                "source_message_ids": [first, second],
                "period_start": 10.0,
                "period_end": 20.0,
                "memory_units_count": 2,
                "user_id": "alice",
            },
            (MemoryType.CONVERSATION_MEMORY, first): {
                "_id": first,
                "role": "user",
                "content": "Original question",
                "user_id": "alice",
            },
            (MemoryType.CONVERSATION_MEMORY, second): {
                "_id": second,
                "role": "assistant",
                "content": "Original answer",
                "user_id": "alice",
            },
        }
    )
    agent = MemAgent(memory_provider=provider)
    agent._current_user_id = "alice"
    result, _ = agent.tool_manager.execute_tool(
        "expand_summary", {"summary_id": "summary-1"}
    )
    assert result["memory_units_count"] == 2
    assert result["original_messages"] == [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "Original answer"},
    ]


@pytest.mark.unit
def test_oracle_vector_index_policy_defaults_to_lazy_and_supports_all_modes():
    default = OracleConfig("user", "password", "dsn")
    assert default.index_policy == "lazy"
    assert default.lazy_vector_indexes is True
    assert OracleConfig("u", "p", "d", index_policy="none").index_policy == "none"
    selected = OracleConfig(
        "u",
        "p",
        "d",
        index_policy="selected",
        selected_vector_indexes=["conversation_memory"],
    )
    assert selected.selected_vector_indexes == {MemoryType.CONVERSATION_MEMORY}
    assert OracleConfig("u", "p", "d", index_policy="eager").index_policy == "eager"
    with pytest.raises(ValueError, match="index_policy"):
        OracleConfig("u", "p", "d", index_policy="surprise")


@pytest.mark.unit
def test_oracle_semantic_cache_refresh_preserves_scope_and_governance_metadata():
    provider = OracleProvider.__new__(OracleProvider)
    provider._generate_embedding_if_needed = lambda **_kwargs: [0.5, 0.25]
    captured = {}

    def refresh(cache_key, payload):
        captured["cache_key"] = cache_key
        captured["payload"] = payload
        return True

    provider._update_semantic_cache_by_key = refresh
    provider._insert_base_row = lambda *_args, **_kwargs: pytest.fail(
        "an existing deterministic cache key must be refreshed, not inserted"
    )

    result = provider._store_semantic_cache(
        {
            "cache_key": "cache-1",
            "query_text": "available inventory",
            "response": "12 units",
            "agent_id": "agent-1",
            "memory_id": "memory-1",
            "session_id": "session-1",
            "user_id": "alice",
            "scope": "local",
            "similarity_threshold": 0.91,
            "hit_count": 4,
            "metadata": {
                "domain": "inventory",
                "fingerprints": {"data_version": "inventory-v4"},
            },
        }
    )

    assert result == "cache-1"
    assert captured["cache_key"] == "cache-1"
    payload = captured["payload"]
    assert payload["agent_id"] == "agent-1"
    assert payload["memory_id"] == "memory-1"
    assert payload["session_id"] == "session-1"
    assert payload["user_id"] == "alice"
    assert payload["scope"] == "local"
    assert payload["similarity_threshold"] == pytest.approx(0.91)
    assert payload["hit_count"] == 4
    assert json.loads(payload["metadata"])["domain"] == "inventory"
    assert payload["embedding"] == [0.5, 0.25]


@pytest.mark.unit
def test_oracle_semantic_cache_resolves_concurrent_insert_as_upsert():
    provider = OracleProvider.__new__(OracleProvider)
    provider._generate_embedding_if_needed = lambda **_kwargs: None
    refresh_results = iter([False, True])
    refresh_calls = []
    provider._update_semantic_cache_by_key = lambda key, payload: (
        refresh_calls.append((key, payload)) or next(refresh_results)
    )

    def concurrent_insert(*_args, **_kwargs):
        raise RuntimeError("ORA-00001: unique constraint violated")

    provider._insert_base_row = concurrent_insert

    assert (
        provider._store_semantic_cache(
            {
                "cache_key": "cache-race",
                "query_text": "same query",
                "response": "latest answer",
            }
        )
        == "cache-race"
    )
    assert len(refresh_calls) == 2
