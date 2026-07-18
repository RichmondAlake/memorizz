"""Oracle workflows accept logical workflow IDs and internal RAW row IDs."""

import uuid
from datetime import datetime

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.memory_provider.oracle.provider import _LIST_SPECS, OracleProvider


class _Cursor:
    def __init__(self, fetch_rows=None):
        self.fetch_rows = list(fetch_rows or [])
        self.executions = []
        self.rowcount = 1

    def execute(self, sql, params=None):
        self.executions.append((" ".join(sql.split()), params or {}))

    def fetchone(self):
        return self.fetch_rows.pop(0) if self.fetch_rows else None

    def close(self):
        pass


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


def _provider(monkeypatch, cursor):
    provider = OracleProvider.__new__(OracleProvider)
    connection = _Connection(cursor)
    monkeypatch.setattr(provider, "_get_connection", lambda: connection)
    monkeypatch.setattr(
        provider, "_get_table_name", lambda _memory_type: "workflow_memory"
    )
    return provider, connection


@pytest.mark.unit
def test_retrieve_workflow_falls_back_to_internal_row_id(monkeypatch):
    row_id = uuid.uuid4()
    cursor = _Cursor(fetch_rows=[None, ("workflow-row",)])
    provider, _connection = _provider(monkeypatch, cursor)
    monkeypatch.setattr(
        provider,
        "_apply_row_fields",
        lambda _fields, row, *_args: {"row": row},
    )

    result = provider.retrieve_by_id(str(row_id), MemoryType.WORKFLOW_MEMORY)

    assert result == {"row": ("workflow-row",)}
    assert len(cursor.executions) == 2
    assert "WHERE workflow_id = :workflow_id" in cursor.executions[0][0]
    assert "WHERE id = :row_id" in cursor.executions[1][0]
    assert cursor.executions[1][1]["row_id"] == row_id.bytes


@pytest.mark.unit
def test_update_workflow_matches_logical_or_internal_id(monkeypatch):
    row_id = uuid.uuid4()
    cursor = _Cursor()
    provider, connection = _provider(monkeypatch, cursor)

    updated = provider.update_by_id(
        str(row_id),
        {"promoted_skill_id": "skill-1"},
        MemoryType.WORKFLOW_MEMORY,
    )

    sql, params = cursor.executions[-1]
    assert updated is True
    assert "(workflow_id = :workflow_id OR id = :row_id)" in sql
    assert params["workflow_id"] == str(row_id)
    assert params["row_id"] == row_id.bytes
    assert params["promoted_skill_id"] == "skill-1"
    assert connection.commits == 1


@pytest.mark.unit
def test_delete_workflow_matches_logical_or_internal_id(monkeypatch):
    row_id = uuid.uuid4()
    cursor = _Cursor()
    provider, connection = _provider(monkeypatch, cursor)

    deleted = provider.delete_by_id(str(row_id), MemoryType.WORKFLOW_MEMORY)

    sql, params = cursor.executions[-1]
    assert deleted is True
    assert "workflow_id = :workflow_id OR id = :row_id" in sql
    assert params == {"workflow_id": str(row_id), "row_id": row_id.bytes}
    assert connection.commits == 1


@pytest.mark.unit
def test_workflow_list_rows_emit_query_scope_and_timestamps():
    provider = OracleProvider.__new__(OracleProvider)
    fields, _order_by = _LIST_SPECS[MemoryType.WORKFLOW_MEMORY]
    created_at = datetime(2026, 7, 17, 12, 30)
    updated_at = datetime(2026, 7, 17, 12, 31)
    values = {
        "id": uuid.uuid4().bytes,
        "workflow_id": "workflow-1",
        "name": "Refund workflow",
        "steps": "{}",
        "outcome": '"success"',
        "memory_id": "memory-1",
        "agent_id": "agent-1",
        "user_id": "user-1",
        "user_query": "Refund order R-1001",
        "canonical_signature": "[]",
        "skills_activated": "[]",
        "created_at": created_at,
        "updated_at": updated_at,
    }
    row = tuple(values.get(field.col) for field in fields)

    document = provider._apply_row_fields(fields, row, include_embedding=False)

    assert document["user_id"] == "user-1"
    assert document["user_query"] == "Refund order R-1001"
    assert document["created_at"] == created_at.isoformat()
    assert document["updated_at"] == updated_at.isoformat()


@pytest.mark.unit
def test_store_workflow_includes_original_user_query(monkeypatch):
    provider = OracleProvider.__new__(OracleProvider)
    inserted = {}
    monkeypatch.setattr(
        provider,
        "_generate_embedding_if_needed",
        lambda *_args, **_kwargs: None,
    )

    def capture_insert(memory_type, required_columns, optional_columns, **_kwargs):
        inserted.update(
            {
                "memory_type": memory_type,
                "required": required_columns,
                "optional": optional_columns,
            }
        )

    monkeypatch.setattr(provider, "_insert_base_row", capture_insert)

    workflow_id = provider._store_workflow_memory(
        {
            "workflow_id": "workflow-1",
            "name": "Refund workflow",
            "description": "A refund run",
            "agent_id": "agent-1",
            "user_id": "user-1",
            "user_query": "Refund order R-1001",
        }
    )

    assert workflow_id == "workflow-1"
    assert inserted["memory_type"] == MemoryType.WORKFLOW_MEMORY
    assert inserted["optional"]["user_id"] == "user-1"
    assert inserted["optional"]["user_query"] == "Refund order R-1001"
