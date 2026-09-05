"""Bounded SQL reads and Oracle migration/driver contract without live services."""

import io
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from memorizz.observability.index import prepare_bundle
from memorizz.observability.sql_index import OracleSpanIndex, SQLiteSpanIndex


def payload(number=0, size=500):
    return {
        "record_id": f"bundle-{number}",
        "record_type": "observability_trace_bundle",
        "type": "trace_bundle",
        "version": 2,
        "agent_id": "agent",
        "memory_id": "memory",
        "user_id": "alice",
        "thread_id": "thread",
        "root_trace_id": "root",
        "turn_id": str(number),
        "run_id": "run",
        "events": [
            {
                "event_id": f"event-{number}-{i}",
                "trace_kind": "model_result",
                "timestamp": "2026-09-04T10:00:00Z",
                "content": "PRIVATE" * 300,
            }
            for i in range(size)
        ],
    }


def test_ten_thousand_event_page_has_bounded_deserialization_and_native_counts(
    tmp_path,
):
    index = SQLiteSpanIndex(tmp_path)
    index.initialize()
    for number in range(20):
        index.write_bundle(payload(number))
    with patch.object(
        index,
        "_json",
        side_effect=AssertionError("Summary must not deserialize trace payloads"),
    ):
        assert index.summaries(user_id="alice")[0]["event_count"] == 10000
    with patch(
        "memorizz.observability.index.normalize_trace_snapshot",
        side_effect=AssertionError("Native reads must not expand historical bundles"),
    ):
        first = index.query(user_id="alice", thread_id="thread", limit=250)
        assert len(first["items"]) == 250 and first["scanned_count"] == 251
        second = index.query(
            user_id="alice", thread_id="thread", limit=250, cursor=first["next_cursor"]
        )
    assert not {e["event_id"] for e in first["items"]} & {
        e["event_id"] for e in second["items"]
    }
    assert first["normalization_errors"] == 0
    with index.connection() as conn:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT event_key FROM obs_spans ORDER BY timestamp DESC,event_key DESC LIMIT 251"
        ).fetchall()
    assert "obs_time" in str(plan)


def test_index_rejects_changed_immutable_bundle_without_losing_old_evidence(tmp_path):
    index = SQLiteSpanIndex(tmp_path)
    index.initialize()
    p = payload(size=1)
    index.write_bundle(p)
    p["events"][0]["status"] = "error"
    with pytest.raises(ValueError, match="Immutable"):
        index.write_bundle(p)
    fresh = SQLiteSpanIndex(tmp_path)
    assert fresh.query()["coverage"] == "untrusted"  # Durable pending checkpoint.
    fresh.write_bundle(payload(size=1))
    assert fresh.query()["coverage"] == "complete"


def test_nonfinite_legacy_metadata_never_reaches_json_index():
    p = payload(size=1)
    p["events"][0]["duration_ms"] = float("nan")
    _, rows = prepare_bundle(p)
    assert "NaN" not in json.dumps(rows)


def oracle(cursor):
    conn = Mock()
    conn.cursor.return_value = cursor

    @contextmanager
    def connection():
        yield conn

    return (
        OracleSpanIndex(
            SimpleNamespace(
                config=SimpleNamespace(schema="OBS_TEST", user="OBS_TEST"),
                _get_connection=connection,
            )
        ),
        conn,
    )


def test_oracle_migration_is_additive_explicit_qualified_and_idempotent():
    cursor = Mock()
    cursor.fetchone.return_value = ("1",)
    index, conn = oracle(cursor)
    assert cursor.execute.call_count == 0
    assert index.initialize()["ready"]
    index.initialize()
    sql = cursor.execute.call_args_list[0].args[0]
    assert "CREATE TABLE OBS_TEST.obs_spans" in sql
    assert "__SCHEMA__" not in sql
    assert "shared_memory" not in sql.lower() and "DROP " not in sql
    assert "IF SQLCODE != -955 THEN RAISE" in sql
    assert sql.rstrip().endswith("END;")
    assert conn.commit.call_count == 2


def test_oracle_queries_are_bound_compound_and_clob_aware():
    pytest.importorskip("oracledb")
    cursor = Mock()
    cursor.fetchone.return_value = ("1",)
    cursor.fetchall.return_value = [
        ("key", "bundle", "time", io.StringIO('{"event_id":"event"}'))
    ]
    index, _ = oracle(cursor)
    rows = index.select(
        {"agent_ids": ["agent' OR 1=1 --"], "user_id": "alice"},
        boundary=["time", "key"],
        limit=251,
    )
    sql, params = cursor.execute.call_args.args
    assert "OR 1=1" not in sql and "FETCH FIRST :page_size ROWS ONLY" in sql
    assert "event_key <" in sql
    assert params["page_size"] == 251
    assert rows[0]["metadata"] == {"event_id": "event"}
    index._insert(cursor, "obs_spans", {"payload": "x" * 50000})
    assert "payload" in cursor.setinputsizes.call_args.kwargs
    with pytest.raises(ValueError):
        OracleSpanIndex(
            SimpleNamespace(config=SimpleNamespace(schema="unsafe;DROP", user="user"))
        )


def test_sqlite_read_never_creates_tables(tmp_path):
    index = SQLiteSpanIndex(tmp_path)
    assert index.capabilities()["ready"] is False
    with pytest.raises(RuntimeError):
        index.query()
    assert not (Path(tmp_path) / "_observability").exists()
