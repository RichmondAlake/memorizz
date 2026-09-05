"""Explicit disposable-target gates; never discover or start host services.

Set MEMORIZZ_OBSERVABILITY_LIVE=1 and the documented isolated endpoints.
Mongo uses a generated database. Oracle requires a dedicated, empty schema
named MEMORIZZ_OBS_TEST_*. Only the objects created by this fixture are removed.
"""

import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy, deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from memorizz.observability.mongo_index import MongoSpanIndex
from memorizz.observability.sql_index import TABLES, OracleSpanIndex


@pytest.fixture(params=["mongodb", "oracle"])
def live_index(request):
    if os.getenv("MEMORIZZ_OBSERVABILITY_LIVE") != "1":
        pytest.skip(
            "Live observability validation requires explicitly supplied disposable targets"
        )
    if request.param == "mongodb":
        uri = os.getenv("MEMORIZZ_OBS_TEST_MONGODB_URI")
        if not uri:
            pytest.fail("MEMORIZZ_OBS_TEST_MONGODB_URI is required for the live gate")
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        name = "memorizz_obs_test_" + uuid.uuid4().hex
        try:
            client.admin.command("ping")
            yield MongoSpanIndex(client[name])
        finally:
            client.drop_database(name)
            client.close()
    else:
        import oracledb

        user = os.getenv("MEMORIZZ_OBS_TEST_ORACLE_USER", "")
        if not re.fullmatch(r"MEMORIZZ_OBS_TEST_[A-Z0-9_]{1,80}", user.upper()):
            pytest.fail("Supply a dedicated MEMORIZZ_OBS_TEST_* Oracle schema")
        dsn = os.getenv("MEMORIZZ_OBS_TEST_ORACLE_DSN")
        password = os.getenv("MEMORIZZ_OBS_TEST_ORACLE_PASSWORD")
        if not dsn or not password:
            pytest.fail("Explicit Oracle test DSN and password are required")

        # Match the production provider's pooled connection contract. Creating
        # a new dedicated server for every indexed read overwhelms a small Free
        # instance's listener registration during concurrent retry tests.
        pool = oracledb.create_pool(
            user=user, password=password, dsn=dsn, min=1, max=4, increment=1
        )
        request.addfinalizer(lambda: pool.close(force=True))

        @contextmanager
        def connection():
            with pool.acquire() as conn:
                yield conn

        with connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user_tables")
            if cursor.fetchone()[0]:
                pytest.fail(
                    "Oracle test schema must be empty; existing data will not be touched"
                )
        index = OracleSpanIndex(
            SimpleNamespace(
                config=SimpleNamespace(user=user, schema=user),
                _get_connection=connection,
            )
        )
        try:
            yield index
        finally:
            with connection() as conn:
                cursor = conn.cursor()
                for table in TABLES:
                    try:
                        cursor.execute(f"DROP TABLE {index.table(table)} PURGE")
                    except oracledb.DatabaseError as exc:
                        if getattr(exc.args[0], "code", None) != 942:
                            raise


def test_real_database_native_index_contract(live_index):
    index = live_index
    assert not index.ready()
    index.initialize()
    index.initialize()
    for user in ("alice", "bob"):
        for number, size in enumerate((12, 36, 26)):
            payload = {
                "record_id": f"bundle-{user}-{number}",
                "record_type": "observability_trace_bundle",
                "type": "trace_bundle",
                "version": 2,
                "agent_id": "agent",
                "memory_id": "memory",
                "thread_id": "thread",
                "user_id": user,
                "root_trace_id": f"root-{number}",
                "turn_id": "turn",
                "run_id": "run",
                "events": [
                    {
                        "event_id": f"e-{number}-{i}",
                        "trace_kind": "tool_call" if i < 4 else "model_result",
                        "timestamp": "2026-09-04T00:00:00Z",
                        "content": "PRIVATE PREVIEW",
                    }
                    for i in range(size)
                ],
            }
            index.write_bundle(payload)
            index.write_bundle(payload)
    rows, cursor = [], None
    while True:
        page = index.query(
            user_id="alice",
            agent_ids=["agent"],
            memory_ids=["unrelated"],
            limit=7,
            cursor=cursor,
        )
        rows.extend(page["items"])
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert len(rows) == len({row["event_id"] for row in rows}) == 74
    assert all(row["user_id"] == "alice" for row in rows)
    assert "PRIVATE" not in str(rows)
    assert index.summaries(user_id="alice")[0]["bundle_count"] == 3
    assert index.preview("e-0-0", user_id="alice") == "PRIVATE PREVIEW"
    assert index.preview("e-0-0", user_id="unauthorized") is None
    assert index.retention()["dry_run"] is True


def _payload(number=0, size=12, timestamp="2026-09-04T00:00:00Z"):
    return {
        "record_id": f"lifecycle-{number}",
        "record_type": "observability_trace_bundle",
        "type": "trace_bundle",
        "version": 2,
        "application_id": "application",
        "agent_id": "agent",
        "memory_id": "memory",
        "thread_id": "thread",
        "user_id": "alice",
        "root_trace_id": "root",
        "turn_id": str(number),
        "run_id": "run",
        "timestamp": timestamp,
        "events": [
            {
                "event_id": f"event-{number}-{i}",
                "trace_kind": "tool_call",
                "status": "success",
                "timestamp": timestamp,
                "content": "PRIVATE PREVIEW",
            }
            for i in range(size)
        ],
    }


def test_real_database_concurrent_retries_and_scoped_cursor(live_index):
    index = live_index
    index.initialize()
    payload = _payload()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(index.write_bundle, [deepcopy(payload) for _ in range(8)])
        )
    assert all(result["event_count"] == 12 for result in results)
    assert index.pending_count() == 0
    assert (
        index.query(application_id="application", user_id="alice")["coverage"]
        == "complete"
    )
    first = index.query(application_id="application", user_id="alice", limit=3)
    assert first["coverage"] == "partial"  # The page deliberately excludes nine events.
    assert first["normalization_errors"] == 0
    index.write_bundle(_payload(1, timestamp="2026-09-05T00:00:00Z"))
    following = index.query(
        application_id="application", user_id="alice", cursor=first["next_cursor"]
    )
    assert len(following["items"]) == 9
    assert not {e["event_id"] for e in first["items"]} & {
        e["event_id"] for e in following["items"]
    }
    for changed_scope in (
        {"application_id": "other", "user_id": "alice"},
        {"application_id": "application", "user_id": "bob"},
    ):
        with pytest.raises(ValueError, match="scope"):
            index.query(cursor=first["next_cursor"], **changed_scope)
    assert index.summaries(user_id="alice")[0]["event_count"] == 24


def test_real_database_partial_write_recovery(live_index):
    index = live_index
    index.initialize()
    payload = _payload()
    # Fail after a real child write, not before entering the persistence path.
    target, method = (
        (index.spans, "replace_one")
        if isinstance(index, MongoSpanIndex)
        else (index, "_insert")
    )
    original = getattr(target, method)
    writes = 0

    def interrupted(*args, **kwargs):
        nonlocal writes
        if isinstance(index, MongoSpanIndex) or args[1] == "obs_spans":
            writes += 1
            if writes == 2:
                raise RuntimeError("synthetic child-write interruption")
        return original(*args, **kwargs)

    with patch.object(target, method, side_effect=interrupted):
        with pytest.raises(RuntimeError, match="synthetic"):
            index.write_bundle(payload)
    # A separate reader must detect the durable checkpoint without relying on
    # the writer's process-local failed-bundle set.
    reader = copy(index)
    reader._failed_bundles = set()
    assert reader.pending_count() == 1
    assert reader.query()["coverage"] == "untrusted"
    reader.write_bundle(payload)
    assert reader.pending_count() == 0
    page = reader.query()
    assert page["coverage"] == "complete"
    assert len(page["items"]) == len({e["event_id"] for e in page["items"]}) == 12


def test_real_database_retention_and_typed_resource_lookup(live_index):
    index = live_index
    index.initialize()
    payload = _payload(timestamp="2026-01-01T00:00:00Z")
    payload["events"][0] = {
        "schema_version": 3,
        "event_id": "verified",
        "span_id": "verified-span",
        "event_kind": "verified_outcome",
        "operation": "task.outcome",
        "timestamp": payload["timestamp"],
        "status": "success",
        "input_refs": [{"resource_type": "analysis", "ref": "analysis-1"}],
        "attributes": {"verified": True},
    }
    index.write_bundle(payload)
    assert len(index.query(resource_refs=["analysis-1"], user_id="alice")["items"]) == 1
    assert not index.query(resource_refs=["analysis-1"], user_id="bob")["items"]
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    plan = index.retention(now=now)
    assert plan["dry_run"] and plan["metadata_events"] == 11 and plan["previews"] == 11
    assert len(index.query()["items"]) == 12
    assert index.preview("event-0-1", user_id="alice") == "PRIVATE PREVIEW"
    applied = index.retention(now=now, dry_run=False)
    assert applied["source_bundles_unchanged"]
    assert [e["event_id"] for e in index.query()["items"]] == ["verified"]
    assert index.preview("event-0-1", user_id="alice") is None
    assert index.summaries(user_id="alice")[0]["event_count"] == 1
    assert index.retention(now=now, dry_run=False)["metadata_events"] == 0
