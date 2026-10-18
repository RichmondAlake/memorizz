"""UTC stamps on write; legacy offset-less values read by their writer's convention.

Every test that reads a stored time runs under non-UTC local zones (UTC-7 and
UTC+9 in October), so reading local time as UTC (or UTC as local) shows up as
an hours-wide error instead of passing by accident on a UTC machine.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from memorizz._time import to_epoch, to_utc
from memorizz.enums import MemoryType, Role
from memorizz.observability import TraceSnapshot, normalize_trace_snapshot
from memorizz.observability.normalization import select_trace_events

pytestmark = pytest.mark.unit


@pytest.fixture(params=["America/Los_Angeles", "Asia/Tokyo"])
def local_tz(request):
    previous = os.environ.get("TZ")
    os.environ["TZ"] = request.param
    time.tzset()
    try:
        yield request.param
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def legacy_local(moment: datetime) -> str:
    """``moment`` as ``datetime.now().isoformat()`` would have written it."""
    return moment.astimezone().replace(tzinfo=None).isoformat()


def legacy_utc(moment: datetime) -> str:
    """``moment`` as ``datetime.utcnow().isoformat()`` would have written it."""
    return moment.astimezone(timezone.utc).replace(tzinfo=None).isoformat()


def _has_utc_offset(value) -> bool:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    return parsed.utcoffset() == timedelta(0)


class _CapturingProvider:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.stored = []

    def store(self, data, memory_store_type=None, **_):
        self.stored.append((memory_store_type, dict(data)))
        return f"id-{len(self.stored)}"

    def list_all(self, memory_type=None, **_):
        return list(self.rows)


def test_to_utc_reads_z_suffix():
    assert to_utc("2026-10-18T09:00:00Z") == datetime(
        2026, 10, 18, 9, tzinfo=timezone.utc
    )


# --------------------------------------------------------------------- D1


def test_conversation_and_tool_log_writers_stamp_utc(local_tz):
    from memorizz.memagent.managers.memory_manager import MemoryManager

    provider = _CapturingProvider()
    manager = MemoryManager(provider)
    unit = manager.create_conversation_memory_unit(
        role=Role.USER, content="hi", thread_id="t", memory_id="m"
    )
    assert _has_utc_offset(unit.timestamp)
    manager.store_tool_log("lookup", {}, "ok", memory_id="m")
    assert _has_utc_offset(provider.stored[-1][1]["timestamp"])


def _bundle(at: str):
    event = {
        "trace_kind": "tool_call",
        "trace_id": "call-1",
        "span_id": "span-1",
        "timestamp": at,
        "logical_tool_name": "lookup",
    }
    return {
        "_id": "bundle-1",
        "record_type": "observability_trace_bundle",
        "agent_id": "agent",
        "memory_id": "m",
        "thread_id": "t",
        "root_trace_id": "root-1",
        "timestamp": at,
        "content": json.dumps(
            {"type": "trace_bundle", "version": 2, "events": [event]}
        ),
    }


def test_trace_timeline_reads_legacy_conversation_rows_as_local(local_tz):
    asked = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
    conversation = {
        "_id": "turn-1",
        "agent_id": "agent",
        "memory_id": "m",
        "thread_id": "t",
        "role": "user",
        "content": "look it up",
        "timestamp": legacy_local(asked),
    }
    tool_log = {
        "_id": "log-1",
        "agent_id": "agent",
        "memory_id": "m",
        "thread_id": "t",
        "tool_name": "lookup",
        "success": True,
        "timestamp": legacy_local(asked + timedelta(seconds=10)),
    }
    window = normalize_trace_snapshot(
        TraceSnapshot(
            conversation_rows=[conversation],
            tool_rows=[tool_log],
            bundle_rows=[_bundle("2026-10-18T09:00:30Z")],
        )
    )
    by_kind = {event["kind"]: event for event in window.events}
    assert by_kind["conversation"]["timestamp"] == "2026-10-18T09:00:00.000000+00:00"
    assert by_kind["execution_log"]["timestamp"] == "2026-10-18T09:00:10.000000+00:00"
    kinds = [event["kind"] for event in window.events]
    assert kinds == ["conversation", "execution_log", "tool_call"]

    selected = select_trace_events(window.events, start_time="2026-10-18T08:59:00Z")
    assert [event["kind"] for event in selected] == kinds


# --------------------------------------------------------------------- D2


def test_entity_and_filesystem_default_stamps_are_utc(tmp_path):
    from memorizz.long_term.semantic.entity_memory import EntityMemory
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    assert _has_utc_offset(EntityMemory._timestamp())
    provider = FileSystemProvider(FileSystemConfig(root_path=tmp_path / "fs"))
    assert _has_utc_offset(provider._prepare_document({"content": "x"})["timestamp"])


def test_recency_reads_each_source_by_its_writer(local_tz):
    from memorizz.memagent.utils.context_dedup import _candidate_timestamp

    written = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)
    expected = written.timestamp()
    entity = {"created_at": legacy_utc(written)}
    turn = {"timestamp": legacy_local(written)}
    # Filesystem KB chunk: default ``timestamp`` (utcnow) beside a local created_at.
    chunk = {"created_at": legacy_local(written)}
    fs_chunk = {"timestamp": legacy_utc(written), "created_at": legacy_local(written)}
    assert _candidate_timestamp(entity, "entity_memory") == pytest.approx(expected)
    assert _candidate_timestamp(turn, "episodic") == pytest.approx(expected)
    assert _candidate_timestamp(chunk, "knowledge_base") == pytest.approx(expected)
    assert _candidate_timestamp(fs_chunk, "knowledge_base") == pytest.approx(expected)
    aware = {"timestamp": written.isoformat()}
    assert _candidate_timestamp(aware, "episodic") == pytest.approx(expected)


def test_retention_ages_legacy_entities_from_utc(local_tz):
    from memorizz.learning.retention import RetentionPlanner, _created_at

    written = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=1)
    entity = {"_id": "e1", "created_at": legacy_utc(written)}
    assert _created_at(entity, MemoryType.ENTITY_MEMORY) == pytest.approx(
        written.timestamp()
    )
    planner = RetentionPlanner(_CapturingProvider(), agent_id="agent")
    parts = planner.score(
        entity,
        max_access=0,
        now=time.time(),
        memory_type=MemoryType.ENTITY_MEMORY,
    )
    assert parts["age_days"] * 24 < 0.1
    assert parts["recency"] > 0.999


# --------------------------------------------------------------------- D3


def test_evidence_freshness_reads_conversation_rows_as_local(local_tz):
    from memorizz.learning.evidence import EvidencePlanner
    from memorizz.learning.models import LearningControlPlaneConfig

    planner = EvidencePlanner(
        _CapturingProvider(), agent_id="agent", config=LearningControlPlaneConfig()
    )
    said = datetime.now(timezone.utc) - timedelta(minutes=2)
    row = {
        "_id": "turn-1",
        "agent_id": "agent",
        "memory_id": "m",
        "thread_id": "t",
        "user_id": None,
        "content": "the release checklist lives in docs/release.md",
        "timestamp": legacy_local(said),
        "score": 0.9,
    }
    item = planner._candidate_from_row(
        query="release checklist",
        rank=1,
        source_type="conversation_memory",
        row=row,
        memory_id="m",
        user_id=None,
        thread_id="t",
        tombstones=set(),
        history_hashes=set(),
        freshness={"conversation_memory": 3600},
    )
    assert item is not None
    assert not item.stale
    assert 100 < item.age_seconds < 140
    assert _has_utc_offset(item.timestamp)


def test_knowledge_base_chunks_are_stamped_utc():
    from memorizz.long_term.semantic.knowledge_base import KnowledgeBase

    provider = _CapturingProvider()
    KnowledgeBase(provider).ingest_knowledge(
        "one short fact.", namespace="docs", embeddings="off"
    )
    _, chunk = provider.stored[0]
    assert _has_utc_offset(chunk["created_at"])


# --------------------------------------------------------------------- D6


def test_session_summary_periods_are_epoch_seconds(local_tz, monkeypatch):
    import memorizz.embeddings as embeddings
    from memorizz.episodic_capture import summarize_thread

    def no_embedder(_text):
        raise RuntimeError("no embedder in tests")

    monkeypatch.setattr(embeddings, "get_embedding", no_embedder)
    first = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
    turns = [
        # The new (UTC) turn sorts after the legacy (local) one by instant,
        # whatever their strings say.
        {
            "_id": "2",
            "memory_id": "m",
            "thread_id": "t",
            "role": "assistant",
            "content": "answer",
            "timestamp": (first + timedelta(minutes=5)).isoformat(),
        },
        {
            "_id": "1",
            "memory_id": "m",
            "thread_id": "t",
            "role": "user",
            "content": "question",
            "timestamp": legacy_local(first),
        },
    ]
    provider = _CapturingProvider(turns)

    class _Model:
        def generate(self, messages):
            return "- asked a question"

    assert summarize_thread(provider, memory_id="m", thread_id="t", model=_Model())
    _, summary = provider.stored[0]
    assert summary["period_start"] == pytest.approx(first.timestamp())
    assert summary["period_end"] == pytest.approx(first.timestamp() + 300)
    assert summary["source_message_ids"] == ["1", "2"]


def test_summary_listing_orders_legacy_iso_periods_by_time(local_tz):
    from memorizz.memagent.managers.memory_manager import MemoryManager

    newest = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
    docs = [
        {
            "_id": "older",
            "memory_id": "m",
            "content": "older",
            "period_end": (newest - timedelta(hours=1)).timestamp(),
        },
        {
            "_id": "session",
            "memory_id": "m",
            "content": "legacy session summary",
            "period_end": legacy_local(newest),
        },
    ]
    manager = MemoryManager(_CapturingProvider(docs))
    listed = manager.load_summaries_for_thread("m", limit=1)
    assert [item["summary_id"] for item in listed] == ["session"]


# --------------------------------------------------------------------- D7


def test_semantic_cache_age_reads_string_created_at(local_tz):
    from memorizz.short_term_memory.semantic_cache import SemanticCache

    cached = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=3)
    expected = cached.timestamp()
    # Legacy Oracle: ISO without an offset, in the session's (local) zone.
    legacy_oracle = {"created_at": legacy_local(cached)}
    assert SemanticCache._stored_timestamp(legacy_oracle) == pytest.approx(expected)
    assert SemanticCache._stored_timestamp(
        {"created_at": cached.isoformat()}
    ) == pytest.approx(expected)
    # pymongo hands BSON dates back naive UTC.
    mongo = {"created_at": cached.replace(tzinfo=None)}
    assert SemanticCache._stored_timestamp(mongo) == pytest.approx(expected)
    assert SemanticCache._stored_timestamp({"timestamp": expected}) == expected


def test_semantic_cache_usage_update_sends_last_accessed():
    from memorizz.memory_unit.semantic_cache_entry import SemanticCacheEntry
    from memorizz.short_term_memory.semantic_cache import SemanticCache

    sent = []
    cache = SemanticCache.__new__(SemanticCache)
    cache._update_in_memory_provider = lambda record_id, data: sent.append(data) or True
    entry = SemanticCacheEntry(
        query="q",
        response="r",
        embedding=[],
        timestamp=100.0,
        last_accessed=250.0,
        usage_count=3,
        metadata={"_id": "row-1"},
    )
    assert cache._update_usage_in_memory_provider(entry)
    assert sent == [{"hit_count": 3, "usage_count": 3, "last_accessed": 250.0}]


# ------------------------------------------------------------ also checked


def test_cron_next_run_is_never_before_the_anchor_when_clocks_go_back():
    from memorizz.automation.schedule import next_cron_time

    # 01:30 PST, the second pass through 01:xx on 2026-11-01.
    after = datetime(2026, 11, 1, 9, 30, tzinfo=timezone.utc)
    nxt = next_cron_time("*/15 * * * *", after_utc=after, tz_name="America/Los_Angeles")
    assert nxt > after


def test_cli_conversation_times_read_legacy_turns_as_local(local_tz):
    from memorizz.cli.conversations import conversation_row_parts

    said = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
    parts = conversation_row_parts(
        {"memory_id": "m", "role": "user", "timestamp": legacy_local(said)}
    )
    assert parts["timestamp"] == said


def test_tool_logs_and_mongo_turns_order_by_instant(local_tz):
    from memorizz.memory_provider.base import filter_tool_log_rows
    from memorizz.memory_provider.mongodb.provider import _turn_epoch

    legacy_at = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
    rows = [
        {"tool_log_id": "legacy", "timestamp": legacy_local(legacy_at)},
        {
            "tool_log_id": "new",
            "timestamp": (legacy_at + timedelta(minutes=1)).isoformat(),
        },
    ]
    newest_first = filter_tool_log_rows(rows, limit=0)
    assert [row["tool_log_id"] for row in newest_first] == ["new", "legacy"]
    assert sorted(rows, key=lambda row: _turn_epoch(row["timestamp"]))[0] is rows[0]


def test_workflow_and_skill_stamps_mix_legacy_and_utc(local_tz):
    from memorizz.long_term.procedural.skillbox.promotion import _since
    from memorizz.long_term.procedural.skillbox.skill import Skill
    from memorizz.long_term.procedural.workflow.canonicalization import (
        aggregate_trajectory_stats,
    )

    ran = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=1)
    docs = [
        {
            "workflow_id": "w1",
            "canonical_hash": "h",
            "outcome": "success",
            "user_query": "a",
            "created_at": legacy_local(ran),
        },
        {
            "workflow_id": "w2",
            "canonical_hash": "h",
            "outcome": "success",
            "user_query": "b",
            "created_at": (ran + timedelta(minutes=30)).isoformat(),
        },
    ]

    class _Docs:
        def list_all(self, memory_store_type=None, **_):
            return docs

    (stat,) = aggregate_trajectory_stats(_Docs())
    assert stat.first_seen == ran
    assert stat.last_seen == ran + timedelta(minutes=30)
    naive_local = datetime.fromisoformat(legacy_local(ran))
    for moment in (stat.first_seen, naive_local):
        assert abs(_since(moment) - timedelta(hours=1)) < timedelta(seconds=5)
    skill = Skill.from_dict(
        {"name": "s", "description": "d", "demoted_at": legacy_local(ran)}
    )
    assert skill.demoted_at == ran
    fresh = Skill(name="s", description="d", content="c")
    assert _has_utc_offset(fresh.created_at)
    assert to_epoch(fresh.created_at) == pytest.approx(time.time(), abs=5)
