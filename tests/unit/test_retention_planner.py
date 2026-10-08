"""Governed suppression of primary memories: plan, approve, honour, reverse."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from memorizz import FileSystemConfig, FileSystemProvider
from memorizz.enums import MemoryType
from memorizz.learning.retention import RetentionConfig, RetentionPlanner
from memorizz.memagent.utils.context_dedup import RetrievalScoring, dedupe_and_select

DAY = 86_400.0


@pytest.fixture()
def provider(tmp_path):
    return FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "retention",
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )


def _store(provider, record_id, text, *, days_old, **fields):
    row = {
        "_id": record_id,
        "content": text,
        "name": record_id,
        "agent_id": "agent-1",
        "memory_id": "kb-1",
        "timestamp": time.time() - days_old * DAY,
        "embedding": [1.0, 0.0],
    }
    row.update(fields)
    provider.store(row, MemoryType.KNOWLEDGE_BASE)
    return record_id


@pytest.fixture()
def planner(provider):
    config = RetentionConfig(
        enabled=True, min_retention=0.3, grace_days=30, memory_types=["knowledge_base"]
    )
    return RetentionPlanner(
        provider, agent_id="agent-1", config=config, scoring=RetrievalScoring()
    )


@pytest.mark.unit
def test_plan_selects_only_old_unused_unimportant_unprotected_records(
    provider, planner
):
    _store(provider, "stale", "forgotten detail", days_old=120, importance=0.1)
    _store(
        provider, "pinned", "keep forever", days_old=120, importance=0.1, pinned=True
    )
    _store(
        provider,
        "verified",
        "proven outcome",
        days_old=120,
        importance=0.1,
        verified=True,
    )
    _store(provider, "recent", "new but dull", days_old=5, importance=0.1)
    _store(
        provider,
        "used",
        "recalled often",
        days_old=120,
        importance=0.1,
        last_accessed_at=time.time() - DAY,
        access_count=20,
    )
    _store(provider, "cited", "source of a decision", days_old=120, importance=0.1)
    _store(
        provider,
        "decision",
        "derived decision",
        days_old=10,
        importance=0.9,
        source_ids=["cited"],
    )

    report = planner.plan()
    assert report.dry_run is True
    assert [c.target_id for c in report.candidates] == ["stale"]
    assert report.candidates[0].target_type == "knowledge_base"
    assert "retention" in report.candidates[0].reason
    assert report.retained == 6


@pytest.mark.unit
def test_apply_requires_an_approver_and_suppresses_reversibly(provider, planner):
    _store(provider, "stale", "forgotten detail", days_old=120, importance=0.1)
    report = planner.plan()
    with pytest.raises(ValueError):
        planner.apply(report, approved_by="")

    applied = planner.apply(report, approved_by="operator-7", reason="quarterly tidy")
    assert applied.dry_run is False and applied.tombstoned == 1 and not applied.errors
    row = provider.retrieve_by_id("stale", MemoryType.KNOWLEDGE_BASE)
    assert row["retention_state"] == "suppressed"
    assert row["suppressed_by"] == report.plan_id
    assert row["suppression_approver"] == "operator-7"
    assert row["content"] == "forgotten detail"  # nothing deleted

    # Retrieval honours suppression ...
    selected = dedupe_and_select([("knowledge_base", row)], max_items=3)
    assert selected == []
    # ... re-planning does not list it again ...
    assert planner.plan().candidates == ()
    assert planner.suppressed()[0]["record_id"] == "stale"
    # ... and the decision can be reversed.
    assert (
        planner.unsuppress("stale", MemoryType.KNOWLEDGE_BASE, approved_by="operator-7")
        is True
    )
    row = provider.retrieve_by_id("stale", MemoryType.KNOWLEDGE_BASE)
    assert row["retention_state"] == "active"
    assert dedupe_and_select([("knowledge_base", row)], max_items=3)[0]["id"] == "stale"


@pytest.mark.unit
def test_apply_rejects_a_tampered_plan(provider, planner):
    _store(provider, "stale", "forgotten detail", days_old=120, importance=0.1)
    report = planner.plan()
    tampered = type(report)(
        plan_id="forget-plan-bogus", dry_run=True, candidates=report.candidates
    )
    with pytest.raises(ValueError):
        planner.apply(tampered, approved_by="operator-7")


@pytest.mark.unit
def test_retention_config_from_env_and_mapping(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_RETENTION_ENABLED", "true")
    monkeypatch.setenv("MEMORIZZ_RETENTION_MIN_SCORE", "0.2")
    monkeypatch.setenv("MEMORIZZ_RETENTION_GRACE_DAYS", "45")
    monkeypatch.setenv(
        "MEMORIZZ_RETENTION_MEMORY_TYPES", "knowledge_base, summaries, bogus"
    )
    config = RetentionConfig.from_env()
    assert config.enabled is True
    assert config.min_retention == 0.2 and config.grace_days == 45.0
    assert config.memory_types == ("knowledge_base", "summaries")
    merged = RetentionConfig.from_mapping({"protect_pinned": "false"}, base=config)
    assert merged.protect_pinned is False and merged.min_retention == 0.2


@pytest.mark.unit
def test_touch_many_records_recalls_without_changing_timestamps(provider):
    _store(provider, "fact", "recalled fact", days_old=10, importance=0.5)
    before = provider.retrieve_by_id("fact", MemoryType.KNOWLEDGE_BASE)
    touched = provider.touch_many(["fact", "missing"], MemoryType.KNOWLEDGE_BASE)
    after = provider.retrieve_by_id("fact", MemoryType.KNOWLEDGE_BASE)
    assert touched == 1
    assert after["access_count"] == 1 and after["last_accessed_at"]
    assert after["timestamp"] == before["timestamp"]
    provider.touch_many(["fact"], MemoryType.KNOWLEDGE_BASE)
    assert (
        provider.retrieve_by_id("fact", MemoryType.KNOWLEDGE_BASE)["access_count"] == 2
    )
