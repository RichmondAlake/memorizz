"""Generative-Agents retrieval scoring: recency since last access, importance, relevance."""

from __future__ import annotations

import time

import pytest

from memorizz.memagent.utils.context_dedup import RetrievalScoring, dedupe_and_select

HOUR = 3600.0
QUERY = [1.0, 0.0, 0.0]
EMB_A = [0.8, 0.6, 0.0]  # cosine 0.8 to QUERY
EMB_B = [0.8, 0.0, 0.6]  # cosine 0.8 to QUERY, 0.64 to EMB_A (not a near-duplicate)


def _row(
    id_,
    text,
    *,
    created_hours_ago,
    accessed_hours_ago=None,
    importance=None,
    embedding=None,
):
    now = time.time()
    row = {
        "_id": id_,
        "content": text,
        "timestamp": now - created_hours_ago * HOUR,
        "embedding": embedding or EMB_A,
    }
    if accessed_hours_ago is not None:
        row["last_accessed_at"] = now - accessed_hours_ago * HOUR
    if importance is not None:
        row["importance"] = importance
    return row


@pytest.mark.unit
def test_recency_is_anchored_on_last_access_not_creation():
    old_but_used = _row(
        "used", "fact used recently", created_hours_ago=24 * 30, accessed_hours_ago=1
    )
    newer_unused = _row(
        "unused", "fact never recalled", created_hours_ago=24, embedding=EMB_B
    )
    rows = [("episodic", old_but_used), ("episodic", newer_unused)]

    selected = dedupe_and_select(
        rows,
        query_embedding=QUERY,
        max_items=1,
        scoring=RetrievalScoring(recency_anchor="last_accessed"),
    )
    assert [item["id"] for item in selected] == ["used"]

    selected = dedupe_and_select(
        rows,
        query_embedding=QUERY,
        max_items=1,
        scoring=RetrievalScoring(recency_anchor="created"),
    )
    assert [item["id"] for item in selected] == ["unused"]


@pytest.mark.unit
def test_importance_breaks_ties_between_equally_relevant_memories():
    rows = [
        (
            "episodic",
            _row("trivial", "weather small talk", created_hours_ago=2, importance=0.1),
        ),
        (
            "episodic",
            _row(
                "key",
                "release owner changed",
                created_hours_ago=2,
                importance=0.9,
                embedding=EMB_B,
            ),
        ),
    ]
    selected = dedupe_and_select(rows, query_embedding=QUERY, max_items=1)
    assert [item["id"] for item in selected] == ["key"]
    parts = selected[0]["scoring"]
    assert set(parts) >= {"recency", "importance", "relevance", "score"}
    assert 0.0 <= parts["recency"] <= 1.0 and 0.0 <= parts["importance"] <= 1.0


@pytest.mark.unit
def test_missing_importance_uses_the_configured_default():
    rows = [
        ("episodic", _row("plain", "no rating", created_hours_ago=2)),
        (
            "episodic",
            _row(
                "rated",
                "rated low",
                created_hours_ago=2,
                importance=0.2,
                embedding=EMB_B,
            ),
        ),
    ]
    selected = dedupe_and_select(
        rows,
        query_embedding=QUERY,
        max_items=1,
        scoring=RetrievalScoring(default_importance=0.5),
    )
    assert [item["id"] for item in selected] == ["plain"]


@pytest.mark.unit
def test_weights_can_turn_off_recency():
    rows = [
        (
            "episodic",
            _row(
                "old_relevant",
                "exact answer",
                created_hours_ago=24 * 10,
                embedding=[1.0, 0.0, 0.0],
            ),
        ),
        (
            "episodic",
            _row(
                "new_vague",
                "loosely related",
                created_hours_ago=1,
                embedding=[0.6, 0.8, 0.0],
            ),
        ),
    ]
    selected = dedupe_and_select(
        rows,
        query_embedding=QUERY,
        max_items=1,
        scoring=RetrievalScoring(
            alpha_recency=0.0, alpha_importance=0.0, alpha_relevance=1.0
        ),
    )
    assert [item["id"] for item in selected] == ["old_relevant"]


@pytest.mark.unit
def test_scoring_applies_without_a_query_embedding_too():
    rows = [
        ("episodic", _row("stale", "same provider score", created_hours_ago=24 * 20)),
        (
            "episodic",
            _row("fresh", "same provider score", created_hours_ago=1, embedding=EMB_B),
        ),
    ]
    for row in rows:
        row[1]["score"] = 0.5
        row[1]["content"] = row[1]["content"] + " " + row[1]["_id"]
    selected = dedupe_and_select(rows, max_items=1)
    assert [item["id"] for item in selected] == ["fresh"]


@pytest.mark.unit
def test_scoring_from_env_and_mapping(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_RECENCY_DECAY_PER_HOUR", "0.9")
    monkeypatch.setenv("MEMORIZZ_RECENCY_ANCHOR", "created")
    monkeypatch.setenv("MEMORIZZ_ALPHA_IMPORTANCE", "2")
    scoring = RetrievalScoring.from_env()
    assert scoring.recency_decay_per_hour == 0.9
    assert scoring.recency_anchor == "created"
    assert scoring.alpha_importance == 2.0
    merged = RetrievalScoring.from_mapping(
        {"alpha_relevance": "0.5", "recency_anchor": "bogus"}, base=scoring
    )
    assert merged.alpha_relevance == 0.5
    assert merged.recency_anchor == "created"  # invalid values keep the base
    assert merged.alpha_importance == 2.0
