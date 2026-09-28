# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Batched trace reads return exactly what record-at-a-time reads returned."""

import json

import pytest

from memorizz.enums import MemoryType
from memorizz.observability.normalization import query_trace_events


def _row(i, events=3):
    return {
        "_id": f"r{i:03d}",
        "memory_id": "m",
        "agent_id": "a",
        "role": "tool",
        "timestamp": f"2026-09-28T00:{i:02d}:00Z",
        "content": json.dumps(
            {
                "type": "trace_bundle",
                "memory_id": "m",
                "agent_id": "a",
                "thread_id": "t",
                "events": [
                    {
                        "trace_kind": "tool_call",
                        "title": f"e{i}-{k}",
                        "content": "",
                        "trace_id": f"b{i}-e{k}",
                        "turn_id": f"turn{i}",
                        "root_trace_id": f"root{i}",
                        "span_id": f"s{i}-{k}",
                        "timestamp": f"2026-09-28T00:{i:02d}:{k:02d}Z",
                    }
                    for k in range(events)
                ],
            }
        ),
    }


class _PlainProvider:
    """Newest first, cursor = last row id, like the MongoDB provider."""

    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda row: row["_id"], reverse=True)
        self.queries = 0

    def query_observability_records(self, memory_type, *, limit, cursor=None, **_):
        self.queries += 1
        start = 0
        if cursor:
            start = (
                next(i for i, row in enumerate(self.rows) if row["_id"] == cursor) + 1
            )
        page = self.rows[start : start + limit]
        more = start + limit < len(self.rows)
        return {
            "items": page,
            "next_cursor": page[-1]["_id"] if more and page else None,
        }


class _BatchedProvider(_PlainProvider):
    def observability_row_cursor(self, row):
        return row["_id"]


def _page_all(provider, limit):
    ids, cursor = [], None
    while True:
        page = query_trace_events(
            provider, MemoryType.CONVERSATION_MEMORY, limit=limit, cursor=cursor
        )
        ids += [event["event_id"] for event in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            return ids


@pytest.mark.unit
def test_batched_paging_matches_record_at_a_time_paging():
    rows = [_row(i) for i in range(7)]
    plain, batched = _PlainProvider(rows), _BatchedProvider(rows)
    # Pages of 4 events split the 3-event bundles mid-way, exercising resume.
    expected = _page_all(plain, limit=4)
    assert len(expected) == 21 and len(set(expected)) == 21
    assert _page_all(batched, limit=4) == expected
    assert batched.queries < plain.queries


@pytest.mark.unit
def test_batched_resume_survives_a_record_inserted_between_pages():
    rows = [_row(i) for i in range(1, 6)]
    provider = _BatchedProvider(rows)
    first = query_trace_events(provider, MemoryType.CONVERSATION_MEMORY, limit=4)
    # A record that sorts before the resume point must not shift the position.
    provider.rows = sorted(
        provider.rows + [_row(0)], key=lambda row: row["_id"], reverse=True
    )
    second = query_trace_events(
        provider,
        MemoryType.CONVERSATION_MEMORY,
        limit=100,
        cursor=first["next_cursor"],
    )
    seen = [e["event_id"] for e in first["items"] + second["items"]]
    assert len(seen) == len(set(seen)), "no event repeats"
    assert len(seen) == 18, "the older inserted bundle is read once, nothing skipped"
