# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Monitor summaries for the Observability pages and the shared figure macros."""

from pathlib import Path
from unittest.mock import patch

import pytest
from jinja2 import Environment, FileSystemLoader

from memorizz.ui.observability_view import (
    compare_tape,
    exact_usd,
    health_tape,
    iso_timestamp,
    navigator_counts,
    navigator_rows,
    navigator_tape,
    relative_age,
    usage_tape,
)

NOW = 1_800_000_000.0
TEMPLATES = Path(__file__).parents[2] / "src/memorizz/ui/templates"


def _by_label(items):
    return {entry["label"]: entry for entry in items}


def _agent(agent_id, events, threads=(), coverage="complete", last_ts=NOW - 7200):
    return {
        "agent_id": agent_id,
        "name": f"Agent {agent_id.upper()}",
        "mode": "assistant",
        "memory_count": 1,
        "tool_count": 2,
        "event_count": events,
        "bundle_count": events // 10,
        "coverage": coverage,
        "last_ts": last_ts if events else 0.0,
        "last_activity": "2027-01-15 08:00:00" if events else "—",
        "thread_rows": [
            {
                "thread_id": thread,
                "memory_id": f"mem-{thread}",
                "event_count": 3,
                "last_ts": NOW - 90,
                "last_activity": "2027-01-15 09:58:30",
            }
            for thread in threads
        ],
    }


@pytest.mark.unit
def test_relative_age_and_timestamps():
    assert relative_age(NOW - 30, NOW) == "just now"
    assert relative_age(NOW - 90, NOW) == "1m ago"
    assert relative_age(NOW - 7200, NOW) == "2h ago"
    assert relative_age(NOW - 3 * 86400, NOW) == "3d ago"
    assert relative_age(NOW + 7200, NOW) == "in 2h"
    assert relative_age(0, NOW) == "—"
    assert relative_age(None, NOW) == "—"
    assert iso_timestamp("2026-09-28T11:51:47Z") == iso_timestamp(
        "2026-09-28T11:51:47+00:00"
    )
    assert iso_timestamp("not a date") is None
    assert iso_timestamp(None) is None


@pytest.mark.unit
def test_exact_usd_keeps_every_recorded_digit():
    assert exact_usd("1.10741417") == "$1.10741417"
    assert exact_usd("0.0011700") == "$0.00117"
    assert exact_usd(0) == "$0"
    assert exact_usd(None) == ""
    assert exact_usd("nan") == ""
    assert exact_usd("junk") == ""


@pytest.mark.unit
def test_navigator_rows_add_filter_text_tags_and_ages():
    rows = navigator_rows(
        [
            _agent("a1", 40, threads=["thread-1", "thread-2"]),
            _agent("b2", 0),
            _agent("c3", 5, coverage="partial"),
        ],
        now=NOW,
    )
    first, idle, partial = rows
    assert first["thread_count"] == 2
    assert first["age"] == "2h ago"
    assert first["thread_rows"][0]["age"] == "1m ago"
    assert first["tags"] == ["active", "threads"]
    assert "thread-2" in first["search"] and "mem-thread-1" in first["search"]
    assert "agent a1" in first["search"]
    assert idle["tags"] == ["idle"] and idle["age"] == "—"
    assert partial["coverage_tone"] == "is-warn" and "partial" in partial["tags"]
    # Source rows are not mutated.
    assert "age" not in _agent("a1", 1)
    assert navigator_counts(rows) == {
        "all": 3,
        "active": 2,
        "idle": 1,
        "threads": 1,
        "partial": 1,
    }


@pytest.mark.unit
def test_navigator_tape_sums_the_loaded_window():
    rows = navigator_rows(
        [_agent("a1", 40, threads=["t1", "t2"]), _agent("b2", 0)], now=NOW
    )
    tape = _by_label(
        navigator_tape(
            rows,
            now=NOW,
            read_only=True,
            content_mode="metadata",
            query_metadata={"provider_native": False, "truncated": True},
        )
    )
    assert tape["Agents"]["value"] == 2
    assert tape["With events"]["value"] == 1
    assert tape["Threads"]["value"] == 2
    assert tape["Trace events"]["value"] == 40
    assert tape["Bundles"]["value"] == 4
    assert tape["Last activity"]["value"] == "2h ago"
    assert tape["Last activity"]["title"].endswith("UTC")
    assert tape["Coverage"]["value"] == "Complete"
    assert tape["Coverage"]["tone"] == "is-good"
    assert tape["Query"]["value"] == "Fallback · truncated"
    assert tape["Query"]["title"] == "Compatibility fallback scan; window truncated"
    assert tape["Query"]["tone"] == "is-warn"
    meta = tape["Content mode metadata"]
    assert meta["value"] == "Read-only provider" and meta["meta"]

    failing = _by_label(
        navigator_tape(
            navigator_rows([_agent("a", 3, coverage="unknown")], now=NOW),
            now=NOW,
            read_only=False,
            content_mode="",
            query_metadata={"provider_native": True, "query_errors": ["trace"]},
        )
    )
    assert failing["Query"]["value"] == "Native"
    assert failing["Query"]["title"].endswith("1 store queries failed")
    assert failing["Query"]["tone"] == "is-bad"
    assert failing["Coverage"]["value"] == "Unknown"
    assert failing["Content mode full"]["value"] == "Writable provider"

    empty = _by_label(navigator_tape([], now=NOW, read_only=False, content_mode="full"))
    assert empty["Agents"]["value"] == 0
    assert empty["Last activity"]["value"] == "—"
    assert empty["Coverage"]["value"] == "—"


@pytest.mark.unit
def test_usage_tape_leads_with_calls_tokens_and_exact_charge():
    usage = {
        "totals": {
            "calls": 303,
            "input_tokens": 1384984,
            "output_tokens": 19680,
            "total_tokens": 1404664,
            "cost_usd": "1.10741417",
            "unpriced_calls": 20,
            "unknown_usage_calls": 0,
            "mean_latency_ms": 1601.79,
            "p95_latency_ms": 3490.676,
        },
        "prompt_cache": {"read_percent": 63.2, "attention": True},
        "coverage": {"read_complete": False, "events_scanned": 2289},
    }
    items = usage_tape(usage)
    assert [entry["label"] for entry in items[:3]] == [
        "Calls",
        "Tokens",
        "Known charge",
    ]
    tape = _by_label(items)
    assert tape["Tokens"]["title"] == "1,384,984 input · 19,680 output"
    assert tape["Known charge"]["kind"] == "usd"
    assert tape["Known charge"]["title"].startswith("Exact $1.10741417;")
    assert tape["Unpriced calls"]["tone"] == "is-warn"
    assert tape["Missing usage"]["tone"] == ""
    assert tape["Cache read"]["value"] == 63.2
    assert tape["Cache read"]["tone"] == "is-warn"
    assert "Read" not in tape
    assert tape["Events read"]["value"] == 2289
    assert tape["Events read"]["tone"] == "is-warn"
    assert tape["Events read"]["title"] == "Read partial or not established"

    empty = _by_label(usage_tape({"totals": {"cost_usd": None}}))
    assert empty["Calls"]["value"] == 0
    assert empty["Known charge"]["value"] is None
    assert empty["Known charge"]["title"] == ""


@pytest.mark.unit
def test_health_tape_signals():
    health = {
        "coverage": "partial",
        "headline": "Stored-event read incomplete",
        "alerts": [{"code": "missing_identity_events", "severity": "warning"}],
        "failed_queries": [],
        "normalization_errors": 0,
        "normalized_events": 1000,
        "missing_identity_events": 94,
        "orphan_spans": 0,
        "index": {"ready": False, "schema_version": 1},
        "latest_event_at": "2027-01-15T09:00:00+00:00",
        "truncated": True,
    }
    tape = _by_label(health_tape(health, now=iso_timestamp("2027-01-15T11:00:00Z")))
    assert tape["Coverage"]["value"] == "Partial"
    assert tape["Coverage"]["title"] == "Stored-event read incomplete"
    assert tape["Alerts"]["tone"] == "is-warn"
    assert tape["Failed queries"]["tone"] == "is-good"
    assert tape["Errors"]["tone"] == "is-good"
    assert tape["Missing identity"]["tone"] == "is-warn"
    assert tape["Span index"]["value"] == "Not ready"
    assert tape["Latest event"]["value"] == "2h ago"
    assert tape["Truncated"]["value"] == "Yes"

    broken = _by_label(
        health_tape(
            {
                "alerts": [{"severity": "critical"}],
                "failed_queries": ["trace"],
                "normalization_errors": 3,
                "index": {"ready": True},
            },
            now=NOW,
        )
    )
    assert broken["Alerts"]["tone"] == "is-bad"
    assert broken["Failed queries"]["tone"] == "is-bad"
    assert broken["Errors"]["tone"] == "is-bad"
    assert broken["Coverage"]["value"] == "Unknown"
    assert broken["Span index"]["tone"] == "is-good"
    assert broken["Latest event"]["value"] == "—"


@pytest.mark.unit
def test_compare_tape_with_and_without_a_comparison():
    waiting = compare_tape(None, agent_id="a", baseline_root=None, candidate_root="c")
    assert [entry["value"] for entry in waiting] == ["a", "—", "c", "Choose two roots"]
    assert waiting[-1]["meta"]

    tape = _by_label(
        compare_tape(
            {
                "baseline": {"normalized_events": 28},
                "candidate": {"normalized_events": 21},
                "findings": {
                    "only_in_baseline": ["contract_parse_failed"],
                    "only_in_candidate": [],
                    "in_both": ["x", "y"],
                },
                "changes": {
                    "operations": {"added": [], "removed": [{"count": 1}]},
                    "artifacts": {"added": [], "removed": []},
                },
            },
            agent_id="a",
            baseline_root="r1",
            candidate_root="r2",
        )
    )
    assert tape["Baseline events"]["value"] == 28
    assert tape["Candidate events"]["value"] == 21
    assert tape["Only in baseline"]["tone"] == "is-warn"
    assert tape["Only in candidate"]["tone"] == ""
    assert tape["In both"]["value"] == 2
    assert tape["Changed categories"]["value"] == 1
    assert tape["Changed categories"]["title"] == "1 of 2 categories differ"


def _macros():
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("_fmt.html").module


@pytest.mark.unit
@pytest.mark.parametrize(
    "value, expected",
    [
        (1.107414, "$1.11"),
        ("1.10741417", "$1.11"),
        (1234.5, "$1,234.50"),
        (0.0295196, "$0.0295"),
        ("0.00117", "$0.001170"),
        (0.002, "$0.002000"),
        (0, "$0.00"),
        (None, "Unknown"),
    ],
)
def test_usd_macro_uses_one_rule_for_charges(value, expected):
    assert str(_macros().usd(value)) == expected


@pytest.mark.unit
def test_figure_macros():
    macros = _macros()
    assert str(macros.count(875799)) == "875,799"
    assert str(macros.count(None)) == "—"
    assert str(macros.ms(1601.79)) == "1,602 ms"
    assert str(macros.ms(8.61)) == "8.6 ms"
    assert str(macros.ms(None)) == "—"
    assert str(macros.percent(63.2)) == "63.2%"
    html = str(
        macros.tape(
            [
                {
                    "label": "Known charge",
                    "value": "1.10741417",
                    "kind": "usd",
                    "tone": "",
                    "title": "Exact $1.10741417",
                    "meta": False,
                },
                {
                    "label": "Tokens",
                    "value": 1404664,
                    "kind": "count",
                    "tone": "is-warn",
                    "title": "",
                    "meta": False,
                },
            ],
            "Usage summary",
            "usage-kpis",
        )
    )
    assert 'class="fleet-tape obs-tape usage-kpis" role="list"' in html
    assert '<strong title="Exact $1.10741417">$1.11</strong>' in html
    assert '<strong class="is-warn">1,404,664</strong>' in html


@pytest.mark.unit
def test_cache_kpis_group_digits_in_a_bare_environment():
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    html = env.from_string(
        "{% from '_cache_kpis.html' import cache_kpis %}"
        "{{ cache_kpis(63.2, 197, 283, 875799, 470639) }}"
    ).render()
    assert "875,799" in html and "470,639" in html
    assert "197 of 283 calls" in html


def _usage_event(index, **updates):
    return {
        "trace_kind": "model_result",
        "span_id": f"call-{index}",
        "event_id": f"result-{index}",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_tokens": 4000,
        "output_tokens": 400,
        "cached_tokens": 1000,
        "duration_ms": 1250.5,
        "agent_id": "agent",
        "thread_id": "thread",
        "root_trace_id": f"root-{index}",
        "turn_id": f"turn-{index}",
        "timestamp": "2026-09-07T00:30:00Z",
        **updates,
    }


@pytest.mark.unit
def test_usage_page_renders_the_tape_with_grouped_figures(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "{}")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))

    from memorizz.observability import aggregate_usage

    events = [_usage_event(i) for i in range(3)]
    with patch.dict(state._state, provider=object(), provider_type="filesystem"), patch(
        "memorizz.ui.routers.usage.query_usage",
        side_effect=lambda provider, **kwargs: aggregate_usage(events),
    ):
        client = TestClient(create_app())
        response = client.get("/traces/usage")
    assert response.status_code == 200, response.text
    html = response.text
    assert html.count('class="fleet-tape obs-tape usage-kpis"') == 1
    assert "<span>Calls</span><strong>3</strong>" in html
    assert "<span>Tokens</span><strong" in html and ">13,200</strong>" in html
    assert "/static/css/pages/observability.css" in html
