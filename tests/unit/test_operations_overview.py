# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The console home page summarizes recent runs from trace events."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from memorizz.observability.overview import build_operations_overview

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def _at(minutes_ago):
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def _run(turn, agent, *, minutes_ago, status="success", seconds=4, tools=(), calls=()):
    events = [
        {
            "trace_kind": "turn_start",
            "turn_id": turn,
            "agent_id": agent,
            "timestamp": _at(minutes_ago),
        }
    ]
    for input_tokens, cached, cost in calls:
        events.append(
            {
                "trace_kind": "model_result",
                "turn_id": turn,
                "agent_id": agent,
                "status": "success",
                "provider": "anthropic",
                "model": "claude-sonnet-5",
                "input_tokens": input_tokens,
                "output_tokens": 10,
                "cached_tokens": cached,
                "cost_usd": cost,
                "timestamp": _at(minutes_ago),
            }
        )
    for name, tool_status, reason in tools:
        events.append(
            {
                "trace_kind": "tool_result",
                "turn_id": turn,
                "agent_id": agent,
                "tool_name": name,
                "status": tool_status,
                "outcome_reason_code": reason,
                "timestamp": _at(minutes_ago),
            }
        )
    if status is not None:
        events.append(
            {
                "trace_kind": "turn_result",
                "turn_id": turn,
                "agent_id": agent,
                "status": status,
                "error_code": "" if status == "success" else "provider_error",
                "timestamp": (
                    NOW - timedelta(minutes=minutes_ago) + timedelta(seconds=seconds)
                ).isoformat(),
            }
        )
    return events


@pytest.mark.unit
def test_runs_are_grouped_and_summarized_for_the_window():
    events = [
        *_run(
            "t1",
            "a1",
            minutes_ago=30,
            seconds=2,
            calls=[(1000, 900, "0.01")],
            tools=[("lookup_order", "error", "invalid_tool_invocation")],
        ),
        *_run("t2", "a1", minutes_ago=90, status="error", seconds=8),
        *_run("t3", "a2", minutes_ago=60 * 30, calls=[(500, 0, "0.02")]),
    ]
    overview = build_operations_overview(events, window="24h", now=NOW)
    totals = overview["totals"]

    assert totals["runs"] == 2, "the 30-hour-old run is outside 24h"
    assert totals["failed"] == 1 and totals["succeeded"] == 1
    assert totals["success_rate"] == 50.0
    assert totals["p95_ms"] == 8000
    assert totals["cache_read_percent"] == 90.0
    assert totals["cost_usd"] == Decimal("0.01")
    assert totals["runs_with_tool_failures"] == 1
    assert [run["key"] for run in overview["recent"]] == ["t1", "t2"]
    assert overview["tools"][0] == {
        "name": "lookup_order",
        "calls": 1,
        "failures": 1,
        "failure_rate": 100.0,
        "top_reason": "invalid_tool_invocation",
    }
    titles = [item["title"] for item in overview["attention"]]
    assert "1 failed run" in titles
    assert "lookup_order failed 1 of 1 calls" in titles
    assert sum(bucket["runs"] for bucket in overview["series"]) == 2
    assert sum(bucket["failed"] for bucket in overview["series"]) == 1


@pytest.mark.unit
def test_runs_without_a_result_are_running_then_incomplete():
    events = [
        *_run("fresh", "a1", minutes_ago=2, status=None),
        *_run("stale", "a1", minutes_ago=120, status=None),
    ]
    runs = {
        run["key"]: run
        for run in build_operations_overview(events, window="24h", now=NOW)["recent"]
    }
    assert runs["fresh"]["status"] == "running"
    assert runs["stale"]["status"] == "incomplete"
    assert runs["stale"]["duration_ms"] is None


@pytest.mark.unit
def test_price_callback_prices_calls_without_a_recorded_cost():
    events = _run("t1", "a1", minutes_ago=5, calls=[(1000, 0, None)])
    default = build_operations_overview(events, window="24h", now=NOW)
    assert default["totals"]["unpriced_calls"] == 1
    priced = build_operations_overview(
        events, window="24h", now=NOW, price=lambda event: Decimal("0.5")
    )
    assert priced["totals"]["unpriced_calls"] == 0
    assert priced["totals"]["cost_usd"] == Decimal("0.5")


@pytest.mark.unit
def test_unknown_window_falls_back_and_buckets_align_to_the_hour():
    overview = build_operations_overview([], window="bogus", now=NOW)
    assert overview["window"] == "7d"
    hourly = build_operations_overview([], window="24h", now=NOW)
    assert all(b["start"].minute == 0 for b in hourly["series"])
    assert hourly["totals"]["success_rate"] is None


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402


class _Events(list):
    coverage = {"source_rows": 3}


@pytest.fixture()
def client(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "ui", lazy_vector_indexes=True)
    )
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    ):
        yield TestClient(create_app(), follow_redirects=False)


@pytest.mark.unit
def test_dashboard_shows_run_health_from_traces(client):
    events = _Events(
        _run(
            "t1",
            "a1",
            minutes_ago=0,
            calls=[(1000, 500, "0.01")],
            tools=[("lookup_order", "error", "invalid_tool_invocation")],
        )
    )
    # The page reads the wall clock, so place the run just before now.
    for event in events:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with patch(
        "memorizz.ui.routers.traces._selection_window", return_value=(None, events)
    ):
        response = client.get("/dashboard?window=24h")
    assert response.status_code == 200
    html = response.text
    assert "Runs over time" in html
    assert "Recent runs" in html
    assert "lookup_order failed 1 of 1 calls" in html
    assert 'aria-current="page">24h<' in html


@pytest.mark.unit
def test_dashboard_without_trace_access_keeps_inventory(client):
    with patch(
        "memorizz.ui.dashboard.require_trace_permission",
        side_effect=HTTPException(status_code=403),
    ):
        response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Run health is unavailable" in response.text
    assert "Platform" in response.text


@pytest.mark.unit
def test_agent_health_ranks_failures_before_tool_errors():
    from memorizz.observability.overview import agent_health

    assert agent_health(None) == "idle"
    assert agent_health({"runs": 3, "last_status": "failed", "success_rate": 95}) == (
        "failing"
    )
    assert agent_health({"runs": 10, "success_rate": 80, "last_status": "success"}) == (
        "failing"
    )
    assert agent_health({"runs": 5, "success_rate": 100, "tool_failures": 1}) == (
        "degraded"
    )
    assert agent_health({"runs": 5, "success_rate": 100}) == "healthy"


@pytest.mark.unit
def test_agent_rows_carry_trend_recent_runs_and_cache_share():
    events = [
        *_run("t1", "a1", minutes_ago=10, calls=[(1000, 750, "0.01")]),
        *_run("t2", "a1", minutes_ago=20, status="error"),
    ]
    row = build_operations_overview(events, window="24h", now=NOW)["agents"][0]
    assert row["health"] == "failing", "the latest run is not the failed one"
    assert row["success_rate"] == 50.0
    assert row["cache_read_percent"] == 75.0
    assert [run["key"] for run in row["recent"]] == ["t1", "t2"]
    assert sum(bucket["runs"] for bucket in row["series"]) == 2


@pytest.mark.unit
def test_agents_page_is_a_fleet_monitor(client):
    created = client.post(
        "/agents/new",
        data={
            "agent_name": "Fleet Agent",
            "instruction": "You are a fleet test agent.",
            "application_mode": "assistant",
            "max_steps": "10",
            "tool_access": "private",
            "llm_provider": "openai",
            "llm_model": "gpt-4.1-mini",
        },
    )
    assert created.status_code in (302, 303), created.text[:300]
    agent_id = state._state["provider"].list_memagents()[0].agent_id
    events = _Events(
        _run(
            "t1",
            agent_id,
            minutes_ago=0,
            calls=[(1000, 500, "0.01")],
            tools=[("lookup_order", "error", "invalid_tool_invocation")],
        )
    )
    for event in events:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with patch(
        "memorizz.ui.routers.traces._selection_window", return_value=(None, events)
    ):
        response = client.get("/agents?window=24h")
    assert response.status_code == 200
    html = response.text
    assert "data-sidebar-scope" not in html
    assert "Fleet Agent" in html
    assert 'data-health-label="degraded"' in html, "a failed tool degrades health"
    assert 'data-runs="1"' in html
    assert "gpt-4.1-mini" in html
    assert f'id="fleet-detail-{agent_id}"' in html
    assert f'id="input-{agent_id}"' in html, "quick chat is kept in the pane"


@pytest.mark.unit
def test_attention_explains_causes_without_trace_content():
    events = [
        *_run(
            "t1",
            "a1",
            minutes_ago=5,
            status="error",
            tools=[("get_analysis", "error", "tool_not_disclosed")],
        ),
        {
            "trace_kind": "model_result",
            "turn_id": "t1",
            "agent_id": "a1",
            "status": "error",
            "provider": "anthropic",
            "model": "claude-opus-4-8",
            "error_code": "RemoteProtocolError",
            "response_chars": 900,
            "timestamp": _at(5),
        },
        {
            "trace_kind": "model_result",
            "turn_id": "t2",
            "agent_id": "a1",
            "status": "success",
            "provider": "anthropic",
            "model": "claude-opus-4-8",
            "input_tokens": 100,
            "output_tokens": 5,
            "timestamp": _at(3),
        },
    ]
    overview = build_operations_overview(
        events,
        window="24h",
        now=NOW,
        price=lambda event: (None, "Cached-token usage was not reported"),
    )
    items = {item["title"]: item["detail"] for item in overview["attention"]}
    assert (
        "RemoteProtocolError: the provider closed the connection"
        in items["1 failed run"]
    )
    assert (
        "called before discover_tools disclosed it"
        in items["get_analysis failed 1 of 1 calls"]
    )
    assert items["1 provider call error"].startswith("RemoteProtocolError ×1")
    assert "0 before any output, 1 mid-response" in items["1 provider call error"]
    assert items["1 anthropic / claude-opus-4-8 call without a price"].startswith(
        "Cached-token usage was not reported"
    )


@pytest.mark.unit
def test_a_retried_provider_error_counts_its_failed_run_once():
    error = {
        "trace_kind": "model_result",
        "turn_id": "t1",
        "agent_id": "a1",
        "status": "error",
        "provider": "ollama",
        "model": "qwen2.5:7b",
        "error_code": "ProviderStreamError",
        "timestamp": _at(5),
    }
    events = [*_run("t1", "a1", minutes_ago=5, status="error"), error, dict(error)]
    overview = build_operations_overview(events, window="24h", now=NOW)
    failed = next(
        item for item in overview["attention"] if "failed run" in item["title"]
    )
    assert "(in 1 of 1 failed run)" in failed["detail"]
    assert "2 of 1" not in failed["detail"]


@pytest.mark.unit
def test_windows_include_90_days_and_a_custom_range():
    from memorizz.observability.overview import resolve_window

    ninety = resolve_window("90d", now=NOW)
    assert (ninety["key"], NOW - ninety["start"]) == ("90d", timedelta(days=90))

    custom = resolve_window(
        "custom", start="2026-09-27T06:00", end="2026-09-27T18:00", now=NOW
    )
    assert custom["key"] == "custom" and not custom["error"]
    assert custom["start"] == datetime(2026, 9, 27, 6, tzinfo=timezone.utc)
    assert custom["bucket"] == timedelta(minutes=15), "12 hours in 48 bars"

    future = resolve_window("custom", start="2026-09-28", end="2026-12-01", now=NOW)
    assert future["end"] == NOW, "a range ends at now at the latest"

    for start, end in (
        ("", ""),
        ("2026-09-28T13:00", ""),
        ("2026-09-27", "2026-09-26"),
        ("2024-01-01", "2026-09-01"),
    ):
        fallback = resolve_window("custom", start=start, end=end, now=NOW)
        assert fallback["key"] == "7d" and fallback["error"], (start, end)


@pytest.mark.unit
def test_a_custom_range_counts_only_its_runs_and_each_bar_knows_its_span():
    from memorizz.observability.overview import resolve_window

    events = [
        *_run("t1", "a1", minutes_ago=30),
        *_run("t2", "a1", minutes_ago=180, status="error"),
        *_run("t3", "a1", minutes_ago=190),
    ]
    span = resolve_window("custom", start=_at(240), end=_at(120), now=NOW)
    overview = build_operations_overview(events, now=NOW, span=span)
    assert overview["totals"]["runs"] == 2, "the run 30 minutes ago is after the end"
    busy = [bucket for bucket in overview["series"] if bucket["runs"]]
    assert all(b["end"] - b["start"] == span["bucket"] for b in overview["series"])
    assert [(b["runs"], b["run"]["key"]) for b in busy] == [
        (1, "t3"),
        (1, "t2"),
    ], "a lone run is kept for a direct link"


@pytest.mark.unit
def test_attention_items_link_to_where_each_is_fixed():
    from memorizz.ui.dashboard import link_attention

    events = [
        *_run(
            "t1",
            "a1",
            minutes_ago=5,
            status="error",
            tools=[
                ("lookup_order", "error", "unknown_tool"),
                ("entity_memory_upsert", "error", "unknown_tool"),
            ],
        ),
        *_run("t2", "plugin", minutes_ago=6, tools=[("save", "error", "unreported")]),
        {
            "trace_kind": "model_result",
            "turn_id": "t1",
            "agent_id": "a1",
            "status": "error",
            "provider": "openai",
            "model": "gpt-6.1-sol",
            "error_code": "ReadTimeout",
            "timestamp": _at(5),
        },
        {
            "trace_kind": "model_result",
            "turn_id": "t2",
            "agent_id": "plugin",
            "status": "success",
            "provider": "ollama",
            "model": "gemma4:latest",
            "input_tokens": 10,
            "timestamp": _at(6),
        },
    ]
    overview = build_operations_overview(events, window="24h", now=NOW)
    items = {
        item["kind"]: item
        for item in link_attention(overview["attention"], editable_agent_ids={"a1"})
    }
    links = {kind: [a["href"] for a in item["actions"]] for kind, item in items.items()}
    assert links["failed_runs"] == ["/traces?agent_id=a1&turn_id=t1#trace-timeline"]
    tools = {
        i["tool"]: i["actions"]
        for i in overview["attention"]
        if i.get("kind") == "tool"
    }
    assert [a["href"] for a in tools["lookup_order"]] == [
        "/agents/a1/edit#section-tools",
        "/traces?agent_id=a1&turn_id=t1#trace-timeline",
    ], "a missing tool is fixed in the agent's tool settings"
    assert tools["entity_memory_upsert"][0] == {
        "label": "Fix the agent's memory settings",
        "href": "/agents/a1/edit#section-memory",
    }, "entity memory tools come from the memory settings"
    assert [a["href"] for a in tools["save"]] == [
        "/traces?agent_id=plugin&turn_id=t2#trace-timeline"
    ], "an agent that isn't saved here has no settings to link"
    assert links["provider_errors"] == [
        "/traces?agent_id=a1&turn_id=t1#trace-timeline",
        "/agents/a1/edit#section-model",
    ]
    assert links["unpriced"] == [
        "/traces/usage/rates?provider=ollama&model=gemma4%3Alatest"
        "&service_tier=default"
    ]


@pytest.mark.unit
def test_dashboard_offers_90_days_a_custom_range_and_bars_that_open_runs(client):
    events = _Events(
        _run("t1", "a1", minutes_ago=0, tools=[("lookup_order", "error", "x")])
    )
    for event in events:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with patch(
        "memorizz.ui.routers.traces._selection_window", return_value=(None, events)
    ):
        ninety = client.get("/dashboard?window=90d").text
        custom = client.get(
            "/dashboard?window=custom&start=2026-01-01T00:00&end=2026-01-02T00:00"
        ).text
        broken = client.get("/dashboard?window=custom&start=nope").text
    assert 'aria-current="page">90d<' in ninety
    assert 'name="start"' in ninety and 'type="datetime-local"' in ninety
    assert (
        'class="ov-bar ov-bar--end" href="/traces?agent_id=a1&amp;turn_id=t1#trace-timeline"'
        in (ninety)
    ), "a bar with one run opens it"
    assert "Click to open this run" in ninety
    assert "Open the run" in ninety, "the failing tool links to its run"
    assert "Jan 01 – Jan 02, 2026" in custom
    assert "No runs in this window" in custom
    assert "Choose a start and an end" in broken
    assert 'aria-current="page">7d<' in broken
