# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The Usage & cost page: the window against the one before, usage over time,
by agent, model and run, and its exports."""

import csv
import io
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from memorizz.observability import aggregate_usage
from memorizz.observability.analytics import period_start
from memorizz.ui import usage_view

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _call(index, *, hours_ago, agent="agent", model="gpt-4o-mini", **updates):
    return {
        "trace_kind": "model_result",
        "span_id": f"call-{index}",
        "event_id": f"result-{index}",
        "provider": "openai",
        "model": model,
        "input_tokens": 4000,
        "output_tokens": 400,
        "cached_tokens": 0,
        "duration_ms": 1250.0,
        "agent_id": agent,
        "thread_id": "thread",
        "root_trace_id": f"root-{index}",
        "turn_id": f"turn-{index}",
        "timestamp": (NOW - timedelta(hours=hours_ago)).isoformat(),
        **updates,
    }


@pytest.mark.unit
def test_day_buckets_start_at_midnight_in_the_chosen_zone():
    moment = datetime(2026, 10, 22, 1, 30, tzinfo=timezone.utc)
    new_york = ZoneInfo("America/New_York")
    assert period_start(moment, 3600) == datetime(2026, 10, 22, 1, tzinfo=timezone.utc)
    assert period_start(moment, 86400, new_york) == datetime(
        2026, 10, 21, 4, tzinfo=timezone.utc
    ), "01:30 UTC is still Oct 21 in New York"
    usage = aggregate_usage(
        [_call(1, hours_ago=1), _call(2, hours_ago=1.2), _call(3, hours_ago=30)],
        interval_seconds=3600,
    )
    assert [row["calls"] for row in usage["series"]] == [1, 2]
    assert all(row["started_at"] for row in usage["interactions"])


@pytest.mark.unit
def test_a_day_chart_fills_quiet_days_and_follows_daylight_saving():
    # Clocks go back in London on 2026-10-25, so that day lasts 25 hours.
    london = ZoneInfo("Europe/London")
    start = datetime(2026, 10, 23, tzinfo=london).astimezone(timezone.utc)
    end = datetime(2026, 10, 28, tzinfo=london).astimezone(timezone.utc)
    busy = start + timedelta(days=2, hours=6)
    usage = aggregate_usage(
        [{**_call(1, hours_ago=0), "timestamp": busy.isoformat()}],
        timezone_name="Europe/London",
        interval_seconds=86400,
    )
    chart = usage_view.series(
        usage,
        start=start,
        end=end,
        interval=timedelta(days=1),
        timezone_name="Europe/London",
        metric="tokens",
    )
    days = [bar["start"].astimezone(london) for bar in chart["bars"]]
    assert [day.day for day in days] == [23, 24, 25, 26, 27]
    assert all(day.hour == 0 for day in days), "each bar starts at local midnight"
    assert [bar["tokens"] for bar in chart["bars"]] == [0, 0, 4400, 0, 0]
    assert chart["bars"][2]["height"] == 100.0 and chart["day_buckets"]


@pytest.mark.unit
def test_figures_compare_with_the_period_before():
    current = aggregate_usage([_call(i, hours_ago=1) for i in range(3)])
    previous = aggregate_usage([_call(i, hours_ago=30) for i in range(2)])
    figures = {
        k["label"]: k for k in usage_view.kpis(current, previous, runs=3, prev_runs=2)
    }
    assert figures["Model calls"]["delta"] == pytest.approx(50.0)
    assert figures["Tokens"]["note"] == "12,000 in · 1,200 out"
    assert figures["Known spend"]["note"] == "Every call priced"
    assert figures["Latency p95"]["delta"] == 0
    page = usage_view.page(
        current,
        aggregate_usage([]),
        span={"key": "7d", "start": NOW - timedelta(days=7), "end": NOW},
        interval=timedelta(hours=6),
        timezone_name="UTC",
        metric="calls",
        names={},
        query={},
        now=NOW,
    )
    assert page["previous_empty"] and not page["compared"]
    assert all(k["delta"] is None for k in page["kpis"]), "nothing to compare with"


@pytest.mark.unit
def test_rows_name_agents_rank_by_tokens_and_link_unpriced_models():
    usage = aggregate_usage(
        [
            _call(1, hours_ago=1, agent="a1"),
            _call(2, hours_ago=1, agent="a2", input_tokens=9000),
            _call(3, hours_ago=1, agent="a2", model="private-reader"),
        ]
    )
    agents = usage_view.agent_rows(usage, {"a2": "Support desk"}, {"model": "x"})
    assert [(row["name"], row["saved"]) for row in agents] == [
        ("Support desk", True),
        ("a1", False),
    ]
    assert agents[0]["filter_url"] == "/traces/usage?model=x&agent_id=a2"
    models = {row["model"]: row for row in usage_view.model_rows(usage, {})}
    assert models["private-reader"]["price_url"] == (
        "/traces/usage/rates?provider=openai&model=private-reader"
    )
    assert not models["gpt-4o-mini"]["price_url"]
    runs = usage_view.run_rows(usage, {"a2": "Support desk"})
    assert runs[0]["agent_name"] == "Support desk", "the costliest run first"


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402


@pytest.fixture()
def page(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "{}")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    events = [
        _call(1, hours_ago=2, agent="a1"),
        _call(2, hours_ago=3, agent="=HYPERLINK(1)"),
        _call(3, hours_ago=24 * 40),
    ]
    reads = []

    def query_usage(provider, **kwargs):
        reads.append(kwargs["filters"])
        low = kwargs["filters"].get("start_time", "")
        high = kwargs["filters"].get("end_time", "~")
        window = [e for e in events if low <= e["timestamp"] <= high]
        return aggregate_usage(window, interval_seconds=kwargs.get("interval_seconds"))

    with patch.dict(state._state, provider=object(), provider_type="filesystem"), patch(
        "memorizz.ui.routers.usage.query_usage", side_effect=query_usage
    ):
        client = TestClient(create_app())
        client.reads = reads
        yield client


@pytest.mark.unit
def test_the_page_opens_on_30_days_against_the_30_before(page):
    html = page.get("/traces/usage").text
    assert 'aria-current="page">30d<' in html and ">All<" in html
    current, before = page.reads
    span = datetime.fromisoformat(current["end_time"]) - datetime.fromisoformat(
        current["start_time"]
    )
    assert span == timedelta(days=30)
    assert before["end_time"] == current["start_time"], "the 30 days just before"
    assert "Tokens over time" in html and "By agent" in html and "By model" in html
    assert "Runs by spend and tokens" in html
    assert "Compared with the 30 days before" in html, "a call 40 days ago"


@pytest.mark.unit
def test_the_api_keeps_all_recorded_usage_and_refuses_bad_bounds(page):
    usage = page.get("/traces/usage.json").json()
    assert usage["totals"]["calls"] == 3, "JSON still covers everything recorded"
    assert "start_time" not in page.reads[-1]
    assert page.get("/traces/usage.json?start_time=invalid").status_code == 400
    assert page.get("/traces/usage.json?window=custom&start=nope").status_code == 400
    old_link = page.get(
        f"/traces/usage?start_time={(NOW - timedelta(days=2)):%Y-%m-%dT%H:%M}"
    )
    assert 'aria-current="page" title="Choose a start and an end">' in old_link.text


@pytest.mark.unit
def test_csv_exports_rows_with_names_and_no_formulas(page):
    response = page.get("/traces/usage.csv?group=agent")
    assert response.headers["content-type"].startswith("text/csv")
    assert "memorizz-usage-by-agent.csv" in response.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0][:3] == ["agent_id", "agent_name", "calls"]
    agents = {row[0] for row in rows[1:]}
    assert "'=HYPERLINK(1)" in agents, "a cell a spreadsheet would run is quoted"
    runs = list(csv.reader(io.StringIO(page.get("/traces/usage.csv").text)))
    assert runs[0][0] == "started_at" and len(runs) == 4


@pytest.mark.unit
def test_a_bar_and_a_measure_keep_the_page_filters(page):
    html = page.get("/traces/usage?window=7d&metric=spend&model=gpt-4o-mini").text
    assert "Known spend over time" in html
    assert (
        'href="/traces/usage?model=gpt-4o-mini&amp;metric=tokens&amp;window=7d"' in html
    )
    assert "metric=spend&amp;window=custom&amp;start=" in html, "bars zoom in"
    broken = page.get("/traces/usage?window=custom&start=2026-13-01").text
    assert "Choose a start and an end for a custom range." in broken
