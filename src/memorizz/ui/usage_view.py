"""The Usage page's figures: the window against the one before it, then usage
over time, by agent, by model and by run. Pure data shaping for the template.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from ..observability.analytics import period_start
from .dashboard import format_duration

# Runs listed on the page; the CSV export has every run in the window.
RUNS_SHOWN = 25
# Bars drawn for "all recorded" usage, newest last.
MAX_BARS = 120
METRICS = ("tokens", "spend", "calls")


def _decimal(value: Any) -> Optional[Decimal]:
    return None if value in (None, "") else Decimal(str(value))


def change(current: Any, previous: Any) -> Optional[float]:
    """Percent change from ``previous``; None when there is nothing to compare."""
    if current is None or previous in (None, 0, Decimal(0)):
        return None
    return (
        float((Decimal(str(current)) - Decimal(str(previous))) / Decimal(str(previous)))
        * 100
    )


def _kpi(label, value, kind, *, previous=None, note="", href="", worse_when_up=False):
    delta = change(value, previous)
    tone = ""
    if worse_when_up and delta is not None and abs(delta) >= 5:
        tone = "is-bad" if delta > 0 else "is-good"
    return {
        "label": label,
        "value": value,
        "kind": kind,
        "delta": delta,
        "delta_tone": tone,
        "note": note,
        "href": href,
    }


def kpis(current: Mapping, previous: Optional[Mapping], *, runs: int, prev_runs: int):
    totals = current["totals"]
    prev = (previous or {}).get("totals") or {}
    spend = _decimal(totals.get("cost_usd"))
    prev_spend = _decimal(prev.get("cost_usd"))
    unpriced = int(totals.get("unpriced_calls") or 0)
    calls = int(totals.get("calls") or 0)
    if not calls:
        spend_note = "No model calls"
    elif unpriced == calls:
        spend_note = "No call has a price yet"
    elif unpriced:
        spend_note = f"{unpriced:,} of {calls:,} calls have no price"
    else:
        spend_note = "Every call priced"
    cache = current.get("prompt_cache") or {}
    per_run = spend / runs if spend is not None and runs else None
    prev_per_run = (
        prev_spend / prev_runs if prev_spend is not None and prev_runs else None
    )
    return [
        _kpi(
            "Known spend",
            spend,
            "usd",
            previous=prev_spend if previous else None,
            note=spend_note,
            href="/traces/usage/rates" if unpriced else "",
        ),
        _kpi(
            "Tokens",
            totals.get("total_tokens"),
            "count",
            previous=prev.get("total_tokens") if previous else None,
            note="{:,} in · {:,} out".format(
                int(totals.get("input_tokens") or 0),
                int(totals.get("output_tokens") or 0),
            ),
        ),
        _kpi(
            "Model calls",
            calls,
            "count",
            previous=prev.get("calls") if previous else None,
            note=f"{runs:,} run{'' if runs == 1 else 's'}",
        ),
        _kpi(
            "Spend per run",
            per_run,
            "usd",
            previous=prev_per_run if previous else None,
            note="Known spend over runs",
        ),
        _kpi(
            "Latency p95",
            totals.get("p95_latency_ms"),
            "duration",
            previous=prev.get("p95_latency_ms") if previous else None,
            note=(
                "Mean " + format_duration(totals.get("mean_latency_ms"))
                if totals.get("mean_latency_ms") is not None
                else "No latency recorded"
            ),
            worse_when_up=True,
        ),
        _kpi(
            "Prompt cache",
            cache.get("read_percent"),
            "percent",
            note=(
                "of input tokens read from cache"
                if cache.get("read_percent") is not None
                else "Not reported by these models"
            ),
            href="#provider-prompt-cache",
        ),
    ]


def series(
    usage: Mapping,
    *,
    start: Optional[datetime],
    end: datetime,
    interval: timedelta,
    timezone_name: str,
    metric: str,
) -> Dict[str, Any]:
    """One bar per interval from ``start`` to ``end``, gaps included."""
    zone = ZoneInfo(timezone_name)
    rows = {row["label"]: row for row in usage.get("series") or []}
    if start is None:
        # All recorded usage: from the first period with calls.
        start = min(
            (datetime.fromisoformat(label) for label in rows),
            default=end - interval,
        )
    seconds = interval.total_seconds()
    # Day buckets run midnight to midnight in the zone: 23 or 25 hours across a
    # daylight-saving change, so the next one is found from the zone's clock.
    days = seconds >= 86400 and not seconds % 86400
    cursor = period_start(start, seconds, zone)
    bars: List[Dict[str, Any]] = []
    while cursor < end:
        following = (
            period_start(cursor + interval + timedelta(hours=3), seconds, zone)
            if days
            else cursor + interval
        )
        row = rows.get(cursor.isoformat()) or {}
        cost = _decimal(row.get("cost_usd"))
        bars.append(
            {
                "start": cursor,
                "end": min(following, end),
                "calls": int(row.get("calls") or 0),
                "tokens": int(row.get("total_tokens") or 0),
                "input_tokens": int(row.get("input_tokens") or 0),
                "output_tokens": int(row.get("output_tokens") or 0),
                "cost": cost,
                "unpriced": int(row.get("unpriced_calls") or 0),
            }
        )
        cursor = following
    clipped = max(0, len(bars) - MAX_BARS)
    bars = bars[clipped:]
    value = {
        "tokens": lambda bar: bar["tokens"],
        "spend": lambda bar: float(bar["cost"] or 0),
        "calls": lambda bar: bar["calls"],
    }[metric]
    peak = max((value(bar) for bar in bars), default=0) or 1
    for bar in bars:
        bar["value"] = value(bar)
        bar["height"] = round(100 * bar["value"] / peak, 1)
    return {
        "bars": bars,
        "peak": peak,
        "metric": metric,
        "clipped": clipped,
        "day_buckets": days,
    }


def _share(rows: List[Dict[str, Any]], field: str) -> None:
    total = sum(row[field] or 0 for row in rows) or 1
    for row in rows:
        row["share"] = 100 * (row[field] or 0) / total


def agent_rows(usage: Mapping, names: Mapping[str, str], query: Mapping[str, str]):
    rows = []
    for row in usage.get("agents") or []:
        agent_id = row["label"]
        known = agent_id != "Unknown agent"
        rows.append(
            {
                **row,
                "agent_id": agent_id if known else "",
                "name": names.get(agent_id) or (agent_id[:8] if known else "Unknown"),
                "saved": agent_id in names,
                "filter_url": (
                    "/traces/usage?" + urlencode({**query, "agent_id": agent_id})
                    if known
                    else ""
                ),
                "cost": _decimal(row.get("cost_usd")),
            }
        )
    rows.sort(key=lambda row: (-(row["total_tokens"] or 0), -row["calls"], row["name"]))
    _share(rows, "total_tokens")
    return rows


def model_rows(usage: Mapping, query: Mapping[str, str]):
    rows = []
    for row in usage.get("models") or []:
        provider, _, model = row["label"].partition(" / ")
        rows.append(
            {
                **row,
                "provider": provider,
                "model": model,
                "cost": _decimal(row.get("cost_usd")),
                "filter_url": "/traces/usage?" + urlencode({**query, "model": model}),
                "price_url": (
                    "/traces/usage/rates?"
                    + urlencode({"provider": provider, "model": model})
                    if row.get("unpriced_calls")
                    else ""
                ),
            }
        )
    rows.sort(
        key=lambda row: (-(row["total_tokens"] or 0), -row["calls"], row["label"])
    )
    _share(rows, "total_tokens")
    return rows


def run_rows(usage: Mapping, names: Mapping[str, str]):
    """Runs by spend, then tokens: where the money and tokens went."""
    rows = []
    for row in usage.get("interactions") or []:
        agent_id = row.get("agent_id") or ""
        started = row.get("started_at")
        rows.append(
            {
                **row,
                "agent_name": names.get(agent_id) or (agent_id[:8] or "Unknown"),
                "started": datetime.fromisoformat(started) if started else None,
                "cost": _decimal(row.get("cost_usd")),
            }
        )
    rows.sort(
        key=lambda row: (
            -(row["cost"] or 0),
            -(row["total_tokens"] or 0),
            row["label"],
        )
    )
    return rows


def coverage(usage: Mapping) -> Dict[str, Any]:
    window = usage.get("coverage") or {}
    totals = usage.get("totals") or {}
    return {
        "events": int(window.get("events_scanned") or 0),
        "complete": bool(window.get("read_complete")),
        "missing_usage": int(totals.get("unknown_usage_calls") or 0),
        "truncated": bool(window.get("aggregation_truncated")),
    }


def page(
    current: Mapping,
    previous: Optional[Mapping],
    *,
    span: Mapping[str, Any],
    interval: timedelta,
    timezone_name: str,
    metric: str,
    names: Mapping[str, str],
    query: Mapping[str, str],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Everything the Usage page draws, in the order it draws it."""
    now = now or datetime.now(timezone.utc)
    runs = run_rows(current, names)
    # A period before with no calls has nothing to compare against.
    previous_empty = previous is not None and not previous["totals"].get("calls")
    if previous_empty:
        previous = None
    prev_runs = len((previous or {}).get("interactions") or [])
    return {
        "kpis": kpis(current, previous, runs=len(runs), prev_runs=prev_runs),
        "previous_empty": previous_empty,
        "series": series(
            current,
            start=span.get("start") if span["key"] != "all" else None,
            end=span.get("end") or now,
            interval=interval,
            timezone_name=timezone_name,
            metric=metric,
        ),
        "agents": agent_rows(current, names, query),
        "models": model_rows(current, query),
        "runs": runs[:RUNS_SHOWN],
        "run_count": len(runs),
        "coverage": coverage(current),
        "compared": previous is not None,
    }
