"""Bounded, provider-neutral usage analytics over normalized trace events.

One pass over the supplied iterable; no storage, network, tokenizer or prompt
content access. Scope events to the authorized tenant *before* calling this API.
Totals cover recorded model results, not uninstrumented retries or embeddings.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from itertools import islice
from zoneinfo import ZoneInfo

from .pricing import DEFAULT_PRICING, token_count

MEMORY_FIELDS = (
    "memory_type",
    "memory_chars",
    "memory_tokens_estimate",
    "token_estimation_method",
)
MEMORY_TYPES = frozenset(
    {
        "history",
        "episodic",
        "knowledge_base",
        "entity",
        "summaries",
        "skills",
        "personalization",
        "persona",
        "tool_log",
        "evidence_pack",
        "other",
    }
)


def memory_type(value):
    if not isinstance(value, str):
        return "other"
    aliases = {
        "conversation_memory": "episodic",
        "semantic": "knowledge_base",
        "entity_memory": "entity",
        "procedural": "skills",
    }
    value = aliases.get(value, value)
    return value if value in MEMORY_TYPES else "other"


def finite_number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _bucket(label):
    return {
        "label": label,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "measured_calls": 0,
        "priced_calls": 0,
        "cost_usd": Decimal(0),
        "durations": [],
    }


def _add(bucket, event, quote):
    bucket["calls"] += 1
    measured = True
    for field in ("input_tokens", "output_tokens", "cached_tokens"):
        count = token_count(event.get(field))
        if count is not None:
            bucket[field] += count
        elif field != "cached_tokens":
            measured = False
    bucket["measured_calls"] += measured
    if quote.get("cost_usd") is not None:
        bucket["priced_calls"] += 1
        bucket["cost_usd"] += Decimal(quote["cost_usd"])
    duration = finite_number(event.get("duration_ms"))
    if duration is not None:
        bucket["durations"].append(duration)


def _finish(bucket):
    result = {key: value for key, value in bucket.items() if key != "durations"}
    durations = sorted(bucket["durations"])
    result.update(
        cost_usd=str(bucket["cost_usd"]) if bucket["priced_calls"] else None,
        total_tokens=bucket["input_tokens"] + bucket["output_tokens"],
        unknown_usage_calls=bucket["calls"] - bucket["measured_calls"],
        unpriced_calls=bucket["calls"] - bucket["priced_calls"],
        mean_latency_ms=sum(durations) / len(durations) if durations else None,
        p95_latency_ms=durations[math.ceil(len(durations) * 0.95) - 1]
        if durations
        else None,
        latency_samples=len(durations),
    )
    return result


def _quote(event, pricing):
    # Frozen runtime quotes remain stable when the catalog changes. Legacy costs
    # without provenance are not silently promoted to verified charges.
    if (
        event.get("cost_status") in {"calculated", "provider_reported"}
        and event.get("pricing_version")
        and event.get("pricing_source")
    ):
        try:
            cost = Decimal(str(event.get("cost_usd")))
            if cost.is_finite() and cost >= 0:
                return {
                    "cost_usd": str(cost),
                    "cost_status": event["cost_status"],
                    "pricing_version": event["pricing_version"],
                    "pricing_source": event["pricing_source"],
                    "pricing_as_of": event.get("pricing_as_of"),
                    "basis": "recorded_snapshot",
                }
        except InvalidOperation:
            pass
    return {**pricing.quote(event), "basis": "current_catalog"}


def aggregate_usage(
    events,
    *,
    pricing=DEFAULT_PRICING,
    timezone_name="UTC",
    max_events=10_000,
    coverage=None,
    model=None,
) -> dict:
    """Aggregate a bounded iterable. ``coverage`` describes the supplied window.

    Deduplication uses scoped model span identities, never a parent root's cost.
    Memory tokens use rendered characters / 4 and are explicitly approximate;
    provider usage is never apportioned to memory types as if it were billing.
    """
    if type(max_events) is not int or not 1 <= max_events <= 100_000:
        raise ValueError("max_events must be between 1 and 100000")
    zone = ZoneInfo(timezone_name)
    totals = _bucket("Recorded model calls")
    groups = {"agents": {}, "models": {}, "daily": {}, "interactions": {}}
    memories, seen, prices, reasons = {}, set(), {}, {}
    scanned = duplicates = missing_timestamps = 0
    truncated = False
    for index, raw in enumerate(islice(events, max_events + 1)):
        if index == max_events:
            truncated = True
            break
        scanned += 1
        if not isinstance(raw, dict):
            continue
        event = {**(raw.get("attributes") or {}), **raw}
        kind = event.get("trace_kind") or event.get("event_kind") or event.get("kind")
        if kind == "model_call" and event.get("phase") == "result":
            kind = "model_result"
        if kind not in {"model_result", "memory_supply", "memory_retrieval"}:
            continue
        if model and event.get("model") != model:
            continue
        scope = tuple(
            str(event.get(key) or "")
            for key in (
                "application_id",
                "user_id",
                "agent_id",
                "thread_id",
                "root_trace_id",
                "turn_id",
            )
        )
        identity = (
            event.get("span_id")
            if kind == "model_result"
            else event.get("event_id") or event.get("trace_id")
        )
        identity = identity or event.get("event_id") or event.get("trace_id")
        if identity:
            key = (scope, kind, str(identity))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
        if kind.startswith("memory_"):
            category = memory_type(event.get("memory_type"))
            row = memories.setdefault(
                category,
                {
                    "label": category,
                    "memory_chars": 0,
                    "memory_tokens_estimate": 0,
                    "supply_samples": 0,
                    "retrieval_samples": 0,
                    "retrieval_errors": 0,
                    "retrieval_total_ms": 0.0,
                    "retrieval_durations": [],
                },
            )
            if kind == "memory_supply":
                chars = token_count(event.get("memory_chars"))
                estimate = token_count(event.get("memory_tokens_estimate"))
                if chars is not None and estimate is not None:
                    row["supply_samples"] += 1
                    row["memory_chars"] += chars
                    row["memory_tokens_estimate"] += estimate
            else:
                duration = finite_number(event.get("duration_ms"))
                if duration is not None:
                    row["retrieval_samples"] += 1
                    row["retrieval_total_ms"] += duration
                    row["retrieval_durations"].append(duration)
                row["retrieval_errors"] += event.get("status") == "error"
            continue
        quote = _quote(event, pricing)
        if quote.get("pricing_version"):
            prices[(quote["pricing_version"], quote["basis"])] = {
                key: quote.get(key)
                for key in (
                    "pricing_version",
                    "pricing_source",
                    "pricing_as_of",
                    "basis",
                )
            }
        if quote.get("cost_usd") is None:
            reason = quote.get("cost_reason") or "Unknown charge"
            reasons[reason] = reasons.get(reason, 0) + 1
        agent = str(event.get("agent_id") or "Unknown agent")[:240]
        model_label = (
            f"{event.get('provider') or 'unknown'} / {event.get('model') or 'unknown'}"[
                :480
            ]
        )
        try:
            moment = datetime.fromisoformat(
                str(event.get("timestamp")).replace("Z", "+00:00")
            )
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            day = moment.astimezone(zone).date().isoformat()
        except (ValueError, TypeError, OverflowError):
            day = "Unknown date"
            missing_timestamps += 1
        interaction = (
            " / ".join(filter(None, (event.get("root_trace_id"), event.get("turn_id"))))
            or "Unattributed interaction"
        )
        _add(totals, event, quote)
        for group, key, label in (
            ("agents", (scope[0], scope[1], agent), agent),
            ("models", model_label, model_label),
            ("daily", day, day),
            ("interactions", scope, interaction),
        ):
            bucket = groups[group].setdefault(key, _bucket(label))
            if group == "interactions":
                bucket.update(
                    agent_id=event.get("agent_id"),
                    thread_id=event.get("thread_id"),
                    root_trace_id=event.get("root_trace_id"),
                    turn_id=event.get("turn_id"),
                )
            _add(bucket, event, quote)
    memory_rows = []
    for row in memories.values():
        durations = sorted(row.pop("retrieval_durations"))
        row["retrieval_mean_ms"] = (
            sum(durations) / len(durations) if durations else None
        )
        row["retrieval_p95_ms"] = (
            durations[math.ceil(len(durations) * 0.95) - 1] if durations else None
        )
        memory_rows.append(row)
    window = dict(
        coverage if coverage is not None else getattr(events, "coverage", {}) or {}
    )
    window.update(
        events_scanned=scanned,
        duplicates_removed=duplicates,
        aggregation_truncated=truncated,
        missing_timestamps=missing_timestamps,
    )
    # A complete database read is not proof of end-to-end instrumentation.
    # Preserve that distinction, but consume the normalizer's actual field.
    read_state = window.get("read_completeness", window.get("coverage"))
    window["read_complete"] = (
        read_state == "complete"
        or (read_state is None and window.get("read_complete") is True)
    ) and not any(
        window.get(key)
        for key in (
            "truncated",
            "window_truncated",
            "aggregation_truncated",
            "normalization_errors",
            "query_errors",
            "errors",
        )
    )
    return {
        "schema_version": 1,
        "timezone": timezone_name,
        "totals": _finish(totals),
        **{
            group: [
                _finish(bucket)
                for bucket in sorted(rows.values(), key=lambda row: row["label"])
            ]
            for group, rows in groups.items()
        },
        "memory": sorted(
            memory_rows, key=lambda row: (-row["memory_tokens_estimate"], row["label"])
        ),
        "pricing": list(prices.values()),
        "unpriced_reasons": reasons,
        "coverage": window,
        "notes": [
            "Charges cover recorded text-model calls only, not invoices, taxes, tools, embeddings or unobserved retries.",
            "Memory tokens are approximate rendered characters / 4, counted each time supplied to a model; they are not billed-token attribution.",
            "Retrieval latency is measured per operation, not memory's causal contribution to generation latency. Missing measurements are unknown.",
        ],
    }
