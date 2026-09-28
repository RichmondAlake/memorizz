"""Provider-cache health from reported usage; missing counters are not misses."""

from datetime import datetime

from .pricing import token_count


def cache_state(event):
    if event.get("prompt_cache_warning"):
        return "configuration_warning"
    if event.get("prompt_cache_enabled") is False:
        return "disabled"
    cached = token_count(event.get("cached_tokens"))
    if cached is None:
        return "unknown"
    if cached:
        return "hit"
    if token_count(event.get("cache_write_tokens")):
        return "write"
    return "no_reuse"


def add_call_usage(totals, call):
    """Fold one model call's reported usage into run totals (in place).

    Cache reads count toward the read rate only when the call reported both
    its cache and input counters, so a provider that omits cache usage is not
    shown as a miss. Returns ``totals`` with a derived ``read_percent``.
    """
    for key in (
        "calls",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "measured_input_tokens",
        "cache_reported_calls",
        "cache_hit_calls",
    ):
        totals.setdefault(key, 0)
    totals["calls"] += 1
    inputs = token_count(call.get("input_tokens"))
    cached = token_count(call.get("cached_tokens"))
    for key, count in (
        ("input_tokens", inputs),
        ("output_tokens", token_count(call.get("output_tokens"))),
        ("cache_write_tokens", token_count(call.get("cache_write_tokens"))),
    ):
        totals[key] += count or 0
    if cached is not None and inputs is not None and cached <= inputs:
        totals["cache_reported_calls"] += 1
        totals["cache_hit_calls"] += int(cached > 0)
        totals["cached_tokens"] += cached
        totals["measured_input_tokens"] += inputs
    measured = totals["measured_input_tokens"]
    totals["read_percent"] = (
        round(100 * totals["cached_tokens"] / measured, 1) if measured else None
    )
    return totals


def _group_warnings(warnings):
    """Collapse repeated drops per model and cause; keep the first event."""
    grouped = {}
    for warning in warnings:
        key = (warning["model"], warning["code"])
        if key in grouped:
            grouped[key]["count"] += 1
        else:
            grouped[key] = {**warning, "count": 1}
    return list(grouped.values())


def summarize_prompt_cache(events):
    """Input is already scoped, deduplicated model-result events."""
    counts = dict(
        hit=0, write=0, no_reuse=0, disabled=0, unknown=0, configuration_warning=0
    )
    cached_tokens = measured_input_tokens = cache_write_tokens = 0
    models, previous, warnings = {}, {}, []
    for event in sorted(events, key=lambda e: str(e.get("timestamp") or "")):
        state = cache_state(event)
        counts[state] += 1
        label = (
            f"{event.get('provider') or 'unknown'} / {event.get('model') or 'unknown'}"
        )
        row = models.setdefault(label, {"label": label, **dict.fromkeys(counts, 0)})
        row[state] += 1
        cached = token_count(event.get("cached_tokens"))
        inputs = token_count(event.get("input_tokens"))
        if cached is not None and inputs is not None and cached <= inputs:
            cached_tokens += cached
            measured_input_tokens += inputs
        cache_write_tokens += token_count(event.get("cache_write_tokens")) or 0
        scope = tuple(
            event.get(k)
            for k in (
                "application_id",
                "user_id",
                "agent_id",
                "thread_id",
                "provider",
                "model",
                "response_model",
                "operation",
            )
        )
        prior = previous.get(scope)
        try:
            moment = datetime.fromisoformat(
                str(event.get("timestamp")).replace("Z", "+00:00")
            )
        except ValueError:
            moment = None
        # Only compare nearby calls after an observed hit. A cold first request,
        # short prompt, changed model, unknown timestamp or expired cache is not
        # evidence that cache reuse has broken.
        if prior and moment and event.get("status") == "success" and cached == 0:
            before, at = prior
            try:
                nearby = 0 <= (moment - at).total_seconds() < 240
            except TypeError:
                nearby = False
            if (
                nearby
                and before.get("prompt_cache_prefix")
                and event.get("prompt_cache_prefix")
            ):
                changed = before["prompt_cache_prefix"] != event["prompt_cache_prefix"]
                key_changed = before.get("prompt_cache_key") != event.get(
                    "prompt_cache_key"
                )
                warnings.append(
                    {
                        "model": label,
                        "code": "prefix_changed"
                        if changed
                        else "key_changed"
                        if key_changed
                        else "reuse_dropped",
                        "message": "Cache reuse dropped after the instructions, tools or settings changed."
                        if changed
                        else "Cache reuse dropped after the routing key changed."
                        if key_changed
                        else "No cache read after a recent hit. Check message history, cache boundaries and provider eviction.",
                        "event_id": event.get("event_id"),
                    }
                )
                previous.pop(scope, None)
        if (
            state == "hit"
            and moment
            and event.get("status") == "success"
            and event.get("agent_id")
            and event.get("thread_id")
        ):
            previous[scope] = (event, moment)
    measured_calls = counts["hit"] + counts["write"] + counts["no_reuse"]
    return {
        **counts,
        "calls": sum(counts.values()),
        # Share of calls with reported cache usage that read from the cache.
        "hit_rate_percent": round(100 * counts["hit"] / measured_calls, 1)
        if measured_calls
        else None,
        "cache_write_tokens": cache_write_tokens,
        "cached_tokens": cached_tokens,
        "measured_input_tokens": measured_input_tokens,
        "read_percent": round(100 * cached_tokens / measured_input_tokens, 1)
        if measured_input_tokens
        else None,
        "attention": bool(
            counts["disabled"] or counts["configuration_warning"] or warnings
        ),
        "warnings": _group_warnings(warnings)[:20],
        "models": list(models.values()),
    }
