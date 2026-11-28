"""Lightweight process-wide hooks for auxiliary model spend (background & embeddings).

Chat turns report usage through the streaming event loop. Background extraction
(such as post-turn entity-fact filing via :func:`text_generator`) and embedding
calls run outside that stream; callers that enforce a real-time USD budget can
register a listener here to observe those calls without polling provider bills.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .pricing import DEFAULT_PRICING, PricingRegistry

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_listeners: Dict[int, Callable[[Dict[str, Any]], None]] = {}
_next_id = 0
_thread_state = threading.local()


def clear_thread_usage() -> None:
    """Clear the current thread's recorded model-call usage."""
    _thread_state.last_usage = None


def record_thread_usage(model: Any, usage: Optional[Mapping[str, Any]]) -> None:
    """Store the current thread's usage for ``model`` without cross-thread races."""
    copied = dict(usage) if isinstance(usage, Mapping) else None
    _thread_state.last_usage = (id(model), copied)


def consume_thread_usage(model: Any) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Return ``(recorded_on_this_thread, usage)`` for ``model`` and clear the slot."""
    entry = getattr(_thread_state, "last_usage", None)
    _thread_state.last_usage = None
    if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == id(model):
        return True, entry[1]
    return False, None


def register_usage_listener(
    callback: Callable[[Dict[str, Any]], None],
) -> Callable[[], None]:
    """Register ``callback`` for auxiliary model usage events; return an unregister handle."""
    global _next_id
    with _lock:
        _next_id += 1
        token = _next_id
        _listeners[token] = callback

    def _unregister() -> None:
        with _lock:
            _listeners.pop(token, None)

    return _unregister


def _int_value(raw: Any) -> int:
    if isinstance(raw, bool) or raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def normalize_provider_usage(usage: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    """Normalize token keys across OpenAI, Anthropic, Gemini, and local providers."""
    if not isinstance(usage, Mapping):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }
    input_tokens = _int_value(
        usage.get("input_tokens")
        if usage.get("input_tokens") is not None
        else usage.get("prompt_tokens")
    )
    output_tokens = _int_value(
        usage.get("output_tokens")
        if usage.get("output_tokens") is not None
        else usage.get("completion_tokens")
    )
    cached_tokens = _int_value(
        usage.get("cached_tokens")
        if usage.get("cached_tokens") is not None
        else usage.get("cache_read_input_tokens")
    )
    cache_write_tokens = _int_value(
        usage.get("cache_write_tokens")
        if usage.get("cache_write_tokens") is not None
        else usage.get("cache_creation_input_tokens")
    )
    if input_tokens == 0 and output_tokens == 0 and usage.get("total_tokens"):
        input_tokens = _int_value(usage.get("total_tokens"))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
    }


def report_auxiliary_usage(
    *,
    kind: str,
    provider: str,
    model: str,
    usage: Optional[Mapping[str, Any]] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    pricing: Optional[PricingRegistry] = None,
) -> Optional[Dict[str, Any]]:
    """Quote one auxiliary model call and dispatch it to registered listeners.

    Returns immediately when no listeners are active.
    """
    with _lock:
        if not _listeners:
            return None
        snapshot = list(_listeners.values())

    counts = (
        normalize_provider_usage(usage)
        if usage is not None
        else {
            "input_tokens": _int_value(input_tokens),
            "output_tokens": _int_value(output_tokens),
            "cached_tokens": _int_value(cached_tokens),
            "cache_write_tokens": _int_value(cache_write_tokens),
        }
    )
    if counts["input_tokens"] == 0 and counts["output_tokens"] == 0:
        return None

    prov = str(provider or "").strip().lower()
    model_name = str(model or "").strip()
    cost_usd: Optional[float] = None
    unpriced = False
    if prov and model_name:
        try:
            registry = pricing or DEFAULT_PRICING
            quote = registry.quote(
                {
                    "provider": prov,
                    "model": model_name,
                    "input_tokens": counts["input_tokens"],
                    "output_tokens": counts["output_tokens"],
                    "cached_tokens": counts["cached_tokens"],
                    "cache_write_tokens": counts["cache_write_tokens"],
                }
            )
            if quote.get("cost_usd") is not None:
                cost_usd = float(quote["cost_usd"])
            else:
                unpriced = True
        except Exception:
            unpriced = True
    else:
        unpriced = True

    event = {
        "kind": str(kind or "background"),
        "provider": prov,
        "model": model_name,
        "cost_usd": cost_usd,
        "input_tokens": counts["input_tokens"],
        "output_tokens": counts["output_tokens"],
        "cached_tokens": counts["cached_tokens"],
        "cache_write_tokens": counts["cache_write_tokens"],
        "unpriced": unpriced,
    }
    for callback in snapshot:
        try:
            callback(event)
        except Exception:
            logger.debug("Auxiliary usage listener failed", exc_info=True)
    return event
