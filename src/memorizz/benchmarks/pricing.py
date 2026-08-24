"""Small, explicit pricing registry for bounded benchmark forecasts.

The registry is deliberately not a general billing system.  It contains only
models used by MemoRizz benchmark adapters, records the source URL and date,
and fails closed when a caller asks it to price an unknown model.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class OpenAITextPricing:
    """Published token prices in US dollars per million tokens."""

    model: str
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    source_url: str
    as_of: str
    long_context_threshold: int | None = None
    long_input_multiplier: float = 1.0
    long_output_multiplier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OPENAI_TEXT_PRICING: Mapping[str, OpenAITextPricing] = {
    "gpt-5.6-terra": OpenAITextPricing(
        model="gpt-5.6-terra",
        input_per_million=2.0,
        cached_input_per_million=0.2,
        output_per_million=12.0,
        source_url="https://developers.openai.com/api/docs/models/gpt-5.6-terra",
        as_of="2026-08-23",
        long_context_threshold=272_000,
        long_input_multiplier=2.0,
        long_output_multiplier=1.5,
    ),
    "gpt-5.6-luna": OpenAITextPricing(
        model="gpt-5.6-luna",
        input_per_million=0.2,
        cached_input_per_million=0.02,
        output_per_million=1.2,
        source_url="https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        as_of="2026-08-23",
    ),
    "gpt-5.5": OpenAITextPricing(
        model="gpt-5.5",
        input_per_million=5.0,
        cached_input_per_million=0.5,
        output_per_million=30.0,
        source_url="https://developers.openai.com/api/docs/models/gpt-5.5",
        as_of="2026-08-23",
    ),
    "gpt-5-mini": OpenAITextPricing(
        model="gpt-5-mini",
        input_per_million=0.25,
        cached_input_per_million=0.025,
        output_per_million=2.0,
        source_url="https://developers.openai.com/api/docs/models/gpt-5-mini",
        as_of="2026-08-23",
    ),
    "gpt-5.2": OpenAITextPricing(
        model="gpt-5.2",
        input_per_million=1.75,
        cached_input_per_million=0.175,
        output_per_million=14.0,
        source_url="https://developers.openai.com/api/docs/models/gpt-5.2",
        as_of="2026-08-23",
    ),
}


OPENAI_EMBEDDING_PRICING: Mapping[str, dict[str, Any]] = {
    "text-embedding-3-small": {
        "input_per_million": 0.02,
        "source_url": (
            "https://developers.openai.com/api/docs/models/" "text-embedding-3-small"
        ),
        "as_of": "2026-08-23",
    }
}


def resolve_openai_text_pricing(model: str) -> OpenAITextPricing:
    """Resolve a base or dated model name, rejecting unpriced models."""

    normalized = str(model or "").strip().lower()
    for base_name in sorted(OPENAI_TEXT_PRICING, key=len, reverse=True):
        suffix = normalized.removeprefix(base_name + "-")
        dated_snapshot = normalized.startswith(base_name + "-") and bool(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[.-].+)?", suffix)
        )
        if normalized == base_name or dated_snapshot:
            return OPENAI_TEXT_PRICING[base_name]
    supported = ", ".join(sorted(OPENAI_TEXT_PRICING))
    raise ValueError(
        f"No benchmark pricing is registered for OpenAI model {model!r}. "
        f"Supported models: {supported}."
    )


def estimate_openai_text_cost(model: str, usage: Mapping[str, Any]) -> float:
    """Estimate one response from provider-reported token usage."""

    pricing = resolve_openai_text_pricing(model)
    prompt_tokens = max(0, int(usage.get("prompt_tokens") or 0))
    completion_tokens = max(0, int(usage.get("completion_tokens") or 0))
    cached_tokens = min(
        prompt_tokens,
        max(0, int(usage.get("cached_tokens") or 0)),
    )
    uncached_tokens = prompt_tokens - cached_tokens
    is_long = bool(
        pricing.long_context_threshold is not None
        and prompt_tokens > pricing.long_context_threshold
    )
    input_multiplier = pricing.long_input_multiplier if is_long else 1.0
    output_multiplier = pricing.long_output_multiplier if is_long else 1.0
    return (
        uncached_tokens * pricing.input_per_million * input_multiplier
        + cached_tokens * pricing.cached_input_per_million * input_multiplier
        + completion_tokens * pricing.output_per_million * output_multiplier
    ) / 1_000_000


def estimate_openai_embedding_cost(model: str, input_tokens: int) -> float:
    """Estimate embedding input cost, rejecting unknown models."""

    normalized = str(model or "").strip().lower()
    try:
        pricing = OPENAI_EMBEDDING_PRICING[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(OPENAI_EMBEDDING_PRICING))
        raise ValueError(
            f"No benchmark pricing is registered for embedding model {model!r}. "
            f"Supported models: {supported}."
        ) from exc
    return max(0, int(input_tokens)) * float(pricing["input_per_million"]) / 1_000_000


__all__ = [
    "OPENAI_EMBEDDING_PRICING",
    "OPENAI_TEXT_PRICING",
    "OpenAITextPricing",
    "estimate_openai_embedding_cost",
    "estimate_openai_text_cost",
    "resolve_openai_text_pricing",
]
