"""Offline, auditable text-token pricing. Calculated charges are not invoices.

Rates are USD per million tokens. Unknown usage/models/tiers are never priced
as zero. Register separate cards for Azure, Bedrock, discounts or other regions;
direct OpenAI list prices must not be applied to those deployments implicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit

from .privacy import validate_opaque

PRICING_FIELDS = (
    "cost_usd",
    "cost_status",
    "cost_reason",
    "pricing_version",
    "pricing_source",
    "pricing_as_of",
    "service_tier",
    "cache_write_tokens",
)


def token_count(value):
    """Accept measured non-negative integer counts, never booleans or estimates."""
    return value if type(value) is int and value >= 0 else None


@dataclass(frozen=True)
class RateCard:
    provider: str
    model: str
    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal
    source_url: str
    as_of: str
    version: str
    service_tier: str = "default"
    cache_write_per_million: Decimal | None = None
    # Refuse unsupported long-context schedules instead of silently undercharging.
    max_input_tokens: int | None = None
    long_context_threshold: int | None = None
    long_input_multiplier: Decimal = Decimal(1)
    long_output_multiplier: Decimal = Decimal(1)

    def __post_init__(self):
        for key in (
            "input_per_million",
            "output_per_million",
            "cached_input_per_million",
            "cache_write_per_million",
            "long_input_multiplier",
            "long_output_multiplier",
        ):
            value = getattr(self, key)
            if value is None and key == "cache_write_per_million":
                continue
            try:
                number = Decimal(str(value))
            except InvalidOperation as exc:
                raise ValueError(f"Invalid rate: {key}") from exc
            if not number.is_finite() or number < 0:
                raise ValueError(f"Invalid rate: {key}")
            object.__setattr__(self, key, number)
        if not all(
            (self.provider, self.model, self.source_url, self.as_of, self.version)
        ):
            raise ValueError("Rate cards require identity and pricing provenance")
        date.fromisoformat(self.as_of)
        source = urlsplit(self.source_url)
        if (
            source.scheme != "https"
            or not source.hostname
            or source.query
            or source.fragment
            or source.username
            or source.password
        ):
            raise ValueError(
                "Pricing sources must be public HTTPS URLs without credentials or query strings"
            )
        for value in (
            self.provider,
            self.model,
            self.version,
            self.service_tier,
            self.source_url.removeprefix("https://"),
        ):
            validate_opaque(value)
            if len(value) > 240:
                raise ValueError(
                    "Pricing identifiers must be bounded to 240 characters"
                )


@dataclass(frozen=True)
class PricingRegistry:
    """Immutable registry safe to share between concurrent agent runs.

    Pass a custom registry to ``aggregate_usage(..., pricing=registry)`` or set
    ``agent.usage_pricing = registry`` to snapshot charges when calls complete.
    """

    cards: Mapping

    def __init__(self, cards=()):
        object.__setattr__(
            self,
            "cards",
            MappingProxyType(
                {
                    (card.provider.lower(), card.model, card.service_tier): card
                    for card in cards
                }
            ),
        )

    def quote(self, usage: Mapping) -> dict:
        def unknown(reason):
            return {"cost_usd": None, "cost_status": "unknown", "cost_reason": reason}

        provider = str(usage.get("provider") or "").lower()
        model = str(usage.get("model") or "")
        tier = str(usage.get("service_tier") or "default")
        # Only OpenAI's dated snapshots inherit a base model's list rate.
        card = self.cards.get((provider, model, tier))
        if card is None and provider == "openai":
            base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
            card = self.cards.get((provider, base, tier))
        if card is None:
            return unknown("No rate card for this provider, model and service tier")
        inputs = token_count(usage.get("input_tokens"))
        outputs = token_count(usage.get("output_tokens"))
        cached = token_count(usage.get("cached_tokens"))
        if inputs is None or outputs is None or cached is None:
            return unknown("Input, output or cached-token usage was not reported")
        writes = token_count(usage.get("cache_write_tokens"))
        if writes is None:
            if card.cache_write_per_million is not None:
                return unknown("Cache-write usage was not reported")
            writes = 0
        if cached + writes > inputs:
            return unknown("Cached and cache-write tokens exceed input tokens")
        if card.max_input_tokens is not None and inputs > card.max_input_tokens:
            return unknown("Long-context pricing requires a matching custom rate card")
        long = (
            card.long_context_threshold is not None
            and inputs > card.long_context_threshold
        )
        input_multiplier = card.long_input_multiplier if long else Decimal(1)
        output_multiplier = card.long_output_multiplier if long else Decimal(1)
        write_rate = card.cache_write_per_million
        if write_rate is None:
            write_rate = card.input_per_million
        cost = (
            (
                (inputs - cached - writes) * card.input_per_million
                + cached * card.cached_input_per_million
                + writes * write_rate
            )
            * input_multiplier
            + outputs * card.output_per_million * output_multiplier
        ) / Decimal(1_000_000)
        return {
            "cost_usd": str(cost),
            "cost_status": "calculated",
            "cost_reason": "List-rate calculation; not reconciled to an invoice"
            + ("; standard tier assumed" if not usage.get("service_tier") else ""),
            "pricing_version": card.version,
            # A scheme-free public source reference fits the content-free trace
            # contract, which deliberately prohibits arbitrary application URLs.
            "pricing_source": card.source_url.removeprefix("https://"),
            "pricing_as_of": card.as_of,
        }


def _openai_cards():
    rates = {
        "gpt-5.5": (5, 0.5, 30),
        "gpt-5.4": (2.5, 0.25, 15),
        "gpt-5.4-mini": (0.75, 0.075, 4.5),
        "gpt-5.4-nano": (0.2, 0.02, 1.25),
        "gpt-5.2": (1.75, 0.175, 14),
        "gpt-5.1": (1.25, 0.125, 10),
        "gpt-5": (1.25, 0.125, 10),
        "gpt-5-mini": (0.25, 0.025, 2),
        "gpt-5-nano": (0.05, 0.005, 0.4),
        "gpt-4.1": (2, 0.5, 8),
        "gpt-4.1-mini": (0.4, 0.1, 1.6),
        "gpt-4.1-nano": (0.1, 0.025, 0.4),
        "gpt-4o": (2.5, 1.25, 10),
        "gpt-4o-mini": (0.15, 0.075, 0.6),
        "o3": (2, 0.5, 8),
        "o4-mini": (1.1, 0.275, 4.4),
        "gpt-6-astra": (10, 1, 50),
        "gpt-5.6-sol": (4, 0.4, 20),
        "gpt-5.6-terra": (2, 0.2, 12),
        "gpt-5.6-luna": (0.2, 0.02, 1.2),
    }
    for model, (inputs, cached, outputs) in rates.items():
        new_cache = model.startswith(("gpt-5.6", "gpt-6-"))
        long = model in {"gpt-5.5", "gpt-5.4"} or new_cache
        yield RateCard(
            provider="openai",
            model=model,
            input_per_million=inputs,
            cached_input_per_million=cached,
            output_per_million=outputs,
            cache_write_per_million=Decimal(str(inputs)) * Decimal("1.25")
            if new_cache
            else None,
            source_url="https://developers.openai.com/api/docs/pricing",
            as_of="2026-09-07",
            version="openai-standard-2026-09-07",
            long_context_threshold=272_000 if long else None,
            long_input_multiplier=Decimal(2) if long else Decimal(1),
            long_output_multiplier=Decimal("1.5") if long else Decimal(1),
        )


DEFAULT_PRICING = PricingRegistry(_openai_cards())
