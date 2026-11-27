"""List rates for Gemini and for the newest OpenAI and Anthropic models."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from memorizz.llms.model_catalog import LLM_MODEL_CATALOG
from memorizz.observability.pricing import (
    DEFAULT_PRICING,
    PricingRegistry,
    _gemini_cards,
)

pytestmark = pytest.mark.unit

MILLION = {
    "input_tokens": 1_000_000,
    "output_tokens": 1_000_000,
    "cached_tokens": 0,
    "cache_write_tokens": 0,
}


def _cost(provider, model, registry=DEFAULT_PRICING, **usage):
    quote = registry.quote({"provider": provider, "model": model, **MILLION, **usage})
    return None if quote["cost_usd"] is None else Decimal(quote["cost_usd"])


def test_gemini_runs_are_priced_from_googles_list():
    small = {"input_tokens": 100_000, "output_tokens": 10_000, "cached_tokens": 40_000}
    # 60k at $3, 40k cached at $0.15 and 10k out at $15, per million.
    assert _cost("gemini", "gemini-4-neon", **small) == Decimal("0.336")
    assert _cost("gemini", "gemini-3.5-flash-lite", **small) == Decimal("0.0442")
    quote = DEFAULT_PRICING.quote(
        {"provider": "gemini", "model": "gemini-4-hydrogen", **MILLION}
    )
    assert quote["cost_usd"] == "9.000"
    assert quote["pricing_source"] == "aistudio.google.com/docs/pricing"


def test_gemini_3_1_pro_charges_more_for_a_prompt_over_200k():
    under = {"input_tokens": 200_000, "output_tokens": 1_000, "cached_tokens": 0}
    over = {"input_tokens": 300_000, "output_tokens": 1_000, "cached_tokens": 100_000}
    assert _cost("gemini", "gemini-3.1-pro-preview", **under) == Decimal("0.412")
    # 200k at $4, 100k cached at $0.40 and 1k out at $18.
    assert _cost("gemini", "gemini-3.1-pro-preview", **over) == Decimal("0.858")


@pytest.mark.parametrize(
    "model, change, before, after",
    [
        ("gemini-4-carbon", date(2026, 12, 1), "12", "24"),
        ("gemini-4-argon", date(2026, 12, 1), "12", "24"),
        ("gemini-3.8-flash", date(2027, 1, 1), "4.5", "9"),
        ("gemini-3.6-flash", date(2027, 1, 1), "4.5", "9"),
    ],
)
def test_gemini_rates_google_has_dated_change_on_the_day(model, change, before, after):
    earlier = PricingRegistry(tuple(_gemini_cards(date(2026, 11, 26))))
    later = PricingRegistry(tuple(_gemini_cards(change)))

    assert _cost("gemini", model, earlier) == Decimal(before)
    assert _cost("gemini", model, later) == Decimal(after)
    assert later.cards[("gemini", model, "default")].as_of == change.isoformat()


def test_the_newest_openai_and_anthropic_models_are_priced():
    # Over 272k input tokens GPT-6 doubles input and adds half to output.
    assert _cost("openai", "gpt-6.2-sol") == Decimal("19")
    small = {"input_tokens": 100_000, "output_tokens": 10_000}
    assert _cost("openai", "gpt-6.2-sol", **small) == Decimal("0.3")
    assert _cost("openai", "gpt-6.2-astra", **small) == Decimal("1.2")
    assert _cost("openai", "gpt-6.1-terra", **small) == Decimal("0.15")
    assert _cost("openai", "gpt-6.2-sol", service_tier="fast", **small) == Decimal(
        "0.6"
    )
    assert _cost("openai", "gpt-6.2-astra", service_tier="batch", **small) == Decimal(
        "0.6"
    )
    # Flex is not sold on GPT-6.2, so it is unpriced rather than guessed.
    assert _cost("openai", "gpt-6.2-sol", service_tier="flex", **small) is None
    assert _cost("openai", "gpt-6.1-terra", service_tier="flex", **small) == Decimal(
        "0.075"
    )
    # 80k at $8, 10k cache writes at $10, 10k cache reads at $0.20, 10k out at $40.
    assert _cost(
        "anthropic",
        "claude-fable-5-2",
        input_tokens=100_000,
        output_tokens=10_000,
        cached_tokens=10_000,
        cache_write_tokens=10_000,
    ) == Decimal("1.142")


def test_every_model_in_the_pickers_has_a_price_or_a_stated_reason():
    unpriced = {
        (provider, row["value"])
        for provider in ("openai", "anthropic", "gemini")
        for row in LLM_MODEL_CATALOG[provider]
        if _cost(provider, row["value"], input_tokens=10, output_tokens=1) is None
    }
    # Google's price page does not list these two.
    assert unpriced == {("gemini", "gemini-3.7-flash"), ("gemini", "gemini-3.5-flash")}
