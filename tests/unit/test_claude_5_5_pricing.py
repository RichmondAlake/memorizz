"""Claude Haiku 5.5 is priced by prompt length; Sonnet 5.5 cache hits are 0.05x."""

from __future__ import annotations

from decimal import Decimal

import pytest

from memorizz.benchmarks.measurement import token_price, usage_cost
from memorizz.observability.pricing import DEFAULT_PRICING

pytestmark = pytest.mark.unit


def _quote(model, **usage):
    return DEFAULT_PRICING.quote({"provider": "anthropic", "model": model, **usage})


def test_haiku_5_5_pays_five_times_over_100k_prompt_tokens():
    usage = dict(cached_tokens=40_000, cache_write_tokens=10_000, output_tokens=2_000)
    short = _quote("claude-haiku-5-5", input_tokens=100_000, **usage)
    long = _quote("claude-haiku-5-5", input_tokens=100_001, **usage)
    # 50k input x $0.10, 40k cache reads x $0.01, 10k writes x $0.125, 2k x $0.50
    assert Decimal(short["cost_usd"]) == Decimal("0.00765")
    # The same request just over the line: every rate x5.
    assert Decimal(long["cost_usd"]) == Decimal("0.0382505")
    assert short["pricing_as_of"] == "2026-10-21"
    over_200k = _quote(
        "claude-haiku-5-5",
        input_tokens=300_000,
        cached_tokens=0,
        cache_write_tokens=0,
        output_tokens=0,
    )
    assert Decimal(over_200k["cost_usd"]) == Decimal("0.15")


def test_sonnet_5_5_cache_hits_are_ten_cents_per_million():
    quote = _quote(
        "claude-sonnet-5-5",
        input_tokens=1_000_000,
        cached_tokens=1_000_000,
        cache_write_tokens=0,
        output_tokens=0,
    )
    assert Decimal(quote["cost_usd"]) == Decimal("0.10")


def test_benchmark_prices_match_including_one_hour_writes():
    price = token_price("anthropic", "claude-haiku-5-5")
    usage = {
        "prompt_tokens": 100_001,
        "cached_tokens": 40_000,
        "completion_tokens": 2_000,
        "cache_write_tokens": 10_000,
        "cache_write_1h_tokens": 4_000,
    }
    # 50,001 x $0.50 + 40k x $0.05 + 6k x $0.625 + 4k x $1.00 + 2k x $2.50
    assert usage_cost("anthropic", usage, price) == pytest.approx(0.0397505)
    assert usage_cost(
        "anthropic", {**usage, "prompt_tokens": 100_000}, price
    ) == pytest.approx(0.00795)
    assert token_price("anthropic", "claude-sonnet-5-5")["cached_input"] == 0.1
