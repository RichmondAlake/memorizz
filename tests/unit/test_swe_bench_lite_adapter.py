import pytest

from memorizz.benchmarks.swe_bench_lite import estimate_gpt_5_4_mini_cost


def test_swe_bench_cost_estimate_accounts_for_cached_input():
    cost = estimate_gpt_5_4_mini_cost(
        {
            "prompt_tokens": 1_000,
            "cached_tokens": 400,
            "completion_tokens": 100,
        }
    )

    assert cost == pytest.approx(0.00093)
