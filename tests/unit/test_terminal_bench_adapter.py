import time

import pytest

pytest.importorskip("harbor")

from memorizz.benchmarks.pricing import (  # noqa: E402
    estimate_openai_text_cost,
    resolve_openai_text_pricing,
)
from memorizz.benchmarks.terminal_bench import (  # noqa: E402
    _bounded_text,
    _TrackedOpenAI,
    estimate_terra_cost,
)


def test_bounded_text_preserves_both_ends():
    value = "abcdefghijklmnopqrstuvwxyz"

    result = _bounded_text(value, 12)

    assert result.startswith("abcdef")
    assert result.endswith("uvwxyz")
    assert "14 characters omitted" in result


def test_terra_cost_accounts_for_cached_input():
    cost = estimate_terra_cost(
        {
            "prompt_tokens": 1_000,
            "cached_tokens": 500,
            "completion_tokens": 100,
        }
    )

    assert cost == pytest.approx(0.0023)


def test_terra_cost_applies_long_context_multiplier():
    cost = estimate_terra_cost(
        {
            "prompt_tokens": 300_000,
            "cached_tokens": 0,
            "completion_tokens": 10_000,
        }
    )

    assert cost == pytest.approx(1.38)


def test_benchmark_cost_uses_the_selected_model_price():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 100_000}

    assert estimate_openai_text_cost("gpt-5.6-luna", usage) == pytest.approx(0.32)
    assert estimate_openai_text_cost("gpt-5.5", usage) == pytest.approx(8.0)
    assert resolve_openai_text_pricing("gpt-5-mini-2026-08-01").model == "gpt-5-mini"


def test_benchmark_cost_rejects_unknown_models():
    with pytest.raises(ValueError, match="No benchmark pricing"):
        estimate_openai_text_cost("future-unpriced-model", {})
    with pytest.raises(ValueError, match="No benchmark pricing"):
        estimate_openai_text_cost("gpt-5.5-pro", {})


def test_deadline_policy_forces_tool_free_finalization_without_mutating_history():
    provider = object.__new__(_TrackedOpenAI)
    provider.deadline_monotonic = time.monotonic() + 5
    provider.finalization_reserve_sec = 30
    provider.forced_finalization = False
    messages = [{"role": "user", "content": "Finish the task"}]
    tools = [{"type": "function", "function": {"name": "terminal_exec"}}]

    (
        effective_messages,
        effective_tools,
        tool_choice,
        forced,
    ) = provider._apply_deadline_policy(messages, tools, "auto")

    assert forced is True
    assert provider.forced_finalization is True
    assert effective_tools is None
    assert tool_choice == "auto"
    assert messages == [{"role": "user", "content": "Finish the task"}]
    assert effective_messages[-1]["role"] == "developer"
    assert "Do not call another tool" in effective_messages[-1]["content"]


def test_deadline_policy_warns_before_reserve_without_disabling_tools():
    provider = object.__new__(_TrackedOpenAI)
    provider.deadline_monotonic = time.monotonic() + 50
    provider.finalization_reserve_sec = 30
    provider.forced_finalization = False
    messages = [{"role": "user", "content": "Finish the task"}]
    tools = [{"type": "function", "function": {"name": "terminal_exec"}}]

    effective_messages, effective_tools, _, forced = provider._apply_deadline_policy(
        messages, tools, "auto"
    )

    assert forced is False
    assert effective_tools is tools
    assert "seconds remain" in effective_messages[-1]["content"]
