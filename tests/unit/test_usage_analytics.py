"""Pricing, bounded aggregation, actual runtime capture and authorized UI reads."""

from decimal import Decimal
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz import MemAgent
from memorizz.llms.response_metadata import response_metadata
from memorizz.observability import (
    DEFAULT_PRICING,
    PricingRegistry,
    RateCard,
    aggregate_usage,
    query_usage,
)
from memorizz.ui import state
from memorizz.ui.analytics import chart, eval_charts
from memorizz.ui.app import create_app


def result(**updates):
    return {
        "trace_kind": "model_result",
        "span_id": "call",
        "event_id": "result",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_tokens": 1000,
        "output_tokens": 100,
        "cached_tokens": 200,
        "duration_ms": 25,
        "agent_id": "agent",
        "thread_id": "thread",
        "root_trace_id": "root",
        "turn_id": "turn",
        "timestamp": "2026-09-07T00:30:00Z",
        **updates,
    }


def test_exact_decimal_cache_rates_and_long_context():
    assert DEFAULT_PRICING.quote(result())["cost_usd"] == "0.000195"
    quote = DEFAULT_PRICING.quote(
        result(model="gpt-5.5", input_tokens=300000, cached_tokens=100000)
    )
    assert Decimal(quote["cost_usd"]) == Decimal("2.1045")
    assert (
        DEFAULT_PRICING.quote(result(model="gpt-4o-mini-2024-07-18"))["cost_status"]
        == "calculated"
    )


def test_cache_writes_and_zero_usage():
    usage = result(
        model="gpt-5.6-terra",
        input_tokens=1000,
        cached_tokens=400,
        cache_write_tokens=200,
    )
    assert Decimal(DEFAULT_PRICING.quote(usage)["cost_usd"]) == Decimal(".00258")
    assert (
        DEFAULT_PRICING.quote(result(input_tokens=0, output_tokens=0, cached_tokens=0))[
            "cost_usd"
        ]
        == "0.000"
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"input_tokens": None},
        {"output_tokens": -1},
        {"cached_tokens": None},
        {"input_tokens": True},
        {"output_tokens": 2.5},
        {"cached_tokens": 1001},
        {"model": "unknown"},
        {"provider": "azure"},
        {"provider": "openai_compatible"},
        {"service_tier": "priority"},
        {"model": "gpt-5.6-luna"},
    ],
)
def test_unknown_is_never_free(updates):
    quote = DEFAULT_PRICING.quote(result(**updates))
    assert quote["cost_usd"] is None
    assert quote["cost_status"] == "unknown"


def test_custom_registry_and_recorded_snapshot_do_not_reprice():
    registry = PricingRegistry(
        [
            RateCard(
                "private",
                "reader",
                "1",
                "2",
                ".1",
                "https://example.test/rates",
                "2026-09-07",
                "contract-v1",
            )
        ]
    )
    event = result(provider="private", model="reader")
    event.update(registry.quote(event))
    usage = aggregate_usage([event])
    assert usage["totals"]["priced_calls"] == 1
    assert usage["pricing"][0]["basis"] == "recorded_snapshot"
    assert DEFAULT_PRICING.quote(result(provider="private"))["cost_usd"] is None
    with pytest.raises(TypeError):
        registry.cards[("a", "b", "c")] = None
    with pytest.raises(ValueError):
        RateCard("a", "b", "NaN", 1, 1, "source", "date", "version")


def test_start_parent_duplicate_and_tenant_scope_are_not_double_counted():
    events = [
        result(trace_kind="model_call"),
        result(trace_kind="run_result", cost_usd=99),
        result(),
        result(event_id="duplicate"),
        result(user_id="another-user"),
        result(span_id="second", input_tokens=None),
    ]
    usage = aggregate_usage(events)
    assert usage["totals"]["calls"] == 3
    assert usage["totals"]["priced_calls"] == 2
    assert usage["totals"]["unknown_usage_calls"] == 1
    assert usage["coverage"]["duplicates_removed"] == 1
    assert len(usage["interactions"]) == 2
    assert usage["totals"]["cost_usd"] == "0.000390"


def test_v3_attributes_day_boundaries_and_missing_timestamps():
    event = result()
    event.pop("trace_kind")
    event.update(event_kind="model_call", phase="result")
    attributes = {
        key: event.pop(key)
        for key in (
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
        )
    }
    event["attributes"] = attributes
    usage = aggregate_usage(
        [event, result(span_id="bad-time", timestamp="invalid")],
        timezone_name="America/Los_Angeles",
    )
    assert usage["daily"][0]["label"] == "2026-09-06"
    assert usage["coverage"]["missing_timestamps"] == 1
    assert usage["totals"]["calls"] == 2


def test_memory_supply_repeat_consumption_and_separate_retrieval():
    events = [
        result(
            trace_kind="memory_retrieval",
            memory_type="knowledge_base",
            event_id="read",
            duration_ms=40,
        ),
        result(
            trace_kind="memory_supply",
            memory_type="knowledge_base",
            event_id="supply1",
            memory_chars=400,
            memory_tokens_estimate=100,
        ),
        result(
            trace_kind="memory_supply",
            memory_type="knowledge_base",
            event_id="supply2",
            memory_chars=400,
            memory_tokens_estimate=100,
        ),
    ]
    row = aggregate_usage(events)["memory"][0]
    assert row["memory_tokens_estimate"] == 200
    assert row["retrieval_total_ms"] == 40
    assert row["retrieval_mean_ms"] == 40
    assert row["supply_samples"] == 2


def test_bounded_generator_is_not_exhausted():
    consumed = []

    def events():
        for index in range(1000):
            consumed.append(index)
            yield result(span_id=str(index))

    usage = aggregate_usage(events(), max_events=10)
    assert len(consumed) == 11
    assert usage["coverage"]["aggregation_truncated"]
    assert usage["totals"]["calls"] == 10


def test_provider_response_captures_cache_writes_tier_and_usage():
    metadata = response_metadata(
        NS(
            service_tier="priority",
            usage=NS(
                input_tokens=1000,
                output_tokens=50,
                total_tokens=1050,
                input_tokens_details=NS(cached_tokens=300, cache_write_tokens=100),
            ),
        )
    )
    assert metadata["cache_write_tokens"] == 100
    assert metadata["service_tier"] == "priority"
    assert metadata["input_tokens"] == 1000


def test_new_model_long_context_includes_cache_write_multiplier():
    usage = result(
        model="gpt-5.6-terra",
        input_tokens=300000,
        cached_tokens=100000,
        cache_write_tokens=100000,
        output_tokens=1000,
    )
    assert Decimal(DEFAULT_PRICING.quote(usage)["cost_usd"]) == Decimal(".958")


class FakeModel:
    def get_config(self):
        return {"model": "gpt-4o-mini", "provider": "openai"}

    def get_last_usage(self):
        return {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 200}

    def generate(self, messages, tools=None):
        return "answer"

    def generate_stream(self, messages, tools=None, **kwargs):
        if tools and any(
            tool.get("function", {}).get("name") == "memorizz_finalize_answer"
            for tool in tools
        ):
            from memorizz.llms.streaming import tool_response

            yield {
                "type": "tool_calls",
                "response": tool_response(
                    [
                        {
                            "id": "finalize",
                            "name": "memorizz_finalize_answer",
                            "arguments": "{}",
                        }
                    ],
                    "",
                ),
            }
            return
        yield {"type": "content", "content": "answer"}
        yield {"type": "usage", "usage": self.get_last_usage()}
        yield {"type": "done", "content": "answer"}


@pytest.mark.parametrize("streaming", [False, True])
def test_full_agent_delivery_persistence_and_usage_endpoint(
    tmp_path, monkeypatch, streaming
):
    from memorizz.enums import MemoryType
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "{}")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_DUAL_WRITE", "true")
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_READ_PATH", "index")
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    provider.get_observability_index().initialize()
    agent = MemAgent(
        model=FakeModel(),
        memory_provider=provider,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
        name="priced-agent",
        application_id="usage-app",
        context_policy={"progressive_tool_disclosure": False},
    )
    agent._schedule_conversation_embedding_backfill = lambda rows: None
    kwargs = {"memory_id": "memory", "thread_id": "thread", "user_id": "alice"}
    try:
        if streaming:
            events = list(agent.run_stream_events("question", **kwargs))
            assert events[-1]["status"] == "completed", events[-1]
            assert (
                "".join(e["delta"] for e in events if e["type"] == "answer.delta")
                == "answer"
            )
        else:
            assert agent.run("question", **kwargs) == "answer"
        with patch.dict(state._state, provider=provider):
            client = TestClient(create_app())
            response = client.get(
                "/traces/usage.json",
                params={
                    "agent_id": agent.agent_id,
                    "user_id": "alice",
                    "application_id": "usage-app",
                },
            )
            assert response.status_code == 200, response.text
            usage = response.json()
            assert usage["totals"]["calls"] == (2 if streaming else 1)
            assert Decimal(usage["totals"]["cost_usd"]) == Decimal("0.000195") * (
                2 if streaming else 1
            )
            assert usage["pricing"][0]["basis"] == "recorded_snapshot"
            assert "question" not in response.text
    finally:
        provider.close()


def test_public_recorder_accepts_quote_and_memory_fields(tmp_path):
    from memorizz.observability.models import validate_attributes

    attributes = {
        **DEFAULT_PRICING.quote(result()),
        "memory_type": "history",
        "memory_chars": 400,
        "memory_tokens_estimate": 100,
        "token_estimation_method": "rendered_chars_div_4",
    }
    assert validate_attributes(attributes) == attributes


def test_context_copy_does_not_mutate_parent_memory_counters():
    from contextvars import copy_context

    agent = MemAgent(model=FakeModel())
    agent._record_memory_chars("history", "parent")
    copy_context().run(agent._record_memory_chars, "history", "child")
    assert agent._memory_usage_chars == {"history": 6}


def test_runtime_prompt_unchanged_memory_timings_and_persisted_quotes():
    agent = MemAgent(model=FakeModel())
    agent._begin_trace_turn(None)
    context = {
        "retrieved_memories": [
            {"source": "knowledge_base", "text": "PRIVATE-MEMORY", "id": "one"}
        ],
        "conversation_history": [{"role": "user", "content": "history"}],
    }
    with patch.object(agent, "_record_memory_chars"):
        baseline = agent._build_prompt_messages("system", "question", context)
    messages = agent._build_prompt_messages("system", "question", context)
    assert messages == baseline
    assert agent._timed_memory_read("knowledge_base", lambda: ["private"]) == [
        "private"
    ]
    with pytest.raises(RuntimeError):
        agent._timed_memory_read(
            "entity", lambda: (_ for _ in ()).throw(RuntimeError())
        )
    for iteration in range(2):
        agent._generate_with_trace(
            messages, tools=None, iteration=iteration, stage="test"
        )
    usage = agent.last_usage_analytics()
    assert usage["totals"]["calls"] == 2
    assert usage["totals"]["priced_calls"] == 2
    memory = {row["label"]: row for row in usage["memory"]}
    assert memory["knowledge_base"]["supply_samples"] == 2
    assert memory["knowledge_base"]["retrieval_samples"] == 1
    assert memory["entity"]["retrieval_errors"] == 1
    assert "PRIVATE-MEMORY" not in str(usage)
    rows = agent._build_trace_bundle_events(agent._stream_trace_events)
    assert next(row for row in rows if row["trace_kind"] == "model_result")[
        "pricing_version"
    ]
    assert (
        next(row for row in rows if row["trace_kind"] == "memory_supply")[
            "memory_chars"
        ]
        > 0
    )
    agent._begin_trace_turn(None)
    assert not agent._memory_usage_chars


def test_paginated_reads_are_bounded_and_report_unvisited_stores(monkeypatch):
    calls = []

    class Provider:
        def query_trace_events(self, **kwargs):
            calls.append(kwargs)
            return {
                "items": [result(span_id=str(i)) for i in range(kwargs["limit"])],
                "next_cursor": "next",
            }

    usage = query_usage(
        Provider(), filters={"application_id": "app", "user_id": "alice"}, max_events=12
    )
    assert len(calls) == 1
    assert calls[0]["application_id"] == "app"
    assert usage["totals"]["calls"] == 12
    assert usage["coverage"]["read_complete"] is False


def test_chart_models_skip_nonfinite_and_empty_values():
    data = chart(
        [
            {"label": "bad", "value": float("nan")},
            {"label": "missing"},
            {"label": "zero", "value": 0},
        ],
        "value",
    )
    assert len(data["rows"]) == 1
    assert data["total"] == 0
    assert not eval_charts(None, [])["cost"]["rows"]


def test_ui_empty_page_validation_and_custom_pricing(tmp_path, monkeypatch):
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "{}")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    with patch.dict(state._state, provider=provider):
        client = TestClient(create_app())
        response = client.get("/traces/usage")
        assert response.status_code == 200, response.text
        assert "Usage &amp; memory analytics" in response.text
        assert "Unknown" in response.text
        assert client.get("/traces/usage.json").json()["totals"]["cost_usd"] is None
        assert client.get("/traces/usage.json?timezone_name=invalid").status_code == 400
        assert client.get("/traces/usage.json?max_events=10001").status_code == 422
        assert client.get("/traces/usage.json?start_time=invalid").status_code == 400
    provider.close()


@pytest.mark.parametrize(
    "metadata,complete",
    [
        ({"read_completeness": "complete"}, True),
        ({"read_completeness": "complete", "normalization_errors": 1}, False),
        ({"read_completeness": "complete", "truncated": True}, False),
        ({"read_completeness": "partial", "read_complete": True}, False),
        ({}, False),
    ],
)
def test_read_completeness_uses_normalized_metadata_not_capture_claims(
    metadata, complete
):
    assert (
        aggregate_usage([], coverage=metadata)["coverage"]["read_complete"] is complete
    )
