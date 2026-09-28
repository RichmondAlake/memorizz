"""Controlled experiments: accounting, fairness, failures and durable UI data."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from memorizz.benchmarks.comparison import (
    ComparisonConfig,
    ComparisonRunner,
    ModelSpec,
    atomic_json,
    case_json,
    demo_cases,
    load_cases,
    run_comparison,
)
from memorizz.benchmarks.measurement import (
    MeasuredModel,
    MeasurementLedger,
    summarize_calls,
    token_price,
    usage_cost,
)
from memorizz.benchmarks.rerankers import Reranker
from memorizz.ui import state
from memorizz.ui.routers import comparisons as routes


def config(**kwargs):
    return ComparisonConfig(
        readers=[ModelSpec(provider="ollama", model="test")], **kwargs
    )


def test_cost_accounts_for_cache_without_double_counting_reasoning():
    price = token_price(
        "anthropic", "test", {"input": 2, "cached_input": 0.2, "output": 10}
    )
    usage = {
        "prompt_tokens": 1000,
        "cached_tokens": 200,
        "completion_tokens": 100,
        "reasoning_tokens": 80,
    }
    assert usage_cost("anthropic", usage, price) == pytest.approx(0.00264)
    assert usage_cost("anthropic", {}, price) is None
    assert token_price("openai", "not-a-known-model") is None
    assert usage_cost("ollama", {}, None) == 0


def test_latency_percentiles_interpolate_and_use_actual_median():
    from memorizz.benchmarks.memory_suite.runner import _percentile

    assert _percentile([1, 9], 0.5) == 5
    assert _percentile([1, 9], 0.95) == pytest.approx(8.6)
    assert _percentile([], 0.5) is None


def test_full_comparison_records_fixed_judge_oracle_and_serving_cost(
    tmp_path, monkeypatch
):
    class Model:
        def generate_text(self, prompt, instructions=None):
            if "grader" in (instructions or ""):
                return '{"score":1,"reason":"supported"}'
            return '{"answer":"tea","source_ids":["m1"],"abstained":false}'

        def get_last_usage(self):
            return {"prompt_tokens": 1000, "completion_tokens": 100}

    class Embeddings:
        def get_embedding(self, text):
            return [1.0, 0.0]

        def get_embeddings(self, texts):
            return [self.get_embedding(t) for t in texts]

    monkeypatch.setattr(
        "memorizz.llms.llm_factory.create_llm_provider", lambda config: Model()
    )
    dataset = tmp_path / "labeled.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "case_id": "q1",
                    "corpus_id": "c1",
                    "category": "preference",
                    "question": "Drink?",
                    "answers": ["tea"],
                    "scorer": "llm_judge",
                    "documents": [{"source_id": "m1", "content": "Asha prefers tea."}],
                    "relevant_source_ids": ["m1"],
                }
            ]
        )
    )
    pricing = {"input": 2, "cached_input": 0, "output": 10}
    c = ComparisonConfig(
        dataset="custom",
        data_path=str(dataset),
        oracle_reader=True,
        readers=[ModelSpec(provider="openai", model="test", pricing=pricing)],
        judge=ModelSpec(provider="anthropic", model="fixed-judge", pricing=pricing),
    )

    def factory(**kwargs):
        return ComparisonRunner(**kwargs, embedding_manager=Embeddings())

    result = run_comparison(c, tmp_path / "run", runner_factory=factory)
    assert result["status"] == "completed", result
    row = result["runs"][0]
    assert row["summary"]["accuracy"] == 1
    assert row["summary"]["serving_cost_usd"] == pytest.approx(0.003)
    # Oracle generation and grading, plus the retrieved answer's grading.
    assert row["summary"]["evaluation_cost_usd"] == pytest.approx(0.009)
    assert {call["lane"] for call in row["calls"]} == {"reader", "judge", "oracle"}
    assert (
        json.loads((tmp_path / "run" / "result.json").read_text())["runs"][0]["calls"]
        == row["calls"]
    )


def test_ledger_keeps_unknown_failed_cost_and_splits_overhead():
    saved = []
    ledger = MeasurementLedger(on_append=lambda: saved.append(True))
    for lane, status, cost in [
        ("reader", "completed", 1),
        ("judge", "completed", 2),
        ("reranker", "failed", None),
    ]:
        ledger.lane = lane
        ledger.append(
            provider="openai",
            model="test",
            seconds=1,
            usage={},
            price=None,
            status=status,
            cost=cost,
        )
    summary = summarize_calls(ledger.calls)
    assert summary["total_cost_usd"] is None
    assert summary["serving_cost_usd"] is None
    assert summary["evaluation_cost_usd"] == 2
    assert summary["known_cost_usd"] == 3
    assert summary["unpriced_calls"] == 1
    assert not summary["usage_complete"]
    assert len(saved) == 3


def test_stream_requires_completion_and_measures_visible_ttft():
    class Stream:
        def generate_stream(self, messages):
            yield {"type": "content", "content": "hello"}
            yield {
                "type": "usage",
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            }
            yield {"type": "done", "content": "hello"}

    ledger = MeasurementLedger()
    model = MeasuredModel(
        {"provider": "ollama", "model": "test", "stream": True}, ledger, model=Stream()
    )
    assert model.generate_text("hi") == "hello"
    assert ledger.calls[0]["ttft_seconds"] is not None

    class Broken:
        def generate_stream(self, messages):
            yield {"type": "content", "content": "partial"}

    model.wrapped = Broken()
    with pytest.raises(RuntimeError, match="without a completed answer"):
        model.generate_text("hi")
    assert ledger.calls[-1]["status"] == "failed"
    assert ledger.calls[-1]["cost_usd"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": 30, "candidate_pool_size": 10},
        {"rerankers": [ModelSpec(provider="heuristic")]},
        {"dataset": "custom"},
        {"max_cost_usd": 1, "judge": ModelSpec(provider="anthropic", model="unpriced")},
    ],
)
def test_invalid_experiments_are_rejected(overrides):
    with pytest.raises(ValidationError):
        config(**overrides)


def test_credentials_cannot_be_persisted_as_model_options():
    with pytest.raises(ValidationError, match="credentials"):
        ModelSpec(provider="openai", model="test", options={"api_key": "secret"})


def test_custom_dataset_unique_ids_and_seed(tmp_path):
    path = tmp_path / "cases.json"
    rows = [case_json(c) for c in demo_cases()]
    path.write_text(json.dumps(rows))
    c = config(dataset="custom", data_path=str(path), seed=42, limit=4)
    first = load_cases(c)
    assert [x.case_id for x in first] == [x.case_id for x in load_cases(c)]
    assert len(first) == 4
    path.write_text(json.dumps([rows[0], rows[0]]))
    with pytest.raises(ValueError, match="unique"):
        load_cases(c)


def test_reranking_preserves_candidates_and_stable_ties():
    rows = [
        {"source_id": "a", "content": "tea"},
        {"source_id": "b", "content": "tea"},
        {"source_id": "c", "content": "coffee"},
    ]
    ledger = MeasurementLedger()
    ranked = Reranker({"provider": "heuristic"}, ledger).rank("tea", rows, 2)
    assert [r["source_id"] for r in ranked] == ["a", "b"]
    assert "_retrieval" not in rows[0]
    assert ledger.calls[0]["lane"] == "reranker"


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.3}],
        [
            {"index": 0, "relevance_score": float("nan")},
            {"index": 1, "relevance_score": 0.3},
        ],
    ],
)
def test_cohere_rejects_bad_indices_and_scores(monkeypatch, results):
    monkeypatch.setenv("COHERE_API_KEY", "test")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: SimpleNamespace(ok=True, json=lambda: {"results": results}),
    )
    ledger = MeasurementLedger()
    with pytest.raises(ValueError):
        Reranker({"provider": "cohere", "model": "test"}, ledger).rank(
            "q", [{"content": "a"}, {"content": "b"}], 1
        )
    assert ledger.calls[-1]["status"] == "failed"


def test_cohere_and_jev_use_billing_and_original_indices(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")

    def post(url, **kwargs):
        if "cohere" in url:
            return SimpleNamespace(
                ok=True,
                json=lambda: {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.1},
                    ],
                    "meta": {"billed_units": {"search_units": 2}},
                },
            )
        assert len(kwargs["json"]["questions"]) == 2
        return SimpleNamespace(
            ok=True,
            json=lambda: {
                "answers": {
                    "0": {"type": "noul", "noul": 0.1},
                    "1": {"type": "noul", "noul": 0.9},
                },
                "usage": {"input_tokens": 1000, "output_tokens": 0},
            },
        )

    monkeypatch.setattr("requests.post", post)
    for provider, rate, expected in [("cohere", 0.002, 0.004), ("jev", None, 0.000042)]:
        ledger = MeasurementLedger()
        rows = Reranker(
            {"provider": provider, "model": "test", "search_unit_usd": rate}, ledger
        ).rank(
            "q",
            [{"source_id": "a", "content": "a"}, {"source_id": "b", "content": "b"}],
            1,
        )
        assert rows[0]["source_id"] == "b"
        assert ledger.calls[0]["cost_usd"] == pytest.approx(expected)


def test_frozen_retrieval_is_reused_across_arms(monkeypatch):
    from memorizz.benchmarks.memory_suite import FusionConfig, MemorySuiteRunner

    snapshots, invoked = {}, []

    def retrieve(self, case, corpus):
        invoked.append(case.case_id)
        return (
            [{"source_id": "a", "content": "tea"}],
            {"semantic_seconds": 0.1, "lexical_seconds": 0.2, "fusion_seconds": 0.3},
            ["q"],
        )

    monkeypatch.setattr(MemorySuiteRunner, "_retrieve", retrieve)
    for _ in range(2):
        runner = ComparisonRunner.__new__(ComparisonRunner)
        runner.frozen_candidates = snapshots
        runner.fusion_config = FusionConfig()
        runner.candidate_pool_size, runner.top_k = 10, 1
        runner.reranker = Reranker({"provider": "none"}, MeasurementLedger())
        rows, timing, _ = runner._retrieve(demo_cases()[0], {})
        assert rows[0]["source_id"] == "a"
        assert timing["semantic_seconds"] == 0.1
    assert len(invoked) == 1


def test_failed_run_and_cancel_are_durable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memorizz.benchmarks.comparison.MeasuredModel", lambda *a: object()
    )

    def fail(**kwargs):
        raise ValueError("Invalid test configuration")

    result = run_comparison(config(), tmp_path / "failed", runner_factory=fail)
    assert result["status"] == "completed_with_errors"
    assert result["runs"][0]["error"] == "Invalid test configuration"
    assert (tmp_path / "failed" / "result.json").exists()
    path = tmp_path / "cancelled"
    path.mkdir()
    (path / "cancel").touch()
    assert run_comparison(config(), path, runner_factory=fail)["status"] == "cancelled"


def test_routes_history_restart_export_and_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_COMPARISON_HOME", str(tmp_path))
    monkeypatch.setitem(state._state, "provider", object())
    monkeypatch.setitem(state._state, "read_only", True)
    run_id = "a" * 32
    atomic_json(tmp_path / run_id / "config.json", config().model_dump())
    atomic_json(tmp_path / run_id / "result.json", {"status": "running", "runs": []})
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    assert client.get("/evalground/compare").status_code == 200
    assert (
        client.get("/evalground/comparisons").json()["experiments"][0]["status"]
        == "interrupted"
    )
    assert (
        client.get(f"/evalground/comparisons/{run_id}/export").json()["status"]
        == "interrupted"
    )
    assert (
        "run,status"
        in client.get(f"/evalground/comparisons/{run_id}/export?format=csv").text
    )
    assert (
        client.post("/evalground/comparisons", json=config().model_dump()).status_code
        == 403
    )
    assert client.post(f"/evalground/comparisons/{run_id}/stop").status_code == 403
    assert client.get("/evalground/comparisons/invalid").status_code == 404


def test_anthropic_cache_writes_are_priced_separately():
    from memorizz.llms.anthropic import Anthropic

    raw = SimpleNamespace(
        input_tokens=300,
        output_tokens=100,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=500,
        cache_creation=SimpleNamespace(
            ephemeral_5m_input_tokens=400, ephemeral_1h_input_tokens=100
        ),
    )
    usage = Anthropic._usage_to_dict(raw)
    assert usage["prompt_tokens"] == 1000
    assert usage["cache_write_1h_tokens"] == 100
    price = token_price("anthropic", "claude-haiku-4-5-20251001")
    assert usage_cost("anthropic", usage, price) == pytest.approx(0.00152)
    assert (
        usage_cost("anthropic", usage, {"input": 1, "cached_input": 0.1, "output": 5})
        is None
    )
    assert token_price("anthropic", "claude-haiku-4-5-not-a-snapshot") is None


def test_strict_checks_reject_a_wrong_answer_that_contains_the_right_word():
    from memorizz.benchmarks.comparison import accuracy_interval, memory_check_cases
    from memorizz.benchmarks.memory_suite.scoring import score_answer

    cases = memory_check_cases()
    assert len(cases) == 6
    assert len({c.category for c in cases}) == 6
    for c in cases:
        assert score_answer(c, c.answers[0])["correct"]
        assert not score_answer(c, c.answers[0] + " or something else")["correct"]
    interval = accuracy_interval(6, 6)
    assert interval["lower"] == pytest.approx(0.6096657121)
    assert interval["upper"] == pytest.approx(1)
    assert accuracy_interval(0, 6)["upper"] == pytest.approx(0.3903342879)
    assert accuracy_interval(0, 0) is None


def test_model_catalog_uses_account_list_filters_and_prices(monkeypatch):
    from memorizz.benchmarks import model_catalog

    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-secret")
    seen = []

    def get(url, **kwargs):
        seen.append(url)
        assert kwargs["headers"]["Authorization"] == "Bearer not-a-real-secret"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "data": [
                    {"id": "gpt-4.1-mini-2025-04-14"},
                    {"id": "text-embedding-3-small"},
                    {"id": "gpt-realtime"},
                    {"id": "gpt-image-1"},
                    {"id": "gpt-5-mini"},
                ]
            },
        )

    monkeypatch.setattr(model_catalog.requests, "get", get)
    result = model_catalog.discover_models("openai", refresh=True)
    assert [m["id"] for m in result["models"]] == [
        "gpt-4.1-mini-2025-04-14",
        "gpt-5-mini",
    ]
    assert result["models"][0]["pricing"]["input"] == 0.4
    assert result["source"] == "provider_api"
    assert "not-a-real-secret" not in json.dumps(result)
    model_catalog.discover_models("openai")
    assert len(seen) == 1
    monkeypatch.setenv("OPENAI_API_KEY", "different-key")
    assert model_catalog.discover_models("openai")["source"] == "unavailable"


def test_model_discovery_failure_is_not_a_fake_available_list(monkeypatch):
    from memorizz.benchmarks import model_catalog

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = model_catalog.discover_models("anthropic", refresh=True)
    assert result["models"] == []
    assert result["source"] == "unavailable"
    assert "ANTHROPIC_API_KEY" in result["message"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sensitive-key")

    def fail(*a, **k):
        raise RuntimeError("sensitive-key")

    monkeypatch.setattr(model_catalog.requests, "get", fail)
    result = model_catalog.discover_models("anthropic", refresh=True)
    assert "sensitive-key" not in json.dumps(result)


def test_anthropic_catalog_paginates(monkeypatch):
    from memorizz.benchmarks import model_catalog

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-pagination")

    def get(url, **kwargs):
        second = kwargs["params"].get("after_id") == "claude-a"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "data": [{"id": "claude-b" if second else "claude-a"}],
                "has_more": not second,
                "last_id": "claude-b" if second else "claude-a",
            },
        )

    monkeypatch.setattr(model_catalog.requests, "get", get)
    assert len(model_catalog.discover_models("anthropic", refresh=True)["models"]) == 2


def test_ttft_does_not_mix_reranker_and_reader_calls():
    ledger = MeasurementLedger()
    for lane, ttft in [("reranker", 10), ("reader", 1)]:
        ledger.lane = lane
        ledger.append(
            provider="ollama",
            model="test",
            seconds=11,
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            price=None,
            ttft=ttft,
        )
    assert summarize_calls(ledger.calls)["ttft_p50_seconds"] == 1


def test_nonstandard_service_tier_without_override_has_unknown_cost():
    class Model:
        def generate_text(self, *a, **k):
            return "answer"

        def get_last_usage(self):
            return {"prompt_tokens": 100, "completion_tokens": 10}

        def get_last_response_metadata(self):
            return {
                "service_tier": "priority",
                "response_model": "gpt-4.1-mini-2025-04-14",
            }

    ledger = MeasurementLedger()
    MeasuredModel(
        {"provider": "openai", "model": "gpt-4.1-mini"}, ledger, model=Model()
    ).generate_text("q")
    assert ledger.calls[0]["cost_usd"] is None
    assert ledger.calls[0]["response_metadata"]["service_tier"] == "priority"


def test_reader_prompt_does_not_receive_scoring_labels():
    from dataclasses import replace

    from memorizz.benchmarks.memory_suite import MemorySuiteRunner

    case = replace(demo_cases()[0], answers=("SECRET_SCORING_LABEL",))
    runner = MemorySuiteRunner.__new__(MemorySuiteRunner)
    prompt = runner._evidence_prompt(case, "Retrieved evidence only")
    assert "SECRET_SCORING_LABEL" not in prompt
    assert case.question in prompt
    assert "Retrieved evidence only" in prompt
