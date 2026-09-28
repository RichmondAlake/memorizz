"""System One adapter contracts; provider responses below are unit-test fixtures."""
import json
import math
from types import SimpleNamespace

import pytest

from memorizz.benchmarks.comparison import ComparisonConfig, ModelSpec
from memorizz.benchmarks.jev_reranking import question_payload, scores_from_answers
from memorizz.benchmarks.measurement import MeasurementLedger, token_price, usage_cost
from memorizz.benchmarks.rerankers import Reranker
from memorizz.benchmarks.snapshot_comparison import (
    fingerprint,
    load_snapshot,
    retrieval_metrics,
)


def test_full_gold_and_requested_k_denominators():
    result = retrieval_metrics(["b"], {"a": 3, "b": 1}, ["b", "c"], 3)
    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["recall_at_k"] == 0.5
    assert result["candidate_recall"] == 0.5
    assert result["ndcg_at_k"] == pytest.approx(1 / (7 + 1 / math.log2(3)))
    with pytest.raises(ValueError):
        retrieval_metrics(["a", "a"], {"a": 3}, ["a"], 3)


def test_score_distribution_and_choice_do_not_confuse_confidence():
    answer = {
        "0": {
            "type": "score",
            "confidence": 0.99,
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4},
        }
    }
    assert scores_from_answers(answer, 1, "score") == pytest.approx([2 / 3])
    choice = {
        "rank": {"type": "choice", "probabilities": {"0": 0.1, "1": 0.2, "none": 0.7}}
    }
    assert scores_from_answers(choice, 2, "choice") == [0.1, 0.2]
    choice["rank"]["probabilities"]["none"] = 0.9
    with pytest.raises(ValueError):
        scores_from_answers(choice, 2, "choice")
    with pytest.raises(ValueError):
        question_payload("q", ["x"] * 255, "choice")


def test_voyage_preserves_ids_and_prices_provider_token_count(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "unit-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **kw: SimpleNamespace(
            ok=True,
            json=lambda: {
                "data": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ],
                "usage": {"total_tokens": 2000},
            },
        ),
    )
    ledger = MeasurementLedger()
    rows = Reranker({"provider": "voyage", "model": "rerank-2.5"}, ledger).rank(
        "q",
        [
            {"source_id": "a", "content": "alpha", "metadata": {"v": 1}},
            {"source_id": "b", "content": "beta"},
        ],
        2,
    )
    assert [r["source_id"] for r in rows] == ["b", "a"]
    assert rows[1]["metadata"] == {"v": 1}
    assert ledger.calls[0]["cost_usd"] == pytest.approx(0.0001)
    assert usage_cost("voyage", {}, token_price("voyage", "rerank-2.5")) is None


def test_snapshot_checksum_and_unique_candidates(tmp_path):
    cases = [
        {
            "case_id": "q",
            "question": "Where?",
            "gold": {"a": 3},
            "required_keywords": ["Bath"],
            "candidates": [{"source_id": "a", "content": "Bath"}],
        }
    ]
    data = {
        "format": "system-one-oracle-v1",
        "cases": cases,
        "fingerprint": fingerprint(cases),
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(data))
    assert load_snapshot(path, 1)[1] == cases
    cases[0]["question"] = "Changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprint"):
        load_snapshot(path, 1)
    cases[0]["candidates"] *= 2
    data["fingerprint"] = fingerprint(cases)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        load_snapshot(path, 1)


def test_provider_and_recipe_configuration_is_typed():
    assert (
        ModelSpec(provider="jev", model="jev-1.13.0", jev_method="score").jev_method
        == "score"
    )
    with pytest.raises(ValueError):
        ModelSpec(provider="jev", model="jev-1.13.0", jev_method="made-up")
    cfg = ComparisonConfig(
        experiment_type="reranker",
        readers=[ModelSpec(provider="openai", model="gpt-4.1-mini")],
        rerankers=[ModelSpec(provider="voyage", model="rerank-2.5")],
        max_cost_usd=1,
    )
    assert cfg.rerankers[0].provider == "voyage"


def test_gpt6_cache_writes_are_separate_from_reads_and_uncached_input():
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "cached_tokens": 100,
        "cache_write_tokens": 200,
    }
    assert usage_cost(
        "openai", usage, token_price("openai", "gpt-6-sol")
    ) == pytest.approx(0.00292)
    assert usage_cost(
        "openai", usage, token_price("openai", "gpt-6-luna")
    ) == pytest.approx(0.000146)
    assert usage_cost("openai", usage, token_price("openai", "gpt-4.1-mini")) is None


def test_anthropic_effort_is_forwarded(monkeypatch):
    from memorizz.llms.anthropic import Anthropic

    model = Anthropic(
        api_key="unit-test-key",
        model="claude-opus-5-5",
        effort="medium",
        enable_prompt_caching=False,
    )
    assert model._request_options["output_config"] == {"effort": "medium"}
    assert not model._enable_prompt_caching
    with pytest.raises(ValueError):
        Anthropic(api_key="unit-test-key", effort="invalid")


def test_saved_evidence_and_home_run_library(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from memorizz.benchmarks.comparison import atomic_json
    from memorizz.ui import state
    from memorizz.ui.routers import comparisons, evalground

    monkeypatch.setenv("MEMORIZZ_COMPARISON_HOME", str(tmp_path))
    monkeypatch.setitem(
        state._state, "provider", SimpleNamespace(list_memagents=lambda: [])
    )
    monkeypatch.setattr(
        evalground,
        "_build_eval_run_history_rows",
        lambda **kw: [
            {
                "run_id": "b" * 32,
                "agent_name": "Memory assistant",
                "benchmark": "locomo",
                "status": "completed",
                "created_at": "2026-09-25T10:00:00Z",
                "num_samples": 10,
                "evaluated_samples": 8,
                "overall_accuracy": 75,
                "cost_usd": None,
                "model": "test-model",
            }
        ],
    )
    run_id = "a" * 32
    atomic_json(
        tmp_path / run_id / "config.json",
        {
            "name": "Ranking check",
            "experiment_type": "reranker",
            "top_k": 3,
            "limit": 6,
            "readers": [{"model": "reader"}],
            "rerankers": [{"model": "jev"}],
            "repeats": 1,
        },
    )
    atomic_json(
        tmp_path / run_id / "result.json",
        {
            "status": "completed",
            "started_at": 100,
            "case_ids": ["q"],
            "runs": [
                {
                    "cases": [{"case_id": "q"}],
                    "summary": {"total_cost_usd": 0.1, "ndcg_at_k": 0.9},
                }
            ],
        },
    )
    atomic_json(
        tmp_path / run_id / "cases.json",
        [
            {
                "case_id": "q",
                "gold": {"a": 3, "b": 0},
                "candidates": [{"source_id": "a", "content": "source"}],
            }
        ],
    )
    # A malformed saved experiment must not hide valid rows.
    (tmp_path / ("c" * 32)).mkdir()
    (tmp_path / ("c" * 32) / "config.json").write_text("{")
    app = FastAPI()
    app.include_router(comparisons.router)
    client = TestClient(app)
    pool = client.get(f"/evalground/comparisons/{run_id}/evidence?case_id=q").json()
    assert pool["relevant_source_ids"] == ["a"]
    assert pool["gold"] == {"a": 3, "b": 0}
    assert (
        client.get(
            f"/evalground/comparisons/{run_id}/evidence?case_id=missing"
        ).status_code
        == 404
    )
    result = client.get("/evalground/run-library").json()
    assert len(result["runs"]) == 2 and result["warnings"]
    agent, comparison = result["runs"]
    assert (
        agent["kind"] == "agent" and agent["completed"] == 8 and agent["planned"] == 10
    )
    assert agent["quality_min"] == 0.75 and agent["cost_usd"] is None
    assert comparison["completed"] == comparison["planned"] == 1
    assert comparison["quality_label"] == "nDCG@3" and comparison["cost_usd"] == 0.1
    assert comparison["url"].endswith(run_id)


def test_standard_saved_pool_is_available_without_snapshot(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from memorizz.benchmarks.comparison import atomic_json
    from memorizz.ui.routers import comparisons

    monkeypatch.setenv("MEMORIZZ_COMPARISON_HOME", str(tmp_path))
    run_id = "d" * 32
    atomic_json(
        tmp_path / run_id / "cases.json",
        [{"case_id": "q", "relevant_source_ids": ["b"]}],
    )
    atomic_json(
        tmp_path / run_id / "candidates.json",
        {"q": {"rows": [{"source_id": "b", "content": "text"}]}},
    )
    app = FastAPI()
    app.include_router(comparisons.router)
    response = TestClient(app).get(
        f"/evalground/comparisons/{run_id}/evidence?case_id=q"
    )
    assert response.status_code == 200
    assert response.json()["candidates"][0]["source_id"] == "b"
    assert response.json()["labels_available"] is True
