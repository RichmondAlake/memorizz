"""Typed Decisions API contracts and paired reranker measurements."""

import copy
from types import SimpleNamespace

import pytest

from memorizz import OpenAIDecisions
from memorizz.benchmarks.measurement import MeasurementLedger, token_price, usage_cost
from memorizz.benchmarks.rerankers import Reranker


def response(monkeypatch, payload):
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-only")

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(ok=True, json=lambda: copy.deepcopy(payload))

    monkeypatch.setattr("requests.post", post)
    return calls


def test_predicate_requests_use_dedicated_endpoint_and_preserve_order(monkeypatch):
    calls = response(
        monkeypatch,
        {
            "model": "gpt-6-luna",
            "answers": [
                {"name": "a", "type": "predicate", "probability": 0.9},
                {"name": "b", "type": "refusal"},
            ],
        },
    )
    questions = [
        {"type": "predicate", "name": name, "instructions": "Useful?"}
        for name in ("a", "b")
    ]
    result = OpenAIDecisions().evaluate("evidence", questions)
    assert result["answers"][1]["type"] == "refusal"
    assert calls[0][0] == "https://api.openai.com/v1/decisions"
    assert calls[0][1]["json"] == {
        "model": "gpt-6-luna",
        "input": "evidence",
        "questions": questions,
    }


@pytest.mark.parametrize("bad", [True, -0.1, 1.1, float("nan"), None])
def test_invalid_probabilities_do_not_become_scores(monkeypatch, bad):
    response(
        monkeypatch,
        {"answers": [{"name": "a", "type": "predicate", "probability": bad}]},
    )
    with pytest.raises(ValueError, match="probability"):
        OpenAIDecisions().evaluate(
            "x", [{"name": "a", "type": "predicate", "instructions": "?"}]
        )


def test_choice_values_preserve_boolean_vs_string_and_reject_duplicate_distribution(
    monkeypatch,
):
    answer = {
        "type": "choice",
        "name": "a",
        "choice": True,
        "confidence": 0.8,
        "probabilities": [
            {"value": True, "probability": 0.8},
            {"value": "True", "probability": 0.2},
        ],
    }
    question = {
        "name": "a",
        "type": "choice",
        "instructions": "?",
        "choices": [{"value": True}, {"value": "True"}],
    }
    response(monkeypatch, {"answers": [answer]})
    assert OpenAIDecisions().evaluate("x", [question])["answers"][0]["choice"] is True
    answer["probabilities"][1]["value"] = True
    with pytest.raises(ValueError, match="coverage"):
        OpenAIDecisions().evaluate("x", [question])


def test_paired_score_reranker_uses_expected_grade_not_confidence(monkeypatch):
    answers = [
        {
            "name": str(i),
            "type": "score",
            "score": value,
            "confidence": 0.99,
            "probabilities": [
                {"label": str(j), "value": j, "probability": float(j == value)}
                for j in range(4)
            ],
        }
        for i, value in enumerate((0, 3))
    ]
    calls = response(
        monkeypatch,
        {
            "answers": answers,
            "usage": {
                "input_tokens": 1000,
                "input_tokens_details": {
                    "cached_tokens": 200,
                    "cache_write_tokens": 100,
                },
                "output_tokens": 0,
            },
        },
    )
    ledger = MeasurementLedger()
    rows = Reranker(
        {"provider": "openai_decisions", "model": "gpt-6-luna", "jev_method": "score"},
        ledger,
    ).rank(
        "q",
        [{"id": "a", "content": "irrelevant"}, {"id": "b", "content": "evidence"}],
        2,
    )
    assert [row["id"] for row in rows] == ["b", "a"]
    assert rows[0]["_retrieval"]["reranker_score"] == 1
    assert calls[0][1]["json"]["questions"][0]["levels"][3]["label"] == "3"
    assert ledger.calls[0]["cost_usd"] == pytest.approx(0.00007)
    assert (
        usage_cost(
            "openai_decisions", {}, token_price("openai_decisions", "gpt-6-luna")
        )
        is None
    )


def test_refusal_fails_reranking_and_http_errors_do_not_expose_body(monkeypatch):
    response(monkeypatch, {"answers": [{"name": "0", "type": "refusal"}]})
    ledger = MeasurementLedger()
    with pytest.raises(RuntimeError, match="declined"):
        Reranker({"provider": "openai_decisions", "model": "gpt-6-luna"}, ledger).rank(
            "q", [{"content": "x"}], 1
        )
    assert ledger.calls[0]["status"] == "failed"
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **kw: SimpleNamespace(
            ok=False, status_code=403, text="credential material"
        ),
    )
    with pytest.raises(RuntimeError, match="HTTP 403") as exc:
        OpenAIDecisions().evaluate(
            "x", [{"name": "a", "type": "predicate", "instructions": "?"}]
        )
    assert "credential" not in str(exc.value)
