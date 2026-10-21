"""An agent's own reranker: hosted (Voyage, Cohere, OpenAI Decisions),
fine-tuned, off, or the store's."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memorizz import reranking
from memorizz.reranking import RerankerConfig, reranker_for

pytestmark = pytest.mark.unit


def test_config_values():
    assert RerankerConfig.from_value(None) is None
    assert RerankerConfig.from_value("") is None
    assert RerankerConfig.from_value("off").provider == "off"
    voyage = RerankerConfig.from_value("voyage")
    assert voyage.model_name == "rerank-2.5" and voyage.label == "Voyage · rerank-2.5"
    assert RerankerConfig.from_value("voyage:rerank-2.5-lite").to_dict() == {
        "provider": "voyage",
        "model": "rerank-2.5-lite",
    }
    # The fine-tuning page's per-agent record reads the same.
    assert RerankerConfig.from_value({"mode": "run", "run_id": "r1"}).to_dict() == {
        "provider": "finetuned",
        "run_id": "r1",
    }
    with pytest.raises(ValueError):
        RerankerConfig.from_value("finetuned")
    with pytest.raises(ValueError):
        RerankerConfig.from_value("bm42")


def test_voyage_scores_follow_the_returned_indices(monkeypatch):
    calls = []

    def post(url, headers, json, timeout):
        calls.append((url, json))
        return SimpleNamespace(
            ok=True,
            json=lambda: {
                "data": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", post)
    score = reranker_for("refunds", config="voyage:rerank-2.5-lite")
    assert score(["shipping times", "refund policy"]) == [0.2, 0.9]
    url, body = calls[0]
    assert url.endswith("/v1/rerank") and body["model"] == "rerank-2.5-lite"
    assert body["query"] == "refunds"


def test_a_missing_key_or_bad_answer_fails_the_call(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    score = reranker_for("q", config="cohere")
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        score(["a"])
    with pytest.raises(ValueError):
        reranking._from_results([{"index": 0, "relevance_score": 1}], 2)


def test_off_and_follow_the_store(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "memorizz.finetune.reranking.rerank_for",
        lambda query, **kw: seen.append(kw) or (lambda texts: [1.0] * len(texts)),
    )
    assert reranker_for("q", config="off") is None
    assert reranker_for("q", config=None, agent_id="a1")(["x"]) == [1.0]
    assert seen[-1]["agent_id"] == "a1"
    assert reranker_for("  ", config="voyage") is None


def test_a_saved_agent_setting_is_used_for_its_memory(monkeypatch):
    from memorizz.finetune import agent_models

    agent_models.forget()

    class Provider:
        def list_memagents(self):
            return [{"agent_id": "a1", "memory_ids": ["m1"]}]

        def retrieve_memagent(self, agent_id):
            return {"agent_id": "a1", "reranker": {"provider": "openai_decisions"}}

    used = []
    monkeypatch.setattr(
        reranking,
        "_decisions",
        lambda model: used.append(model) or (lambda query, texts: [0.5] * len(texts)),
    )
    score = reranker_for("q", provider=Provider(), memory_id="m1")
    assert score(["x", "y"]) == [0.5, 0.5] and used == ["gpt-6-luna"]
    agent_models.forget()


def test_the_agent_passes_its_own_reranker_to_recall(monkeypatch):
    from memorizz.memagent import core

    captured = {}

    def fake(query, *, config, provider, agent_id):
        captured.update(config=config, agent_id=agent_id)
        return None

    monkeypatch.setattr("memorizz.reranking.reranker_for", fake)
    core._memory_reranker("q", agent_id="a1", config=RerankerConfig("voyage"))
    assert captured == {"config": RerankerConfig("voyage"), "agent_id": "a1"}


def test_available_reports_keys():
    assert reranking.available({"VOYAGE_API_KEY": "x"}) == {
        "voyage": True,
        "cohere": False,
        "openai_decisions": False,
    }
