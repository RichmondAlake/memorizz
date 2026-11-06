"""A Hugging Face reranker, your own reranker, and reranked knowledge-base
lookups."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from memorizz import reranking
from memorizz.memagent import MemAgent
from memorizz.memagent.persistence import _refresh_memory_models
from memorizz.reranking import RerankerConfig, rerank_rows, reranker_for
from tests.mocks.mock_providers import MockMemoryProvider

pytestmark = pytest.mark.unit

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _hf_post(calls: List[Dict[str, Any]], logits_for):
    """A fake Hugging Face text-ranking endpoint: rows sorted by score, as
    the real one returns them."""

    def post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json})
        texts = json["inputs"]["texts"]
        rows = [
            {"index": index, "score": logits_for(text)}
            for index, text in enumerate(texts)
        ]
        rows.sort(key=lambda row: -row["score"])
        return SimpleNamespace(ok=True, status_code=200, json=lambda: rows)

    return post


def test_huggingface_settings():
    default = RerankerConfig.from_value("huggingface")
    assert default.model_name == MODEL
    assert default.label == f"Hugging Face · {MODEL}"
    chosen = RerankerConfig.from_value("huggingface:BAAI/bge-reranker-base")
    assert chosen.to_dict() == {
        "provider": "huggingface",
        "model": "BAAI/bge-reranker-base",
    }


def test_huggingface_scores_are_logits_read_as_probabilities(monkeypatch):
    calls: List[Dict[str, Any]] = []
    logits = {"carry-on": 2.4, "breakfast": -11.4, "batteries": -8.0}
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("requests.post", _hf_post(calls, lambda text: logits[text]))

    score = reranker_for("power bank on a plane?", config="huggingface")
    scores = score(["carry-on", "breakfast", "batteries"])

    # The rows come back sorted by score; each lands on its own text.
    assert scores[0] > scores[2] > scores[1]
    assert all(0.0 < value < 1.0 for value in scores)
    assert scores[0] == pytest.approx(reranking._sigmoid(2.4))
    call = calls[0]
    assert call["url"] == f"https://router.huggingface.co/hf-inference/models/{MODEL}"
    assert call["headers"]["Authorization"] == "Bearer hf_test"
    assert call["json"] == {
        "inputs": {
            "query": "power bank on a plane?",
            "texts": ["carry-on", "breakfast", "batteries"],
        }
    }


def test_huggingface_sends_long_lists_in_batches(monkeypatch):
    calls: List[Dict[str, Any]] = []
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("requests.post", _hf_post(calls, lambda text: float(text)))
    texts = [str(value) for value in range(reranking.HF_BATCH + 5)]

    scores = reranker_for("q", config="huggingface")(texts)

    assert [len(call["json"]["inputs"]["texts"]) for call in calls] == [
        reranking.HF_BATCH,
        5,
    ]
    assert scores == sorted(scores)  # each score stays on its own text


def test_huggingface_needs_a_token_and_reports_its_error(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # No token saved by `hf auth login` either (and no huggingface_hub needed).
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(get_token=lambda: None)
    )
    with pytest.raises(ValueError, match="HF_TOKEN"):
        reranker_for("q", config="huggingface")(["a"])

    monkeypatch.setenv("HF_TOKEN", "hf_bad")
    monkeypatch.setattr(
        "requests.post",
        lambda url, headers, json, timeout: SimpleNamespace(
            ok=False,
            status_code=401,
            json=lambda: {"error": "Invalid username or password."},
        ),
    )
    with pytest.raises(RuntimeError, match="HTTP 401: Invalid username"):
        reranker_for("q", config="huggingface")(["a"])


def test_your_own_function_is_a_reranker():
    def by_length(query, texts):
        return [len(text) / 10 for text in texts]

    config = RerankerConfig.from_value(by_length)
    assert config.provider == "custom" and config.label == "custom · by_length"
    # Only the name is saved: the function stays in this process.
    assert config.to_dict() == {"provider": "custom", "model": "by_length"}
    assert reranker_for("q", config=config)(["ab", "abcd"]) == [0.2, 0.4]


def test_your_own_scores_outside_0_to_1_are_squashed_in_order():
    config = RerankerConfig.from_value(lambda query, texts: [3.0, -2.0, 0.5])
    scores = reranker_for("q", config=config)(["a", "b", "c"])
    assert scores[0] > scores[2] > scores[1]
    assert all(0.0 < value < 1.0 for value in scores)

    wrong = RerankerConfig.from_value(lambda query, texts: [1.0])
    with pytest.raises(ValueError, match="wrong number"):
        reranker_for("q", config=wrong)(["a", "b"])


def test_a_cross_encoder_object_is_a_reranker():
    class CrossEncoder:
        model_name = "my-cross-encoder"

        def __call__(self, *args):  # torch modules are callable too
            raise AssertionError("predict() is the reranking entry point")

        def predict(self, pairs):
            return [0.9 if "refund" in text else 0.1 for _query, text in pairs]

    config = RerankerConfig.from_value(CrossEncoder())
    assert config.label == "custom · my-cross-encoder"
    assert reranker_for("refunds", config=config)(["shipping", "refund policy"]) == [
        0.1,
        0.9,
    ]


def test_a_saved_custom_reranker_without_its_function_does_not_rerank():
    saved = RerankerConfig.from_value({"provider": "custom", "model": "by_length"})
    assert saved.function is None
    assert reranker_for("q", config=saved) is None


def test_refreshing_from_storage_keeps_the_function_passed_in():
    def mine(query, texts):
        return [0.5] * len(texts)

    agent = SimpleNamespace(
        reranker=RerankerConfig.from_value(mine),
        _register_memory_routing_tools=lambda: None,
    )
    _refresh_memory_models(agent, None, {"provider": "custom", "model": "mine"}, None)
    assert agent.reranker.function is mine
    # A saved change to another reranker still applies.
    _refresh_memory_models(agent, None, {"provider": "voyage"}, None)
    assert agent.reranker == RerankerConfig("voyage")


def test_rerank_rows_orders_by_score_and_keeps_order_on_failure():
    rows = [{"t": "a"}, {"t": "b"}, {"t": "c"}]
    ranked = rerank_rows(rows, lambda texts: [0.2, 0.9, 0.5], text_of=lambda r: r["t"])
    assert [row["t"] for _score, row in ranked] == ["b", "c", "a"]

    def broken(texts):
        raise RuntimeError("down")

    assert rerank_rows(rows, broken, text_of=lambda r: r["t"]) is None
    assert rerank_rows(rows, None, text_of=lambda r: r["t"]) is None


class _KB:
    calls: List[int] = []
    rows: List[Dict[str, Any]] = []

    def __init__(self, memory_provider):
        self.memory_provider = memory_provider

    def retrieve_knowledge_by_query(
        self, query, namespace=None, limit=5, knowledge_base_ids=None, user_id=None
    ):
        type(self).calls.append(limit)
        return type(self).rows[:limit]


def _kb_agent(monkeypatch, reranker) -> MemAgent:
    _KB.calls = []
    _KB.rows = [
        {"content": f"passage {index}", "knowledge_base_id": "kb-1"}
        for index in range(30)
    ]
    monkeypatch.setattr("memorizz.long_term.semantic.knowledge_base.KnowledgeBase", _KB)
    agent = MemAgent(
        memory_provider=MockMemoryProvider(), auto_register=False, reranker=reranker
    )
    agent.knowledge_base_ids = ["kb-1"]
    return agent


def test_knowledge_base_lookup_reranks_a_wider_fetch(monkeypatch):
    # Vector search put passage 17 far down; the reranker brings it first.
    def prefers_17(query, texts):
        return [1.0 if text == "passage 17" else 0.1 for text in texts]

    agent = _kb_agent(monkeypatch, prefers_17)
    result, _ = agent.tool_manager.execute_tool(
        "knowledge_base_lookup", {"query": "budget", "limit": 3}
    )

    assert _KB.calls == [20]  # more candidates than it returns
    assert [match["content"] for match in result["matches"]] == [
        "passage 17",
        "passage 0",
        "passage 1",
    ]
    assert result["matches"][0]["rerank_score"] == 1.0
    agent.close()


def test_knowledge_base_lookup_keeps_vector_order_without_a_reranker(monkeypatch):
    agent = _kb_agent(monkeypatch, "off")
    result, _ = agent.tool_manager.execute_tool(
        "knowledge_base_lookup", {"query": "budget", "limit": 2}
    )
    assert _KB.calls == [2]
    assert [match["content"] for match in result["matches"]] == [
        "passage 0",
        "passage 1",
    ]
    assert "rerank_score" not in result["matches"][0]
    agent.close()


def test_knowledge_base_lookup_survives_a_failing_reranker(monkeypatch):
    def down(query, texts):
        raise RuntimeError("Hugging Face is down")

    agent = _kb_agent(monkeypatch, down)
    result, _ = agent.tool_manager.execute_tool(
        "knowledge_base_lookup", {"query": "budget", "limit": 2}
    )
    assert [match["content"] for match in result["matches"]] == [
        "passage 0",
        "passage 1",
    ]
    agent.close()
