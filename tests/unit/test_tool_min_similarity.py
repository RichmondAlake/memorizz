"""ContextPolicy.tool_min_similarity: tool routing offers no tool to a
request unrelated to every stored tool."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from memorizz.memagent.managers.tool_manager import ToolManager
from memorizz.memagent.persistence import _refresh_policies
from memorizz.tooling import ContextPolicy, SemanticToolRouter

pytestmark = pytest.mark.unit


def get_weather(city: str) -> str:
    """Get the weather forecast for a city."""
    return city


def convert_currency(amount: float, currency: str) -> str:
    """Convert an amount of money into another currency."""
    return currency


def public_holidays(country: str) -> str:
    """List the public holidays of a country."""
    return country


class _Toolbox:
    """Rows as the stores return them: ranked, each with its cosine score."""

    def __init__(self, scores: Dict[str, Dict[str, float]]):
        self.scores = scores

    def get_most_similar_tools(self, query, limit=25, **kwargs):
        ranked = sorted(self.scores[query].items(), key=lambda item: -item[1])
        return [
            {"name": name, "score": score, "user_id": None, "agent_id": None}
            for name, score in ranked[:limit]
        ]


SCORES = {
    "Will it rain in Lisbon?": {
        "get_weather": 0.62,
        "public_holidays": 0.31,
        "convert_currency": 0.18,
    },
    # Unrelated to every tool, but its words still appear in a description.
    "Find me a direct flight to the country": {
        "public_holidays": 0.21,
        "get_weather": 0.19,
        "convert_currency": 0.12,
    },
}


def _router(min_similarity=None, top_k=2) -> SemanticToolRouter:
    manager = ToolManager()
    for tool in (get_weather, convert_currency, public_holidays):
        manager.add_tool(tool)
    router = SemanticToolRouter(
        manager, toolbox=_Toolbox(SCORES), top_k=top_k, min_similarity=min_similarity
    )
    router.begin_turn()
    return router


def test_without_a_cut_off_every_request_gets_the_closest_tools():
    router = _router()
    assert router.preview("Find me a direct flight to the country") == [
        "public_holidays",
        "get_weather",
    ]


def test_a_cut_off_offers_only_close_tools():
    router = _router(min_similarity=0.4)
    assert router.preview("Will it rain in Lisbon?") == ["get_weather"]


def test_an_unrelated_request_is_offered_no_tools():
    router = _router(min_similarity=0.4)
    # "country" is in public_holidays' description, yet keyword matching
    # does not offer it back once meaning has turned every tool away.
    assert router.preview("Find me a direct flight to the country") == []
    schemas = router.schemas_for_turn("Find me a direct flight to the country")
    names = {schema["function"]["name"] for schema in schemas}
    assert names.isdisjoint({"get_weather", "convert_currency", "public_holidays"})


def test_rows_without_a_score_are_kept():
    router = _router(min_similarity=0.9)
    router.toolbox = SimpleNamespace(
        get_most_similar_tools=lambda query, limit=25, **kw: [
            {"name": "get_weather", "user_id": None, "agent_id": None}
        ]
    )
    assert router.preview("Will it rain in Lisbon?") == ["get_weather"]


def test_an_empty_toolbox_still_falls_back_to_keywords():
    router = _router(min_similarity=0.4)
    router.toolbox = SimpleNamespace(get_most_similar_tools=lambda *a, **kw: [])
    assert router.preview("convert this amount of money") == ["convert_currency"]


def test_the_policy_carries_the_cut_off():
    policy = ContextPolicy(tool_min_similarity=0.35)
    assert policy.to_dict()["tool_min_similarity"] == 0.35
    assert ContextPolicy.from_value(policy.to_dict()) == policy
    assert "tool_min_similarity" not in ContextPolicy().to_dict()
    assert ContextPolicy(tool_min_similarity=1.7).tool_min_similarity == 1.0
    assert ContextPolicy(tool_min_similarity=-0.2).tool_min_similarity == 0.0


def test_the_agent_router_follows_its_policy():
    from memorizz.memagent import MemAgent
    from tests.mocks.mock_providers import MockMemoryProvider

    agent = MemAgent(
        memory_provider=MockMemoryProvider(),
        auto_register=False,
        context_policy=ContextPolicy(tool_min_similarity=0.3),
    )
    try:
        assert agent.semantic_tool_router.min_similarity == 0.3
        # A policy saved elsewhere reaches the live router on refresh.
        _refresh_policies(agent, None, {"tool_min_similarity": 0.5}, None, None)
        assert agent.semantic_tool_router.min_similarity == 0.5
        _refresh_policies(agent, None, {}, None, None)
        assert agent.semantic_tool_router.min_similarity is None
    finally:
        agent.close()


def test_mongodb_toolbox_scores_are_cosine(monkeypatch):
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.toolbox_collection = object()
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding", lambda text, **kw: [1.0], raising=False
    )
    monkeypatch.setattr(
        provider, "_build_vector_search_pipeline", lambda *a, **kw: [], raising=False
    )
    rows: List[Dict[str, Any]] = [{"name": "get_weather", "score": 0.81}]
    monkeypatch.setattr(
        provider, "_run_vector_search", lambda *a, **kw: rows, raising=False
    )
    result = provider.retrieve_toolbox_item("Will it rain?", limit=1)
    assert result[0]["score"] == pytest.approx(0.62)
