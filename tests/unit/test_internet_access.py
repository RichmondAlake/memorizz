"""Tests for internet access integration."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from memorizz.internet_access import (
    InternetAccessProvider,
    InternetPageContent,
    InternetSearchResult,
    OfflineInternetProvider,
    register_provider,
)
from memorizz.memagent.core import MemAgent
from memorizz.memagent.managers.internet_access_manager import InternetAccessManager
from memorizz.memagent.models import MemAgentModel
from memorizz.tooling import ToolOutcomeStatus, normalize_tool_result


class _DummyProvider(InternetAccessProvider):
    provider_name = "dummy-provider"

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> List[InternetSearchResult]:
        return [
            InternetSearchResult(
                url=f"https://example.com/{idx}",
                title=f"Result {idx}",
                snippet=query,
                score=1.0,
            )
            for idx in range(max_results)
        ]

    def fetch_url(self, url: str, **kwargs) -> InternetPageContent:
        return InternetPageContent(url=url, title="Example", content="Example body")


register_provider(_DummyProvider.provider_name, _DummyProvider)


@pytest.mark.unit
def test_internet_access_manager_serializes_results():
    provider = _DummyProvider()
    manager = InternetAccessManager(provider)

    results = manager.search("memorizz", max_results=2)
    assert len(results) == 2
    assert results[0]["url"].startswith("https://example.com/")

    page = manager.fetch_url("https://memorizz.ai")
    assert page["content"] == "Example body"


@pytest.mark.unit
def test_memagent_registers_internet_tools():
    provider = MagicMock()
    provider.get_provider_name.return_value = "dummy"
    provider.get_config.return_value = {"api_key": "test"}
    provider.search.return_value = [{"url": "https://example.com"}]
    provider.fetch_url.return_value = {
        "url": "https://example.com",
        "content": "Body",
    }

    agent = MemAgent(instruction="Internet agent", internet_access_provider=provider)

    assert agent.has_internet_access() is True
    assert "internet_search" in agent.tool_manager.tools
    assert agent.search_internet("python")
    provider.search.assert_called_once()


@pytest.mark.unit
def test_memagent_disables_internet_access():
    provider = MagicMock()
    provider.get_provider_name.return_value = "dummy"
    provider.get_config.return_value = {}
    provider.search.return_value = []
    provider.fetch_url.return_value = {}

    agent = MemAgent(instruction="toggle agent", internet_access_provider=provider)
    assert agent.has_internet_access() is True

    agent.with_internet_access_provider(None)
    assert agent.has_internet_access() is False
    assert "internet_search" not in agent.tool_manager.tools


@pytest.mark.unit
def test_offline_internet_tool_is_provider_error_not_false_success():
    agent = MemAgent(
        instruction="Offline internet agent",
        internet_access_provider=OfflineInternetProvider("No provider credentials"),
    )

    raw, _workflow = agent.tool_manager.execute_tool(
        "internet_search", {"query": "latest release", "max_results": 3}
    )
    payload, outcome = normalize_tool_result(raw)

    assert payload["results"][0]["metadata"]["status"] == "offline"
    assert outcome.status is ToolOutcomeStatus.PROVIDER_ERROR
    assert outcome.ok is False
    assert outcome.provider == "offline"


@pytest.mark.unit
def test_memagent_load_rehydrated_provider(monkeypatch):
    memory_provider = MagicMock()
    saved = MemAgentModel(
        instruction="Load",
        internet_access_provider=_DummyProvider.provider_name,
        internet_access_config={"custom": "value"},
    )
    memory_provider.retrieve_memagent.return_value = saved

    agent = MemAgent.load(
        agent_id="agent-123",
        memory_provider=memory_provider,
    )

    assert agent.has_internet_access() is True
    assert agent.get_internet_access_provider_name() == _DummyProvider.provider_name
