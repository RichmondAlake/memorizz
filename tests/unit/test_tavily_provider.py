"""Unit tests for the Tavily internet provider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memorizz.internet_access.providers.tavily import TavilyProvider


@pytest.mark.unit
def test_tavily_search_normalizes_results():
    provider = TavilyProvider(
        api_key="test-key",
        base_url="https://api.tavily.com",
        config={"include_raw_results": True},
    )
    provider._post = MagicMock(
        return_value={
            "results": [
                {
                    "url": "https://example.com/doc",
                    "title": "Example Doc",
                    "content": "Snippet",
                    "score": 0.8,
                    "site": "example.com",
                    "published_date": "2024-01-01",
                }
            ]
        }
    )

    results = provider.search("Example query", max_results=2)

    assert len(results) == 1
    assert results[0].url == "https://example.com/doc"
    assert results[0].metadata["site"] == "example.com"
    assert results[0].raw["title"] == "Example Doc"
    provider._post.assert_called_once()


@pytest.mark.unit
def test_tavily_fetch_truncates_content_and_returns_raw():
    provider = TavilyProvider(
        api_key="test-key",
        base_url="https://api.tavily.com",
        config={"max_content_chars": 20, "include_raw_page": True},
    )
    provider._post = MagicMock(
        return_value={
            "results": [
                {
                    "url": "https://example.com/doc",
                    "title": "Example Doc",
                    "content": "A" * 40,
                    "metadata": {"lang": "en"},
                    "site": "example.com",
                }
            ]
        }
    )

    page = provider.fetch_url("https://example.com/doc")

    assert page.metadata["content_truncated"] is True
    assert page.metadata["content_returned_characters"] == 20
    assert len(page.content) == 20
    assert page.raw["title"] == "Example Doc"


@pytest.mark.unit
def test_tavily_search_can_return_each_pages_text():
    provider = TavilyProvider(
        api_key="test-key",
        config={"include_raw_content": True, "max_content_chars": 30},
    )
    provider._post = MagicMock(
        return_value={
            "results": [
                {
                    "url": "https://example.com/rules",
                    "title": "Baggage rules",
                    "content": "Power banks go in carry-on.",
                    "raw_content": "# Baggage rules\n" + "Full page text. " * 10,
                },
                {"url": "https://example.com/empty", "content": "Snippet only"},
            ]
        }
    )

    results = provider.search("power bank rules")

    payload = provider._post.call_args.args[1]
    assert payload["include_raw_content"] == "markdown"
    first, second = results
    assert first.snippet == "Power banks go in carry-on."
    assert first.content == ("# Baggage rules\n" + "Full page text. " * 10)[:30]
    assert first.metadata["content_truncated"] is True
    assert first.to_dict()["content"] == first.content
    # A result without page text keeps the usual shape.
    assert second.content is None and "content" not in second.to_dict()


@pytest.mark.unit
def test_tavily_page_text_is_off_unless_asked_and_per_call():
    provider = TavilyProvider(api_key="test-key")
    provider._post = MagicMock(return_value={"results": []})

    provider.search("q")
    assert "include_raw_content" not in provider._post.call_args.args[1]
    provider.search("q", include_raw_content="text")
    assert provider._post.call_args.args[1]["include_raw_content"] == "text"
    assert provider.get_config()["include_raw_content"] is False
