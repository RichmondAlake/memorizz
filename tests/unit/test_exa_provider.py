"""Unit tests for the Exa internet provider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memorizz.internet_access import (
    create_internet_access_provider,
    get_default_internet_access_provider,
)
from memorizz.internet_access.providers.exa import ExaProvider


@pytest.mark.unit
def test_exa_search_asks_for_passages_and_normalizes_results():
    provider = ExaProvider(api_key="test-key", config={"include_raw_results": True})
    provider._post = MagicMock(
        return_value={
            "results": [
                {
                    "id": "doc_1",
                    "url": "https://example.com/doc",
                    "title": "Example Doc",
                    "publishedDate": "2025-03-01T00:00:00.000Z",
                    "author": "Ada",
                    "highlights": ["First passage.", "Second passage."],
                }
            ]
        }
    )

    results = provider.search(
        "vector databases", max_results=2, include_domains=["example.com"]
    )

    path, payload = provider._post.call_args.args
    assert path == "/search"
    assert payload["numResults"] == 2 and payload["type"] == "auto"
    assert payload["includeDomains"] == ["example.com"]
    assert "highlights" in payload["contents"] and "text" not in payload["contents"]
    assert len(results) == 1
    assert results[0].snippet == "First passage. … Second passage."
    assert results[0].content is None
    assert results[0].metadata == {
        "published_at": "2025-03-01T00:00:00.000Z",
        "author": "Ada",
        "site": "example.com",
    }
    assert results[0].raw["id"] == "doc_1"


@pytest.mark.unit
def test_exa_search_can_carry_each_pages_text():
    provider = ExaProvider(
        api_key="test-key", config={"include_text": True, "max_content_chars": 10}
    )
    provider._post = MagicMock(
        return_value={"results": [{"url": "https://example.com", "text": "A" * 40}]}
    )

    result = provider.search("anything")[0]

    assert provider._post.call_args.args[1]["contents"]["text"] == {"maxCharacters": 10}
    assert result.content == "A" * 10
    assert result.metadata["content_truncated"] is True
    assert result.metadata["content_original_characters"] == 40


@pytest.mark.unit
def test_exa_fetch_returns_the_page_and_says_why_when_it_cannot():
    provider = ExaProvider(api_key="test-key", config={"include_raw_page": True})
    provider._post = MagicMock(
        return_value={
            "results": [
                {"url": "https://example.com/doc", "title": "Doc", "text": "Body"}
            ],
            "statuses": [{"id": "https://example.com/doc", "status": "success"}],
        }
    )

    page = provider.fetch_url(" https://example.com/doc ")

    path, payload = provider._post.call_args.args
    assert path == "/contents" and payload["urls"] == ["https://example.com/doc"]
    assert (page.title, page.content) == ("Doc", "Body")
    assert page.raw["title"] == "Doc"

    provider._post = MagicMock(
        return_value={
            "results": [],
            "statuses": [
                {
                    "id": "https://example.com/gone",
                    "status": "error",
                    "error": {"tag": "CRAWL_NOT_FOUND", "httpStatusCode": 404},
                }
            ],
        }
    )
    with pytest.raises(RuntimeError, match="CRAWL_NOT_FOUND 404"):
        provider.fetch_url("https://example.com/gone")


@pytest.mark.unit
def test_exa_sends_its_key_in_a_header_and_never_saves_it():
    provider = ExaProvider(api_key="secret-key")
    response = MagicMock(content=b"{}")
    response.json.return_value = {"results": []}
    provider._session.post = MagicMock(return_value=response)

    provider.search("anything")

    assert provider._session.headers["x-api-key"] == "secret-key"
    assert "secret-key" not in str(provider._session.post.call_args.kwargs["json"])
    config = provider.get_config()
    assert config["api_key_set"] is True
    assert "secret-key" not in str(config)


@pytest.mark.unit
def test_exa_is_registered_and_is_the_default_when_only_its_key_is_set(monkeypatch):
    for name in (
        "MEMORIZZ_DEFAULT_INTERNET_PROVIDER",
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EXA_API_KEY", "test-key")

    assert isinstance(create_internet_access_provider("exa", {}), ExaProvider)
    assert get_default_internet_access_provider().get_provider_name() == "exa"

    monkeypatch.delenv("EXA_API_KEY")
    with pytest.raises(ValueError, match="EXA_API_KEY"):
        ExaProvider()


@pytest.mark.unit
def test_exa_page_text_can_be_asked_for_one_search_and_says_when_it_was_cut():
    provider = ExaProvider(api_key="test-key", config={"max_content_chars": 10})
    provider._post = MagicMock(
        return_value={
            "results": [
                {"url": "https://example.com", "highlights": ["H"], "text": "A" * 10}
            ]
        }
    )

    result = provider.search("anything", include_text=True)[0]

    assert provider._post.call_args.args[1]["contents"]["text"] == {"maxCharacters": 10}
    # Exa already cut the text to the limit, so its real length is unknown.
    assert result.content == "A" * 10
    assert result.metadata["content_truncated"] is True
    assert provider.search("anything")[0].content is None

    page = provider.fetch_url("https://example.com")
    assert page.metadata["content_truncated"] is True
    assert "content_original_characters" not in page.metadata
