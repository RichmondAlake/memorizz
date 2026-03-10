# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tavily internet access provider."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..base import InternetAccessProvider, register_provider
from ..models import InternetPageContent, InternetSearchResult

logger = logging.getLogger(__name__)


class TavilyProvider(InternetAccessProvider):
    """Internet provider backed by Tavily's search and extract APIs."""

    provider_name = "tavily"
    DEFAULT_BASE_URL = "https://api.tavily.com"
    DEFAULT_SEARCH_DEPTH = "basic"
    DEFAULT_MAX_RESULTS = 5
    MAX_RESULTS_CAP = 10
    DEFAULT_MAX_CONTENT_CHARS = 12000

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        config: Optional[Dict[str, Any]] = None,
    ):
        merged_config: Dict[str, Any] = dict(config or {})
        if api_key:
            merged_config["api_key"] = api_key
        if base_url:
            merged_config["base_url"] = base_url
        merged_config.setdefault("timeout", timeout)

        resolved_api_key = merged_config.get("api_key") or os.getenv("TAVILY_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "TavilyProvider requires an API key via parameter or TAVILY_API_KEY"
            )

        resolved_base_url = (
            merged_config.get("base_url") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        resolved_timeout = int(merged_config.get("timeout", timeout))
        resolved_search_depth = merged_config.get(
            "search_depth", self.DEFAULT_SEARCH_DEPTH
        )
        resolved_max_results = merged_config.get(
            "default_max_results", self.DEFAULT_MAX_RESULTS
        )
        resolved_max_content_chars = self._coerce_positive_int(
            merged_config.get("max_content_chars", self.DEFAULT_MAX_CONTENT_CHARS)
        )
        resolved_include_raw_results = self._to_bool(
            merged_config.get("include_raw_results", False)
        )
        resolved_include_raw_page = self._to_bool(
            merged_config.get("include_raw_page", False)
        )

        super().__init__(
            {
                "api_key": resolved_api_key,
                "base_url": resolved_base_url,
                "timeout": resolved_timeout,
                "search_depth": resolved_search_depth,
                "default_max_results": resolved_max_results,
                "max_content_chars": resolved_max_content_chars,
                "include_raw_results": resolved_include_raw_results,
                "include_raw_page": resolved_include_raw_page,
            }
        )

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url
        self.timeout = resolved_timeout
        self.search_depth = resolved_search_depth
        self.default_max_results = max(
            1,
            min(
                int(resolved_max_results or self.DEFAULT_MAX_RESULTS),
                self.MAX_RESULTS_CAP,
            ),
        )
        self.max_content_chars = resolved_max_content_chars
        self.include_raw_results = resolved_include_raw_results
        self.include_raw_page = resolved_include_raw_page
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def search(
        self, query: str, max_results: int = 5, **kwargs: Any
    ) -> List[InternetSearchResult]:
        if not query or not query.strip():
            raise ValueError("Search query must be provided")

        limit = max(
            1, min(max_results or self.default_max_results, self.MAX_RESULTS_CAP)
        )
        payload: Dict[str, Any] = {
            "query": query.strip(),
            "search_depth": kwargs.get("search_depth", self.search_depth),
            "max_results": limit,
        }
        for key in (
            "include_domains",
            "exclude_domains",
            "include_images",
            "include_image_descriptions",
            "include_answer",
        ):
            if key in kwargs:
                payload[key] = kwargs[key]

        response = self._post("/search", payload, timeout=kwargs.get("timeout"))
        results = response.get("results") or response.get("data") or []
        normalized = []
        for item in results[:limit]:
            normalized.append(self._parse_search_result(item))
        return normalized

    def fetch_url(self, url: str, **kwargs: Any) -> InternetPageContent:
        if not url or not url.strip():
            raise ValueError("URL must be provided")

        payload: Dict[str, Any] = {
            "urls": [url.strip()],
        }
        for key in ("include_images", "include_links", "include_summary"):
            if key in kwargs:
                payload[key] = kwargs[key]

        response = self._post("/extract", payload, timeout=kwargs.get("timeout"))
        entry = self._extract_page_entry(response)
        content = (
            entry.get("content") or entry.get("markdown") or entry.get("raw_content")
        )
        title = entry.get("title")
        metadata = entry.get("metadata") or {}
        metadata = dict(metadata) if isinstance(metadata, dict) else {}

        if "site" not in metadata and entry.get("site"):
            metadata["site"] = entry.get("site")

        if "score" in entry:
            metadata.setdefault("score", entry.get("score"))

        content, truncated, original_length = self._truncate_text(
            content, self.max_content_chars
        )
        if truncated:
            metadata.update(
                {
                    "content_truncated": True,
                    "content_char_limit": self.max_content_chars,
                    "content_original_characters": original_length,
                    "content_returned_characters": len(content) if content else 0,
                }
            )

        return InternetPageContent(
            url=entry.get("url") or url.strip(),
            title=title,
            content=content,
            metadata=metadata or None,
            raw=entry if self.include_raw_page else None,
        )

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    # Internal helpers -------------------------------------------------
    def _post(
        self, path: str, payload: Dict[str, Any], timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        request_payload = {"api_key": self.api_key}
        request_payload.update(payload)
        try:
            response = self._session.post(
                url,
                json=request_payload,
                timeout=timeout or self.timeout,
            )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()
        except requests.HTTPError as exc:
            response_text = exc.response.text if exc.response is not None else ""
            logger.error(
                "Tavily HTTP error: %s - %s", exc, response_text.strip() or "<empty>"
            )
            raise
        except requests.RequestException as exc:
            logger.error("Tavily request failed: %s", exc)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Tavily returned invalid JSON: %s", exc)
            raise

    def _parse_search_result(self, item: Any) -> InternetSearchResult:
        if not isinstance(item, dict):
            return InternetSearchResult(url=str(item))

        metadata = {
            "published_at": item.get("published_date") or item.get("publishedAt"),
            "site": item.get("site"),
        }
        if "answer" in item:
            metadata["answer"] = item.get("answer")

        raw_payload = item if self.include_raw_results else None
        return InternetSearchResult(
            url=item.get("url") or "",
            title=item.get("title"),
            snippet=item.get("content") or item.get("snippet") or item.get("summary"),
            score=item.get("score"),
            metadata=metadata,
            raw=raw_payload,
        )

    def _extract_page_entry(self, response: Dict[str, Any]) -> Dict[str, Any]:
        candidates: Any = None
        for key in ("results", "data", "items"):
            if key in response:
                candidates = response[key]
                break
        if isinstance(candidates, list) and candidates:
            return candidates[0]
        if isinstance(candidates, dict):
            return candidates
        return response

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            result = int(value)
        except (TypeError, ValueError):
            return None
        return result if result > 0 else None

    @staticmethod
    def _truncate_text(
        value: Optional[str], limit: Optional[int]
    ) -> Tuple[Optional[str], bool, Optional[int]]:
        if not value:
            return value, False, None
        if not limit or limit <= 0:
            return value, False, len(value)
        original_length = len(value)
        if original_length <= limit:
            return value, False, original_length
        truncated = value[:limit]
        return truncated, True, original_length


# Register provider on import
register_provider(TavilyProvider.provider_name, TavilyProvider)
