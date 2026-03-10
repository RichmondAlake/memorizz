# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Firecrawl internet access provider."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..base import InternetAccessProvider, register_provider
from ..models import InternetPageContent, InternetSearchResult

logger = logging.getLogger(__name__)


class FirecrawlProvider(InternetAccessProvider):
    """Internet access provider backed by Firecrawl's search + crawl APIs."""

    provider_name = "firecrawl"
    DEFAULT_BASE_URL = "https://api.firecrawl.dev/v1"
    DEFAULT_MAX_CONTENT_CHARS = 16000
    DEFAULT_MAX_RAW_CHARS = 2000
    _TRUNCATION_NOTE = (
        "[MemoRizz trimmed the page to fit inside the model's context window]"
    )

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

        resolved_api_key = merged_config.get("api_key") or os.getenv(
            "FIRECRAWL_API_KEY"
        )
        if not resolved_api_key:
            raise ValueError(
                "FirecrawlProvider requires an API key via parameter or FIRECRAWL_API_KEY"
            )

        resolved_base_url = merged_config.get("base_url", self.DEFAULT_BASE_URL)
        resolved_timeout = int(merged_config.get("timeout", timeout))
        resolved_max_content_chars = self._coerce_positive_int(
            merged_config.get("max_content_chars", self.DEFAULT_MAX_CONTENT_CHARS)
        )
        include_raw = merged_config.get("include_raw_response", False)
        resolved_include_raw = (
            include_raw.lower() in {"1", "true", "yes", "on"}
            if isinstance(include_raw, str)
            else bool(include_raw)
        )
        resolved_max_raw_chars = self._coerce_positive_int(
            merged_config.get("max_raw_chars", self.DEFAULT_MAX_RAW_CHARS)
        )

        super().__init__(
            {
                "api_key": resolved_api_key,
                "base_url": resolved_base_url,
                "timeout": resolved_timeout,
                "max_content_chars": resolved_max_content_chars,
                "include_raw_response": resolved_include_raw,
                "max_raw_chars": resolved_max_raw_chars,
            }
        )

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = resolved_timeout
        self.max_content_chars = resolved_max_content_chars
        self.include_raw_response = resolved_include_raw
        self.max_raw_chars = resolved_max_raw_chars
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> List[InternetSearchResult]:
        if not query or not query.strip():
            raise ValueError("Search query must be provided")

        payload = {
            "query": query.strip(),
            "limit": max(1, min(max_results or 5, 20)),
        }
        if "filters" in kwargs:
            payload["filters"] = kwargs["filters"]

        response = self._post("/search", payload, **kwargs)
        results = response.get("results") or response.get("data") or []
        normalized = [
            self._parse_search_result(item) for item in results[: payload["limit"]]
        ]
        return normalized

    def fetch_url(self, url: str, **kwargs) -> InternetPageContent:
        if not url or not url.strip():
            raise ValueError("URL must be provided")

        payload: Dict[str, Any] = {"url": url.strip()}
        if "formats" in kwargs:
            payload["formats"] = kwargs["formats"]
        else:
            # Firecrawl expects specific enums; keep markdown + rawHtml by default.
            payload["formats"] = ["markdown", "rawHtml"]

        response = self._post("/scrape", payload, **kwargs)
        return self._parse_page_content(response, url.strip())

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    # Internal helpers -------------------------------------------------
    def _post(self, path: str, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        timeout = kwargs.get("timeout", self.timeout)
        try:
            response = self._session.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()
        except requests.HTTPError as exc:
            response_text = exc.response.text if exc.response is not None else ""
            logger.error(
                "Firecrawl HTTP error: %s - %s", exc, response_text.strip() or "<empty>"
            )
            raise
        except requests.RequestException as exc:
            logger.error("Firecrawl request failed: %s", exc)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Firecrawl returned invalid JSON: %s", exc)
            raise

    def _parse_search_result(self, item: Dict[str, Any]) -> InternetSearchResult:
        if not isinstance(item, dict):
            return InternetSearchResult(url=str(item))

        return InternetSearchResult(
            url=item.get("url") or item.get("link") or "",
            title=item.get("title"),
            snippet=item.get("snippet") or item.get("description"),
            score=item.get("score") or item.get("relevance"),
            metadata={
                "source": item.get("source"),
                "published_at": item.get("publishedAt") or item.get("published_at"),
            },
            raw=item,
        )

    def _parse_page_content(
        self, response: Dict[str, Any], url: str
    ) -> InternetPageContent:
        title = response.get("title") or response.get("metadata", {}).get("title")
        markdown_content = response.get("markdown") or response.get("content")
        markdown_content = self._normalize_content(markdown_content)
        metadata = self._extract_metadata(response)

        content = markdown_content
        truncated = False
        returned_chars = None
        original_length = len(content) if content else None
        if content:
            content, truncated, returned_chars = self._apply_content_budget(content)
            if truncated:
                metadata = dict(metadata or {})
                metadata.update(
                    {
                        "content_truncated": True,
                        "content_char_limit": self.max_content_chars,
                        "content_original_characters": original_length,
                        "content_returned_characters": returned_chars,
                    }
                )

        return InternetPageContent(
            url=url,
            title=title,
            content=content,
            metadata=metadata,
            raw=self._prepare_raw_response(response),
        )

    def _extract_metadata(self, response: Dict[str, Any]) -> Dict[str, Any]:
        metadata = response.get("metadata") or {}
        if not metadata and response.get("data") and isinstance(response["data"], dict):
            metadata = response["data"].get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    def _normalize_content(self, raw_content: Any) -> Optional[str]:
        if raw_content is None:
            return None
        if isinstance(raw_content, list):
            return "\n".join(str(chunk) for chunk in raw_content if chunk)
        if isinstance(raw_content, str):
            return raw_content
        return str(raw_content)

    def _apply_content_budget(self, content: str) -> Tuple[str, bool, int]:
        truncated_content, was_truncated, _ = self._truncate_text(
            content, self.max_content_chars
        )
        returned_chars = len(truncated_content)
        if was_truncated:
            truncated_content = f"{truncated_content}\n\n---\n{self._TRUNCATION_NOTE}"
            logger.debug(
                "Firecrawl truncated page content from %s to %s characters",
                len(content),
                len(truncated_content),
            )
        return truncated_content, was_truncated, returned_chars

    def _prepare_raw_response(
        self, response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not self.include_raw_response or not isinstance(response, dict):
            return None
        return self._sanitize_nested_payload(response)

    def _sanitize_nested_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._sanitize_nested_payload(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_nested_payload(item) for item in value]
        if isinstance(value, str):
            truncated, was_truncated, original = self._truncate_text(
                value, self.max_raw_chars
            )
            if was_truncated and original is not None:
                return f"{truncated}\n\n[Firecrawl raw field truncated from {original} characters]"
            return truncated
        return value

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return None
        return coerced if coerced > 0 else None

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
        return value[:limit], True, original_length


# Register provider on import
register_provider(FirecrawlProvider.provider_name, FirecrawlProvider)
