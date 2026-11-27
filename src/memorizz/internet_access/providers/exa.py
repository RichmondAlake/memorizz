# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Exa internet access provider."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from ..base import InternetAccessProvider, register_provider
from ..models import InternetPageContent, InternetSearchResult

logger = logging.getLogger(__name__)


class ExaError(RuntimeError):
    """Exa refused a Websets or Monitors request, or could not be reached."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class ExaProvider(InternetAccessProvider):
    """Internet provider backed by Exa's search and contents APIs."""

    provider_name = "exa"
    DEFAULT_BASE_URL = "https://api.exa.ai"
    DEFAULT_SEARCH_TYPE = "auto"
    DEFAULT_MAX_RESULTS = 5
    MAX_RESULTS_CAP = 25
    DEFAULT_MAX_CONTENT_CHARS = 12000
    DEFAULT_HIGHLIGHT_CHARS = 600
    DEFAULT_WEBSET_MAX = 25

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        config: Optional[Dict[str, Any]] = None,
    ):
        merged: Dict[str, Any] = dict(config or {})
        if api_key:
            merged["api_key"] = api_key
        if base_url:
            merged["base_url"] = base_url
        merged.setdefault("timeout", timeout)

        resolved_api_key = merged.get("api_key") or os.getenv("EXA_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "ExaProvider requires an API key via parameter or EXA_API_KEY"
            )

        self.api_key = resolved_api_key
        self.base_url = (merged.get("base_url") or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = int(merged.get("timeout", timeout))
        self.search_type = str(merged.get("search_type") or self.DEFAULT_SEARCH_TYPE)
        self.default_max_results = max(
            1,
            min(
                int(merged.get("default_max_results") or self.DEFAULT_MAX_RESULTS),
                self.MAX_RESULTS_CAP,
            ),
        )
        self.max_content_chars = self._coerce_positive_int(
            merged.get("max_content_chars", self.DEFAULT_MAX_CONTENT_CHARS)
        )
        self.highlight_chars = (
            self._coerce_positive_int(merged.get("highlight_chars"))
            or self.DEFAULT_HIGHLIGHT_CHARS
        )
        # Search results carry each page's text (``content``), not just the
        # passages Exa picked out of it.
        self.include_text = self._to_bool(merged.get("include_text", False))
        self.include_raw_results = self._to_bool(
            merged.get("include_raw_results", False)
        )
        self.include_raw_page = self._to_bool(merged.get("include_raw_page", False))
        # The most a webset an agent starts may ask for; each row is billed.
        self.webset_max_count = max(
            1, int(merged.get("webset_max_count") or self.DEFAULT_WEBSET_MAX)
        )
        # Where Exa reports a monitor's runs. It can carry a token, so like
        # the key it stays on the instance and out of the saved config.
        self.monitor_webhook_url = str(
            merged.get("monitor_webhook_url")
            or os.getenv("EXA_MONITOR_WEBHOOK_URL")
            or ""
        ).strip()
        # With no address of your own, make a temporary webhook.site one.
        self.temporary_webhooks = (
            True
            if merged.get("temporary_webhooks") is None
            else self._to_bool(merged.get("temporary_webhooks"))
        )

        # The key is held on the instance only; ``get_config()`` (and so the
        # saved agent record) reports ``api_key_set`` instead of the value.
        super().__init__(
            {
                "base_url": self.base_url,
                "timeout": self.timeout,
                "search_type": self.search_type,
                "default_max_results": self.default_max_results,
                "max_content_chars": self.max_content_chars,
                "highlight_chars": self.highlight_chars,
                "include_text": self.include_text,
                "include_raw_results": self.include_raw_results,
                "include_raw_page": self.include_raw_page,
                "webset_max_count": self.webset_max_count,
                "temporary_webhooks": self.temporary_webhooks,
            }
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-api-key": self.api_key,
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
        contents: Dict[str, Any] = {
            "highlights": {"maxCharacters": self.highlight_chars}
        }
        with_text = self._to_bool(kwargs.get("include_text", self.include_text))
        if with_text:
            contents["text"] = self._text_options()
        payload: Dict[str, Any] = {
            "query": query.strip(),
            "type": kwargs.get("search_type", self.search_type),
            "numResults": limit,
            "contents": contents,
        }
        # The agent's search tool speaks in snake case; Exa's API does not.
        for name, field in (
            ("include_domains", "includeDomains"),
            ("exclude_domains", "excludeDomains"),
            ("start_published_date", "startPublishedDate"),
            ("end_published_date", "endPublishedDate"),
            ("category", "category"),
        ):
            if kwargs.get(name):
                payload[field] = kwargs[name]

        response = self._post("/search", payload, timeout=kwargs.get("timeout"))
        results = response.get("results") or []
        return [self._parse_search_result(item, with_text) for item in results[:limit]]

    def fetch_url(self, url: str, **kwargs: Any) -> InternetPageContent:
        if not url or not url.strip():
            raise ValueError("URL must be provided")
        target = url.strip()

        payload: Dict[str, Any] = {"urls": [target], "text": self._text_options()}
        if kwargs.get("max_age_hours") is not None:
            payload["maxAgeHours"] = kwargs["max_age_hours"]

        response = self._post("/contents", payload, timeout=kwargs.get("timeout"))
        results = response.get("results") or []
        if not results:
            raise RuntimeError(self._fetch_problem(response, target))
        entry = results[0] if isinstance(results[0], dict) else {}

        metadata = self._metadata(entry)
        content, truncated, original_length = self._truncate_text(
            entry.get("text") or None, self.max_content_chars
        )
        cut_by_exa = bool(
            content
            and not truncated
            and self.max_content_chars
            and len(content) >= self.max_content_chars
        )
        if truncated or cut_by_exa:
            metadata.update(
                {
                    "content_truncated": True,
                    "content_char_limit": self.max_content_chars,
                    "content_returned_characters": len(content) if content else 0,
                }
            )
            if truncated:
                metadata["content_original_characters"] = original_length
        return InternetPageContent(
            url=entry.get("url") or target,
            title=entry.get("title"),
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
    # --- Websets: lists of things found, checked and filled in by Exa ---

    WEBSETS_PATH = "/websets/v0"
    WEBSET_ENTITIES = ("company", "person", "article", "research_paper")
    ENRICHMENT_FORMATS = ("text", "date", "number", "options", "email", "phone")
    MONITOR_EVENTS = ("monitor.run.completed",)

    def create_webset(
        self,
        query: str,
        *,
        count: int = 10,
        entity: Optional[str] = None,
        criteria: Optional[List[str]] = None,
        enrichments: Optional[List[Any]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a webset: Exa searches for ``count`` things matching
        ``query``, checks each against the ``criteria`` and fills in the
        ``enrichments`` (extra columns). It runs for minutes; read it with
        :meth:`get_webset` and :meth:`list_webset_items`.

        ``enrichments`` are column descriptions, or mappings with
        ``description`` and ``format`` (text, date, number, email, phone).
        """
        search: Dict[str, Any] = {
            "query": str(query).strip(),
            "count": max(1, int(count)),
        }
        kind = str(entity or "").strip().lower()
        if kind in self.WEBSET_ENTITIES:
            search["entity"] = {"type": kind}
        elif kind:
            search["entity"] = {"type": "custom", "description": str(entity).strip()}
        wanted = [str(item).strip() for item in criteria or [] if str(item).strip()]
        if wanted:
            search["criteria"] = [{"description": item} for item in wanted[:5]]
        payload: Dict[str, Any] = {"search": search}
        columns = []
        for item in enrichments or []:
            column = (
                dict(item) if isinstance(item, dict) else {"description": str(item)}
            )
            description = str(column.get("description") or "").strip()
            if not description:
                continue
            made = {"description": description}
            if column.get("format") in self.ENRICHMENT_FORMATS:
                made["format"] = column["format"]
            if column.get("options"):
                made["options"] = [
                    option if isinstance(option, dict) else {"label": str(option)}
                    for option in column["options"]
                ]
                made["format"] = "options"
            columns.append(made)
        if columns:
            payload["enrichments"] = columns
        if title:
            payload["title"] = str(title)[:200]
        return self._request("POST", f"{self.WEBSETS_PATH}/websets", payload)

    def get_webset(self, webset_id: str) -> Dict[str, Any]:
        return self._request("GET", f"{self.WEBSETS_PATH}/websets/{webset_id}")

    def list_webset_items(
        self, webset_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """The items found so far, oldest first, up to ``limit``."""
        items: List[Dict[str, Any]] = []
        cursor = None
        while len(items) < limit:
            params: Dict[str, Any] = {"limit": min(100, limit - len(items))}
            if cursor:
                params["cursor"] = cursor
            page = self._request(
                "GET", f"{self.WEBSETS_PATH}/websets/{webset_id}/items", params=params
            )
            items.extend(
                item for item in page.get("data") or [] if isinstance(item, dict)
            )
            cursor = page.get("nextCursor")
            if not page.get("hasMore") or not cursor:
                break
        return items

    def wait_for_webset(
        self, webset_id: str, seconds: float = 60, every: float = 3
    ) -> Dict[str, Any]:
        """Wait up to ``seconds`` for the webset to finish; return it either way."""
        import time

        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            webset = self.get_webset(webset_id)
            if webset.get("status") == "idle" or time.monotonic() >= deadline:
                return webset
            time.sleep(max(0.5, every))

    def cancel_webset(self, webset_id: str) -> Dict[str, Any]:
        return self._request(
            "POST", f"{self.WEBSETS_PATH}/websets/{webset_id}/cancel", {}
        )

    # --- Monitors: a search Exa runs again on a schedule ---

    def create_monitor(
        self,
        query: str,
        *,
        webhook_url: Optional[str] = None,
        period: str = "1d",
        name: Optional[str] = None,
        num_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Have Exa run ``query`` every ``period`` (``"1h"`` to ``"30d"``).

        Exa's API requires a public HTTPS address to call when a run ends:
        ``webhook_url``, or ``EXA_MONITOR_WEBHOOK_URL``. With neither, a
        temporary webhook.site address is made (:meth:`temporary_webhook`),
        as Exa's dashboard does when the field is left blank. The runs are
        read with :meth:`list_monitor_runs`, so the address only has to
        accept the calls. Anyone holding a webhook.site address can read
        what is sent to it; use your own for private topics, or pass
        ``temporary_webhooks=False`` in the config to refuse.
        """
        search: Dict[str, Any] = {
            "query": str(query).strip(),
            "numResults": max(1, min(int(num_results), 100)),
            "contents": {"highlights": True},
        }
        if include_domains:
            search["includeDomains"] = list(include_domains)
        if exclude_domains:
            search["excludeDomains"] = list(exclude_domains)
        payload: Dict[str, Any] = {
            "search": search,
            "trigger": {"type": "interval", "period": str(period)},
        }
        address = str(webhook_url or self.monitor_webhook_url or "").strip()
        if not address and self.temporary_webhooks:
            address = self.temporary_webhook()
        if not address:
            raise ExaError(
                "Exa needs a public HTTPS address to call when a run ends. "
                "Set EXA_MONITOR_WEBHOOK_URL where MemoRizz runs."
            )
        payload["webhook"] = {"url": address, "events": list(self.MONITOR_EVENTS)}
        if name:
            payload["name"] = str(name)[:200]
        if metadata:
            payload["metadata"] = {str(k): str(v) for k, v in metadata.items()}
        return self._request("POST", "/monitors", payload)

    TEMPORARY_WEBHOOK_SERVICE = "https://webhook.site"
    TEMPORARY_WEBHOOK_RETRY_SECONDS = 4
    # A free webhook.site address lasts a week.
    TEMPORARY_WEBHOOK_SECONDS = 7 * 24 * 3600

    def temporary_webhook(self) -> str:
        """A throwaway public address from webhook.site for a monitor to
        report to. Nothing about the monitor or this account is sent to get
        it, and the Exa key never goes there."""
        import time

        problem = "A temporary address for the monitor to report to could not be made"
        token = ""
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.TEMPORARY_WEBHOOK_SERVICE}/token",
                    headers={"Accept": "application/json"},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise ExaError(
                    f"{problem}. Set EXA_MONITOR_WEBHOOK_URL to your own."
                ) from exc
            if response.status_code == 429 and attempt < 2:
                # webhook.site limits how fast addresses are handed out.
                time.sleep(self.TEMPORARY_WEBHOOK_RETRY_SECONDS * (attempt + 1))
                continue
            if response.status_code == 429:
                raise ExaError(
                    f"{problem}: webhook.site is limiting requests. Try again "
                    "in a minute, or set EXA_MONITOR_WEBHOOK_URL to your own."
                )
            try:
                response.raise_for_status()
                token = str(response.json().get("uuid") or "").strip()
            except (requests.RequestException, ValueError, AttributeError) as exc:
                raise ExaError(
                    f"{problem}. Set EXA_MONITOR_WEBHOOK_URL to your own."
                ) from exc
            break
        if not token:
            raise ExaError(f"{problem}. Set EXA_MONITOR_WEBHOOK_URL to your own.")
        return f"{self.TEMPORARY_WEBHOOK_SERVICE}/{token}"

    def list_monitors(self, limit: int = 50) -> List[Dict[str, Any]]:
        page = self._request("GET", "/monitors", params={"limit": limit})
        return [item for item in page.get("data") or [] if isinstance(item, dict)]

    def get_monitor(self, monitor_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/monitors/{monitor_id}")

    def update_monitor(self, monitor_id: str, **changes: Any) -> Dict[str, Any]:
        """Change a monitor, for example ``status="paused"``."""
        return self._request("PATCH", f"/monitors/{monitor_id}", changes)

    def delete_monitor(self, monitor_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/monitors/{monitor_id}")

    def trigger_monitor(self, monitor_id: str) -> Dict[str, Any]:
        """Run the monitor now, outside its schedule."""
        return self._request("POST", f"/monitors/{monitor_id}/trigger", {})

    def list_monitor_runs(
        self, monitor_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        page = self._request(
            "GET", f"/monitors/{monitor_id}/runs", params={"limit": limit}
        )
        return [item for item in page.get("data") or [] if isinstance(item, dict)]

    def agent_tools(self, agent: Any) -> list:
        """Websets and Monitors as tools, for an agent whose provider is Exa."""
        from ..exa_tools import build_exa_tools

        return build_exa_tools(agent, self)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """One call to Exa. An error carries what Exa said was wrong."""
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        try:
            response = self._session.request(
                method,
                url,
                json=payload if method in {"POST", "PATCH"} else None,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ExaError(f"Exa could not be reached: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            said = ""
            try:
                body = response.json()
                said = str(body.get("error") or body.get("message") or "")
            except ValueError:
                said = response.text
            raise ExaError(
                f"Exa refused the request ({response.status_code})"
                + (f": {said.strip()[:300]}" if said.strip() else ""),
                status=response.status_code,
            )
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise ExaError("Exa returned something that is not JSON") from exc
        return data if isinstance(data, dict) else {"data": data}

    def _text_options(self) -> Any:
        if self.max_content_chars:
            return {"maxCharacters": self.max_content_chars}
        return True

    def _post(
        self, path: str, payload: Dict[str, Any], timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        try:
            response = self._session.post(
                url, json=payload, timeout=timeout or self.timeout
            )
            response.raise_for_status()
            if not response.content:
                return {}
            data = response.json()
            return data if isinstance(data, dict) else {}
        except requests.HTTPError as exc:
            response_text = exc.response.text if exc.response is not None else ""
            logger.error(
                "Exa HTTP error: %s - %s", exc, response_text.strip() or "<empty>"
            )
            raise
        except requests.RequestException as exc:
            logger.error("Exa request failed: %s", exc)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Exa returned invalid JSON: %s", exc)
            raise

    @staticmethod
    def _fetch_problem(response: Dict[str, Any], url: str) -> str:
        """Why Exa returned no page, from its per-URL status when it gave one."""
        for status in response.get("statuses") or []:
            if not isinstance(status, dict) or status.get("status") != "error":
                continue
            error = status.get("error") or {}
            tag = error.get("tag") if isinstance(error, dict) else None
            code = error.get("httpStatusCode") if isinstance(error, dict) else None
            detail = " ".join(str(part) for part in (tag, code) if part)
            return f"Exa could not read {url}" + (f" ({detail})" if detail else "")
        return f"Exa returned no content for {url}"

    @staticmethod
    def _metadata(item: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if item.get("publishedDate"):
            metadata["published_at"] = item["publishedDate"]
        if item.get("author"):
            metadata["author"] = item["author"]
        host = urlparse(str(item.get("url") or "")).netloc
        if host:
            metadata["site"] = host
        return metadata

    def _parse_search_result(
        self, item: Any, with_text: Optional[bool] = None
    ) -> InternetSearchResult:
        if not isinstance(item, dict):
            return InternetSearchResult(url=str(item))

        if with_text is None:
            with_text = self.include_text
        metadata = self._metadata(item)
        highlights = [
            str(part).strip() for part in item.get("highlights") or [] if part
        ]
        text = item.get("text") or None
        snippet = " … ".join(highlights) or item.get("summary") or None
        if not snippet and text:
            snippet = text[: self.highlight_chars]
        content, truncated, original_length = self._truncate_text(
            text if with_text or not highlights else None,
            self.max_content_chars,
        )
        # Exa cuts the text to the limit it was asked for, so a text of
        # exactly that length was longer.
        truncated = truncated or bool(
            content
            and self.max_content_chars
            and len(content) >= self.max_content_chars
        )
        if truncated:
            metadata.update(
                {
                    "content_truncated": True,
                    "content_char_limit": self.max_content_chars,
                    "content_original_characters": original_length,
                }
            )
        return InternetSearchResult(
            url=item.get("url") or "",
            title=item.get("title"),
            snippet=snippet,
            score=item.get("score"),
            metadata=metadata,
            raw=item if self.include_raw_results else None,
            content=content,
        )

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)


# Register provider on import
register_provider(ExaProvider.provider_name, ExaProvider)
