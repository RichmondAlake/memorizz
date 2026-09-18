"""Bounded, rate-limited Notion REST transport (no agent/LLM in the CRUD path)."""

import json
import math
import threading
import time
import uuid
from urllib.parse import urlparse

import requests

API_VERSION = "2026-03-11"


def validate_transport_options(timeout, requests_per_second, max_retries):
    for name, value, ceiling in (
        ("timeout", timeout, 60),
        ("requests_per_second", requests_per_second, 3),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= ceiling
        ):
            raise ValueError(f"{name} must be between 0 and {ceiling}")
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or not 0 <= max_retries <= 8
    ):
        raise ValueError("max_retries must be between 0 and 8")


class NotionError(RuntimeError):
    """A Notion operation failed; error messages never contain response bodies."""


class NotionAPIError(NotionError):
    def __init__(self, status, code="request_failed"):
        self.status = status
        allowed = {
            "invalid_json",
            "invalid_request_url",
            "invalid_request",
            "validation_error",
            "missing_version",
            "unauthorized",
            "restricted_resource",
            "object_not_found",
            "conflict_error",
            "rate_limited",
            "internal_server_error",
            "service_unavailable",
            "service_overload",
            "database_connection_unavailable",
            "gateway_timeout",
        }
        self.code = (
            code if isinstance(code, str) and code in allowed else "request_failed"
        )
        super().__init__(f"Notion API request failed (HTTP {status}, {self.code})")


class NotionWriteUncertain(NotionError):
    """A create may have committed. Reconcile its stable ID; do not blindly retry."""


class NotionClient:
    def __init__(
        self,
        token,
        *,
        timeout=30.0,
        requests_per_second=2.5,
        max_retries=4,
        session=None,
        base_url="https://api.notion.com/v1",
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        if not isinstance(token, str) or not token.strip():
            raise ValueError("A Notion integration token is required")
        validate_transport_options(timeout, requests_per_second, max_retries)
        parsed = urlparse(base_url)
        # Loopback HTTP exists solely for deterministic wire-level tests. Never
        # send integration credentials to redirects or arbitrary remote origins.
        if base_url != "https://api.notion.com/v1" and not (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and not parsed.username
            and not parsed.password
        ):
            raise ValueError(
                "Notion base URL must be the official API or a loopback test server"
            )
        self._headers = {
            "Authorization": "Bearer " + token,
            "Notion-Version": API_VERSION,
            "Content-Type": "application/json",
        }
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._interval = 1.0 / requests_per_second
        self._max_retries = max_retries
        self._sleep, self._clock = sleep, clock
        self._next_request = 0.0
        self._lock = threading.RLock()

    def request(self, method, path, *, body=None, params=None):
        if not path.startswith("/") or "://" in path or ".." in path:
            raise ValueError("A relative Notion API path is required")
        encoded = (
            json.dumps(body, ensure_ascii=False, allow_nan=False)
            if body is not None
            else None
        )
        if encoded is not None and len(encoded.encode("utf-8")) > 450_000:
            raise ValueError(
                "Notion request is too large; split this memory into smaller records"
            )
        safe = method in {"GET", "DELETE", "PATCH"} or path.endswith("/query")
        uncertain_update = False
        with self._lock:
            for attempt in range(self._max_retries + 1):
                delay = max(0.0, self._next_request - self._clock())
                if delay:
                    self._sleep(delay)
                self._next_request = self._clock() + self._interval
                try:
                    response = self._session.request(
                        method,
                        self._base_url + path,
                        data=encoded,
                        params=params,
                        headers=self._headers,
                        timeout=self._timeout,
                        allow_redirects=False,
                    )
                except requests.RequestException:
                    if not safe:
                        raise NotionWriteUncertain(
                            "Notion create outcome is unknown; reconcile before retrying"
                        ) from None
                    if attempt == self._max_retries:
                        if method == "PATCH":
                            raise NotionWriteUncertain(
                                "Notion update outcome is unknown; reconcile before retrying"
                            ) from None
                        raise NotionError(
                            "Notion request could not reach the service"
                        ) from None
                    uncertain_update = method == "PATCH"
                    self._next_request = self._clock() + min(2**attempt, 30)
                    continue
                if 200 <= response.status_code < 300:
                    try:
                        result = response.json()
                    except ValueError:
                        if not safe or method == "PATCH":
                            raise NotionWriteUncertain(
                                "Notion write returned an unreadable response; reconcile before retrying"
                            ) from None
                        raise NotionError("Notion returned invalid JSON") from None
                    if not isinstance(result, dict):
                        if not safe or method == "PATCH":
                            raise NotionWriteUncertain(
                                "Notion write returned an invalid response; reconcile before retrying"
                            )
                        raise NotionError("Notion returned an invalid response object")
                    if not safe or method == "PATCH":
                        try:
                            uuid.UUID(str(result["id"]))
                        except (ValueError, KeyError, TypeError):
                            raise NotionWriteUncertain(
                                "Notion write returned no valid resource ID; reconcile before retrying"
                            ) from None
                    return result
                retryable = response.status_code in {429, 529} or (
                    safe and response.status_code in {500, 502, 503, 504}
                )
                if retryable and attempt < self._max_retries:
                    try:
                        retry_after = float(
                            response.headers.get("Retry-After", 2**attempt)
                        )
                    except (TypeError, ValueError):
                        retry_after = float(2**attempt)
                    # Never shorten Retry-After. Surface long waits for the
                    # durable sync worker instead of blocking a request thread.
                    if math.isfinite(retry_after) and 0 <= retry_after <= 60:
                        self._next_request = self._clock() + retry_after
                        continue
                if (not safe or method == "PATCH") and response.status_code in {
                    500,
                    502,
                    503,
                    504,
                }:
                    raise NotionWriteUncertain(
                        "Notion write outcome is unknown; reconcile before retrying"
                    )
                if uncertain_update:
                    raise NotionWriteUncertain(
                        "Notion update outcome is unknown; reconcile before retrying"
                    )
                try:
                    code = response.json().get("code", "request_failed")
                except (ValueError, AttributeError):
                    code = "request_failed"
                raise NotionAPIError(response.status_code, code)
        raise NotionError("Notion retry budget exhausted")  # pragma: no cover

    def close(self):
        if self._owns_session:
            self._session.close()
