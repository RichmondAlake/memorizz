"""Shared HTTP plumbing for the REST voice providers."""

from __future__ import annotations

from typing import Any, Optional

from ..base import VoiceError

TIMEOUT = (10, 90)  # connect, read


def session(http: Optional[Any] = None) -> Any:
    """The injected HTTP client (tests) or the ``requests`` module."""
    if http is not None:
        return http
    import requests

    return requests


def check(response: Any, provider: str) -> Any:
    """Return ``response`` when it succeeded; raise a readable VoiceError if not."""
    status = getattr(response, "status_code", 200)
    if status < 400:
        return response
    detail = ""
    try:
        body = response.json()
        detail = body.get("detail") or body.get("error") or body.get("message") or ""
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("status") or str(detail)
    except Exception:
        detail = (getattr(response, "text", "") or "")[:200]
    code = {
        401: "voice_unauthorized",
        403: "voice_forbidden",
        429: "voice_rate_limited",
    }
    raise VoiceError(
        f"{provider} answered {status}" + (f": {detail}" if detail else ""),
        code=code.get(status, "voice_provider_error"),
    )
