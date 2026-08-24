"""Security controls for inspecting traces through the local Memorizz UI."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from fastapi import Request

logger = logging.getLogger(__name__)

_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|"
    r"password|passwd|secret|cookie|private[_-]?key)",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_KEY_TOKEN = re.compile(r"\b(?:sk-(?:ant-)?|gh[oprsu]_|xox[baprs]-)[A-Za-z0-9_-]{8,}\b")
_URI_CREDENTIALS = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@", re.I
)
_QUERY_SECRET = re.compile(r"(?i)([?&](?:api[_-]?key|token|secret|password)=)[^&#\s]+")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def ui_read_only() -> bool:
    return _env_bool("MEMORIZZ_UI_READ_ONLY", False)


class ReadOnlyProviderProxy:
    """Fail closed on provider mutations while preserving all read APIs."""

    _MUTATION_PREFIXES = (
        "store",
        "update",
        "delete",
        "clear",
        "invalidate",
        "create",
        "save",
        "upsert",
        "add_",
        "approve",
        "reject",
        "promote",
        "demote",
    )

    def __init__(self, provider: Any):
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "read_only", True)

    @property
    def wrapped_provider(self) -> Any:
        return object.__getattribute__(self, "_provider")

    def close(self) -> Any:
        return self.wrapped_provider.close()

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.wrapped_provider, name)
        if callable(value) and name.lower().startswith(self._MUTATION_PREFIXES):

            def _denied(*_args: Any, **_kwargs: Any) -> Any:
                raise PermissionError(
                    f"Provider mutation '{name}' is disabled in read-only UI mode"
                )

            return _denied
        return value


class UIAccessController:
    """Token authentication with a short-lived signed HttpOnly session."""

    cookie_name = "memorizz_ui_session"

    def __init__(self) -> None:
        self.token = os.getenv("MEMORIZZ_UI_AUTH_TOKEN", "").strip()
        explicit_secret = os.getenv("MEMORIZZ_UI_SESSION_SECRET", "").strip()
        seed = explicit_secret or self.token or secrets.token_urlsafe(32)
        self._secret = hashlib.sha256(
            f"memorizz-ui-session:{seed}".encode("utf-8")
        ).digest()
        try:
            self.ttl_seconds = max(
                300, min(int(os.getenv("MEMORIZZ_UI_SESSION_SECONDS", "28800")), 86400)
            )
        except ValueError:
            self.ttl_seconds = 28800

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def token_matches(self, candidate: str) -> bool:
        return bool(self.token) and secrets.compare_digest(
            candidate.encode("utf-8"), self.token.encode("utf-8")
        )

    def issue_session(self) -> str:
        expires_at = str(int(time.time()) + self.ttl_seconds)
        signature = hmac.new(
            self._secret, expires_at.encode("ascii"), hashlib.sha256
        ).hexdigest()
        raw = f"{expires_at}.{signature}".encode("ascii")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def session_is_valid(self, value: Optional[str]) -> bool:
        if not value:
            return False
        try:
            padded = value + "=" * (-len(value) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode("ascii")
            expires_at, signature = decoded.split(".", 1)
            expected = hmac.new(
                self._secret, expires_at.encode("ascii"), hashlib.sha256
            ).hexdigest()
            return int(expires_at) >= int(time.time()) and secrets.compare_digest(
                signature, expected
            )
        except Exception:
            return False

    def request_is_authenticated(self, request: Request) -> bool:
        if not self.enabled:
            return True
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            return self.token_matches(authorization[7:].strip())
        return self.session_is_valid(request.cookies.get(self.cookie_name))


def trace_content_mode() -> str:
    configured = os.getenv("MEMORIZZ_UI_TRACE_CONTENT_MODE", "").strip().lower()
    if configured in {"full", "redacted", "metadata"}:
        return configured
    return "redacted" if ui_read_only() else "full"


def _redact_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = _URI_CREDENTIALS.sub(
        lambda match: f"{match.group('scheme')}[redacted]@", value
    )
    text = _BEARER.sub("Bearer [redacted]", text)
    text = _KEY_TOKEN.sub("[redacted-key]", text)
    text = _QUERY_SECRET.sub(r"\1[redacted]", text)
    text = _EMAIL.sub("[redacted-email]", text)
    return text


def redact_value(value: Any, *, key: str = "") -> Any:
    """Recursively mask credentials and common personal identifiers."""
    if _SECRET_KEY.search(key):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(child_key): redact_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return _redact_scalar(value)


def redact_trace_events(
    events: Iterable[Dict[str, Any]], *, mode: Optional[str] = None
) -> list[Dict[str, Any]]:
    """Return display-safe copies; never mutate provider documents."""
    resolved_mode = mode or trace_content_mode()
    safe_events: list[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        safe = dict(event)
        if safe.get("user_id"):
            digest = hashlib.sha256(str(safe["user_id"]).encode("utf-8")).hexdigest()
            safe["user_id"] = f"user:{digest[:12]}"
        if resolved_mode == "metadata":
            safe["content"] = "[content hidden by metadata-only mode]"
        elif resolved_mode == "redacted":
            content = safe.get("content")
            if isinstance(content, str) and content.strip()[:1] in {"{", "["}:
                try:
                    content = json.dumps(
                        redact_value(json.loads(content)),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                except (TypeError, ValueError):
                    content = _redact_scalar(content)
            else:
                content = _redact_scalar(content)
            safe["content"] = content
        safe_events.append(safe)
    return safe_events


def audit_trace_view(
    request: Request,
    action: str,
    *,
    agent_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    result_count: int = 0,
    content_mode: Optional[str] = None,
) -> None:
    """Append a content-free trace-access event to a permission-restricted log."""
    configured = os.getenv("MEMORIZZ_UI_AUDIT_LOG", "").strip()
    path = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".memorizz" / "audit" / "trace_views.jsonl"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        salt = os.getenv("MEMORIZZ_UI_AUDIT_SALT", "").strip() or os.getenv(
            "MEMORIZZ_UI_AUTH_TOKEN", "local"
        )
        client_host = request.client.host if request.client else "unknown"
        client_hash = hashlib.sha256(
            f"{salt}:{client_host}".encode("utf-8")
        ).hexdigest()[:16]
        row = {
            "schema_version": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": str(action)[:80],
            "agent_id": str(agent_id)[:160] if agent_id else None,
            "thread_id": str(thread_id)[:160] if thread_id else None,
            "result_count": max(0, int(result_count)),
            "content_mode": content_mode or trace_content_mode(),
            "client_hash": client_hash,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except Exception as exc:
        logger.warning("Unable to append Memorizz UI audit event: %s", exc)


__all__ = [
    "ReadOnlyProviderProxy",
    "UIAccessController",
    "audit_trace_view",
    "redact_trace_events",
    "redact_value",
    "trace_content_mode",
    "ui_read_only",
]
