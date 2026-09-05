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

from ..observability.privacy import pseudonym, validate_opaque
from .trace_access import PERMISSIONS, TracePrincipal, current_principal

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

    def get_observability_index(self):
        index = self.wrapped_provider.get_observability_index()
        return ReadOnlyTraceIndex(index) if index is not None else None

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.wrapped_provider, name)
        if callable(value) and name.lower().startswith(self._MUTATION_PREFIXES):

            def _denied(*_args: Any, **_kwargs: Any) -> Any:
                raise PermissionError(
                    f"Provider mutation '{name}' is disabled in read-only UI mode"
                )

            return _denied
        return value


class ReadOnlyTraceIndex:
    """Do not leak an index mutation API through the read-only provider."""

    def __init__(self, index):
        self._index = index

    def __getattr__(self, name):
        if name in {
            "capabilities",
            "ready",
            "query",
            "summaries",
            "preview",
            "pending_count",
        }:
            return getattr(self._index, name)
        if name == "retention":

            def retention(*, dry_run=True, **policy):
                if not dry_run:
                    raise PermissionError("Retention is disabled in read-only UI mode")
                return self._index.retention(dry_run=True, **policy)

            return retention
        raise PermissionError(
            f"Index operation '{name}' is disabled in read-only UI mode"
        )


class UIAccessController:
    """Token authentication with a short-lived signed HttpOnly session."""

    cookie_name = "memorizz_ui_session"

    def __init__(self) -> None:
        self.token = os.getenv("MEMORIZZ_UI_AUTH_TOKEN", "").strip()
        configured = os.getenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "").strip()
        try:
            self.accounts = json.loads(configured) if configured else {}
            if not isinstance(self.accounts, dict):
                raise ValueError()
            for name, account in self.accounts.items():
                if name == "__legacy__" or len(name) > 64:
                    raise ValueError()
                validate_opaque(name)
                if not isinstance(account, dict) or set(account) - {
                    "token",
                    "role",
                    "application_id",
                    "user_id",
                }:
                    raise ValueError()
                if (
                    account.get("role") not in PERMISSIONS
                    or not isinstance(account.get("token"), str)
                    or len(account["token"]) < 16
                ):
                    raise ValueError()
                for key in ("application_id", "user_id"):
                    if account.get(key) is not None and (
                        not isinstance(account[key], str)
                        or not account[key]
                        or len(account[key]) > 240
                    ):
                        raise ValueError()
                    validate_opaque(account.get(key))
            tokens = [entry["token"] for entry in self.accounts.values()]
            if len(tokens) != len(set(tokens)) or (self.token and self.token in tokens):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError(
                "Invalid MEMORIZZ_UI_AUTH_ACCOUNTS configuration"
            ) from None
        explicit_secret = os.getenv("MEMORIZZ_UI_SESSION_SECRET", "").strip()
        seed = explicit_secret or self.token or configured or secrets.token_urlsafe(32)
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
        return bool(self.token or self.accounts)

    def _principal(self, name):
        if name == "__legacy__":
            return TracePrincipal(principal_id="local", role="admin")
        account = self.accounts.get(name)
        if account is None:
            return None
        return TracePrincipal(
            principal_id=name,
            role=account["role"],
            application_id=account.get("application_id"),
            user_id=account.get("user_id"),
            user_bound="user_id" in account,
        )

    def _token_principal(self, candidate):
        if self.token and secrets.compare_digest(
            candidate.encode(), self.token.encode()
        ):
            return "__legacy__"
        for name, account in self.accounts.items():
            if secrets.compare_digest(candidate.encode(), account["token"].encode()):
                return name
        return None

    def token_matches(self, candidate: str) -> bool:
        return self._token_principal(candidate) is not None

    def issue_session(self, *, token=None) -> str:
        principal = self._token_principal(token) if token is not None else "__legacy__"
        if principal is None or (
            principal == "__legacy__" and self.accounts and not self.token
        ):
            raise ValueError("Valid account token required")
        payload = json.dumps(
            {"principal": principal, "expires": int(time.time()) + self.ttl_seconds},
            sort_keys=True,
        )
        signature = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        return (
            base64.urlsafe_b64encode((payload + "." + signature).encode())
            .decode()
            .rstrip("=")
        )

    def _session_principal(self, value):
        try:
            if not value or len(value) > 8192:
                return None
            raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
            payload, signature = raw.rsplit(".", 1)
            expected = hmac.new(
                self._secret, payload.encode(), hashlib.sha256
            ).hexdigest()
            if not secrets.compare_digest(signature, expected):
                return None
            data = json.loads(payload)
            if not isinstance(data, dict) or data.get("expires", 0) < int(time.time()):
                return None
            name = data.get("principal")
            if name == "__legacy__" and self.accounts and not self.token:
                return None
            return self._principal(name)
        except Exception:
            return None

    def session_is_valid(self, value: Optional[str]) -> bool:
        return self._session_principal(value) is not None

    def principal_for_request(self, request):
        if not self.enabled:
            return TracePrincipal()
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            name = self._token_principal(authorization[7:].strip())
            return self._principal(name) if name is not None else None
        return self._session_principal(request.cookies.get(self.cookie_name))

    def request_is_authenticated(self, request: Request) -> bool:
        return self.principal_for_request(request) is not None


def trace_content_mode() -> str:
    if not current_principal.get().allows("trace.reveal"):
        return "metadata"
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
    if key in {"user_id", "userId"} and value:
        return pseudonym(value)
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
        if resolved_mode == "full" and safe.get("user_id"):
            safe["user_id"] = pseudonym(safe["user_id"])
        if resolved_mode == "metadata":
            from ..observability.models import ATTRIBUTE_FIELDS, TraceEventV3
            from ..observability.normalization import LEGACY_FIELDS

            allowed = (
                LEGACY_FIELDS
                | set(TraceEventV3.model_fields)
                | ATTRIBUTE_FIELDS
                | {
                    "_id",
                    "record_type",
                    "trace_memory_id",
                    "event_count",
                    "event_kind_counts",
                    "tool_names",
                    "models",
                    "has_error",
                    "started_at",
                    "ended_at",
                    "schema_versions",
                }
            )
            safe = {key: value for key, value in safe.items() if key in allowed}
            # Legacy titles can contain prompts. Build display labels from the
            # event taxonomy instead of shipping arbitrary historical titles.
            safe["title"] = (
                str(safe.get("kind") or safe.get("trace_kind") or "trace")
                .replace("_", " ")
                .title()
            )
            if safe.get("logical_tool_name") or safe.get("tool_name"):
                safe["title"] += (
                    " · "
                    + str(safe.get("logical_tool_name") or safe["tool_name"])[:240]
                )
            if isinstance(safe.get("attributes"), dict):
                safe["attributes"] = {
                    key: value
                    for key, value in safe["attributes"].items()
                    if key in ATTRIBUTE_FIELDS
                }
            safe = redact_value(safe)
            safe["content"] = "[content hidden by metadata-only mode]"
            for key in ("arguments", "result", "embedding", "text", "message"):
                safe.pop(key, None)
        elif resolved_mode == "redacted":
            safe = redact_value(safe)
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
) -> bool:
    """Append a content-free trace-access event to a permission-restricted log."""
    configured = os.getenv("MEMORIZZ_UI_AUDIT_LOG", "").strip()
    path = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".memorizz" / "audit" / "trace_views.jsonl"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        client_host = request.client.host if request.client else "unknown"
        client_hash = pseudonym(client_host, scope="client")
        row = {
            "schema_version": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": str(action)[:80],
            "agent_id": str(agent_id)[:160] if agent_id else None,
            "thread_id": str(thread_id)[:160] if thread_id else None,
            "result_count": max(0, int(result_count)),
            "content_mode": content_mode or trace_content_mode(),
            "client_hash": client_hash,
            "principal": pseudonym(
                current_principal.get().principal_id, scope="operator"
            ),
            "role": current_principal.get().role,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return True
    except Exception as exc:
        logger.warning(
            "Unable to append Memorizz UI audit event (%s)", type(exc).__name__
        )
        return False


__all__ = [
    "ReadOnlyProviderProxy",
    "UIAccessController",
    "audit_trace_view",
    "redact_trace_events",
    "redact_value",
    "trace_content_mode",
    "ui_read_only",
]
