"""Provider-neutral tool outcomes without changing model-visible payloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class ToolOutcomeStatus(str, Enum):
    """Terminal state for one tool execution."""

    SUCCESS = "success"
    EMPTY = "empty"
    DEGRADED = "degraded"
    FALLBACK = "fallback"
    PROVIDER_ERROR = "provider_error"
    ERROR = "error"


_USABLE_OUTCOMES = frozenset(
    {
        ToolOutcomeStatus.SUCCESS,
        ToolOutcomeStatus.EMPTY,
        ToolOutcomeStatus.DEGRADED,
        ToolOutcomeStatus.FALLBACK,
    }
)
_PROVIDER_ERROR_CODES = frozenset(
    {
        "authentication_required",
        "authorization_required",
        "connection_error",
        "missing_api_key",
        "oauth_required",
        "provider_authentication_error",
        "provider_error",
        "provider_timeout",
        "provider_unavailable",
        "rate_limit_exceeded",
        "transport_error",
    }
)


@dataclass(frozen=True)
class ToolOutcome:
    """Content-free evidence about a tool result.

    ``status`` drives the primary UI label. ``fallback_used`` and ``degraded``
    remain orthogonal because a fallback can itself offer reduced capability.
    """

    status: ToolOutcomeStatus = ToolOutcomeStatus.SUCCESS
    reason_code: Optional[str] = None
    provider: Optional[str] = None
    primary_provider: Optional[str] = None
    fallback_provider: Optional[str] = None
    retryable: Optional[bool] = None
    result_count: Optional[int] = None
    fallback_used: bool = False
    degraded: bool = False

    def __post_init__(self) -> None:
        status = self.status
        if not isinstance(status, ToolOutcomeStatus):
            status = ToolOutcomeStatus(str(status).strip().lower())
            object.__setattr__(self, "status", status)
        if status is ToolOutcomeStatus.SUCCESS and self.fallback_used:
            status = ToolOutcomeStatus.FALLBACK
            object.__setattr__(self, "status", status)
        elif status is ToolOutcomeStatus.SUCCESS and self.degraded:
            status = ToolOutcomeStatus.DEGRADED
            object.__setattr__(self, "status", status)
        if self.result_count is not None:
            object.__setattr__(self, "result_count", max(0, int(self.result_count)))
        if status is ToolOutcomeStatus.FALLBACK:
            object.__setattr__(self, "fallback_used", True)
        if status is ToolOutcomeStatus.DEGRADED:
            object.__setattr__(self, "degraded", True)

    @property
    def ok(self) -> bool:
        """Whether execution produced a usable result."""
        return self.status in _USABLE_OUTCOMES

    @classmethod
    def from_value(cls, value: Any) -> "ToolOutcome":
        """Build an outcome from a status string or outcome mapping."""
        if isinstance(value, cls):
            return value
        if isinstance(value, ToolOutcomeStatus):
            return cls(status=value)
        if isinstance(value, str):
            return cls(status=ToolOutcomeStatus(value.strip().lower()))
        if not isinstance(value, Mapping):
            raise TypeError(
                "tool outcome must be ToolOutcome, status string, or mapping"
            )
        data = dict(value)
        data.pop("ok", None)
        if "fallback_from" in data and "primary_provider" not in data:
            data["primary_provider"] = data.pop("fallback_from")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a stable JSON-safe representation."""
        value: Dict[str, Any] = {
            "status": self.status.value,
            "ok": self.ok,
            "fallback_used": self.fallback_used,
            "degraded": self.degraded,
        }
        for key in (
            "reason_code",
            "provider",
            "primary_provider",
            "fallback_provider",
            "retryable",
            "result_count",
        ):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        return value


@dataclass(frozen=True)
class ToolResult:
    """A tool payload with an explicit outcome.

    MemAgent unwraps ``value`` before model serialization, preserving the
    tool's established result schema.
    """

    value: Any
    outcome: ToolOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ToolOutcome):
            object.__setattr__(self, "outcome", ToolOutcome.from_value(self.outcome))


def _as_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        mapped = to_dict()
    except Exception:
        return None
    return dict(mapped) if isinstance(mapped, Mapping) else None


def _result_count(value: Any, mapped: Optional[Mapping[str, Any]]) -> Optional[int]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    if not mapped:
        return None
    for key in ("results", "items", "matches", "records"):
        items = mapped.get(key)
        if isinstance(items, (list, tuple)):
            if not items and any(
                mapped.get(payload_key)
                for payload_key in ("output", "content", "stdout", "data", "value")
            ):
                return None
            return len(items)
    return None


def _is_provider_error(error_code: Any) -> bool:
    normalized = str(error_code or "").strip().lower()
    return normalized in _PROVIDER_ERROR_CODES or normalized.startswith("provider_")


def normalize_tool_result(value: Any) -> tuple[Any, ToolOutcome]:
    """Return ``(model_payload, outcome)`` for an explicit or legacy result.

    Legacy inference is deliberately conservative: known failure flags,
    existing retrieval diagnostics, and empty result collections are handled;
    arbitrary prose is not interpreted.
    """
    if isinstance(value, ToolResult):
        return value.value, value.outcome

    mapped = _as_mapping(value)
    if mapped and "_memorizz_outcome" in mapped:
        explicit = mapped.pop("_memorizz_outcome")
        return mapped, ToolOutcome.from_value(explicit)

    count = _result_count(value, mapped)
    if isinstance(value, str) and value.lstrip().lower().startswith("error"):
        return value, ToolOutcome(status=ToolOutcomeStatus.ERROR)
    if mapped is None:
        status = ToolOutcomeStatus.EMPTY if count == 0 else ToolOutcomeStatus.SUCCESS
        return value, ToolOutcome(status=status, result_count=count)

    error_code = mapped.get("error_code")
    raw_status = str(mapped.get("status") or "").strip().lower()
    failed = (
        mapped.get("ok") is False
        or mapped.get("success") is False
        or (bool(mapped.get("error")) and mapped.get("ok") is not True)
        or raw_status in {"failed", "failure", "error", "provider_error"}
    )
    if failed:
        status = (
            ToolOutcomeStatus.PROVIDER_ERROR
            if raw_status == "provider_error" or _is_provider_error(error_code)
            else ToolOutcomeStatus.ERROR
        )
        provider = mapped.get("provider") or mapped.get("server_name")
        return value, ToolOutcome(
            status=status,
            reason_code=str(error_code) if error_code else None,
            provider=str(provider) if provider else None,
            retryable=(
                mapped["retryable"]
                if isinstance(mapped.get("retryable"), bool)
                else None
            ),
            result_count=count,
        )

    signals = [mapped]
    if isinstance(mapped.get("retrieval"), Mapping):
        signals.append(mapped["retrieval"])
    fallback_used = any(bool(item.get("fallback_used")) for item in signals)
    degraded = any(bool(item.get("degraded")) for item in signals)
    reason = next(
        (
            item.get("degraded_reason") or item.get("fallback_reason")
            for item in signals
            if item.get("degraded_reason") or item.get("fallback_reason")
        ),
        None,
    )
    if fallback_used:
        return value, ToolOutcome(
            status=ToolOutcomeStatus.FALLBACK,
            reason_code=str(reason) if reason else None,
            provider=str(mapped["provider"]) if mapped.get("provider") else None,
            primary_provider=(
                str(mapped["primary_provider"])
                if mapped.get("primary_provider")
                else None
            ),
            fallback_provider=(
                str(mapped["fallback_provider"])
                if mapped.get("fallback_provider")
                else None
            ),
            result_count=count,
            degraded=degraded,
        )
    if degraded:
        return value, ToolOutcome(
            status=ToolOutcomeStatus.DEGRADED,
            reason_code=str(reason) if reason else None,
            provider=str(mapped["provider"]) if mapped.get("provider") else None,
            result_count=count,
        )
    status = ToolOutcomeStatus.EMPTY if count == 0 else ToolOutcomeStatus.SUCCESS
    return value, ToolOutcome(status=status, result_count=count)


__all__ = [
    "ToolOutcome",
    "ToolOutcomeStatus",
    "ToolResult",
    "normalize_tool_result",
]
