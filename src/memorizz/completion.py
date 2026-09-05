# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Host-enforced completion policies for agent harnesses.

Prompt instructions are useful guidance, but they are not an enforcement
boundary.  This module lets a host decide whether a model's proposed final
answer is acceptable while preserving the original conversation and tool
state for a bounded retry.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class CompletionCandidate:
    """A model-proposed final response and the evidence available to its host."""

    query: str
    response: str
    iteration: int
    tool_call_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def response_sha256(self) -> str:
        return hashlib.sha256(
            self.response.encode("utf-8", errors="replace")
        ).hexdigest()


@dataclass(frozen=True)
class CompletionDecision:
    """Serializable evidence explaining why a completion was accepted or rejected."""

    accepted: bool
    code: str = "accepted"
    reason: str = "Completion accepted."
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    response_sha256: Optional[str] = None
    response_chars: Optional[int] = None
    iteration: Optional[int] = None
    tool_call_count: Optional[int] = None

    def with_candidate(self, candidate: CompletionCandidate) -> "CompletionDecision":
        return CompletionDecision(
            accepted=self.accepted,
            code=self.code,
            reason=self.reason,
            metadata=dict(self.metadata),
            evaluated_at=self.evaluated_at,
            response_sha256=candidate.response_sha256,
            response_chars=len(candidate.response),
            iteration=candidate.iteration,
            tool_call_count=candidate.tool_call_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "code": self.code,
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "evaluated_at": self.evaluated_at,
            "response_sha256": self.response_sha256,
            "response_chars": self.response_chars,
            "iteration": self.iteration,
            "tool_call_count": self.tool_call_count,
        }


class CompletionRejectedError(RuntimeError):
    """Raised when a fail-closed completion policy exhausts its retry budget."""

    def __init__(self, decision: CompletionDecision, rejection_count: int):
        self.decision = decision
        self.rejection_count = int(rejection_count)
        super().__init__(
            f"Completion rejected ({decision.code}) after {rejection_count} "
            f"attempt(s): {decision.reason}"
        )


ValidatorResult = Union[bool, CompletionDecision, Tuple[Any, ...], Mapping[str, Any]]
CompletionValidator = Callable[[CompletionCandidate], ValidatorResult]


@dataclass
class CompletionPolicy:
    """Bounded, auditable final-response acceptance policy.

    ``validator`` is deliberately runtime-only: serializing arbitrary code is
    unsafe.  When ``validator_required`` is persisted, a reloaded agent rejects
    completion until the trusted host rebinds a validator.
    """

    enabled: bool = False
    max_rejections: int = 2
    fail_closed: bool = True
    require_tool_calls: bool = False
    forbidden_response_patterns: Sequence[str] = field(default_factory=tuple)
    validator: Optional[CompletionValidator] = field(default=None, repr=False)
    validator_name: Optional[str] = None
    validator_required: bool = False
    delivery_mode: str = "buffered"
    retry_instruction: str = (
        "The host rejected that proposed final response: {reason} "
        "Continue working from the current state, use the available tools, and "
        "propose a new final response only after the acceptance criteria pass."
    )

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.max_rejections = max(0, int(self.max_rejections))
        self.fail_closed = bool(self.fail_closed)
        self.require_tool_calls = bool(self.require_tool_calls)
        self.forbidden_response_patterns = tuple(
            str(pattern) for pattern in self.forbidden_response_patterns if str(pattern)
        )
        if self.validator is not None and not callable(self.validator):
            raise TypeError("completion validator must be callable or None")
        if self.validator is not None:
            self.validator_required = True
            if not self.validator_name:
                self.validator_name = getattr(self.validator, "__name__", None)
        self.validate_delivery_mode(self.delivery_mode)

    def validate_delivery_mode(self, mode):
        if mode not in {"buffered", "final_stream"}:
            raise ValueError("delivery_mode must be buffered or final_stream")
        if (
            mode == "final_stream"
            and self.enabled
            and (
                self.validator is not None
                or self.validator_required
                or self.forbidden_response_patterns
            )
        ):
            raise ValueError(
                "final_stream is incompatible with complete-answer validators or forbidden-response patterns"
            )

    @classmethod
    def from_value(
        cls, value: Optional["CompletionPolicy" | Mapping[str, Any] | bool]
    ) -> "CompletionPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, Mapping):
            allowed = {
                "enabled",
                "max_rejections",
                "fail_closed",
                "require_tool_calls",
                "forbidden_response_patterns",
                "validator",
                "validator_name",
                "validator_required",
                "delivery_mode",
                "retry_instruction",
            }
            return cls(**{key: item for key, item in value.items() if key in allowed})
        raise TypeError(
            "completion_policy must be CompletionPolicy, dict, bool, or None"
        )

    @staticmethod
    def _normalize_result(result: ValidatorResult) -> CompletionDecision:
        if isinstance(result, CompletionDecision):
            return result
        if isinstance(result, bool):
            return CompletionDecision(
                accepted=result,
                code="accepted" if result else "validator_rejected",
                reason=(
                    "Runtime validator accepted completion."
                    if result
                    else "Runtime validator rejected completion."
                ),
            )
        if isinstance(result, tuple):
            if not result:
                raise TypeError("completion validator returned an empty tuple")
            accepted = bool(result[0])
            reason = (
                str(result[1])
                if len(result) > 1
                else (
                    "Runtime validator accepted completion."
                    if accepted
                    else "Runtime validator rejected completion."
                )
            )
            metadata = (
                result[2] if len(result) > 2 and isinstance(result[2], Mapping) else {}
            )
            return CompletionDecision(
                accepted=accepted,
                code="accepted" if accepted else "validator_rejected",
                reason=reason,
                metadata=dict(metadata),
            )
        if isinstance(result, Mapping):
            accepted = bool(result.get("accepted", result.get("ok", False)))
            return CompletionDecision(
                accepted=accepted,
                code=str(
                    result.get("code")
                    or ("accepted" if accepted else "validator_rejected")
                ),
                reason=str(
                    result.get("reason")
                    or result.get("message")
                    or (
                        "Runtime validator accepted completion."
                        if accepted
                        else "Runtime validator rejected completion."
                    )
                ),
                metadata=(
                    dict(result.get("metadata") or {})
                    if isinstance(result.get("metadata"), Mapping)
                    else {}
                ),
            )
        raise TypeError(
            "completion validator must return bool, tuple, mapping, or CompletionDecision"
        )

    def evaluate(self, candidate: CompletionCandidate) -> CompletionDecision:
        """Evaluate one candidate without mutating agent or host state."""
        if not self.enabled:
            return CompletionDecision(True).with_candidate(candidate)

        if self.require_tool_calls and candidate.tool_call_count < 1:
            return CompletionDecision(
                False,
                code="tool_evidence_required",
                reason="At least one tool call is required before completion.",
            ).with_candidate(candidate)

        for pattern in self.forbidden_response_patterns:
            try:
                matched = re.search(pattern, candidate.response, flags=re.IGNORECASE)
            except re.error as exc:
                decision = CompletionDecision(
                    accepted=not self.fail_closed,
                    code="invalid_policy_pattern",
                    reason=f"Invalid completion-policy pattern {pattern!r}: {exc}",
                )
                return decision.with_candidate(candidate)
            if matched:
                return CompletionDecision(
                    False,
                    code="forbidden_response_pattern",
                    reason=f"The proposed response matched forbidden pattern {pattern!r}.",
                    metadata={"pattern": pattern},
                ).with_candidate(candidate)

        if self.validator_required and self.validator is None:
            return CompletionDecision(
                accepted=not self.fail_closed,
                code="validator_not_bound",
                reason=(
                    f"Required runtime completion validator "
                    f"{self.validator_name or '<unnamed>'!r} is not bound."
                ),
            ).with_candidate(candidate)

        if self.validator is None:
            return CompletionDecision(True).with_candidate(candidate)

        try:
            decision = self._normalize_result(self.validator(candidate))
        except Exception as exc:
            decision = CompletionDecision(
                accepted=not self.fail_closed,
                code="validator_error",
                reason=f"Completion validator failed: {type(exc).__name__}: {exc}",
            )
        return decision.with_candidate(candidate)

    def retry_message(self, decision: CompletionDecision) -> str:
        try:
            return self.retry_instruction.format(
                reason=decision.reason,
                code=decision.code,
            )
        except (KeyError, ValueError):
            return f"Host completion check rejected the response: {decision.reason}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration without serializing executable validator code."""
        return {
            "enabled": self.enabled,
            "max_rejections": self.max_rejections,
            "fail_closed": self.fail_closed,
            "require_tool_calls": self.require_tool_calls,
            "forbidden_response_patterns": list(self.forbidden_response_patterns),
            "validator_name": self.validator_name,
            "validator_required": self.validator_required,
            "delivery_mode": self.delivery_mode,
            "retry_instruction": self.retry_instruction,
        }
