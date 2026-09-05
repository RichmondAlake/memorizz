"""Bounded, content-free public contracts for host instrumentation.

Resource IDs must be opaque application IDs or keyed fingerprints, never URLs,
emails, prompts, transcripts, or credentials. Attribute names are allowlisted;
values are bounded scalars, not arbitrary application payloads.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .privacy import validate_opaque

IDENTITY_FIELDS = (
    "application_id",
    "agent_id",
    "user_id",
    "memory_id",
    "thread_id",
    "run_id",
    "turn_id",
    "root_trace_id",
    "parent_span_id",
    "caused_by_event_id",
)
ATTRIBUTE_FIELDS = frozenset(
    {
        "fallback_reason",
        "side_effect",
        "credit_decision",
        "error_code",
        "task_id",
        "expected_artifact_types",
        "required_contracts",
        "coverage_profile",
        "source_independent",
        "persistence_verified",
        "artifact_status",
        "delivery_target",
        "required",
        "contract_name",
        "opening_marker_found",
        "closing_marker_found",
        "parser_status",
        "recovery_used",
        "recovery_status",
        "nodes",
        "edges",
        "emitted",
        "acknowledged",
        "output_limit",
        "verified",
        "context_missing",
        "queue_status",
        "job_ref",
        "candidate_rank",
        "relevance_score",
        "selected",
        "selection_reason",
        "ownership_verified",
        "content_version",
        "instrumentation_version",
        "model",
        "provider",
        "request_id",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "max_output_tokens",
        "finish_reason",
        "response_chars",
        "response_bytes",
        "ttft_ms",
        "stream_duration_ms",
        "retry_count",
        "fallback_count",
    }
)


def validate_attributes(value: Any) -> dict:
    if not isinstance(value, dict) or len(value) > 64:
        raise ValueError("attributes must be a dictionary with at most 64 fields")
    if set(value) - ATTRIBUTE_FIELDS:
        raise ValueError("unknown observability attribute")
    for item in value.values():
        values = item if isinstance(item, list) else [item]
        if len(values) > 32:
            raise ValueError("attribute lists are limited to 32 values")
        for scalar in values:
            if scalar is not None and type(scalar) not in (str, bool, int, float):
                raise ValueError("attributes must contain only JSON scalars")
            if isinstance(scalar, str) and len(scalar) > 240:
                raise ValueError("attribute strings are limited to 240 characters")
            validate_opaque(scalar)
            if isinstance(scalar, float) and not math.isfinite(scalar):
                raise ValueError("attributes must contain finite numbers")
    return value


class _BoundedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_max_length=240, frozen=True)


class ResourceRef(_BoundedModel):
    resource_type: Literal[
        "url",
        "ingestion_job",
        "analysis",
        "memory",
        "transcript_chunk",
        "doc",
        "slide_deck",
        "learning_map",
        "ui_message",
        "custom",
    ]
    ref: str = Field(min_length=1)
    version: str | None = None
    role: str | None = None
    ownership_verified: bool | None = None
    provenance_status: Literal["verified", "unverified", "missing"] = "unverified"

    _opaque = field_validator("ref", "version", "role")(validate_opaque)


class SelectionDecision(_BoundedModel):
    resource: ResourceRef
    candidate_rank: int = Field(ge=1, le=100000)
    relevance_score: float | None = Field(default=None, allow_inf_nan=False)
    selected: bool = Field(strict=True)
    selection_reason: Literal[
        "canonical_thread_source",
        "mmr_selected",
        "provider_rank_selected",
        "lower_relevance",
        "wrong_thread",
        "wrong_tenant",
        "stale_version",
        "budget_exceeded",
        "not_ready",
        "exact_duplicate",
        "near_duplicate",
        "already_in_history",
        "parent_source_duplicate",
        "selection_not_observed",
    ]


class TraceContext(_BoundedModel):
    agent_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    root_trace_id: str = Field(min_length=1)
    application_id: str | None = None
    memory_id: str | None = None
    user_id: str | None = None
    parent_span_id: str | None = None
    caused_by_event_id: str | None = None

    _opaque_identity = field_validator(*IDENTITY_FIELDS)(validate_opaque)

    def to_carrier(self) -> dict:
        return self.model_dump(include=set(IDENTITY_FIELDS), exclude_none=True)

    @classmethod
    def from_carrier(cls, carrier: dict) -> "TraceContext":
        # Legacy get_trace_context() also contains schema_version/timestamp.
        if not isinstance(carrier, dict) or set(carrier) - set(IDENTITY_FIELDS) - {
            "schema_version",
            "timestamp",
        }:
            raise ValueError("invalid trace carrier")
        return cls.model_validate(
            {key: carrier[key] for key in IDENTITY_FIELDS if key in carrier}
        )


class TraceEventV3(TraceContext):
    schema_version: Literal[3] = 3
    event_id: str = Field(min_length=1)
    event_kind: Literal[
        "intent_plan",
        "context_binding",
        "memory_retrieval",
        "memory_selection",
        "memory_supply",
        "model_call",
        "tool_call",
        "application_action",
        "queue_transition",
        "artifact_persisted",
        "artifact_validated",
        "output_contract",
        "ui_delivery",
        "verified_outcome",
        "trace_data_quality",
    ]
    operation: str = Field(min_length=1)
    phase: Literal["start", "result", "event"] = "event"
    status: Literal["started", "success", "error", "partial", "unknown"] = "unknown"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    span_id: str = Field(min_length=1)
    component: str | None = None
    deployment_id: str | None = None
    worker_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    input_refs: list[ResourceRef] = Field(default_factory=list, max_length=32)
    output_refs: list[ResourceRef] = Field(default_factory=list, max_length=32)
    selection_ledger: list[SelectionDecision] = Field(
        default_factory=list, max_length=64
    )

    _attributes = field_validator("attributes", mode="before")(validate_attributes)


class LastResponseMetadata(_BoundedModel):
    """Optional provider response envelope. Missing values remain unknown."""

    request_id: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    response_chars: int | None = Field(default=None, ge=0)
    response_bytes: int | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    stream_duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    retry_count: int | None = Field(default=None, ge=0)
    fallback_count: int | None = Field(default=None, ge=0)
