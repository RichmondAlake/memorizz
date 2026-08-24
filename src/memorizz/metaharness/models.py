"""Provider-neutral models for running agent harnesses under MemoRizz control."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HarnessStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    INTERRUPTED = "interrupted"
    BUDGET_EXCEEDED = "budget_exceeded"
    VERIFICATION_FAILED = "verification_failed"

    @property
    def terminal(self) -> bool:
        return self in {
            self.SUCCEEDED,
            self.FAILED,
            self.CANCELED,
            self.INTERRUPTED,
            self.BUDGET_EXCEEDED,
            self.VERIFICATION_FAILED,
        }


class HarnessEventType(str, Enum):
    STATUS = "status"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMMAND = "command"
    FILE_CHANGE = "file_change"
    USAGE = "usage"
    APPROVAL = "approval"
    VERIFICATION = "verification"
    ERROR = "error"
    COMPLETE = "complete"
    LOG = "log"


_ACTIONABLE_HARNESS_EVENT_TYPES = frozenset(
    {
        HarnessEventType.TOOL_CALL.value,
        HarnessEventType.COMMAND.value,
        HarnessEventType.FILE_CHANGE.value,
    }
)


def harness_action_identity(event: Any) -> Optional[str]:
    """Return a stable identity for one actionable harness lifecycle.

    Vendor streams may emit both ``in_progress`` and ``completed`` records for
    the same command or tool call. Those records are one action for budgeting,
    not two. Events without a vendor action ID deliberately return ``None`` so
    callers count each record conservatively.
    """

    if isinstance(event, Mapping):
        event_type = event.get("type") or event.get("event_type")
        data = event.get("data") or {}
    else:
        event_type = getattr(event, "type", None)
        data = getattr(event, "data", {}) or {}
    if isinstance(event_type, HarnessEventType):
        event_type = event_type.value
    event_type = str(event_type or "")
    if event_type not in _ACTIONABLE_HARNESS_EVENT_TYPES:
        return None
    if not isinstance(data, Mapping):
        return None
    for key in ("id", "tool_use_id", "call_id", "action_id", "command_id"):
        value = data.get(key)
        if value is not None and str(value).strip():
            return f"{event_type}:{value}"
    return None


def count_harness_steps(events: Iterable[Any]) -> int:
    """Count unique actionable lifecycles in normalized harness events."""

    identities = set()
    anonymous = 0
    for event in events:
        if isinstance(event, Mapping):
            event_type = event.get("type") or event.get("event_type")
        else:
            event_type = getattr(event, "type", None)
        if isinstance(event_type, HarnessEventType):
            event_type = event_type.value
        if str(event_type or "") not in _ACTIONABLE_HARNESS_EVENT_TYPES:
            continue
        identity = harness_action_identity(event)
        if identity is None:
            anonymous += 1
        else:
            identities.add(identity)
    return len(identities) + anonymous


@dataclass
class HarnessBudget:
    max_wall_time_seconds: float = 900.0
    max_steps: int = 80
    max_cost_usd: Optional[float] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_event_chars: int = 50_000
    max_retries: int = 1

    def __post_init__(self) -> None:
        self.max_wall_time_seconds = max(1.0, float(self.max_wall_time_seconds))
        self.max_steps = max(1, int(self.max_steps))
        self.max_event_chars = max(1_024, min(int(self.max_event_chars), 1_000_000))
        self.max_retries = max(0, min(int(self.max_retries), 10))
        if self.max_cost_usd is not None:
            self.max_cost_usd = max(0.0, float(self.max_cost_usd))
        if self.max_input_tokens is not None:
            self.max_input_tokens = max(1, int(self.max_input_tokens))
        if self.max_output_tokens is not None:
            self.max_output_tokens = max(1, int(self.max_output_tokens))

    @classmethod
    def from_value(cls, value: Any) -> "HarnessBudget":
        if isinstance(value, cls):
            return value
        return cls(**dict(value or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessPermissions:
    workspace_mode: str = "read_only"
    allowed_roots: List[str] = field(default_factory=list)
    allow_dirty_workspace: bool = False
    network: str = "none"
    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    allowed_env: List[str] = field(default_factory=list)
    mcp_access: str = "read_only"
    require_approval: Optional[bool] = None

    def __post_init__(self) -> None:
        self.workspace_mode = str(self.workspace_mode or "read_only").lower()
        if self.workspace_mode not in {"read_only", "direct"}:
            raise ValueError("workspace_mode must be read_only or direct")
        self.network = str(self.network or "none").lower()
        if self.network not in {"none", "restricted", "full"}:
            raise ValueError("network must be none, restricted, or full")
        self.mcp_access = str(self.mcp_access or "read_only").lower()
        if self.mcp_access not in {"none", "read_only", "governed_write"}:
            raise ValueError("mcp_access must be none, read_only, or governed_write")
        if self.require_approval is None:
            self.require_approval = bool(
                self.workspace_mode == "direct"
                or self.network != "none"
                or self.allowed_tools
                or self.allowed_env
                or self.mcp_access == "governed_write"
            )

    @classmethod
    def from_value(cls, value: Any) -> "HarnessPermissions":
        if isinstance(value, cls):
            return value
        return cls(**dict(value or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationSpec:
    command: Optional[str] = None
    cwd: Optional[str] = None
    timeout_seconds: int = 300
    required: bool = False

    def __post_init__(self) -> None:
        self.timeout_seconds = max(1, min(int(self.timeout_seconds), 3_600))
        if self.command:
            self.required = True

    @classmethod
    def from_value(cls, value: Any) -> "VerificationSpec":
        if isinstance(value, cls):
            return value
        return cls(**dict(value or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessContextPack:
    query: str
    rendered: str = ""
    records: List[Dict[str, Any]] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    token_estimate: int = 0
    truncated: bool = False
    fingerprint: str = ""
    content_fingerprint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = canonical_hash(
                {"rendered": self.rendered, "source_ids": self.source_ids}
            )
        if not self.content_fingerprint:
            projection: List[Dict[str, Any]] = []
            for record in self.records:
                if not isinstance(record, Mapping):
                    continue
                content_hash = record.get("content_hash")
                if content_hash:
                    projection.append(
                        {
                            "source_type": record.get("source_type")
                            or record.get("memory_type"),
                            "content_hash": str(content_hash),
                        }
                    )
                    continue
                projection.append(
                    {
                        "source_type": record.get("source_type")
                        or record.get("memory_type"),
                        "title": record.get("title"),
                        "content": record.get("content")
                        or record.get("summary")
                        or record.get("text"),
                    }
                )
            if projection:
                self.content_fingerprint = canonical_hash(projection)
            else:
                normalized = re.sub(
                    r"(?i)(?:memory:)?(?:[a-z_]+:)?[0-9a-f]{8}-[0-9a-f-]{27,}",
                    "[SOURCE]",
                    self.rendered,
                )
                self.content_fingerprint = canonical_hash(normalized)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HarnessTask:
    task: str
    workspace: str
    harness: str = "auto"
    memory_id: Optional[str] = None
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    agent_id: Optional[str] = None
    model: Optional[str] = None
    mode: str = "runtime"
    permissions: HarnessPermissions = field(default_factory=HarnessPermissions)
    budget: HarnessBudget = field(default_factory=HarnessBudget)
    verification: VerificationSpec = field(default_factory=VerificationSpec)
    output_schema: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        self.task = str(self.task or "").strip()
        if not self.task:
            raise ValueError("HarnessTask.task is required")
        self.workspace = str(self.workspace or "").strip()
        if not self.workspace:
            raise ValueError("HarnessTask.workspace is required")
        self.harness = str(self.harness or "auto").strip().lower().replace("_", "-")
        self.mode = str(self.mode or "runtime").strip().lower()
        if self.mode not in {"runtime", "delegate", "stage"}:
            raise ValueError("HarnessTask.mode must be runtime, delegate, or stage")
        self.permissions = HarnessPermissions.from_value(self.permissions)
        self.budget = HarnessBudget.from_value(self.budget)
        self.verification = VerificationSpec.from_value(self.verification)
        if self.output_schema is not None:
            if not isinstance(self.output_schema, Mapping):
                raise TypeError("HarnessTask.output_schema must be a JSON object")
            encoded_schema = json.dumps(
                dict(self.output_schema),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            if len(encoded_schema) > 100_000:
                raise ValueError("HarnessTask.output_schema exceeds 100,000 characters")
            self.output_schema = json.loads(encoded_schema)
        self.context = dict(self.context or {})
        self.metadata = dict(self.metadata or {})

    @property
    def writes_workspace(self) -> bool:
        return self.permissions.workspace_mode == "direct"

    def approval_arguments(self, *, workspace_fingerprint: str) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "harness": self.harness,
            "model": self.model,
            "mode": self.mode,
            "workspace": str(Path(self.workspace).resolve()),
            "workspace_fingerprint": workspace_fingerprint,
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "agent_id": self.agent_id,
            "permissions": self.permissions.to_dict(),
            "budget": self.budget.to_dict(),
            "verification": self.verification.to_dict(),
            "output_schema": self.output_schema,
            # Bind non-public context and adapter metadata without exposing
            # their contents in an operator-facing approval proposal.
            "context_fingerprint": canonical_hash(self.context),
            "metadata_fingerprint": canonical_hash(self.metadata),
        }

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["permissions"] = self.permissions.to_dict()
        value["budget"] = self.budget.to_dict()
        value["verification"] = self.verification.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HarnessTask":
        payload = dict(value)
        payload["permissions"] = HarnessPermissions.from_value(
            payload.get("permissions")
        )
        payload["budget"] = HarnessBudget.from_value(payload.get("budget"))
        payload["verification"] = VerificationSpec.from_value(
            payload.get("verification")
        )
        return cls(**payload)


@dataclass
class HarnessCapabilities:
    name: str
    available: bool
    version: Optional[str] = None
    command: Optional[str] = None
    structured_events: bool = True
    resume: bool = False
    mcp: bool = False
    per_action_approvals: bool = False
    requires_external_isolation: bool = False
    usage_reporting: bool = False
    models: List[str] = field(default_factory=list)
    error_code: Optional[str] = None
    error: Optional[str] = None
    remediation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["ready"] = bool(self.available and not self.error)
        return value


@dataclass
class HarnessEvent:
    run_id: str
    type: HarnessEventType | str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utcnow_iso)
    sequence: Optional[int] = None

    def __post_init__(self) -> None:
        self.type = (
            self.type
            if isinstance(self.type, HarnessEventType)
            else HarnessEventType(self.type)
        )
        self.data = dict(self.data or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }


@dataclass
class HarnessResult:
    run_id: str
    harness: str
    status: HarnessStatus | str
    final_response: str = ""
    verified: bool = False
    verification: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    phase_timings_ms: Dict[str, int] = field(default_factory=dict)
    workspace_diff: Optional[str] = None
    workspace_fingerprint_before: Optional[str] = None
    workspace_fingerprint_after: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    context_pack: Optional[HarnessContextPack] = None
    routing: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    error: Optional[str] = None
    remediation: Optional[str] = None

    def __post_init__(self) -> None:
        self.status = (
            self.status
            if isinstance(self.status, HarnessStatus)
            else HarnessStatus(self.status)
        )

    @property
    def ok(self) -> bool:
        return self.status == HarnessStatus.SUCCEEDED and (
            self.verified or not self.verification.get("required", False)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "run_id": self.run_id,
            "harness": self.harness,
            "status": self.status.value,
            "final_response": self.final_response,
            "verified": self.verified,
            "verification": self.verification,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "phase_timings_ms": self.phase_timings_ms,
            "workspace_diff": self.workspace_diff,
            "workspace_fingerprint_before": self.workspace_fingerprint_before,
            "workspace_fingerprint_after": self.workspace_fingerprint_after,
            "artifacts": self.artifacts,
            "checkpoint": self.checkpoint,
            "context_pack": self.context_pack.to_dict() if self.context_pack else None,
            "routing": self.routing,
            "error_code": self.error_code,
            "error": self.error,
            "remediation": self.remediation,
        }


@dataclass
class HarnessRun:
    run_id: str
    task: Dict[str, Any]
    status: HarnessStatus | str
    harness: Optional[str] = None
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    heartbeat_at: Optional[str] = None
    cancel_requested: bool = False
    approval_proposal_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    routing: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = (
            self.status
            if isinstance(self.status, HarnessStatus)
            else HarnessStatus(self.status)
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass
class HarnessStage:
    name: str
    harness: str = "auto"
    instruction: Optional[str] = None
    workspace_mode: str = "read_only"
    verification: Optional[VerificationSpec] = None


@dataclass
class HarnessPlan:
    stages: List[HarnessStage]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("HarnessPlan requires at least one stage")
        writers = [stage for stage in self.stages if stage.workspace_mode == "direct"]
        if len(writers) > 1:
            raise ValueError(
                "A HarnessPlan may contain at most one write-capable stage"
            )


__all__ = [
    "HarnessBudget",
    "HarnessCapabilities",
    "HarnessContextPack",
    "HarnessEvent",
    "HarnessEventType",
    "HarnessPermissions",
    "HarnessPlan",
    "HarnessResult",
    "HarnessRun",
    "HarnessStage",
    "HarnessStatus",
    "HarnessTask",
    "VerificationSpec",
    "canonical_hash",
    "utcnow_iso",
]
