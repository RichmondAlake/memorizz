# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Durable, single-use human approval proposals.

Approval is deliberately outside the model-visible tool schema.  A model may
request an action, but only a host-controlled store can approve or reject the
exact canonical tool call.  Approved proposals are consumed atomically so a
replayed tool call cannot reuse human authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ._env_io import memorizz_home


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).isoformat() if value else None


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def canonical_arguments(arguments: Dict[str, Any]) -> str:
    """Return a stable JSON representation suitable for approval hashing."""
    if not isinstance(arguments, dict):
        raise TypeError("Approval arguments must be a JSON object")
    try:
        return json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("Approval arguments must be JSON serializable") from exc


def argument_hash(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Bind approval to the exact tool name and canonical arguments."""
    payload = f"{str(tool_name).strip()}\n{canonical_arguments(arguments)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"


@dataclass(frozen=True)
class ApprovalProposal:
    """A durable request for a human decision about one exact tool call."""

    proposal_id: str
    owner_id: str
    tool_name: str
    arguments: Dict[str, Any]
    argument_hash: str
    policy_reason: str
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    approver_id: Optional[str] = None
    decision_reason: Optional[str] = None
    decided_at: Optional[datetime] = None
    consumed_at: Optional[datetime] = None

    @property
    def expired(self) -> bool:
        return self.status == ApprovalStatus.EXPIRED or (
            self.status in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}
            and _utcnow() >= self.expires_at
        )

    def to_dict(
        self, *, include_arguments: bool = True, include_checkpoint: bool = False
    ) -> Dict[str, Any]:
        # ``proposal_id`` is also the opaque checkpoint handle used by the
        # host resume API.  Expose only the thread identifier from the stored
        # checkpoint by default; message history remains private unless a
        # trusted host explicitly asks for ``include_checkpoint``.
        thread_id = self.checkpoint.get("thread_id") if self.checkpoint else None
        value: Dict[str, Any] = {
            "proposal_id": self.proposal_id,
            "checkpoint_id": self.proposal_id,
            "thread_id": thread_id,
            "owner_id": self.owner_id,
            "tool_name": self.tool_name,
            "argument_hash": self.argument_hash,
            "policy_reason": self.policy_reason,
            "status": self.status.value,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "approver_id": self.approver_id,
            "decision_reason": self.decision_reason,
            "decided_at": _iso(self.decided_at),
            "consumed_at": _iso(self.consumed_at),
        }
        if include_arguments:
            value["arguments"] = self.arguments
        if include_checkpoint:
            value["checkpoint"] = self.checkpoint
        return {key: item for key, item in value.items() if item is not None}


@runtime_checkable
class ApprovalStore(Protocol):
    """Host-side persistence contract for durable approvals."""

    def propose(
        self,
        *,
        owner_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        policy_reason: str,
        checkpoint: Optional[Dict[str, Any]] = None,
        ttl_seconds: int = 900,
    ) -> ApprovalProposal:
        ...

    def get(self, proposal_id: str) -> Optional[ApprovalProposal]:
        ...

    def list(
        self,
        *,
        owner_id: Optional[str] = None,
        status: Optional[ApprovalStatus | str] = None,
        limit: int = 100,
    ) -> List[ApprovalProposal]:
        ...

    def approve(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        decision_reason: Optional[str] = None,
    ) -> ApprovalProposal:
        ...

    def reject(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        decision_reason: Optional[str] = None,
    ) -> ApprovalProposal:
        ...

    def consume(
        self,
        proposal_id: str,
        *,
        expected_tool_name: Optional[str] = None,
        expected_arguments: Optional[Dict[str, Any]] = None,
    ) -> ApprovalProposal:
        ...


class ApprovalError(RuntimeError):
    """Base error for invalid approval lifecycle operations."""

    code = "approval_error"


class ApprovalNotFound(ApprovalError):
    code = "approval_not_found"


class ApprovalStateError(ApprovalError):
    code = "invalid_approval_state"


class ApprovalMismatch(ApprovalError):
    code = "approval_mismatch"


class ApprovalRequired(RuntimeError):
    """Internal control-flow exception carrying a newly-created proposal."""

    def __init__(self, proposal: ApprovalProposal):
        self.proposal = proposal
        super().__init__(
            f"Human approval required for {proposal.tool_name} "
            f"(proposal {proposal.proposal_id})"
        )


class SQLiteApprovalStore:
    """SQLite-backed approval store with atomic decisions and consumption."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path or (memorizz_home() / "approvals.sqlite3")).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    argument_hash TEXT NOT NULL,
                    policy_reason TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    approver_id TEXT,
                    decision_reason TEXT,
                    decided_at TEXT,
                    consumed_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS approval_owner_status_idx
                ON approval_proposals(owner_id, status, created_at DESC)
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ApprovalProposal:
        return ApprovalProposal(
            proposal_id=row["proposal_id"],
            owner_id=row["owner_id"],
            tool_name=row["tool_name"],
            arguments=json.loads(row["arguments_json"]),
            argument_hash=row["argument_hash"],
            policy_reason=row["policy_reason"],
            checkpoint=json.loads(row["checkpoint_json"] or "{}"),
            status=ApprovalStatus(row["status"]),
            created_at=_parse_time(row["created_at"]) or _utcnow(),
            expires_at=_parse_time(row["expires_at"]) or _utcnow(),
            approver_id=row["approver_id"],
            decision_reason=row["decision_reason"],
            decided_at=_parse_time(row["decided_at"]),
            consumed_at=_parse_time(row["consumed_at"]),
        )

    @staticmethod
    def _normalize_text(value: str, label: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"{label} cannot be empty")
        return normalized

    def _expire(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            UPDATE approval_proposals
            SET status = ?
            WHERE status IN (?, ?) AND expires_at <= ?
            """,
            (
                ApprovalStatus.EXPIRED.value,
                ApprovalStatus.PENDING.value,
                ApprovalStatus.APPROVED.value,
                _iso(_utcnow()),
            ),
        )

    def propose(
        self,
        *,
        owner_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        policy_reason: str,
        checkpoint: Optional[Dict[str, Any]] = None,
        ttl_seconds: int = 900,
    ) -> ApprovalProposal:
        owner = self._normalize_text(owner_id, "Approval owner_id")
        tool = self._normalize_text(tool_name, "Approval tool_name")
        reason = self._normalize_text(policy_reason, "Approval policy_reason")
        canonical = canonical_arguments(arguments)
        checkpoint_value = dict(checkpoint or {})
        try:
            checkpoint_json = json.dumps(
                checkpoint_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("Approval checkpoint must be JSON serializable") from exc
        ttl = max(1, min(int(ttl_seconds), 86_400))
        created = _utcnow()
        expires = created + timedelta(seconds=ttl)
        proposal_id = str(uuid.uuid4())
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO approval_proposals (
                    proposal_id, owner_id, tool_name, arguments_json,
                    argument_hash, policy_reason, checkpoint_json, status,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    owner,
                    tool,
                    canonical,
                    argument_hash(tool, arguments),
                    reason,
                    checkpoint_json,
                    ApprovalStatus.PENDING.value,
                    _iso(created),
                    _iso(expires),
                ),
            )
        proposal = self.get(proposal_id)
        if proposal is None:  # pragma: no cover - defensive database invariant
            raise ApprovalError("Approval proposal was not persisted")
        return proposal

    def get(self, proposal_id: str) -> Optional[ApprovalProposal]:
        normalized = str(proposal_id or "").strip()
        if not normalized:
            return None
        with self._lock, self._connect() as connection:
            self._expire(connection)
            row = connection.execute(
                "SELECT * FROM approval_proposals WHERE proposal_id = ?",
                (normalized,),
            ).fetchone()
        return self._from_row(row) if row else None

    def list(
        self,
        *,
        owner_id: Optional[str] = None,
        status: Optional[ApprovalStatus | str] = None,
        limit: int = 100,
    ) -> List[ApprovalProposal]:
        clauses: List[str] = []
        parameters: List[Any] = []
        if owner_id is not None:
            clauses.append("owner_id = ?")
            parameters.append(str(owner_id))
        if status is not None:
            status_value = ApprovalStatus(str(getattr(status, "value", status))).value
            clauses.append("status = ?")
            parameters.append(status_value)
        sql = "SELECT * FROM approval_proposals"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(max(1, min(int(limit), 500)))
        with self._lock, self._connect() as connection:
            self._expire(connection)
            rows = connection.execute(sql, parameters).fetchall()
        return [self._from_row(row) for row in rows]

    def _decide(
        self,
        proposal_id: str,
        status: ApprovalStatus,
        *,
        approver_id: str,
        decision_reason: Optional[str],
    ) -> ApprovalProposal:
        approver = self._normalize_text(approver_id, "Approver identity")
        now = _utcnow()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire(connection)
            row = connection.execute(
                "SELECT * FROM approval_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ApprovalNotFound(f"Unknown approval proposal '{proposal_id}'")
            current = ApprovalStatus(row["status"])
            if current != ApprovalStatus.PENDING:
                connection.rollback()
                raise ApprovalStateError(
                    f"Approval proposal '{proposal_id}' is {current.value}, not pending"
                )
            connection.execute(
                """
                UPDATE approval_proposals
                SET status = ?, approver_id = ?, decision_reason = ?, decided_at = ?
                WHERE proposal_id = ? AND status = ?
                """,
                (
                    status.value,
                    approver,
                    str(decision_reason).strip() if decision_reason else None,
                    _iso(now),
                    proposal_id,
                    ApprovalStatus.PENDING.value,
                ),
            )
            connection.commit()
        proposal = self.get(proposal_id)
        if proposal is None:  # pragma: no cover
            raise ApprovalNotFound(f"Unknown approval proposal '{proposal_id}'")
        return proposal

    def approve(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        decision_reason: Optional[str] = None,
    ) -> ApprovalProposal:
        return self._decide(
            proposal_id,
            ApprovalStatus.APPROVED,
            approver_id=approver_id,
            decision_reason=decision_reason,
        )

    def reject(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        decision_reason: Optional[str] = None,
    ) -> ApprovalProposal:
        return self._decide(
            proposal_id,
            ApprovalStatus.REJECTED,
            approver_id=approver_id,
            decision_reason=decision_reason,
        )

    def consume(
        self,
        proposal_id: str,
        *,
        expected_tool_name: Optional[str] = None,
        expected_arguments: Optional[Dict[str, Any]] = None,
    ) -> ApprovalProposal:
        now = _utcnow()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire(connection)
            row = connection.execute(
                "SELECT * FROM approval_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ApprovalNotFound(f"Unknown approval proposal '{proposal_id}'")
            proposal = self._from_row(row)
            if proposal.status != ApprovalStatus.APPROVED:
                connection.rollback()
                raise ApprovalStateError(
                    f"Approval proposal '{proposal_id}' is {proposal.status.value}, not approved"
                )
            if expected_tool_name is not None and (
                proposal.tool_name != str(expected_tool_name).strip()
            ):
                connection.rollback()
                raise ApprovalMismatch(
                    "Approved tool name does not match the checkpoint"
                )
            if (
                expected_arguments is not None
                and proposal.argument_hash
                != argument_hash(proposal.tool_name, expected_arguments)
            ):
                connection.rollback()
                raise ApprovalMismatch("Approved arguments do not match the checkpoint")
            updated = connection.execute(
                """
                UPDATE approval_proposals
                SET status = ?, consumed_at = ?
                WHERE proposal_id = ? AND status = ?
                """,
                (
                    ApprovalStatus.CONSUMED.value,
                    _iso(now),
                    proposal_id,
                    ApprovalStatus.APPROVED.value,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise ApprovalStateError("Approval was already consumed")
            connection.commit()
        consumed = self.get(proposal_id)
        if consumed is None:  # pragma: no cover
            raise ApprovalNotFound(f"Unknown approval proposal '{proposal_id}'")
        return consumed

    def close(self) -> None:
        """Compatibility no-op; connections are short-lived per operation."""

    def __enter__(self) -> "SQLiteApprovalStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def default_approval_store() -> SQLiteApprovalStore:
    """Return the package-owned durable host approval store."""
    return SQLiteApprovalStore()


__all__ = [
    "ApprovalError",
    "ApprovalMismatch",
    "ApprovalNotFound",
    "ApprovalProposal",
    "ApprovalRequired",
    "ApprovalStateError",
    "ApprovalStatus",
    "ApprovalStore",
    "SQLiteApprovalStore",
    "argument_hash",
    "canonical_arguments",
    "default_approval_store",
]
