"""Provider-neutral durable storage for learning control-plane records."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Set

from ..enums.memory_type import MemoryType
from .models import LearningEvent, canonical_json, content_digest, utc_now

LEARNING_EVENT_RECORD = "learning_event"
LEARNING_ARTIFACT_RECORD = "learning_artifact"
LEARNING_CHECKPOINT_RECORD = "learning_checkpoint"
LEARNING_TOMBSTONE_RECORD = "learning_tombstone"


def _read_json(value: Any) -> Optional[Dict[str, Any]]:
    if hasattr(value, "read"):
        try:
            value = value.read()
        except Exception:
            return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class LearningRecordConflictError(RuntimeError):
    """Raised when an immutable record ID is reused with different content."""


class LearningControlPlaneStore:
    """Store immutable events and rebuildable projections in private shared memory.

    ``MemoryType.SHARED_MEMORY`` is intentionally used as the physical substrate:
    it already has portable filesystem, MongoDB, and Oracle support and keeps
    control-plane records out of conversational recall.  ``record_type`` remains
    the logical partition and is duplicated in provider-indexable metadata when
    a backend supports it.
    """

    def __init__(self, provider: Any, *, agent_id: str):
        if provider is None:
            raise ValueError("LearningControlPlaneStore requires a memory provider")
        if not str(agent_id or "").strip():
            raise ValueError("agent_id is required")
        self.provider = provider
        self.agent_id = str(agent_id)

    @staticmethod
    def _payload(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        payload = _read_json(row.get("content"))
        if payload is None:
            return None
        payload.setdefault(
            "record_id", row.get("memory_id") or row.get("_id") or row.get("id")
        )
        return payload

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        value = str(record_id or "").strip()
        if not value:
            return None
        for method_name in ("retrieve_by_id", "retrieve_by_name"):
            method = getattr(self.provider, method_name, None)
            if not callable(method):
                continue
            try:
                row = method(value, MemoryType.SHARED_MEMORY)
            except (NotImplementedError, TypeError, ValueError):
                row = None
            if isinstance(row, Mapping):
                payload = self._payload(row)
                if payload and str(payload.get("record_id") or "") == value:
                    return payload
        for payload in self.list_records(limit=10_000):
            if str(payload.get("record_id") or "") == value:
                return payload
        return None

    def _put(self, payload: Mapping[str, Any], *, immutable: bool) -> Dict[str, Any]:
        value = json.loads(canonical_json(dict(payload)))
        record_id = str(value.get("record_id") or "").strip()
        if not record_id:
            raise ValueError("learning record_id is required")
        value["record_id"] = record_id
        value.setdefault("schema_version", 1)
        value.setdefault("agent_id", self.agent_id)
        value.setdefault("created_at", utc_now())
        value["updated_at"] = value.get("created_at") if immutable else utc_now()
        value_hash = content_digest(
            {key: item for key, item in value.items() if key not in {"updated_at"}}
        )
        value["record_hash"] = value_hash

        existing = self.get(record_id)
        if existing:
            existing_hash = existing.get("record_hash")
            if immutable and existing_hash and str(existing_hash) != value_hash:
                raise LearningRecordConflictError(
                    f"Immutable learning record '{record_id}' already exists with "
                    "different content"
                )
            if immutable:
                return existing
            value.setdefault("created_at", existing.get("created_at") or utc_now())

        trace_memory_id = value.get("trace_memory_id") or value.get("memory_id")
        document = {
            "_id": record_id,
            "memory_id": record_id,
            "content": canonical_json(value),
            "memory_type": MemoryType.SHARED_MEMORY.value,
            "scope": "private",
            "owner_agent_id": self.agent_id,
            "record_type": value.get("record_type"),
            "application_id": value.get("application_id"),
            "agent_id": value.get("agent_id"),
            "run_id": value.get("run_id"),
            "turn_id": value.get("turn_id"),
            "root_trace_id": value.get("trace_id") or value.get("root_trace_id"),
            "trace_memory_id": trace_memory_id,
            "thread_id": value.get("thread_id"),
            "user_id": value.get("user_id"),
            "status": value.get("status"),
            "timestamp": value.get("timestamp") or value.get("updated_at"),
        }
        self.provider.store(
            {key: item for key, item in document.items() if item is not None},
            MemoryType.SHARED_MEMORY,
        )
        return value

    def append_event(self, event: LearningEvent) -> LearningEvent:
        if event.agent_id != self.agent_id:
            raise ValueError("event agent_id does not match the control-plane store")
        payload = {
            "record_id": event.event_id,
            "record_type": LEARNING_EVENT_RECORD,
            **event.to_dict(),
            "created_at": event.timestamp,
        }
        stored = self._put(payload, immutable=True)
        return LearningEvent.from_dict(stored)

    def put_artifact(self, artifact: Mapping[str, Any]) -> Dict[str, Any]:
        value = dict(artifact)
        value["record_type"] = LEARNING_ARTIFACT_RECORD
        value.setdefault("artifact_id", value.get("record_id"))
        value.setdefault("record_id", value.get("artifact_id"))
        return self._put(value, immutable=False)

    def put_checkpoint(self, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
        value = dict(checkpoint)
        value["record_type"] = LEARNING_CHECKPOINT_RECORD
        value.setdefault("checkpoint_id", value.get("record_id"))
        value.setdefault("record_id", value.get("checkpoint_id"))
        return self._put(value, immutable=False)

    def put_tombstone(self, tombstone: Mapping[str, Any]) -> Dict[str, Any]:
        value = dict(tombstone)
        value["record_type"] = LEARNING_TOMBSTONE_RECORD
        target_id = str(value.get("target_id") or "").strip()
        if not target_id:
            raise ValueError("tombstone target_id is required")
        value.setdefault("record_id", f"learn-tomb-{content_digest(target_id)[:32]}")
        value.setdefault("tombstoned_at", utc_now())
        return self._put(value, immutable=False)

    def list_records(
        self,
        *,
        record_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Any = ...,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 50_000))
        try:
            rows = self.provider.list_all(MemoryType.SHARED_MEMORY) or []
        except TypeError:
            rows = (
                self.provider.list_all(memory_store_type=MemoryType.SHARED_MEMORY) or []
            )
        output: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            payload = self._payload(row)
            if not payload:
                continue
            logical_type = str(payload.get("record_type") or "")
            if not logical_type.startswith("learning_"):
                continue
            if record_type and logical_type != str(record_type):
                continue
            if str(payload.get("agent_id") or "") != self.agent_id:
                continue
            if memory_id is not None and payload.get("memory_id") != memory_id:
                continue
            if user_id is not ... and payload.get("user_id") != user_id:
                continue
            if thread_id is not None and payload.get("thread_id") != thread_id:
                continue
            if run_id is not None and payload.get("run_id") != run_id:
                continue
            if stream_id is not None and payload.get("stream_id") != stream_id:
                continue
            output.append(payload)
        output.sort(
            key=lambda item: str(
                item.get("timestamp")
                or item.get("updated_at")
                or item.get("created_at")
                or ""
            )
        )
        return output[-safe_limit:]

    def list_events(self, **filters: Any) -> List[LearningEvent]:
        records = self.list_records(record_type=LEARNING_EVENT_RECORD, **filters)
        events: List[LearningEvent] = []
        for record in records:
            try:
                events.append(LearningEvent.from_dict(record))
            except (KeyError, TypeError, ValueError):
                continue
        return events

    def list_artifacts(self, **filters: Any) -> List[Dict[str, Any]]:
        return self.list_records(record_type=LEARNING_ARTIFACT_RECORD, **filters)

    def list_checkpoints(self, **filters: Any) -> List[Dict[str, Any]]:
        return self.list_records(record_type=LEARNING_CHECKPOINT_RECORD, **filters)

    def tombstoned_ids(self, **filters: Any) -> Set[str]:
        return {
            str(item.get("target_id"))
            for item in self.list_records(
                record_type=LEARNING_TOMBSTONE_RECORD, **filters
            )
            if item.get("target_id")
        }

    def latest_checkpoint(self, *, stream_id: str) -> Optional[Dict[str, Any]]:
        checkpoints = self.list_checkpoints(stream_id=stream_id, limit=100)
        return checkpoints[-1] if checkpoints else None

    def statistics(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Any = ...,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rows = self.list_records(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            limit=50_000,
        )
        by_record_type = Counter(
            str(item.get("record_type") or "unknown") for item in rows
        )
        event_types = Counter(
            str(item.get("event_type") or "unknown")
            for item in rows
            if item.get("record_type") == LEARNING_EVENT_RECORD
        )
        artifacts = Counter(
            str(item.get("artifact_kind") or "unknown")
            for item in rows
            if item.get("record_type") == LEARNING_ARTIFACT_RECORD
        )
        return {
            "agent_id": self.agent_id,
            "record_count": len(rows),
            "record_types": dict(sorted(by_record_type.items())),
            "event_types": dict(sorted(event_types.items())),
            "artifact_kinds": dict(sorted(artifacts.items())),
            "latest_at": (
                rows[-1].get("timestamp")
                or rows[-1].get("updated_at")
                or rows[-1].get("created_at")
                if rows
                else None
            ),
        }


__all__ = [
    "LEARNING_ARTIFACT_RECORD",
    "LEARNING_CHECKPOINT_RECORD",
    "LEARNING_EVENT_RECORD",
    "LEARNING_TOMBSTONE_RECORD",
    "LearningControlPlaneStore",
    "LearningRecordConflictError",
]
