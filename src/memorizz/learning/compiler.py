"""Incremental projection compiler for immutable learning events."""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from .models import (
    ArtifactKind,
    CompilerReport,
    LearningEvent,
    LearningEventType,
    canonical_json,
    content_digest,
    utc_now,
)
from .store import LearningControlPlaneStore

_PROJECTABLE = {
    LearningEventType.RUN_COMPLETED,
    LearningEventType.OUTCOME_RECORDED,
    LearningEventType.WORKFLOW_RECORDED,
    LearningEventType.TOOL_EXECUTED,
    LearningEventType.SKILL_CANDIDATE_CREATED,
    LearningEventType.SKILL_PROMOTED,
    LearningEventType.SKILL_DEMOTED,
    LearningEventType.SKILL_DEPRECATED,
}


class MemoryCompiler:
    """Build small, rebuildable learning artifacts from an event stream.

    The fast tier is deterministic and never calls an LLM.  It folds events by
    run/workflow into compact digests suitable for retrieval.  Checkpoints hold
    processed event hashes, so retries and process restarts are idempotent.
    """

    def __init__(
        self,
        store: LearningControlPlaneStore,
        *,
        batch_size: int = 250,
        asynchronous: bool = True,
    ) -> None:
        self.store = store
        self.batch_size = max(1, int(batch_size))
        self.asynchronous = bool(asynchronous)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._in_flight: Dict[str, Future] = {}
        self._last_report: Optional[CompilerReport] = None

    @staticmethod
    def stream_id(
        agent_id: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
        thread_id: Optional[str],
    ) -> str:
        material = canonical_json(
            {
                "agent_id": agent_id,
                "memory_id": memory_id,
                "user_id": user_id,
                "thread_id": thread_id,
            }
        )
        return "learn-stream-" + hashlib.sha256(material.encode()).hexdigest()[:28]

    @staticmethod
    def _checkpoint_id(stream_id: str) -> str:
        return "learn-checkpoint-" + hashlib.sha256(stream_id.encode()).hexdigest()[:28]

    @staticmethod
    def _artifact_id(kind: ArtifactKind, identity: str) -> str:
        return (
            "learn-artifact-"
            + hashlib.sha256(f"{kind.value}:{identity}".encode()).hexdigest()[:32]
        )

    @staticmethod
    def _event_identity(event: LearningEvent) -> str:
        return str(
            event.run_id
            or event.workflow_id
            or event.payload.get("skill_id")
            or event.event_id
        )

    @staticmethod
    def _event_summary(event: LearningEvent) -> str:
        payload = event.payload
        if event.event_type == LearningEventType.TOOL_EXECUTED:
            outcome = payload.get("outcome") or {}
            outcome_status = str(outcome.get("status") or "").strip().lower()
            if outcome_status == "fallback":
                status = "completed via fallback"
            elif outcome_status == "degraded":
                status = "completed with degraded capability"
            elif outcome_status == "empty":
                status = "completed with no results"
            elif outcome_status == "provider_error":
                status = "failed with a provider error"
            else:
                status = "succeeded" if payload.get("success") else "failed"
            return f"Tool {payload.get('tool_name') or 'unknown'} {status}."
        if event.event_type == LearningEventType.OUTCOME_RECORDED:
            authority = "verified" if payload.get("verified") else "unverified"
            return (
                f"{authority.title()} outcome: {payload.get('status', 'unknown')} "
                f"from {payload.get('source', 'unknown')}."
            )
        if event.event_type == LearningEventType.RUN_COMPLETED:
            return (
                f"Run completed with status {payload.get('status', 'unknown')} and "
                f"{payload.get('tool_call_count', 0)} tool call(s)."
            )
        if event.event_type == LearningEventType.WORKFLOW_RECORDED:
            return (
                f"Workflow {payload.get('workflow_id') or event.workflow_id or 'unknown'} "
                f"recorded with outcome {payload.get('outcome', 'unknown')}."
            )
        if event.event_type in {
            LearningEventType.SKILL_CANDIDATE_CREATED,
            LearningEventType.SKILL_PROMOTED,
            LearningEventType.SKILL_DEMOTED,
            LearningEventType.SKILL_DEPRECATED,
        }:
            return (
                f"Skill {payload.get('skill_id') or payload.get('name') or 'unknown'} "
                f"lifecycle event: {event.event_type.value}."
            )
        return event.event_type.value.replace("_", " ")

    def _build_artifacts(self, events: Sequence[LearningEvent]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[LearningEvent]] = {}
        for event in events:
            if event.event_type not in _PROJECTABLE:
                continue
            grouped.setdefault(self._event_identity(event), []).append(event)

        artifacts: List[Dict[str, Any]] = []
        for identity, grouped_events in grouped.items():
            grouped_events.sort(key=lambda item: item.timestamp)
            first = grouped_events[0]
            last = grouped_events[-1]
            outcomes = [
                item.payload
                for item in grouped_events
                if item.event_type == LearningEventType.OUTCOME_RECORDED
            ]
            verified_outcomes = [
                item for item in outcomes if item.get("verified") is True
            ]
            tools = [
                str(item.payload.get("tool_name"))
                for item in grouped_events
                if item.event_type == LearningEventType.TOOL_EXECUTED
                and item.payload.get("tool_name")
            ]
            query = next(
                (
                    str(item.payload.get("query"))
                    for item in grouped_events
                    if item.payload.get("query")
                ),
                "",
            )
            summaries = [self._event_summary(item) for item in grouped_events]
            artifact_kind = (
                ArtifactKind.OUTCOME
                if outcomes and len(grouped_events) == len(outcomes)
                else ArtifactKind.RUN_DIGEST
            )
            artifact_id = self._artifact_id(artifact_kind, identity)
            content = " ".join(
                dict.fromkeys(summary for summary in summaries if summary)
            )
            artifacts.append(
                {
                    "record_id": artifact_id,
                    "artifact_id": artifact_id,
                    "artifact_kind": artifact_kind.value,
                    "agent_id": first.agent_id,
                    "stream_id": first.stream_id,
                    "memory_id": first.memory_id,
                    "user_id": first.user_id,
                    "thread_id": first.thread_id,
                    "run_id": first.run_id,
                    "workflow_id": first.workflow_id,
                    "query": query,
                    "content": content,
                    "tools": list(dict.fromkeys(tools)),
                    "outcomes": outcomes,
                    "verified": bool(verified_outcomes),
                    "authoritative_outcome": (
                        verified_outcomes[-1].get("status")
                        if verified_outcomes
                        else None
                    ),
                    "source_event_ids": [item.event_id for item in grouped_events],
                    "source_event_hashes": [item.event_hash for item in grouped_events],
                    "source_hash": content_digest(
                        [item.event_hash for item in grouped_events]
                    ),
                    "utility": 1.0 if verified_outcomes else 0.55,
                    "created_at": first.timestamp,
                    "updated_at": last.timestamp,
                    "timestamp": last.timestamp,
                }
            )
        return artifacts

    def compile(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        mode: str = "fast",
    ) -> CompilerReport:
        resolved_mode = str(mode or "fast").strip().lower()
        if resolved_mode not in {"fast", "deep"}:
            raise ValueError("compiler mode must be 'fast' or 'deep'")
        started_clock = time.perf_counter()
        started_at = utc_now()
        stream_id = self.stream_id(
            self.store.agent_id,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
        )
        checkpoint_id = self._checkpoint_id(stream_id)
        checkpoint = self.store.get(checkpoint_id) or {}
        processed = {
            str(item)
            for item in (checkpoint.get("processed_event_hashes") or [])
            if item
        }
        events = self.store.list_events(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            limit=max(self.batch_size * 4, self.batch_size),
        )
        scoped = [event for event in events if event.stream_id == stream_id]
        pending = [event for event in scoped if event.event_hash not in processed][
            : self.batch_size
        ]
        artifacts_written = 0
        errors: List[str] = []
        compiled_hashes: List[str] = []
        # An artifact is a complete projection for one run/workflow/skill. If
        # its events cross compiler batches, rebuild that small projection from
        # all events for the affected identity instead of overwriting it with a
        # partial final batch.
        pending_identities = {self._event_identity(event) for event in pending}
        projection_events = [
            event
            for event in scoped
            if self._event_identity(event) in pending_identities
        ]
        for artifact in self._build_artifacts(projection_events):
            try:
                self.store.put_artifact(artifact)
                artifacts_written += 1
            except Exception as exc:
                errors.append(f"artifact {artifact.get('artifact_id')}: {exc}")
        if not errors:
            compiled_hashes = [event.event_hash for event in pending]
            all_processed = list(dict.fromkeys([*processed, *compiled_hashes]))
            # Bound checkpoint size while preserving enough hashes for retries;
            # older events are behind the timestamp watermark.
            all_processed = all_processed[-10_000:]
            self.store.put_checkpoint(
                {
                    "record_id": checkpoint_id,
                    "checkpoint_id": checkpoint_id,
                    "agent_id": self.store.agent_id,
                    "stream_id": stream_id,
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "mode": resolved_mode,
                    "processed_event_hashes": all_processed,
                    "last_event_timestamp": pending[-1].timestamp
                    if pending
                    else checkpoint.get("last_event_timestamp"),
                    "last_event_id": pending[-1].event_id
                    if pending
                    else checkpoint.get("last_event_id"),
                    "compiled_event_count": int(
                        checkpoint.get("compiled_event_count") or 0
                    )
                    + len(pending),
                    "artifact_count": int(checkpoint.get("artifact_count") or 0)
                    + artifacts_written,
                    "timestamp": utc_now(),
                }
            )
        completed_at = utc_now()
        report = CompilerReport(
            checkpoint_id=checkpoint_id,
            mode=resolved_mode,
            scanned_events=len(scoped),
            compiled_events=len(compiled_hashes),
            skipped_events=max(0, len(scoped) - len(pending)),
            artifacts_written=artifacts_written,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=(time.perf_counter() - started_clock) * 1000,
            event_hashes=tuple(compiled_hashes),
            errors=tuple(errors),
        )
        self._last_report = report
        return report

    def submit(self, **scope: Any) -> Optional[Future]:
        if not self.asynchronous:
            future: Future = Future()
            try:
                future.set_result(self.compile(**scope))
            except Exception as exc:
                future.set_exception(exc)
            return future
        stream_id = self.stream_id(
            self.store.agent_id,
            memory_id=scope.get("memory_id"),
            user_id=scope.get("user_id"),
            thread_id=scope.get("thread_id"),
        )
        with self._lock:
            current = self._in_flight.get(stream_id)
            if current is not None and not current.done():
                return current
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="memorizz-memory-compiler"
                )
            future = self._executor.submit(self.compile, **scope)
            self._in_flight[stream_id] = future
            return future

    def drain(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._lock:
                pending = [
                    future for future in self._in_flight.values() if not future.done()
                ]
            if not pending:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            for future in pending:
                try:
                    future.result(timeout=min(remaining, 0.05))
                except TimeoutError:
                    pass
                except Exception:
                    pass

    def close(self) -> None:
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)
            self._executor = None

    @property
    def last_report(self) -> Optional[CompilerReport]:
        return self._last_report


__all__ = ["MemoryCompiler"]
