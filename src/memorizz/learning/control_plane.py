"""The small, provider-neutral MemoRizz Learning Control Plane."""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from ..observability import ObservabilityStore
from .compiler import MemoryCompiler
from .evidence import EvidencePlanner
from .forgetting import ForgettingMechanism
from .models import (
    EvidencePack,
    ForgettingReport,
    LearningControlPlaneConfig,
    LearningEvent,
    LearningEventType,
    OutcomeEvidence,
    canonical_json,
    content_digest,
)
from .store import LearningControlPlaneStore, LearningRecordConflictError

logger = logging.getLogger(__name__)


class LearningControlPlane:
    """Connect memory retrieval, outcomes, compilation and skill governance.

    This class is intentionally a thin coordinator.  It does not replace the
    existing memory manager, observability store, or continual-learning manager;
    it makes their decisions durable and explainable through one SDK contract.
    """

    def __init__(
        self,
        provider: Any,
        *,
        agent_id: str,
        config: Optional[
            Union[LearningControlPlaneConfig, Mapping[str, Any], bool]
        ] = None,
    ) -> None:
        self.config = LearningControlPlaneConfig.from_value(config)
        if not self.config.enabled:
            raise ValueError("LearningControlPlane requires enabled=True")
        self.agent_id = str(agent_id)
        self.store = LearningControlPlaneStore(provider, agent_id=self.agent_id)
        self.observability = ObservabilityStore(provider)
        self.compiler = MemoryCompiler(
            self.store,
            batch_size=self.config.compiler_batch_size,
            asynchronous=self.config.compile_async,
        )
        self.evidence_planner = EvidencePlanner(
            provider,
            agent_id=self.agent_id,
            config=self.config,
            store=self.store,
        )
        self.forgetting = ForgettingMechanism(self.store, config=self.config)
        self._events_since_compile: Dict[str, int] = {}
        self._last_evidence_pack: Optional[EvidencePack] = None
        self._last_forgetting_report: Optional[ForgettingReport] = None
        self._lock = threading.Lock()

    def _stream_id(self, scope: Mapping[str, Any]) -> str:
        return self.compiler.stream_id(
            self.agent_id,
            memory_id=scope.get("memory_id"),
            user_id=scope.get("user_id"),
            thread_id=scope.get("thread_id"),
        )

    @staticmethod
    def _scope(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        raw = dict(value or {})
        return {
            key: raw.get(key)
            for key in (
                "application_id",
                "memory_id",
                "user_id",
                "thread_id",
                "run_id",
                "turn_id",
                "trace_id",
                "root_trace_id",
                "workflow_id",
            )
            if key in raw
        }

    def emit(
        self,
        event_type: Union[LearningEventType, str],
        payload: Optional[Mapping[str, Any]] = None,
        *,
        scope: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        compile_if_due: bool = True,
    ) -> Optional[LearningEvent]:
        if not self.config.capture_events:
            return None
        resolved_scope = self._scope(scope)
        stream_id = self._stream_id(resolved_scope)
        if idempotency_key:
            event_id = (
                "learn-evt-"
                + hashlib.sha256(str(idempotency_key).encode()).hexdigest()[:32]
            )
            existing = self.store.get(event_id)
            if existing:
                try:
                    existing_event = LearningEvent.from_dict(existing)
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    expected_type = (
                        event_type
                        if isinstance(event_type, LearningEventType)
                        else LearningEventType(str(event_type))
                    )
                    if (
                        existing_event.event_type != expected_type
                        or canonical_json(existing_event.payload)
                        != canonical_json(dict(payload or {}))
                        or existing_event.stream_id != stream_id
                    ):
                        raise LearningRecordConflictError(
                            "An idempotency key was reused with different "
                            "learning-event content"
                        )
                    return existing_event
        else:
            event_id = None
        event = LearningEvent.create(
            event_type,
            agent_id=self.agent_id,
            stream_id=stream_id,
            payload=payload,
            event_id=event_id,
            idempotency_key=idempotency_key,
            **resolved_scope,
        )
        stored = self.store.append_event(event)
        if compile_if_due and self.config.compiler_enabled:
            with self._lock:
                count = self._events_since_compile.get(stream_id, 0) + 1
                self._events_since_compile[stream_id] = count
                due = (
                    self.config.compile_every_n_events > 0
                    and count >= self.config.compile_every_n_events
                )
                if due:
                    self._events_since_compile[stream_id] = 0
            if due:
                self.compiler.submit(
                    memory_id=resolved_scope.get("memory_id"),
                    user_id=resolved_scope.get("user_id"),
                    thread_id=resolved_scope.get("thread_id"),
                    mode="fast",
                )
        return stored

    def begin_run(
        self, query: str, *, scope: Optional[Mapping[str, Any]] = None
    ) -> Optional[LearningEvent]:
        resolved = self._scope(scope)
        identity = resolved.get("run_id") or resolved.get("turn_id")
        return self.emit(
            LearningEventType.RUN_STARTED,
            {
                "query": str(query)[:8000],
                "query_hash": content_digest(str(query)),
            },
            scope=resolved,
            idempotency_key=(f"run-start:{identity}" if identity else None),
        )

    def record_cache(
        self,
        *,
        hit: bool,
        query: str,
        reason: Optional[str],
        scope: Optional[Mapping[str, Any]] = None,
    ) -> Optional[LearningEvent]:
        return self.emit(
            LearningEventType.CACHE_HIT if hit else LearningEventType.CACHE_BYPASSED,
            {"query_hash": content_digest(str(query)), "reason": reason},
            scope=scope,
            compile_if_due=False,
        )

    def retrieve_evidence(
        self,
        query: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        history_texts: Optional[Sequence[str]] = None,
    ) -> EvidencePack:
        pack = self.evidence_planner.build(
            query,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            history_texts=history_texts,
        )
        self._last_evidence_pack = pack
        scope = {
            "memory_id": memory_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "turn_id": turn_id,
            "trace_id": trace_id,
        }
        self.emit(
            LearningEventType.EVIDENCE_PACK_BUILT,
            pack.to_dict(include_content=False),
            scope=scope,
            compile_if_due=False,
        )
        return pack

    def record_tool(
        self,
        *,
        tool_name: str,
        arguments: Any,
        result: Any,
        success: bool,
        outcome: Optional[Mapping[str, Any]] = None,
        duration_ms: Optional[float] = None,
        scope: Optional[Mapping[str, Any]] = None,
    ) -> Optional[LearningEvent]:
        result_text = canonical_json(result)
        payload = {
            "tool_name": str(tool_name),
            "argument_hash": content_digest(arguments),
            "result_hash": content_digest(result),
            "result_preview": result_text[:800],
            "success": bool(success),
            "duration_ms": duration_ms,
        }
        if outcome:
            payload["outcome"] = dict(outcome)
        return self.emit(
            LearningEventType.TOOL_EXECUTED,
            payload,
            scope=scope,
        )

    def complete_run(
        self,
        response: str,
        *,
        status: str,
        tool_call_count: int = 0,
        metrics: Optional[Mapping[str, Any]] = None,
        scope: Optional[Mapping[str, Any]] = None,
    ) -> Optional[LearningEvent]:
        resolved = self._scope(scope)
        identity = resolved.get("run_id") or resolved.get("turn_id")
        return self.emit(
            LearningEventType.RUN_COMPLETED,
            {
                "status": str(status),
                "response_hash": content_digest(str(response)),
                "response_chars": len(str(response)),
                "tool_call_count": int(tool_call_count),
                "metrics": dict(metrics or {}),
            },
            scope=resolved,
            idempotency_key=(f"run-complete:{identity}:{status}" if identity else None),
        )

    def record_workflow(
        self,
        *,
        workflow_id: str,
        outcome: str,
        canonical_hash: Optional[str],
        step_count: int,
        skills_activated: Sequence[str],
        scope: Optional[Mapping[str, Any]] = None,
    ) -> Optional[LearningEvent]:
        resolved = {**self._scope(scope), "workflow_id": workflow_id}
        return self.emit(
            LearningEventType.WORKFLOW_RECORDED,
            {
                "workflow_id": workflow_id,
                "outcome": outcome,
                "canonical_hash": canonical_hash,
                "step_count": int(step_count),
                "skills_activated": list(skills_activated),
            },
            scope=resolved,
            idempotency_key=f"workflow:{workflow_id}",
        )

    def record_outcome(
        self,
        outcome: OutcomeEvidence,
        *,
        scope: Mapping[str, Any],
        external_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved = self._scope(scope)
        root_trace_id = (
            resolved.get("trace_id")
            or resolved.get("root_trace_id")
            or resolved.get("run_id")
            or resolved.get("turn_id")
        )
        if not root_trace_id:
            raise ValueError("A trace_id, run_id, or turn_id is required for outcomes")
        trace_context = {
            **resolved,
            "root_trace_id": str(root_trace_id),
            "agent_id": self.agent_id,
        }
        observability_record = self.observability.record_outcome(
            trace_context=trace_context,
            status=outcome.status.value,
            verified=outcome.verified,
            source=outcome.source,
            score=outcome.score,
            metrics=outcome.metrics,
            external_id=external_id,
        )
        outcome_payload = outcome.to_dict()
        # The immutable event timestamp is authoritative. Excluding the
        # convenience timestamp inside the value keeps external-id retries
        # idempotent.
        outcome_payload.pop("timestamp", None)
        event = self.emit(
            LearningEventType.OUTCOME_RECORDED,
            {
                **outcome_payload,
                "observability_record_id": observability_record.get("record_id"),
            },
            scope=resolved,
            idempotency_key=(f"outcome:{external_id}" if external_id else None),
        )
        # Preserve the long-standing ObservabilityStore result at the top
        # level.  The extra evidence is additive, so existing application code
        # that reads ``record_id``/``status`` does not need to change.
        return {
            **observability_record,
            "outcome_evidence": outcome.to_dict(),
            "observability_record": observability_record,
            "learning_event": event.to_dict() if event else None,
        }

    def record_skill_transition(
        self,
        transition: Union[LearningEventType, str],
        *,
        skill_id: str,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        scope: Optional[Mapping[str, Any]] = None,
    ) -> Optional[LearningEvent]:
        event_type = (
            transition
            if isinstance(transition, LearningEventType)
            else LearningEventType(str(transition))
        )
        allowed = {
            LearningEventType.SKILL_CANDIDATE_CREATED,
            LearningEventType.SKILL_PROMOTED,
            LearningEventType.SKILL_DEMOTED,
            LearningEventType.SKILL_DEPRECATED,
        }
        if event_type not in allowed:
            raise ValueError("transition is not a skill lifecycle event")
        return self.emit(
            event_type,
            {"skill_id": skill_id, "reason": reason, **dict(metadata or {})},
            scope=scope,
        )

    def compile(self, **scope: Any):
        return self.compiler.compile(**scope)

    def plan_forgetting(self, **scope: Any) -> ForgettingReport:
        report = self.forgetting.plan(**scope)
        self._last_forgetting_report = report
        self.emit(
            LearningEventType.FORGETTING_PLANNED,
            report.to_dict(),
            scope=scope,
            compile_if_due=False,
        )
        return report

    def get_forgetting_plan(
        self,
        plan_id: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Any = ...,
        thread_id: Optional[str] = None,
    ) -> ForgettingReport:
        value = str(plan_id or "").strip()
        if not value:
            raise ValueError("plan_id is required")
        events = self.store.list_events(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            limit=10_000,
        )
        for event in reversed(events):
            if (
                event.event_type == LearningEventType.FORGETTING_PLANNED
                and event.payload.get("plan_id") == value
            ):
                return ForgettingReport.from_dict(event.payload)
        raise KeyError(f"Forgetting plan not found: {value}")

    def apply_forgetting(
        self,
        report: ForgettingReport,
        *,
        approved_by: str,
        reason: Optional[str] = None,
        scope: Optional[Mapping[str, Any]] = None,
    ) -> ForgettingReport:
        application_key = f"forgetting-applied:{report.plan_id}"
        application_id = (
            "learn-evt-" + hashlib.sha256(application_key.encode()).hexdigest()[:32]
        )
        if self.store.get(application_id):
            raise ValueError("This forgetting plan has already been applied")
        applied = self.forgetting.apply(report, approved_by=approved_by, reason=reason)
        self._last_forgetting_report = applied
        self.emit(
            LearningEventType.MEMORY_FORGOTTEN,
            applied.to_dict(),
            scope=scope,
            idempotency_key=application_key,
            compile_if_due=False,
        )
        return applied

    def explain_last_retrieval(
        self, *, include_content: bool = False
    ) -> Dict[str, Any]:
        if self._last_evidence_pack is None:
            return {"available": False, "reason": "no evidence pack built yet"}
        return {
            "available": True,
            **self._last_evidence_pack.to_dict(include_content=include_content),
        }

    def report(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Any = ...,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        stats = self.store.statistics(
            memory_id=memory_id, user_id=user_id, thread_id=thread_id
        )
        last_compilation = self.compiler.last_report
        return {
            "enabled": True,
            "agent_id": self.agent_id,
            "config": self.config.to_dict(),
            "records": stats,
            "last_evidence_pack": (
                self._last_evidence_pack.to_dict(include_content=False)
                if self._last_evidence_pack
                else None
            ),
            "last_compilation": (
                last_compilation.to_dict() if last_compilation else None
            ),
            "last_forgetting": (
                self._last_forgetting_report.to_dict()
                if self._last_forgetting_report
                else None
            ),
        }

    def close(self) -> None:
        self.compiler.close()


__all__ = ["LearningControlPlane"]
