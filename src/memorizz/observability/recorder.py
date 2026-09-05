"""Host-side spans for application actions, workers, artifacts and delivery."""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

from .models import ResourceRef, TraceContext, TraceEventV3, validate_attributes
from .normalization import timestamp
from .store import ObservabilityStore, _record_id

logger = logging.getLogger(__name__)


def _require_bool(**values):
    if any(type(value) is not bool for value in values.values()):
        raise ValueError("observability evidence flags must be booleans")


class ObservabilityRecorder:
    """Record content-free telemetry independently of a live MemAgent.

    Validation errors are actionable programming errors. Provider write errors
    are fail-soft by default and visible in ``health`` and logging; ``strict``
    makes delivery failures raise. A carrier is correlation, not authorization:
    hosts must validate job ownership before constructing a recorder.
    """

    def __init__(
        self, provider: Any, trace_context: TraceContext | dict, *, strict=False
    ):
        self.store = ObservabilityStore(provider)
        self.context = (
            trace_context
            if isinstance(trace_context, TraceContext)
            else TraceContext.from_carrier(trace_context)
        )
        self.strict = strict
        self.health = {
            "events_written": 0,
            "write_failures": 0,
            "last_error_code": None,
        }
        self._parent = ContextVar(f"observability_parent_{id(self)}", default=None)

    def record_event(
        self,
        operation: str,
        *,
        kind: str = "application_action",
        phase: str = "event",
        status: str = "unknown",
        trace_context=None,
        span_id=None,
        parent_span_id=None,
        attributes=None,
        input_refs=None,
        output_refs=None,
        selection_ledger=None,
        external_id=None,
        duration_ms=None,
        component=None,
        deployment_id=None,
        worker_id=None,
    ) -> dict:
        context = trace_context or self.context
        if isinstance(context, dict):
            context = TraceContext.from_carrier(context)
        identity = context.to_carrier()
        identity["parent_span_id"] = (
            parent_span_id
            or self._parent.get()
            or context.parent_span_id
            or context.root_trace_id
        )
        event_id = (
            _record_id(
                "obs-event",
                context.application_id,
                context.agent_id,
                context.user_id,
                context.thread_id,
                context.root_trace_id,
                context.turn_id,
                external_id,
                operation,
                phase,
            )
            if external_id
            else str(uuid.uuid4())
        )
        event = TraceEventV3(
            **identity,
            event_id=event_id,
            event_kind=kind,
            operation=operation,
            phase=phase,
            status=status,
            span_id=span_id or event_id,
            duration_ms=duration_ms,
            attributes=attributes or {},
            input_refs=input_refs or [],
            output_refs=output_refs or [],
            selection_ledger=selection_ledger or [],
            component=component,
            deployment_id=deployment_id,
            worker_id=worker_id,
        )
        try:
            result = self.store.record_trace_event(event)
            self.health["events_written"] += 1
            return result
        except Exception as exc:
            self.health["write_failures"] += 1
            self.health["last_error_code"] = type(exc).__name__
            logger.warning("Observability event write failed (%s)", type(exc).__name__)
            if self.strict:
                raise
            return {**event.model_dump(mode="json"), "persisted": False}

    def start_span(self, operation: str, **kwargs):
        return ObservedSpan(self, operation, **kwargs)

    @classmethod
    def for_worker(cls, provider, carrier, *, fallback_context=None, strict=False):
        """Restore validated job identity, or report a missing/invalid carrier.

        fallback_context must be a host-authorized scope. Missing carrier evidence
        is an error, never a new unrelated successful trace. It cannot grant access.
        """
        try:
            context = TraceContext.from_carrier(carrier)
        except (ValueError, TypeError):
            if fallback_context is None:
                raise ValueError(
                    "A valid trace carrier or authorized fallback_context is required"
                ) from None
            recorder = cls(provider, fallback_context, strict=strict)
            recorder.record_event(
                "worker.trace_context_missing",
                kind="trace_data_quality",
                status="error",
                attributes={
                    "context_missing": True,
                    "error_code": "trace_context_missing",
                },
            )
            return recorder
        return cls(provider, context, strict=strict)

    def record_selection(self, decisions, *, task_id=None, external_id=None):
        """Record up to 64 typed, content-free candidate decisions."""
        return self.record_event(
            "memory.selection",
            kind="memory_selection",
            status="success",
            selection_ledger=decisions,
            attributes={"task_id": task_id},
            external_id=external_id,
        )

    def record_intent(
        self,
        task_id: str,
        *,
        expected_artifact_types=None,
        required_contracts=None,
        input_refs=None,
        coverage_profile=None,
        external_id=None,
    ):
        for values in (expected_artifact_types, required_contracts):
            if values is not None and (
                not isinstance(values, list)
                or any(not isinstance(item, str) for item in values)
            ):
                raise ValueError("intent expectations must be lists of names")
        return self.record_event(
            "intent.plan",
            kind="intent_plan",
            status="started",
            input_refs=input_refs,
            attributes={
                "task_id": task_id,
                "expected_artifact_types": expected_artifact_types or [],
                "required_contracts": required_contracts or [],
                "coverage_profile": coverage_profile,
            },
            external_id=external_id,
        )

    def reconcile_outcome(
        self, events, *, task_id=None, normalization_metadata=None, external_id=None
    ):
        """Verify only this tenant/turn's current, acknowledged task evidence.

        Supply a complete TraceEvents window, or explicitly pass its query
        metadata with a plain list. An isolated page is not a complete window.
        """
        from .coverage import trace_coverage
        from .diagnostics import diagnose_trace
        from .evidence import (
            event_scope,
            evidence_groups,
            is_delivered,
            is_persisted,
            latest_artifacts,
            latest_named,
            resource_key,
            task_key,
        )

        metadata = dict(
            normalization_metadata
            if normalization_metadata is not None
            else getattr(events, "coverage", {})
        )
        rows = evidence_groups(events).get(event_scope(self.context.to_carrier()), [])
        # Resolve legacy task assignment before narrowing the selected task.
        rows = [{**row, "task_id": task_key(row, rows)} for row in rows]
        if task_id is not None:
            rows = [
                row
                for row in rows
                if row.get("task_id") == task_id
                or (
                    row.get("task_id") is None
                    and row.get("kind")
                    in {
                        "application_action",
                        "model_call",
                        "model_result",
                        "tool_call",
                        "tool_result",
                    }
                )
            ]
        coverage = trace_coverage(rows, metadata)
        remaining_stages = [
            stage
            for stage in coverage.get("missing_stages", [])
            if stage["stage"] != "verified_outcome"
        ]
        evidence_complete = (
            metadata.get("coverage") == "complete"
            and metadata.get("window_complete", True)
            and not remaining_stages
            and not coverage.get("normalization_errors")
            and not coverage.get("truncated")
        )
        findings = diagnose_trace(rows, metadata=metadata)
        intents = [row for row in rows if row.get("kind") == "intent_plan"]
        deliveries = list(latest_named(rows, "ui_delivery").values())
        delivered = all(
            any(
                is_delivered(d) and task_key(d, rows) == task_key(intent, rows)
                for d in deliveries
            )
            for intent in intents
        )
        artifacts_delivered = all(
            any(
                is_delivered(delivery)
                and task_key(delivery, rows) == task_key(artifact, rows)
                and timestamp(delivery.get("timestamp"))
                >= timestamp(artifact.get("timestamp"))
                and all(
                    resource_key(ref, version=True)
                    in {
                        resource_key(r, version=True)
                        for r in delivery.get("input_refs", [])
                    }
                    for ref in artifact.get("output_refs", [])
                )
                for delivery in deliveries
            )
            for artifact in latest_artifacts(rows)
            if is_persisted(artifact)
        )
        success = (
            bool(intents)
            and delivered
            and artifacts_delivered
            and not findings
            and evidence_complete
        )
        return self.record_event(
            "task.outcome",
            kind="verified_outcome",
            status="success" if success else "partial",
            attributes={"verified": success, "task_id": task_id},
            external_id=external_id,
        )

    def record_artifact(
        self,
        artifact: ResourceRef | dict,
        *,
        input_refs: list,
        producing_span_id: str,
        persistence_verified: bool,
        status: str = "created",
        source_independent: bool = False,
        task_id: str | None = None,
        external_id=None,
    ) -> dict:
        _require_bool(
            persistence_verified=persistence_verified,
            source_independent=source_independent,
        )
        if status not in {"created", "updated", "failed", "deleted"}:
            raise ValueError("invalid artifact status")
        if not producing_span_id:
            raise ValueError("producing_span_id is required")
        artifact = ResourceRef.model_validate(artifact)
        # Persist invalid provenance as evidence, so bad application writes
        # remain diagnosable. This records validation, never creates artifacts.
        valid = (
            persistence_verified
            and (source_independent or bool(input_refs))
            and status in {"created", "updated"}
        )
        return self.record_event(
            f"artifact.{artifact.resource_type}",
            kind="artifact_persisted",
            status="success" if valid else "error" if status == "failed" else "partial",
            parent_span_id=producing_span_id,
            input_refs=input_refs,
            output_refs=[artifact],
            attributes={
                "artifact_status": status,
                "persistence_verified": persistence_verified,
                "source_independent": source_independent,
                "task_id": task_id,
            },
            external_id=external_id,
        )

    def record_contract_result(
        self,
        name: str,
        *,
        required: bool = True,
        opening_marker_found: bool,
        closing_marker_found: bool,
        parser_status: str,
        recovery_used: bool = False,
        recovery_status: str | None = None,
        item_counts: dict | None = None,
        output_limit: int | None = None,
        task_id: str | None = None,
        external_id=None,
    ) -> dict:
        _require_bool(
            required=required,
            opening_marker_found=opening_marker_found,
            closing_marker_found=closing_marker_found,
            recovery_used=recovery_used,
        )
        if parser_status not in {"success", "failed", "missing", "partial"}:
            raise ValueError("invalid parser status")
        counts = item_counts or {}
        if set(counts) - {"nodes", "edges"} or any(
            type(v) is not int or v < 0 for v in counts.values()
        ):
            raise ValueError("item_counts supports nonnegative nodes and edges")
        complete = (
            opening_marker_found
            and closing_marker_found
            and parser_status == "success"
            and not recovery_used
        )
        return self.record_event(
            f"contract.{name}",
            kind="output_contract",
            status="success" if complete else "partial" if recovery_used else "error",
            attributes={
                "contract_name": name,
                "task_id": task_id,
                "required": required,
                "opening_marker_found": opening_marker_found,
                "closing_marker_found": closing_marker_found,
                "parser_status": parser_status,
                "recovery_used": recovery_used,
                "recovery_status": recovery_status,
                "output_limit": output_limit,
                **counts,
            },
            external_id=external_id,
        )

    def contract(self, name: str, *, required=True, output_limit=None, task_id=None):
        return ObservedContract(
            self, name, required=required, output_limit=output_limit, task_id=task_id
        )

    def record_delivery(
        self,
        name: str,
        *,
        emitted: bool,
        acknowledged: bool | None = None,
        input_refs=None,
        task_id: str | None = None,
        external_id=None,
    ):
        _require_bool(emitted=emitted)
        if acknowledged is not None:
            _require_bool(acknowledged=acknowledged)
        return self.record_event(
            f"delivery.{name}",
            kind="ui_delivery",
            status="success"
            if emitted and acknowledged is True
            else "partial"
            if emitted
            else "error",
            attributes={
                "contract_name": name,
                "task_id": task_id,
                "emitted": emitted,
                "acknowledged": acknowledged,
            },
            input_refs=input_refs,
            external_id=external_id,
        )


class ObservedSpan:
    def __init__(self, recorder, operation, **kwargs):
        self.recorder, self.operation, self.kwargs = recorder, operation, kwargs
        external_id = kwargs.get("external_id")
        context = kwargs.get("trace_context") or recorder.context
        context = (
            context
            if isinstance(context, TraceContext)
            else TraceContext.from_carrier(context)
        )
        self.span_id = (
            _record_id(
                "obs-span",
                context.application_id,
                context.agent_id,
                context.user_id,
                context.thread_id,
                context.root_trace_id,
                context.turn_id,
                operation,
                external_id,
            )
            if external_id
            else str(uuid.uuid4())
        )
        self.attributes = dict(kwargs.pop("attributes", {}) or {})
        validate_attributes(self.attributes)
        self.outputs = list(kwargs.pop("output_refs", []) or [])
        self.status = "success"
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("a span can only be entered once")
        self._entered = True
        self.started = time.perf_counter()
        self.parent = (
            self.kwargs.pop("parent_span_id", None) or self.recorder._parent.get()
        )
        self.recorder.record_event(
            self.operation,
            phase="start",
            status="started",
            span_id=self.span_id,
            parent_span_id=self.parent,
            attributes=self.attributes,
            **self.kwargs,
        )
        self.token = self.recorder._parent.set(self.span_id)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.recorder._parent.reset(self.token)
        if exc_type:
            self.status = "error"
            self.attributes["error_code"] = exc_type.__name__[:240]
        self.recorder.record_event(
            self.operation,
            phase="result",
            status=self.status,
            span_id=self.span_id,
            parent_span_id=self.parent,
            duration_ms=(time.perf_counter() - self.started) * 1000,
            attributes=self.attributes,
            output_refs=self.outputs,
            **self.kwargs,
        )
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self.__exit__(exc_type, exc, tb)

    def add_output_ref(self, ref):
        self.outputs.append(ResourceRef.model_validate(ref))

    def set_attribute(self, name, value):
        validate_attributes({**self.attributes, name: value})
        self.attributes[name] = value

    def succeed(self):
        self.status = "success"

    def partial(self):
        self.status = "partial"

    def to_carrier(self):
        context = self.kwargs.get("trace_context") or self.recorder.context
        context = (
            context
            if isinstance(context, TraceContext)
            else TraceContext.from_carrier(context)
        )
        return {**context.to_carrier(), "parent_span_id": self.span_id}


class ObservedContract:
    def __init__(self, recorder, name, **kwargs):
        self.recorder, self.name, self.kwargs = recorder, name, kwargs
        self.result = None

    def __enter__(self):
        return self

    def record(self, **kwargs):
        self.result = kwargs

    def __exit__(self, exc_type, exc, tb):
        result = self.result or {
            "opening_marker_found": False,
            "closing_marker_found": False,
            "parser_status": "missing",
        }
        if exc_type:
            result = {**result, "parser_status": "failed"}
        self.recorder.record_contract_result(self.name, **self.kwargs, **result)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self.__exit__(exc_type, exc, tb)
