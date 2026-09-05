"""Durable review, experiment, feedback, and outcome records for traces."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..enums.memory_type import MemoryType

_RECOMMENDATION = "observability_recommendation"
_EXPERIMENT = "observability_experiment"
_FEEDBACK = "observability_feedback"
_OUTCOME = "observability_outcome"
_TRACE_BUNDLE = "observability_trace_bundle"
_DECISIONS = {"accepted", "rejected", "deferred", "pending"}
logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(
        "|".join(str(part or "") for part in parts).encode("utf-8")
    ).hexdigest()[:28]
    return f"{prefix}-{digest}"


def _read_content(value: Any) -> Any:
    if hasattr(value, "read"):
        try:
            value = value.read()
        except Exception:
            return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return None
    try:
        result = json.loads(value)
    except (TypeError, ValueError):
        return None
    return result if isinstance(result, dict) else None


class ObservabilityStore:
    """Provider-neutral control plane layered on private shared memory.

    Recommendations never modify an agent directly. An accepted review creates
    a versioned draft experiment that must still be run and evaluated through
    Evalground before an operator changes a prompt, tool, or policy.
    """

    def __init__(self, provider: Any):
        if provider is None:
            raise ValueError("ObservabilityStore requires a memory provider")
        self.provider = provider

    @staticmethod
    def _payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = _read_content(row.get("content"))
        if not payload:
            return None
        payload.setdefault(
            "record_id",
            row.get("memory_id") or row.get("_id") or row.get("id"),
        )
        return payload

    def _all(self, record_type: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self.provider.list_all(MemoryType.SHARED_MEMORY) or []
        payloads: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            payload = self._payload(row)
            if not payload:
                continue
            if record_type and payload.get("record_type") != record_type:
                continue
            payloads.append(payload)
        payloads.sort(
            key=lambda item: str(
                item.get("updated_at") or item.get("created_at") or ""
            ),
            reverse=True,
        )
        return payloads

    def _get(self, record_id: str) -> Optional[Dict[str, Any]]:
        # Prefer an indexed point lookup. Providers disagree on whether shared
        # memory is addressed through ``retrieve_by_id`` (filesystem/Oracle)
        # or its logical ``memory_id`` via ``retrieve_by_name`` (MongoDB), so
        # try both portable APIs before using the legacy scan fallback.
        for method_name in ("retrieve_by_id", "retrieve_by_name"):
            method = getattr(self.provider, method_name, None)
            if not callable(method):
                continue
            try:
                row = method(record_id, MemoryType.SHARED_MEMORY)
            except (NotImplementedError, TypeError, ValueError):
                row = None
            if isinstance(row, dict):
                payload = self._payload(row)
                if payload and str(payload.get("record_id") or "") == str(record_id):
                    return payload
        for payload in self._all():
            if str(payload.get("record_id") or "") == str(record_id):
                return payload
        return None

    def _put(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from .index import enabled, metadata_envelope, prepare_bundle
        from .pipeline import record_pipeline_metric

        started = time.perf_counter()
        value = dict(payload)
        record_id = str(value.get("record_id") or "").strip()
        if not record_id:
            raise ValueError("Observability record_id is required")
        value.setdefault("schema_version", 1)
        value.setdefault("created_at", _now())
        value["updated_at"] = _now()
        is_bundle = value.get("record_type") == _TRACE_BUNDLE
        dual_write = is_bundle and enabled("MEMORIZZ_OBSERVABILITY_DUAL_WRITE")
        original = value
        if dual_write:
            value = metadata_envelope(value)
        document = {
            "_id": record_id,
            "memory_id": record_id,
            "content": json.dumps(value, ensure_ascii=False, sort_keys=True),
            "memory_type": MemoryType.SHARED_MEMORY.value,
            "scope": "private",
            "owner_agent_id": value.get("agent_id"),
        }
        # Duplicate only bounded, non-content identity fields at the document
        # level. MongoDB can index these without parsing the private JSON
        # payload, while filesystem/Oracle providers continue to round-trip
        # the canonical payload unchanged.
        trace_memory_id = value.get("trace_memory_id") or value.get("memory_id")
        metadata = {
            "record_type": value.get("record_type"),
            "application_id": value.get("application_id"),
            "agent_id": value.get("agent_id"),
            "run_id": value.get("run_id"),
            "turn_id": value.get("turn_id"),
            "root_trace_id": value.get("root_trace_id"),
            "trace_memory_id": trace_memory_id,
            "thread_id": value.get("thread_id"),
            "user_id": value.get("user_id"),
            "verified": value.get("verified"),
            "status": value.get("status"),
            "role": value.get("role"),
            "timestamp": value.get("timestamp") or value.get("updated_at"),
        }
        document.update(
            {key: item for key, item in metadata.items() if item is not None}
        )
        if is_bundle:
            document["immutable_trace"] = True
        for key in (
            "event_count",
            "event_kind_counts",
            "tool_names",
            "models",
            "has_error",
            "started_at",
            "ended_at",
            "schema_versions",
            "resource_ref_hashes",
            "coverage_profiles",
        ):
            if key in value:
                document[key] = value[key]
        try:
            self.provider.store(document, MemoryType.SHARED_MEMORY)
            if is_bundle:
                # Built-in providers use atomic first-write-wins storage. Read
                # back the winner before indexing a potentially concurrent retry.
                value = self._get(record_id) or value
            record_pipeline_metric(
                self.provider,
                "records_written",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            record_pipeline_metric(self.provider, "write_failures")
            raise
        if dual_write:
            try:
                getter = getattr(self.provider, "get_observability_index", None)
                index = getter() if callable(getter) else None
                if index is None:
                    raise NotImplementedError("Provider has no observability index")
                winner = prepare_bundle(value)[0]["fingerprint"]
                candidate = prepare_bundle(original)[0]["fingerprint"]
                index.write_bundle(original if winner == candidate else value)
                record_pipeline_metric(self.provider, "bundles_indexed")
            except Exception as exc:
                record_pipeline_metric(self.provider, "index_write_failures")
                logger.warning(
                    "Observability index write failed (%s); source bundle remains available",
                    type(exc).__name__,
                )
                value["index_persisted"] = False
        return value

    def record_trace_bundle(
        self,
        *,
        trace_context: Dict[str, Any],
        events: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist one private trace bundle outside conversation memory."""
        context = self._validate_trace_context(trace_context)
        normalized_events = [dict(event) for event in events if isinstance(event, dict)]
        if not normalized_events:
            raise ValueError("events must contain at least one trace event")
        root_trace_id = context["root_trace_id"]
        return self._put(
            {
                "record_id": _record_id(
                    "obs-trace",
                    context.get("application_id"),
                    context.get("agent_id"),
                    context.get("user_id"),
                    context.get("thread_id"),
                    root_trace_id,
                    context.get("turn_id"),
                ),
                "record_type": _TRACE_BUNDLE,
                "type": "trace_bundle",
                "version": 2,
                "role": "tool",
                **context,
                "trace_memory_id": context.get("memory_id"),
                "events": normalized_events,
                **self._bundle_summary(normalized_events),
            }
        )

    @staticmethod
    def _bundle_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        from .index import resource_hash
        from .normalization import timestamp

        times = sorted(
            filter(None, (timestamp(event.get("timestamp")) for event in events))
        )
        return {
            "resource_ref_hashes": sorted(
                {
                    resource_hash(ref["ref"])
                    for event in events
                    for side in ("input_refs", "output_refs")
                    for ref in event.get(side, [])
                    if isinstance(ref, dict) and ref.get("ref")
                }
            )[:128],
            "coverage_profiles": sorted(
                {
                    str(
                        event.get("coverage_profile")
                        or (event.get("attributes") or {}).get("coverage_profile")
                    )
                    for event in events
                    if event.get("coverage_profile")
                    or (event.get("attributes") or {}).get("coverage_profile")
                }
            )[:32],
            "event_count": len(events),
            "event_kind_counts": dict(
                Counter(
                    str(
                        e.get("event_kind")
                        or e.get("trace_kind")
                        or e.get("kind")
                        or "unknown"
                    )
                    for e in events
                )
            ),
            "tool_names": sorted(
                {
                    str(e.get("logical_tool_name") or e.get("tool_name"))[:240]
                    for e in events
                    if e.get("logical_tool_name") or e.get("tool_name")
                }
            )[:64],
            "models": sorted({str(e["model"])[:240] for e in events if e.get("model")})[
                :32
            ],
            "has_error": any(
                e.get("status") == "error" or e.get("success") is False for e in events
            ),
            "started_at": times[0] if times else None,
            "ended_at": times[-1] if times else None,
            "schema_versions": sorted(
                {int(e.get("schema_version") or 1) for e in events}
            ),
        }

    def record_trace_event(self, event) -> Dict[str, Any]:
        """Upsert a typed event using a stable per-event envelope.

        The envelope remains a v2 bundle so every existing provider can query
        host events without migrations or dual-write races.
        """
        from .models import TraceEventV3

        typed = TraceEventV3.model_validate(event)
        child = typed.model_dump(mode="json", exclude_none=True)
        context = {
            key: child[key]
            for key in (
                "agent_id",
                "application_id",
                "user_id",
                "memory_id",
                "thread_id",
                "root_trace_id",
                "run_id",
                "turn_id",
                "timestamp",
            )
            if key in child
        }
        return self._put(
            {
                "record_id": _record_id(
                    "obs-event",
                    typed.application_id,
                    typed.agent_id,
                    typed.user_id,
                    typed.thread_id,
                    typed.root_trace_id,
                    typed.turn_id,
                    typed.event_id,
                ),
                "record_type": _TRACE_BUNDLE,
                "type": "trace_bundle",
                "version": 2,
                "role": "tool",
                **context,
                "trace_memory_id": typed.memory_id,
                "events": [child],
                **self._bundle_summary([child]),
            }
        )

    def record_artifact(self, *, trace_context, **kwargs):
        """Typed artifact evidence; see ObservabilityRecorder.record_artifact."""
        from .recorder import ObservabilityRecorder

        return ObservabilityRecorder(
            self.provider, trace_context, strict=True
        ).record_artifact(**kwargs)

    def sync_recommendations(
        self,
        report: Dict[str, Any],
        *,
        evidence_refs: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Upsert analyzer findings without losing prior operator decisions."""
        agent_id = str(report.get("agent_id") or "").strip()
        if not agent_id:
            raise ValueError("report.agent_id is required")
        refs = []
        for ref in evidence_refs or []:
            if not isinstance(ref, dict):
                continue
            refs.append(
                {
                    key: str(ref[key])[:200]
                    for key in ("root_trace_id", "run_id", "turn_id", "thread_id")
                    if ref.get(key)
                }
            )
        unique_refs = list(
            {json.dumps(item, sort_keys=True): item for item in refs if item}.values()
        )[:100]

        stored: List[Dict[str, Any]] = []
        for insight in report.get("insights") or []:
            if not isinstance(insight, dict) or not insight.get("id"):
                continue
            recommendation_id = _record_id("obs-rec", agent_id, insight.get("id"))
            existing = self._get(recommendation_id) or {}
            material = {
                key: insight.get(key)
                for key in (
                    "id",
                    "priority",
                    "component",
                    "title",
                    "finding",
                    "recommendation",
                    "target",
                    "evidence_count",
                    "confidence",
                    "effort",
                )
            }
            material_hash = _canonical_hash(material)
            previous_hash = existing.get("material_hash")
            revision = max(1, int(existing.get("revision") or 1))
            if previous_hash and previous_hash != material_hash:
                revision += 1
            payload = {
                **existing,
                "record_id": recommendation_id,
                "record_type": _RECOMMENDATION,
                "recommendation_id": recommendation_id,
                "agent_id": agent_id,
                "agent_name": str(report.get("agent_name") or "")[:240],
                "scope": str(report.get("scope") or "agent")[:40],
                "insight": material,
                "material_hash": material_hash,
                "revision": revision,
                "review_status": existing.get("review_status") or "pending",
                "review_history": list(existing.get("review_history") or []),
                "evidence_refs": unique_refs,
                "analysis_summary": dict(report.get("summary") or {}),
            }
            stored.append(self._put(payload))
        return stored

    def list_recommendations(
        self, *, agent_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        rows = self._all(_RECOMMENDATION)
        if agent_id:
            rows = [row for row in rows if row.get("agent_id") == agent_id]
        if status:
            rows = [row for row in rows if row.get("review_status") == status]
        return rows

    def review_recommendation(
        self,
        recommendation_id: str,
        *,
        decision: str,
        reviewer_id: str,
        note: Optional[str] = None,
        baseline_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_decision = str(decision or "").strip().lower()
        if resolved_decision not in _DECISIONS - {"pending"}:
            raise ValueError("decision must be accepted, rejected, or deferred")
        reviewer = str(reviewer_id or "").strip()
        if not reviewer:
            raise ValueError("reviewer_id is required")
        record = self._get(recommendation_id)
        if not record or record.get("record_type") != _RECOMMENDATION:
            raise KeyError("Recommendation not found")
        decision_record = {
            "decision": resolved_decision,
            "reviewer_id": reviewer[:200],
            "note": str(note or "")[:2000],
            "timestamp": _now(),
            "revision": int(record.get("revision") or 1),
        }
        history = list(record.get("review_history") or [])
        history.append(decision_record)
        record["review_status"] = resolved_decision
        record["review_history"] = history[-100:]
        record["last_review"] = decision_record
        stored = self._put(record)
        experiment = None
        if resolved_decision == "accepted":
            experiment = self.create_experiment(
                stored,
                reviewer_id=reviewer,
                baseline_config=baseline_config,
            )
        return {"recommendation": stored, "experiment": experiment}

    def create_experiment(
        self,
        recommendation: Dict[str, Any],
        *,
        reviewer_id: str,
        baseline_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        recommendation_id = str(recommendation.get("recommendation_id") or "")
        revision = int(recommendation.get("revision") or 1)
        existing = [
            row
            for row in self._all(_EXPERIMENT)
            if row.get("recommendation_id") == recommendation_id
        ]
        for row in existing:
            if int(row.get("recommendation_revision") or 0) == revision:
                return row
        experiment_version = 1 + max(
            [int(row.get("experiment_version") or 0) for row in existing] or [0]
        )
        experiment_id = _record_id(
            "obs-exp", recommendation_id, revision, experiment_version
        )
        insight = dict(recommendation.get("insight") or {})
        baseline = dict(baseline_config or {})
        payload = {
            "record_id": experiment_id,
            "record_type": _EXPERIMENT,
            "experiment_id": experiment_id,
            "experiment_version": experiment_version,
            "status": "draft",
            "agent_id": recommendation.get("agent_id"),
            "recommendation_id": recommendation_id,
            "recommendation_revision": revision,
            "created_by": str(reviewer_id)[:200],
            "component": insight.get("component"),
            "target": insight.get("target"),
            "hypothesis": insight.get("recommendation"),
            "baseline_config": baseline,
            "baseline_fingerprint": _canonical_hash(baseline),
            "proposed_change": insight.get("recommendation"),
            "evaluation_plan": {
                "source": "verified_trace_evidence",
                "evidence_refs": list(recommendation.get("evidence_refs") or []),
                "primary_metrics": [
                    "verified_task_success_rate",
                    "negative_feedback_rate",
                    "tool_failure_rate",
                ],
                "promotion_gate": "operator review after Evalground comparison",
            },
        }
        return self._put(payload)

    def list_experiments(
        self, *, agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        rows = self._all(_EXPERIMENT)
        if agent_id:
            rows = [row for row in rows if row.get("agent_id") == agent_id]
        return rows

    def create_replay_draft(
        self, events, *, created_by, authorize_resource=None, persist=True
    ):
        """Freeze authorized content-free evidence for an inert Evalground draft.

        This method never invokes tools, models, subprocesses or host side effects.
        Current resource access must be rechecked by the host, not inferred from
        historical ownership flags. Unversioned refs remain explicitly unresolved.
        """
        from .index import _opaque_fields
        from .privacy import validate_opaque
        from .references import event_resource_refs

        metadata = getattr(events, "coverage", {})
        if (
            metadata.get("read_completeness", metadata.get("coverage")) != "complete"
            or metadata.get("truncated")
            or metadata.get("normalization_errors")
            or metadata.get("window_complete") is False
        ):
            raise ValueError("Replay drafts require a complete trusted evidence window")
        validate_opaque(created_by)
        if not isinstance(created_by, str) or not created_by or len(created_by) > 240:
            raise ValueError("created_by must be an opaque operator ID")
        scopes = {
            (event.get("application_id"), event.get("user_id")) for event in events
        }
        if not events or len(scopes) != 1:
            raise ValueError("Select exactly one authorized tenant scope for replay")
        application_id, user_id = next(iter(scopes))
        refs = {}
        for event in events:
            for ref in event_resource_refs(event):
                refs[_canonical_hash(ref)] = ref
        if len(refs) > 256:
            raise ValueError("Replay references exceed the bounded draft limit")
        for ref in refs.values():
            if (
                not callable(authorize_resource)
                or authorize_resource(
                    ref, {"application_id": application_id, "user_id": user_id}
                )
                is not True
            ):
                raise PermissionError("Current resource access was not verified")
        frozen = [_opaque_fields(event) for event in events]
        fingerprint = _canonical_hash(frozen)
        record_id = _record_id(
            "obs-replay", created_by, application_id, user_id, fingerprint
        )
        payload = {
            "record_id": record_id,
            "record_type": _EXPERIMENT,
            "experiment_id": record_id,
            "experiment_version": 1,
            "status": "draft",
            "kind": "trace_replay",
            "agent_id": events[0].get("agent_id"),
            "application_id": application_id,
            "user_id": user_id,
            "created_by": created_by,
            "component": "Trace replay",
            "target": "isolated Evalground review",
            "hypothesis": "Compare a candidate against frozen trace evidence under an approved sandbox policy",
            "baseline_config": {},
            "evidence_fingerprint": fingerprint,
            "resource_refs": list(refs.values()),
            "unversioned_refs": sum(not ref.get("version") for ref in refs.values()),
            "evidence_refs": [
                {
                    key: event.get(key)
                    for key in ("event_id", "root_trace_id", "turn_id", "run_id")
                }
                for event in events
            ],
            "replay_policy": {
                "execution_enabled": False,
                "network": False,
                "side_effects": False,
                "requires_sandbox": True,
                "requires_operator_approval": True,
            },
            "evaluation_plan": {
                "source": "frozen_trace_refs",
                "primary_metrics": [
                    "verified_task_success_rate",
                    "source_provenance",
                    "contract_delivery",
                ],
                "promotion_gate": "explicit operator review",
            },
        }
        return self._put(payload) if persist else payload

    def record_feedback(
        self,
        *,
        trace_context: Dict[str, Any],
        rating: float,
        verified: bool,
        source: str = "user",
        label: Optional[str] = None,
        comment: Optional[str] = None,
        include_comment: bool = False,
        external_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = self._validate_trace_context(trace_context)
        score = float(rating)
        if not math.isfinite(score) or score < -1 or score > 5:
            raise ValueError("rating must be a finite number between -1 and 5")
        record_id = _record_id(
            "obs-feedback",
            external_id or uuid.uuid4(),
            context.get("root_trace_id"),
        )
        payload = {
            "record_id": record_id,
            "record_type": _FEEDBACK,
            **context,
            "rating": score,
            "verified": bool(verified),
            "source": str(source or "user")[:80],
            "label": str(label or "")[:200],
            "comment_sha256": _canonical_hash(str(comment or "")) if comment else None,
        }
        if include_comment and comment:
            payload["comment"] = str(comment)[:4000]
        return self._put(payload)

    def record_outcome(
        self,
        *,
        trace_context: Dict[str, Any],
        status: str,
        verified: bool,
        source: str = "application",
        score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        external_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = self._validate_trace_context(trace_context)
        resolved_status = str(status or "").strip().lower()
        if resolved_status not in {"success", "failure", "partial", "unknown"}:
            raise ValueError("status must be success, failure, partial, or unknown")
        numeric_score = None if score is None else float(score)
        if numeric_score is not None and not math.isfinite(numeric_score):
            raise ValueError("score must be finite")
        record_id = _record_id(
            "obs-outcome",
            external_id or uuid.uuid4(),
            context.get("root_trace_id"),
        )
        return self._put(
            {
                "record_id": record_id,
                "record_type": _OUTCOME,
                **context,
                "status": resolved_status,
                "verified": bool(verified),
                "source": str(source or "application")[:80],
                "score": numeric_score,
                "metrics": dict(metrics or {}),
            }
        )

    @staticmethod
    def _validate_trace_context(value: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("trace_context must be a dict")
        root_trace_id = str(
            value.get("root_trace_id") or value.get("trace_id") or ""
        ).strip()
        if not root_trace_id:
            raise ValueError("trace_context.root_trace_id is required")
        return {
            key: str(value[key])[:240]
            for key in (
                "application_id",
                "agent_id",
                "run_id",
                "turn_id",
                "root_trace_id",
                "memory_id",
                "thread_id",
                "user_id",
                "workflow_id",
            )
            if value.get(key) is not None
        } | {"root_trace_id": root_trace_id}

    def list_signals(
        self,
        *,
        root_trace_ids: Optional[Iterable[str]] = None,
        agent_id: Optional[str] = None,
        verified_only: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        wanted = {str(value) for value in (root_trace_ids or []) if value}

        # One provider read is enough for both signal kinds. This matters for
        # compatibility providers whose bounded control-plane query is not yet
        # native, and prevents feedback + outcome analysis from doubling I/O.
        all_rows = self._all()

        def _select(record_type: str) -> List[Dict[str, Any]]:
            rows = [row for row in all_rows if row.get("record_type") == record_type]
            if wanted:
                rows = [row for row in rows if row.get("root_trace_id") in wanted]
            if agent_id:
                rows = [row for row in rows if row.get("agent_id") == agent_id]
            if verified_only:
                rows = [row for row in rows if row.get("verified") is True]
            return rows

        return {
            "feedback": _select(_FEEDBACK),
            "outcomes": _select(_OUTCOME),
        }


__all__ = ["ObservabilityStore"]
