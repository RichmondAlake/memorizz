"""Provider-neutral trace expansion, identity selection and coverage accounting."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from typing import Any

from .models import IDENTITY_FIELDS, ResourceRef, SelectionDecision, TraceEventV3
from .references import source_ids

_USER_UNSET = object()

# Retain the bounded metadata already captured by legacy MemAgent bundles.
LEGACY_FIELDS = frozenset(
    [
        "schema_version",
        "application_id",
        "agent_id",
        "run_id",
        "turn_id",
        "task_id",
        "root_trace_id",
        "span_id",
        "parent_span_id",
        "memory_id",
        "thread_id",
        "user_id",
        "timestamp",
        "tool_name",
        "logical_tool_name",
        "model_tool_name",
        "tool_call_id",
        "success",
        "status",
        "outcome",
        "outcome_reason_code",
        "tool_provider",
        "primary_provider",
        "fallback_provider",
        "outcome_retryable",
        "result_count",
        "fallback_used",
        "degraded",
        "error_code",
        "duration_ms",
        "model",
        "provider",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "cost_usd",
        "finish_reason",
        "max_output_tokens",
        "response_chars",
        "response_bytes",
        "ttft_ms",
        "stream_duration_ms",
        "retry_count",
        "fallback_count",
        "iteration",
        "stage",
        "total_tokens",
        "request_id",
        "client_page_type",
        "client_page_id",
        "client_title_fingerprint",
        "canonical_page_type",
        "canonical_page_id",
        "canonical_title_fingerprint",
        "thread_binding_status",
        "expected_thread_id",
        "ownership_verified",
        "request_context_present",
        "request_context_fingerprint",
        "request_context_key_count",
        "content_version",
        "grounding_status",
        "grounding_source",
        "grounding_excerpt_count",
        "grounding_source_ids",
        "cache_decision",
        "memory_history_count",
        "memory_candidate_count",
        "memory_supplied_count",
        "memory_referenced_count",
        "memory_injected_chars",
        "memory_degraded",
        "memory_fallback_used",
        "entity_profile_count",
        "preference_count",
        "conversation_memory_count",
        "writing_sample_count",
        "cache_enabled",
        "cache_bypass_reason",
        "event_id",
        "trace_id",
        "trace_kind",
        "kind",
        "role",
        "title",
        "content",
        "timestamp",
        "phase",
        "operation",
        "selection_ledger",
        "input_refs",
        "output_refs",
    ]
)


def timestamp(value: Any) -> str:
    try:
        if isinstance(value, (int, float)):
            date = datetime.fromtimestamp(value, timezone.utc)
        elif isinstance(value, datetime):
            date = value
        else:
            date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return date.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError, OverflowError, OSError):
        return ""


def read_payload(row: dict) -> dict | None:
    raw = row.get("content") or row.get("text")
    if hasattr(raw, "read"):
        try:
            raw = raw.read()
        except Exception:
            return None
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    return raw if isinstance(raw, dict) else None


def is_bundle(row: dict, payload: dict | None = None) -> bool:
    payload = payload if payload is not None else read_payload(row)
    return row.get("record_type") == "observability_trace_bundle" or bool(
        payload and payload.get("type") == "trace_bundle"
    )


def row_identity(row: dict, payload: dict | None = None) -> dict:
    payload = payload or {}
    document_memory_id = row.get("memory_id")
    if (
        payload.get("record_id")
        and document_memory_id == payload["record_id"]
        and is_bundle(row, payload)
    ):
        document_memory_id = None  # A source envelope ID is not an agent memory ID.
    identity = {key: row.get(key) or payload.get(key) for key in IDENTITY_FIELDS}
    identity["agent_id"] = identity.get("agent_id") or row.get("agentId")
    identity["memory_id"] = (
        row.get("trace_memory_id")
        or payload.get("trace_memory_id")
        or payload.get("memory_id")
        or document_memory_id
        or row.get("memoryId")
        or "—"
    )
    identity["thread_id"] = (
        identity.get("thread_id")
        or row.get("threadId")
        or row.get("conversation_id")
        or row.get("conversationId")
        or identity["memory_id"]
    )
    return {key: value for key, value in identity.items() if value is not None}


@dataclass(frozen=True)
class TraceSnapshot:
    conversation_rows: list[dict] = dataclass_field(default_factory=list)
    tool_rows: list[dict] = dataclass_field(default_factory=list)
    bundle_rows: list[dict] = dataclass_field(default_factory=list)
    query_metadata: dict = dataclass_field(default_factory=dict)

    def __iter__(self):
        # Retain the router's private tuple-unpacking contract for extensions.
        yield self.conversation_rows + self.bundle_rows
        yield self.tool_rows
        yield self.query_metadata


class TraceEvents(list):
    """List-compatible timeline carrying coverage to legacy analyzer callers."""

    def __init__(self, events=(), *, coverage=None):
        super().__init__(events)
        self.coverage = coverage or {}


def select_trace_events(
    events,
    *,
    root_trace_id=None,
    thread_id=None,
    thread_memory_id=None,
    turn_id=None,
    task_id=None,
    run_id=None,
    start_time=None,
    end_time=None,
    limit=0,
):
    """Narrow a loaded window without relabelling its parent-query counts."""
    filters = {
        k: v
        for k, v in {
            "root_trace_id": root_trace_id,
            "thread_id": thread_id,
            "memory_id": thread_memory_id,
            "turn_id": turn_id,
            "run_id": run_id,
        }.items()
        if v is not None
    }
    rows = [e for e in events if all(e.get(k) == v for k, v in filters.items())]
    start, end = (
        timestamp(start_time) if start_time else None,
        timestamp(end_time) if end_time else None,
    )
    if (
        (start_time and not start)
        or (end_time and not end)
        or (start and end and start > end)
    ):
        raise ValueError("Invalid trace time range")
    rows = [
        e
        for e in rows
        if (not start or timestamp(e.get("timestamp")) >= start)
        and (
            not end
            or bool(timestamp(e.get("timestamp")))
            and timestamp(e.get("timestamp")) <= end
        )
    ]
    if task_id is not None:
        from .evidence import event_scope, task_key

        groups = {}
        for row in rows:
            groups.setdefault(event_scope(row), []).append(row)
        rows = [
            row for row in rows if task_key(row, groups[event_scope(row)]) == task_id
        ]
    metadata = dict(getattr(events, "coverage", {}))
    truncated = bool(limit and len(rows) > limit)
    if truncated:
        rows = rows[-limit:]
    if filters or task_id is not None or start or end or truncated:
        parent = {
            k: metadata.get(k)
            for k in (
                "source_rows",
                "source_counts",
                "bundle_count",
                "duplicates_removed",
                "normalized_events",
            )
        }
        provenance = {
            (
                e.get("source_store")
                or ("bundle" if e.get("source_bundle_id") else None),
                e.get("source_record_id") or e.get("source_bundle_id"),
            )
            for e in rows
        }
        counts_known = all(store and record for store, record in provenance)
        counts = Counter(store for store, _ in provenance) if counts_known else {}
        metadata.update(
            parent_query_metadata=parent,
            count_scope="selected_event_provenance"
            if counts_known
            else "unknown_parent_counts_available",
            source_counts=dict(counts),
            source_rows=sum(counts.values()) if counts_known else None,
            bundle_count=counts.get("bundle", 0) if counts_known else None,
            duplicates_removed=None,
        )
    metadata.update(
        normalized_events=len(rows),
        schema_counts=dict(
            Counter(str(e.get("schema_version", "unknown")) for e in rows)
        ),
    )
    if truncated:
        read = metadata.get("read_completeness", metadata.get("coverage"))
        metadata.update(
            truncated=True,
            coverage="untrusted" if read == "untrusted" else "partial",
            read_completeness="untrusted" if read == "untrusted" else "partial",
            window_complete=False,
        )
    return TraceEvents(rows, coverage=metadata)


@dataclass
class NormalizedTraceWindow:
    events: TraceEvents
    source_counts: dict
    schema_counts: dict
    normalization_errors: list[dict]
    duplicate_count: int = 0
    truncated: bool = False

    @property
    def metadata(self) -> dict:
        return {
            "source_rows": sum(self.source_counts.values()),
            "bundle_count": self.source_counts.get("bundle", 0),
            "source_counts": self.source_counts,
            "schema_counts": self.schema_counts,
            "normalized_events": len(self.events),
            "normalization_errors": len(self.normalization_errors),
            "errors": self.normalization_errors,
            "duplicates_removed": self.duplicate_count,
            "truncated": self.truncated,
            "coverage": "untrusted"
            if self.normalization_errors
            else "partial"
            if self.truncated
            else "complete",
            "read_completeness": "untrusted"
            if self.normalization_errors
            else "partial"
            if self.truncated
            else "complete",
            "count_scope": "authorized_source_snapshot",
        }


def normalize_trace_snapshot(
    snapshot: TraceSnapshot,
    *,
    agent_ids: set[str] | None = None,
    memory_ids: set[str] | None = None,
    thread_id: str | None = None,
    thread_memory_id: str | None = None,
    root_trace_ids: set[str] | None = None,
    limit: int = 0,
    application_id: str | None = None,
    user_id: Any = _USER_UNSET,
) -> NormalizedTraceWindow:
    """Normalize every returned row; agent and memory are alternative matches.

    Identity filters select within an already authorized provider scope. They
    are not a tenant authorization mechanism. Malformed bundles never become
    conversation events or silently count as zero activity.
    """
    events, errors, seen = [], [], set()
    counts, schemas = Counter(), Counter()
    duplicates = 0

    def in_scope(identity):
        return (
            (
                not (agent_ids or memory_ids or root_trace_ids)
                or identity.get("agent_id") in (agent_ids or set())
                or identity.get("memory_id") in (memory_ids or set())
                or identity.get("root_trace_id") in (root_trace_ids or set())
            )
            and (
                not thread_id
                or thread_id in (identity.get("thread_id"), identity.get("memory_id"))
            )
            and (not thread_memory_id or identity.get("memory_id") == thread_memory_id)
            and (
                application_id is None
                or identity.get("application_id") == application_id
            )
            and (user_id is _USER_UNSET or identity.get("user_id") == user_id)
        )

    for source, rows in (
        ("conversation", snapshot.conversation_rows),
        ("tool", snapshot.tool_rows),
        ("bundle", snapshot.bundle_rows),
    ):
        for row in rows:
            if not isinstance(row, dict):
                errors.append({"code": "invalid_source_row"})
                continue
            payload = read_payload(row)
            bundle = is_bundle(row, payload)
            identity = row_identity(row, payload if bundle else None)
            if not in_scope(identity):
                continue
            counts["bundle" if bundle else source] += 1
            row_id = str(row.get("_id") or row.get("id") or row.get("memory_id") or "")
            if bundle:
                if (
                    not payload
                    or payload.get("version", 1) not in (1, 2, 3)
                    or not isinstance(payload.get("events"), list)
                ):
                    errors.append({"code": "invalid_trace_bundle", "record_id": row_id})
                    continue
                children = payload["events"]
                if not children:
                    errors.append({"code": "empty_trace_bundle", "record_id": row_id})
                if payload.get("event_count", len(children)) != len(children):
                    errors.append(
                        {"code": "bundle_count_mismatch", "record_id": row_id}
                    )
            else:
                children = [row]
            for index, child in enumerate(children):
                if not isinstance(child, dict):
                    errors.append(
                        {
                            "code": "invalid_trace_event",
                            "record_id": row_id,
                            "child_index": index,
                        }
                    )
                    continue
                if any(
                    identity.get(field) is not None
                    and child.get(field, identity[field]) != identity[field]
                    for field in ("user_id", "application_id")
                ):
                    errors.append(
                        {
                            "code": "child_identity_mismatch",
                            "record_id": row_id,
                            "child_index": index,
                        }
                    )
                    continue
                if child.get("event_id") is not None and (
                    not isinstance(child["event_id"], str) or not child["event_id"]
                ):
                    errors.append(
                        {
                            "code": "invalid_event_id",
                            "record_id": row_id,
                            "child_index": index,
                        }
                    )
                    continue
                # Standalone legacy rows may use memory_id as their document
                # key and trace_memory_id as the associated memory. Preserve
                # row_identity's canonical association through child filtering.
                event = {**identity, **child} if bundle else {**child, **identity}
                event_identity = {key: event.get(key) for key in IDENTITY_FIELDS}
                if not in_scope(event_identity):
                    continue
                version = child.get(
                    "schema_version", payload.get("version", 1) if bundle else 1
                )
                if version not in (1, 2, 3):
                    errors.append(
                        {
                            "code": "unsupported_event_schema",
                            "record_id": row_id,
                            "child_index": index,
                        }
                    )
                    continue
                if version == 3:
                    try:
                        if not child.get("timestamp"):
                            raise ValueError("stored v3 events require a timestamp")
                        typed = TraceEventV3.model_validate({**identity, **child})
                        event = typed.model_dump(mode="json", exclude_none=True)
                        event.update(event.get("attributes") or {})
                    except ValueError:
                        errors.append(
                            {
                                "code": "invalid_v3_event",
                                "record_id": row_id,
                                "child_index": index,
                            }
                        )
                        continue
                kind = str(
                    child.get("event_kind")
                    or child.get("trace_kind")
                    or child.get("kind")
                    or (
                        "trace"
                        if bundle
                        else "execution_log"
                        if source == "tool"
                        else "conversation"
                    )
                )
                if child.get("phase") == "result" and kind in (
                    "model_call",
                    "tool_call",
                ):
                    kind = kind.replace("_call", "_result")
                event.update(
                    {
                        "kind": kind,
                        "role": child.get("role")
                        or ("tool" if bundle or source == "tool" else "system"),
                        "title": child.get("title")
                        or child.get("operation")
                        or (
                            "Conversation event"
                            if kind == "conversation"
                            else f"Execution Log · {child.get('tool_name', 'tool')}"
                            if source == "tool"
                            else kind
                        ),
                        "timestamp": timestamp(child.get("timestamp"))
                        or timestamp(row.get("timestamp"))
                        or timestamp((payload or {}).get("timestamp")),
                        "content": child.get("content")
                        or child.get("message")
                        or child.get("text")
                        or "",
                        "schema_version": version,
                    }
                )
                if source == "tool" and not bundle:
                    details = child.get("outcome_details") or {}
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except ValueError:
                            details = {}
                    details = details if isinstance(details, dict) else {}
                    event.update(
                        {
                            "content": json.dumps(
                                {
                                    key: child.get(key)
                                    for key in (
                                        "arguments",
                                        "result",
                                        "error",
                                        "success",
                                        "outcome",
                                        "outcome_details",
                                    )
                                },
                                default=str,
                            ),
                            "trace_id": child.get("tool_call_id") or "",
                            "logical_tool_name": child.get("tool_name"),
                            "outcome": child.get("outcome")
                            or (
                                "error" if child.get("success") is False else "success"
                            ),
                            "outcome_reason_code": details.get("reason_code"),
                            "fallback_provider": details.get("fallback_provider"),
                        }
                    )
                # v3 IDs are global within a tenant. Legacy IDs are scoped to
                # root/span/phase and timestamp so reused call IDs survive.
                fingerprint = [
                    event.get(key)
                    for key in (
                        "root_trace_id",
                        "span_id",
                        "tool_call_id",
                        "trace_id",
                        "phase",
                        "kind",
                        "operation",
                        "timestamp",
                        "agent_id",
                        "thread_id",
                        "memory_id",
                    )
                ]
                if not any(
                    event.get(key)
                    for key in ("root_trace_id", "span_id", "trace_id", "tool_call_id")
                ):
                    fingerprint += [row_id, index, event.get("content")]
                event_id = (
                    event.get("event_id")
                    or "legacy:"
                    + hashlib.sha256(
                        json.dumps(fingerprint, sort_keys=True, default=str).encode()
                    ).hexdigest()
                )
                seen_key = tuple(
                    event.get(key)
                    for key in (
                        "application_id",
                        "agent_id",
                        "user_id",
                        "thread_id",
                        "root_trace_id",
                        "turn_id",
                    )
                ) + (event_id,)
                if seen_key in seen:
                    duplicates += 1
                    continue
                seen.add(seen_key)
                if version != 3:
                    if event.get("task_id") is not None:
                        from .privacy import validate_opaque

                        try:
                            task = event["task_id"]
                            if not isinstance(task, str) or not task or len(task) > 240:
                                raise ValueError()
                            validate_opaque(task)
                        except ValueError:
                            event.pop("task_id", None)
                            errors.append(
                                {"code": "invalid_task_id", "record_id": row_id}
                            )
                    if "grounding_source_ids" in event:
                        try:
                            event["grounding_source_ids"] = source_ids(
                                event["grounding_source_ids"]
                            )
                        except (ValueError, TypeError):
                            event.pop("grounding_source_ids", None)
                            errors.append(
                                {"code": "invalid_source_ids", "record_id": row_id}
                            )
                    for field, model, maximum in (
                        ("selection_ledger", SelectionDecision, 64),
                        ("input_refs", ResourceRef, 32),
                        ("output_refs", ResourceRef, 32),
                    ):
                        if field in event:
                            try:
                                if (
                                    not isinstance(event[field], list)
                                    or len(event[field]) > maximum
                                ):
                                    raise ValueError()
                                event[field] = [
                                    model.model_validate(value).model_dump(
                                        mode="json", exclude_none=True
                                    )
                                    for value in event[field]
                                ]
                            except ValueError:
                                event.pop(field, None)
                                errors.append(
                                    {
                                        "code": "invalid_structured_metadata",
                                        "record_id": row_id,
                                        "field": field,
                                    }
                                )
                allowed = LEGACY_FIELDS
                if version == 3:
                    allowed = (
                        allowed
                        | set(TraceEventV3.model_fields)
                        | set(event.get("attributes") or {})
                    )
                event = {key: value for key, value in event.items() if key in allowed}
                event["event_id"] = event_id
                event["source_record_id"] = row_id or None
                event["source_store"] = "bundle" if bundle else source
                schemas[version] += 1
                events.append(event)
    for query_error in snapshot.query_metadata.get("query_errors", []):
        errors.append({"code": "provider_query_failed", "source": query_error})
    events.sort(key=lambda event: (event["timestamp"], event.get("event_id", "")))

    def result_key(event):
        correlation = str(event.get("tool_call_id") or event.get("trace_id") or "")
        for prefix in ("result:", "call:"):
            if correlation.startswith(prefix):
                correlation = correlation[len(prefix) :]
        return (
            (
                event.get("agent_id"),
                event.get("memory_id"),
                event.get("thread_id"),
                correlation,
            )
            if correlation
            else None
        )

    result_keys = {result_key(e) for e in events if e["kind"] == "tool_result"} - {None}
    unique_events = []
    for event in events:
        if event["kind"] == "execution_log" and result_key(event) in result_keys:
            duplicates += 1
            schemas[event["schema_version"]] -= 1
        else:
            unique_events.append(event)
    events = unique_events
    truncated = bool(snapshot.query_metadata.get("truncated")) or (
        limit > 0 and len(events) > limit
    )
    if limit > 0:
        events = events[-limit:]
    window = NormalizedTraceWindow(
        TraceEvents(events), dict(counts), dict(schemas), errors, duplicates, truncated
    )
    window.events.coverage = window.metadata
    return window


def query_trace_events(provider, memory_type, *, limit=250, cursor=None, **filters):
    """Bounded event pagination with a provider cursor plus child offset.

    A partially consumed bundle is pinned by its event fingerprint; mutation
    is reported as an invalid cursor rather than skipping or repeating spans.
    Coverage describes this page, not the full historical thread.
    """
    limit = max(1, min(int(limit), 1000))
    scope = hashlib.sha256(
        json.dumps([str(memory_type), filters], sort_keys=True, default=str).encode()
    ).hexdigest()
    event_filters = {
        key: filters.pop(key)
        for key in (
            "root_trace_id",
            "run_id",
            "turn_id",
            "event_kinds",
            "statuses",
            "resource_refs",
            "tool_name",
            "success",
            "start_time",
            "end_time",
            "query",
        )
        if key in filters
    }

    # Bundle write time can be much later than a child's event time. Apply
    # temporal predicates to children, not to the enclosing storage document.
    for key in ("start_time", "end_time"):
        if event_filters.get(key) is not None:
            parsed = timestamp(event_filters[key])
            if not parsed:
                raise ValueError(f"invalid {key}")
            event_filters[key] = parsed
    if (
        event_filters.get("start_time")
        and event_filters.get("end_time")
        and event_filters["start_time"] > event_filters["end_time"]
    ):
        raise ValueError("start_time must not be after end_time")

    def selected(event):
        if event_filters.get("query"):
            value = event_filters["query"]
            identifiers = [
                event.get(key)
                for key in (
                    "agent_id",
                    "user_id",
                    "thread_id",
                    "root_trace_id",
                    "run_id",
                    "turn_id",
                    "job_ref",
                    "error_code",
                )
            ]
            identifiers += [
                ref.get("ref")
                for field in ("input_refs", "output_refs")
                for ref in event.get(field, [])
            ]
            identifiers += event.get("grounding_source_ids", [])
            identifiers += [
                entry["resource"]["ref"] for entry in event.get("selection_ledger", [])
            ]
            if value not in identifiers:
                return False
        observed_at = timestamp(event.get("timestamp"))
        if event_filters.get("start_time") and (
            not observed_at or observed_at < event_filters["start_time"]
        ):
            return False
        if event_filters.get("end_time") and (
            not observed_at or observed_at > event_filters["end_time"]
        ):
            return False
        for key in ("root_trace_id", "run_id", "turn_id"):
            if (
                event_filters.get(key) is not None
                and event.get(key) != event_filters[key]
            ):
                return False
        if (
            event_filters.get("event_kinds")
            and event.get("kind") not in event_filters["event_kinds"]
        ):
            return False
        if (
            event_filters.get("statuses")
            and event.get("status") not in event_filters["statuses"]
        ):
            return False
        if event_filters.get("tool_name") and event_filters["tool_name"] not in (
            event.get("tool_name"),
            event.get("logical_tool_name"),
        ):
            return False
        if event_filters.get("success") is not None:
            observed_success = event.get("success")
            if type(observed_success) is not bool:
                observed_success = {
                    "success": True,
                    "error": False,
                    "failed": False,
                }.get(event.get("status"))
            if observed_success is not event_filters["success"]:
                return False
        if event_filters.get("resource_refs"):
            refs = {
                ref.get("ref")
                for ref in [
                    *(event.get("input_refs") or []),
                    *(event.get("output_refs") or []),
                    *(entry["resource"] for entry in event.get("selection_ledger", [])),
                ]
            }
            refs.update(event.get("grounding_source_ids", []))
            if not refs.intersection(event_filters["resource_refs"]):
                return False
        return True

    position = {"cursor": None, "offset": 0}
    if cursor:
        try:
            if len(cursor) > 8192:
                raise ValueError()
            position = json.loads(
                base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            )
            if (
                not isinstance(position, dict)
                or position.get("scope") != scope
                or type(position.get("offset")) is not int
                or position["offset"] < 0
            ):
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            raise ValueError("invalid event cursor") from None
    events, errors = [], []
    counts = Counter()
    duplicate_count = 0
    scanned = 0
    next_position = None
    while len(events) < limit and scanned < 1000:
        page = provider.query_observability_records(
            memory_type, limit=1, cursor=position.get("cursor"), **filters
        )
        rows = page.get("items") or []
        if not rows:
            break
        scanned += 1
        source = (
            "tool_rows"
            if str(getattr(memory_type, "value", memory_type)) == "tool_log"
            else "conversation_rows"
        )
        window = normalize_trace_snapshot(
            TraceSnapshot(**{source: rows}),
            agent_ids=filters.get("agent_ids"),
            memory_ids=filters.get("memory_ids"),
            thread_id=filters.get("thread_id"),
            **{
                key: filters[key]
                for key in ("application_id", "user_id")
                if key in filters
            },
        )
        children = [event for event in window.events if selected(event)]
        checksum = hashlib.sha256(
            json.dumps([e["event_id"] for e in children]).encode()
        ).hexdigest()
        if position.get("checksum") and position["checksum"] != checksum:
            raise ValueError("bundle changed while paginating; restart the event query")
        offset = position["offset"]
        if offset > len(children):
            raise ValueError("invalid event cursor offset")
        counts.update(window.source_counts)
        errors.extend(window.normalization_errors)
        duplicate_count += window.duplicate_count
        take = min(limit - len(events), len(children) - offset)
        events.extend(children[offset : offset + take])
        if offset + take < len(children):
            next_position = {
                "cursor": position.get("cursor"),
                "offset": offset + take,
                "checksum": checksum,
            }
            break
        if not page.get("next_cursor"):
            next_position = None
            break
        position = {"cursor": page["next_cursor"], "offset": 0}
        next_position = position
    next_cursor = None
    if next_position:
        next_cursor = (
            base64.urlsafe_b64encode(
                json.dumps({**next_position, "scope": scope}).encode()
            )
            .decode()
            .rstrip("=")
        )
    return {
        "items": events,
        "source_rows": sum(counts.values()),
        "bundle_count": counts.get("bundle", 0),
        "normalized_events": len(events),
        "normalization_errors": len(errors),
        "errors": errors,
        "duplicates_removed": duplicate_count,
        "coverage": "untrusted" if errors else "partial" if next_cursor else "complete",
        "read_completeness": "untrusted"
        if errors
        else "partial"
        if next_cursor
        else "complete",
        "coverage_scope": "stored_event_page",
        "next_cursor": next_cursor,
        "truncated": bool(next_cursor),
        "window_complete": not cursor and not next_cursor and not errors,
        "limit": limit,
    }
