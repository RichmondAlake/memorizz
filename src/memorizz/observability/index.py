"""Content-free span index contract, keyset cursors and rollout selection.

The immutable source bundles remain the rollback path. Index provisioning and
backfill are explicit operations; opening the UI never creates database objects.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone

from .models import ATTRIBUTE_FIELDS, IDENTITY_FIELDS, ResourceRef, SelectionDecision
from .normalization import (
    LEGACY_FIELDS,
    TraceSnapshot,
    normalize_trace_snapshot,
    timestamp,
)
from .privacy import validate_opaque

EVENT_SCOPE = (
    "application_id",
    "agent_id",
    "user_id",
    "thread_id",
    "root_trace_id",
    "turn_id",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def resource_hash(value):
    return digest(["resource", value])


def enabled(name):
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def read_path():
    value = os.getenv("MEMORIZZ_OBSERVABILITY_READ_PATH", "bundles").lower()
    if value not in {"bundles", "index"}:
        raise ValueError("MEMORIZZ_OBSERVABILITY_READ_PATH must be bundles or index")
    return value


def _opaque_fields(event):
    """Project indexed metadata without content or arbitrary nested payloads."""
    allowed = (
        LEGACY_FIELDS
        | ATTRIBUTE_FIELDS
        | set(IDENTITY_FIELDS)
        | {
            "event_id",
            "kind",
            "event_kind",
            "phase",
            "operation",
            "span_id",
            "parent_span_id",
            "caused_by_event_id",
            "timestamp",
            "schema_version",
            "status",
            "duration_ms",
            "component",
            "deployment_id",
            "worker_id",
        }
    )
    result = {}
    for key in allowed - {"content", "title", "arguments", "result", "message", "text"}:
        value = event.get(key)
        if value is None:
            continue
        values = value[:32] if isinstance(value, list) else [value]
        if any(
            type(item) not in (str, int, float, bool, type(None)) for item in values
        ):
            continue
        if any(isinstance(item, float) and not math.isfinite(item) for item in values):
            continue
        try:
            for item in values:
                validate_opaque(item)
            result[key] = (
                [str(v)[:240] if isinstance(v, str) else v for v in values]
                if isinstance(value, list)
                else value[:240]
                if isinstance(value, str)
                else value
            )
        except ValueError:
            continue
    for direction in ("input_refs", "output_refs"):
        refs = []
        for ref in (event.get(direction) or [])[:32]:
            try:
                refs.append(
                    ResourceRef.model_validate(ref).model_dump(
                        mode="json", exclude_none=True
                    )
                )
            except (TypeError, ValueError):
                continue
        result[direction] = refs
    result["timestamp"] = timestamp(event.get("timestamp"))
    if event.get("selection_ledger"):
        result["selection_ledger"] = [
            SelectionDecision.model_validate(entry).model_dump(
                mode="json", exclude_none=True
            )
            for entry in event["selection_ledger"][:64]
        ]
    return result


def prepare_bundle(payload):
    row = {**payload, "_id": payload["record_id"], "content": json.dumps(payload)}
    window = normalize_trace_snapshot(TraceSnapshot(bundle_rows=[row]))
    if window.normalization_errors:
        raise ValueError("Cannot index a bundle with normalization errors")
    bundle_key = digest([payload.get(k) for k in EVENT_SCOPE] + [payload["record_id"]])
    records = []
    for event in window.events:
        metadata = _opaque_fields(event)
        key = digest([event.get(k) for k in EVENT_SCOPE] + [event["event_id"]])
        # A projection cannot silently lose identity that the index filters on.
        for field in (
            "application_id",
            "user_id",
            "agent_id",
            "memory_id",
            "thread_id",
            "root_trace_id",
            "event_id",
        ):
            if event.get(field) and metadata.get(field) != event[field]:
                raise ValueError("Trace identity is not safe for indexing")
        refs = {
            resource_hash(ref["ref"])
            for side in ("input_refs", "output_refs")
            for ref in metadata.get(side, [])
        }
        refs.update(
            resource_hash(ref) for ref in event.get("grounding_source_ids", [])[:32]
        )
        refs.update(
            resource_hash(entry["resource"]["ref"])
            for entry in metadata.get("selection_ledger", [])
        )
        records.append(
            {
                "key": key,
                "bundle_key": bundle_key,
                "source_record_id": payload["record_id"],
                "user_key": digest(event.get("user_id")),
                "timestamp": metadata.get("timestamp") or "0001-01-01T00:00:00.000000Z",
                "metadata": metadata,
                "preview": str(event.get("content") or "")[:16000],
                "resource_hashes": sorted(refs),
            }
        )
    summary = {
        **{key: payload.get(key) for key in IDENTITY_FIELDS},
        "bundle_key": bundle_key,
        "source_record_id": payload["record_id"],
        "timestamp": max(
            (r["timestamp"] for r in records),
            default=timestamp(payload.get("timestamp"))
            or "0001-01-01T00:00:00.000000Z",
        ),
        "event_count": len(records),
        "fingerprint": digest([r["metadata"] for r in records]),
        "resource_ref_hashes": sorted(
            {h for row in records for h in row["resource_hashes"]}
        ),
    }
    return summary, records


FILTERS = frozenset(
    {
        "agent_ids",
        "memory_ids",
        "application_id",
        "thread_id",
        "user_id",
        "root_trace_id",
        "run_id",
        "turn_id",
        "event_kinds",
        "statuses",
        "resource_refs",
        "start_time",
        "end_time",
        "tool_name",
        "success",
        "query",
    }
)


def validate_filters(filters):
    filters = {k: v for k, v in filters.items() if k != "record_type"}
    if set(filters) - FILTERS:
        raise ValueError("Unsupported indexed trace filter")
    for key, value in filters.items():
        if key == "success":
            if value is not None and type(value) is not bool:
                raise ValueError("success must be a boolean")
            continue
        if key in {"start_time", "end_time"}:
            if value is not None:
                filters[key] = timestamp(value)
                if not filters[key]:
                    raise ValueError(f"Invalid {key}")
            continue
        if key in {
            "agent_ids",
            "memory_ids",
            "event_kinds",
            "statuses",
            "resource_refs",
        }:
            if value is not None and (
                not isinstance(value, (list, tuple, set)) or len(value) > 64
            ):
                raise ValueError("Trace filter lists must contain at most 64 values")
            values = sorted(value or [])
            filters[key] = values
        else:
            values = [value]
        for item in values:
            if item is not None and (not isinstance(item, str) or len(item) > 240):
                raise ValueError("Trace filters require bounded opaque identifiers")
            validate_opaque(item)
    if (
        filters.get("start_time")
        and filters.get("end_time")
        and filters["start_time"] > filters["end_time"]
    ):
        raise ValueError("start_time must not be after end_time")
    return filters


def decode_cursor(cursor, filters):
    if not cursor:
        return None
    try:
        if len(cursor) > 4096:
            raise ValueError()
        value = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if (
            value.get("scope") != digest(filters)
            or value.get("v") != 1
            or len(value["last"]) != 2
            or not all(isinstance(v, str) for v in value["last"])
        ):
            raise ValueError()
        return value["last"]
    except Exception:
        raise ValueError(
            "Invalid indexed trace cursor or changed query scope"
        ) from None


def event_page(records, *, filters, limit, cursor, started):
    more = len(records) > limit
    selected = records[:limit]
    next_cursor = None
    if more:
        last = selected[-1]
        value = {
            "v": 1,
            "scope": digest(filters),
            "last": [last["timestamp"], last["key"]],
        }
        next_cursor = (
            base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
        )
    return {
        "items": [
            {**row["metadata"], "source_bundle_id": row["bundle_key"]}
            for row in selected
        ],
        "next_cursor": next_cursor,
        "truncated": more,
        "window_complete": not cursor and not more,
        "coverage": "partial" if more else "complete",
        "read_completeness": "partial" if more else "complete",
        "coverage_scope": "stored_event_page",
        "normalization_errors": 0,
        "errors": [],
        "duplicates_removed": 0,
        "normalized_events": len(selected),
        "bundle_count": len({row["bundle_key"] for row in selected}),
        "source_rows": len(selected),
        "limit": limit,
        "provider_native": True,
        "read_path": "index",
        "scanned_count": len(records),
        "query_duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def indexed_bundle_rows(page):
    """Rehydrate bounded metadata pages for the shared UI normalizer."""
    from .models import TraceEventV3

    groups = {}
    for event in page.get("items", []):
        bundle_id = event.get("source_bundle_id") or event["event_id"]
        row = groups.setdefault(
            bundle_id,
            {
                "_id": bundle_id,
                "record_type": "observability_trace_bundle",
                **{key: event.get(key) for key in IDENTITY_FIELDS},
                "content": {"type": "trace_bundle", "version": 2, "events": []},
            },
        )
        if event.get("schema_version") == 3:
            child = {
                key: value
                for key, value in event.items()
                if key in TraceEventV3.model_fields
            }
            child["attributes"] = {
                key: value for key, value in event.items() if key in ATTRIBUTE_FIELDS
            }
        else:
            child = {
                **event,
                "trace_kind": event.get("kind"),
                "schema_version": event.get("schema_version", 1),
            }
        row["content"]["events"].append(child)
    return list(groups.values())


def metadata_envelope(payload):
    """Remove raw previews from new dual-written source envelopes as well.

    Raw content has one copy in the separately expirable preview table. Legacy
    bundles are never rewritten by a backfill or read operation.
    """
    _, rows = prepare_bundle(payload)
    groups = indexed_bundle_rows(
        {
            "items": [
                {**row["metadata"], "source_bundle_id": row["bundle_key"]}
                for row in rows
            ]
        }
    )
    return {
        **payload,
        "events": [event for group in groups for event in group["content"]["events"]],
        "preview_storage": "separate",
    }


class SpanIndex:
    """Additive private index, separate from semantic/conversation recall."""

    provider_name = "unknown"

    def capabilities(self):
        return {
            "provider": self.provider_name,
            "span_index": True,
            "compound_cursor": True,
            "resource_filters": True,
            "summary_aggregation": True,
            "retention": True,
            "schema_version": 1,
            "ready": self.ready(),
        }

    def write_bundle(self, payload):
        if not hasattr(self, "_failed_bundles"):
            self._failed_bundles = set()
        source_id = payload["record_id"]
        self._failed_bundles.add(source_id)
        summary, rows = prepare_bundle(payload)
        self.put(summary, rows)
        self._failed_bundles.discard(source_id)
        return {"bundle_count": 1, "event_count": len(rows)}

    def query(self, *, limit=250, cursor=None, **filters):
        started = time.perf_counter()
        filters = validate_filters(filters)
        boundary = decode_cursor(cursor, filters)
        limit = max(1, min(int(limit), 1000))
        records = self.select(filters, boundary=boundary, limit=limit + 1)
        page = event_page(
            records, filters=filters, limit=limit, cursor=cursor, started=started
        )
        if getattr(self, "_failed_bundles", None) or self.pending_count():
            page.update(
                coverage="untrusted",
                window_complete=False,
                normalization_errors=1,
                errors=[{"code": "index_write_incomplete"}],
            )
        return page

    def pending_count(self):
        return 0

    def retention_cutoffs(
        self, *, previews_days=7, metadata_days=30, outcomes_days=365, now=None
    ):
        from datetime import timedelta

        if any(
            type(v) is not int or v < 1
            for v in (previews_days, metadata_days, outcomes_days)
        ):
            raise ValueError("Retention periods must be positive whole days")
        if previews_days > metadata_days or metadata_days > outcomes_days:
            raise ValueError("Retention must satisfy previews <= metadata <= outcomes")
        now = now or datetime.now(timezone.utc)
        return {
            name: timestamp(now - timedelta(days=days))
            for name, days in (
                ("previews", previews_days),
                ("metadata", metadata_days),
                ("outcomes", outcomes_days),
            )
        }
