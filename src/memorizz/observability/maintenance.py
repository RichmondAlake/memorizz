"""Explicit, dry-run-first rollout and retention workflows for trace indexes."""

from __future__ import annotations

from datetime import datetime, timezone

from ..enums import MemoryType
from .index import _opaque_fields, digest, prepare_bundle
from .normalization import read_payload, timestamp


class ObservabilityMaintenance:
    def __init__(self, provider):
        self.provider = provider
        self.index = provider.get_observability_index()
        if self.index is None:
            raise NotImplementedError("Provider has no native observability index")

    def initialize(self):
        """Explicitly provision additive index tables/collections, never on reads."""
        if getattr(self.provider, "read_only", False):
            raise PermissionError(
                "Index provisioning is disabled for read-only providers"
            )
        return self.index.initialize()

    def backfill(self, *, since, limit=100, cursor=None, dry_run=True):
        """Index one recent source page; original bundles are never rewritten."""
        since = timestamp(since)
        if not since:
            raise ValueError("An explicit valid since timestamp is required")
        if not dry_run and getattr(self.provider, "read_only", False):
            raise PermissionError("Backfill is disabled for read-only providers")
        page = self.provider.query_observability_records(
            MemoryType.SHARED_MEMORY,
            record_type="observability_trace_bundle",
            start_time=since,
            limit=max(1, min(int(limit), 1000)),
            cursor=cursor,
        )
        counts = {"bundles": 0, "events": 0, "failed": 0}
        errors = []
        for row in page.get("items", []):
            payload = read_payload(row)
            try:
                if not payload:
                    raise ValueError("invalid bundle")
                payload.setdefault("record_id", row.get("_id") or row.get("memory_id"))
                summary, records = prepare_bundle(payload)
                if not dry_run:
                    self.index.write_bundle(payload)
                counts["bundles"] += 1
                counts["events"] += len(records)
            except Exception as exc:
                counts["failed"] += 1
                errors.append({"code": type(exc).__name__})
        return {
            **counts,
            "dry_run": dry_run,
            "errors": errors,
            "next_cursor": page.get("next_cursor"),
            "source_bundles_unchanged": True,
        }

    def parity(self, *, max_pages=100, **filters):
        """Compare exact content-free event projections before index cutover."""

        def collect(path):
            rows, cursor, complete, errors = {}, None, False, 0
            for _ in range(max(1, min(int(max_pages), 1000))):
                page = self.provider.query_trace_events(
                    read_path=path, limit=1000, cursor=cursor, **filters
                )
                for event in page["items"]:
                    projection = _opaque_fields(event)
                    # Different tenants may deliberately share external event IDs.
                    key = digest(
                        [
                            event.get(k)
                            for k in (
                                "application_id",
                                "agent_id",
                                "user_id",
                                "thread_id",
                                "root_trace_id",
                                "turn_id",
                                "event_id",
                            )
                        ]
                    )
                    rows[key] = digest(projection)
                errors += page.get("normalization_errors", 0)
                cursor = page.get("next_cursor")
                if not cursor:
                    complete = True
                    break
            return rows, complete and not errors

        source, source_complete = collect("bundles")
        indexed, indexed_complete = collect("index")
        missing = len(set(source) - set(indexed))
        unexpected = len(set(indexed) - set(source))
        mismatched = sum(
            source[key] != indexed[key] for key in source.keys() & indexed.keys()
        )
        return {
            "passed": source_complete
            and indexed_complete
            and not (missing or unexpected or mismatched),
            "source_events": len(source),
            "indexed_events": len(indexed),
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
            "complete": source_complete and indexed_complete,
            "rollback": "Set MEMORIZZ_OBSERVABILITY_READ_PATH=bundles; source bundles are unchanged by backfill",
        }

    def retention(self, *, dry_run=True, **policy):
        """Purge only indexed telemetry. Canonical-source expiry is a separate plan."""
        if not dry_run and getattr(self.provider, "read_only", False):
            raise PermissionError("Retention is disabled for read-only providers")
        return self.index.retention(dry_run=dry_run, **policy)

    def plan_source_expiry(self, *, before, limit=100, cursor=None):
        """Freeze exact expired bundle IDs. Caller must retain an archival copy.

        Legacy immutable bundles may contain raw previews. Expire those envelopes
        after indexing; do not rewrite their historical evidence in place.
        """
        before = timestamp(before)
        if not before or before >= timestamp(datetime.now(timezone.utc)):
            raise ValueError("before must be an explicit past timestamp")
        page = self.provider.query_observability_records(
            MemoryType.SHARED_MEMORY,
            record_type="observability_trace_bundle",
            end_time=before,
            limit=max(1, min(int(limit), 1000)),
            cursor=cursor,
        )
        entries = []
        for row in page.get("items", []):
            payload = read_payload(row)
            if payload and payload.get("record_type") == "observability_trace_bundle":
                entries.append(
                    {
                        "record_id": payload.get("record_id")
                        or str(row.get("_id") or row.get("memory_id")),
                        "fingerprint": digest(payload),
                    }
                )
        plan = {
            "before": before,
            "entries": entries,
            "next_cursor": page.get("next_cursor"),
        }
        return {**plan, "confirmation": digest(plan), "dry_run": True}

    def apply_source_expiry(self, plan, *, confirmation, archive):
        """Archive then delete exactly the frozen, unchanged source envelopes.

        An explicit archival callback and confirmation are required. The caller
        owns archive retention/access. No shared-memory data outside the plan
        can be removed. Rerun the plan if records changed between review/apply.
        """
        from .store import ObservabilityStore

        if getattr(self.provider, "read_only", False):
            raise PermissionError("Source expiry is disabled for read-only providers")
        expected = digest({k: plan[k] for k in ("before", "entries", "next_cursor")})
        if (
            confirmation != expected
            or plan.get("confirmation") != expected
            or not callable(archive)
        ):
            raise ValueError(
                "Reviewed plan confirmation and an archival callback are required"
            )
        store = ObservabilityStore(self.provider)
        targets = []
        for entry in plan["entries"]:
            payload = store._get(entry["record_id"])
            if (
                not payload
                or payload.get("record_type") != "observability_trace_bundle"
                or digest(payload) != entry["fingerprint"]
            ):
                raise ValueError("Source bundle changed; create a new expiry plan")
            targets.append(payload)
        for payload in targets:
            # The archive must explicitly acknowledge durable custody.
            if archive(payload) is not True:
                raise RuntimeError(
                    "Archive did not acknowledge persistence; source retained"
                )
        removed = 0
        for payload in targets:
            # Revalidate after archival callbacks before performing each delete.
            if digest(store._get(payload["record_id"])) != digest(payload):
                raise ValueError("Source changed after archival; source retained")
            if not self.provider.delete_observability_bundle(
                payload["record_id"], digest(payload)
            ):
                raise ValueError(
                    "Source changed before deletion; unchanged source retained"
                )
            removed += 1
        return {"removed": removed, "recoverable_from_archive": True}
