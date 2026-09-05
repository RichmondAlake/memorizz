"""Shared content-free evidence helpers. Correlation is not authorization."""

from collections import defaultdict

from .normalization import timestamp
from .references import source_ids


def event_scope(event):
    return tuple(
        event.get(key) or ""
        for key in (
            "application_id",
            "agent_id",
            "user_id",
            "thread_id",
            "root_trace_id",
            "turn_id",
        )
    )


def evidence_rows(events):
    rows = []
    for event in events:
        if not isinstance(event, dict):
            continue
        # Identity cannot be replaced by free-form legacy attributes.
        row = {**(event.get("attributes") or {}), **event}
        if "grounding_source_ids" in row:
            row["grounding_source_ids"] = source_ids(row["grounding_source_ids"])
        kind = row.get("kind") or row.get("event_kind") or row.get("trace_kind")
        if row.get("phase") == "result" and kind in {"model_call", "tool_call"}:
            kind = kind.replace("_call", "_result")
        row["kind"] = kind
        rows.append(row)
    return sorted(rows, key=lambda row: timestamp(row.get("timestamp")))


def evidence_groups(events):
    groups = defaultdict(list)
    for row in evidence_rows(events):
        groups[event_scope(row)].append(row)
    return groups


def task_key(event, rows):
    """Unlabelled legacy evidence belongs only to an unambiguous task."""
    if event.get("task_id"):
        return event["task_id"]
    tasks = {row.get("task_id") for row in rows if row.get("kind") == "intent_plan"}
    return next(iter(tasks)) if len(tasks) == 1 else None


def resource_key(ref, *, version=False):
    key = (ref.get("resource_type"), ref.get("ref"))
    return key + (ref.get("version"),) if version else key


def latest_artifacts(rows):
    """Latest state per artifact, including failed updates and tombstones."""
    current = {}
    for event in rows:
        if event.get("kind") != "artifact_persisted":
            continue
        for ref in event.get("output_refs", []):
            current[resource_key(ref)] = {**event, "output_refs": [ref]}
    return list(current.values())


def latest_named(rows, kind):
    return {
        (task_key(row, rows), row.get("contract_name")): row
        for row in rows
        if row.get("kind") == kind
    }


def is_persisted(event):
    return event.get("persistence_verified") is True and event.get(
        "artifact_status"
    ) in {"created", "updated"}


def is_delivered(event):
    return event.get("emitted") is True and event.get("acknowledged") is True
