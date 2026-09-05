"""Expected evidence profiles, isolated by tenant, turn and task."""

import re

from .evidence import evidence_groups, task_key

PROFILES = {
    "chat_response": {"model_result"},
    "content_ingestion": {"context_binding", "tool_result", "verified_outcome"},
    "content_derived_artifact": {
        "intent_plan",
        "memory_supply",
        "artifact_persisted",
        "ui_delivery",
        "verified_outcome",
    },
    "interactive_response": {"model_result", "output_contract", "ui_delivery"},
}
OPTIONAL_STAGES = {}


def register_coverage_profile(name, *, required, optional=(), replace=False):
    """Register application-specific stage expectations in the current process."""
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
        raise ValueError("Profile names must be bounded lowercase identifiers")
    if name in PROFILES and not replace:
        raise ValueError("Coverage profile already exists")
    required, optional = set(required), set(optional)
    if (
        not required
        or len(required | optional) > 32
        or any(
            not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", kind)
            for kind in required | optional
        )
    ):
        raise ValueError("Coverage stages must be nonempty bounded kind names")
    if required & optional:
        raise ValueError("Required and optional stages must be disjoint")
    PROFILES[name], OPTIONAL_STAGES[name] = required, optional


def trace_coverage(events, metadata=None):
    metadata = dict(
        metadata if metadata is not None else getattr(events, "coverage", {})
    )
    events = list(events)
    roots = {}
    for scope, rows in evidence_groups(events).items():
        for event in rows:
            key = (*scope, task_key(event, rows))
            entry = roots.setdefault(key, {"observed": set(), "profiles": set()})
            kind = event.get("kind")
            entry["observed"].add(
                {
                    "memory_context": "memory_supply",
                    "context_provenance": "context_binding",
                }.get(kind, kind)
            )
            if event.get("coverage_profile"):
                entry["profiles"].add(event["coverage_profile"])
    missing = []
    for key, entry in roots.items():
        for profile in sorted(entry["profiles"]):
            for kind in sorted(
                PROFILES.get(profile, {"unknown_coverage_profile"}) - entry["observed"]
            ):
                missing.append(
                    {
                        "agent_id": key[1],
                        "root_trace_id": key[4],
                        "turn_id": key[5],
                        "task_id": key[6],
                        "profile": profile,
                        "stage": kind,
                    }
                )
    read = metadata.get("read_completeness", metadata.get("coverage", "unknown"))
    if metadata.get("normalization_errors") or read == "untrusted":
        read = "untrusted"
    elif metadata.get("truncated") or metadata.get("window_complete") is False:
        read = "partial"
    profiles = sorted({p for e in roots.values() for p in e["profiles"]})
    instrumentation = (
        "partial"
        if missing
        else "complete"
        if roots and all(e["profiles"] for e in roots.values())
        else "unknown"
    )
    status = (
        "untrusted"
        if read == "untrusted"
        else "partial"
        if read == "partial" or instrumentation == "partial"
        else "complete"
        if read == instrumentation == "complete"
        else "unknown"
    )
    headline = (
        "All stored events loaded; end-to-end coverage unknown"
        if read == "complete" and instrumentation == "unknown"
        else "Stored-event read incomplete or untrusted; end-to-end coverage not established"
        if read != "complete"
        else "All stored events loaded; required instrumentation stages missing"
        if instrumentation == "partial"
        else "All stored events loaded; declared stages observed (not proof of task success)"
    )
    return {
        **metadata,
        "coverage": status,
        "read_completeness": read,
        "instrumentation_coverage": instrumentation,
        "outcome_verification": "recorded"
        if any(
            e.get("kind") == "verified_outcome" and e.get("verified") is True
            for e in events
        )
        else "unknown",
        "headline": headline,
        "missing_stages": missing,
        "observed_stages": [
            {
                "agent_id": key[1],
                "root_trace_id": key[4],
                "turn_id": key[5],
                "task_id": key[6],
                "stages": sorted(str(stage) for stage in entry["observed"] if stage),
            }
            for key, entry in roots.items()
        ],
        "optional_stages": {
            profile: sorted(OPTIONAL_STAGES.get(profile, set()))
            for entry in roots.values()
            for profile in entry["profiles"]
        },
        "instrumentation_versions": sorted(
            {
                str(
                    event.get("instrumentation_version")
                    or (event.get("attributes") or {}).get("instrumentation_version")
                )
                for event in events
                if event.get("instrumentation_version")
                or (event.get("attributes") or {}).get("instrumentation_version")
            }
        ),
        "profiles_declared": profiles,
        "scope": "declared_stages"
        if any(e["profiles"] for e in roots.values())
        else "loaded_events",
    }
