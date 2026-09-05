"""Deterministic lineage and application-boundary diagnostics; no raw content."""

from collections import defaultdict

from .evidence import (
    evidence_groups,
    is_persisted,
    latest_artifacts,
    latest_named,
    resource_key,
    task_key,
)


def diagnose_trace(events, *, metadata=None):
    metadata = metadata if metadata is not None else getattr(events, "coverage", {})
    incomplete = (
        metadata.get("coverage", "complete") != "complete"
        or metadata.get("truncated")
        or metadata.get("normalization_errors")
    )
    findings = defaultdict(list)

    def add(code, event, *, absence=False):
        findings[code].append((event.get("event_id"), absence))

    for rows in evidence_groups(events).values():
        producers = {}
        for row in rows:
            if row.get("kind") in {
                "model_call",
                "model_result",
                "tool_call",
                "tool_result",
                "application_action",
            } and row.get("span_id"):
                previous = producers.get(row["span_id"])
                if not previous or row.get("phase") != "start":
                    producers[row["span_id"]] = row
        expected = defaultdict(set)
        for event in rows:
            if event.get("kind") not in {
                "intent_plan",
                "context_binding",
                "memory_retrieval",
                "memory_selection",
                "memory_supply",
                "context_provenance",
                "memory_context",
            }:
                continue
            key = task_key(event, rows)
            for ref in event.get("input_refs", []):
                if ref.get("role") == "authoritative_current_source":
                    expected[key].add(resource_key(ref, version=True))
            for source_id in event.get("grounding_source_ids", []):
                expected[key].add((None, source_id, event.get("content_version")))

        artifacts = latest_artifacts(rows)
        for event in artifacts:
            if event.get("artifact_status") == "deleted":
                continue
            if not is_persisted(event):
                add("artifact_persistence_unverified", event)
            references = [*event.get("input_refs", []), *event.get("output_refs", [])]
            if any(ref.get("ownership_verified") is False for ref in references):
                add("artifact_ownership_unverified", event)
            if any(ref.get("provenance_status") == "missing" for ref in references):
                add("artifact_provenance_invalid", event)
            expected_sources = expected[task_key(event, rows)] or expected[None]
            sources = {
                resource_key(r, version=True) for r in event.get("input_refs", [])
            }
            if not event.get("source_independent"):
                if not sources:
                    add("artifact_missing_source_provenance", event)
                if (
                    expected_sources
                    and sources
                    and not all(
                        any(
                            a[1] == b[1]
                            and (a[0] is None or a[0] == b[0])
                            and (a[2] is None or a[2] == b[2])
                            for b in sources
                        )
                        for a in expected_sources
                    )
                ):
                    add("artifact_source_mismatch", event)
            producer = producers.get(event.get("parent_span_id"))
            if producer is None:
                add("artifact_without_producer_span", event, absence=True)
            elif producer.get("status") == "error":
                add("artifact_persisted_after_failed_action", event)
            elif (
                producer.get("status") != "success" or producer.get("phase") == "start"
            ):
                add("artifact_producer_incomplete", event, absence=True)

        contracts = latest_named(rows, "output_contract")
        deliveries = latest_named(rows, "ui_delivery")
        required = set()
        for index, event in enumerate(rows):
            if event.get("kind") == "trace_data_quality" and event.get(
                "context_missing"
            ):
                add("trace_context_missing", event)
            task = task_key(event, rows)
            if event.get("kind") == "intent_plan":
                for name in event.get("required_contracts", []):
                    required.add((task, name))
                    if (task, name) not in contracts:
                        add("required_contract_missing", event, absence=True)
                for artifact_type in event.get("expected_artifact_types", []):
                    if not any(
                        is_persisted(a)
                        and task_key(a, rows) == task
                        and any(
                            r.get("resource_type") == artifact_type
                            for r in a.get("output_refs", [])
                        )
                        for a in artifacts
                    ):
                        add("requested_artifact_missing", event, absence=True)
            if (
                event.get("kind") == "application_action"
                and event.get("phase") != "start"
            ):
                if event.get("side_effect") and not any(
                    e.get("kind") in {"intent_plan", "turn_start"}
                    and task_key(e, rows) == task
                    for e in rows
                ):
                    add("orphan_application_action", event, absence=True)
                if event.get("credit_decision") in {
                    "charged",
                    "charged_after_persistence",
                }:
                    if not any(
                        is_persisted(a) and task_key(a, rows) == task
                        for a in latest_artifacts(rows[: index + 1])
                    ):
                        add("credit_before_verified_persistence", event, absence=True)
        for key, event in contracts.items():
            if event.get("required"):
                required.add(key)
            if key in required and event.get("parser_status") == "missing":
                add("required_contract_missing", event)
            if (
                event.get("opening_marker_found")
                and event.get("closing_marker_found") is False
            ):
                add("output_contract_truncated", event)
            if event.get("parser_status") == "failed":
                add("contract_parse_failed", event)
            if event.get("recovery_used") or event.get("parser_status") == "partial":
                add("contract_recovered_partial", event)
            if key in required and not (
                event.get("opening_marker_found") is True
                and event.get("closing_marker_found") is True
                and event.get("parser_status") == "success"
            ):
                add("required_contract_incomplete", event)
            delivery = deliveries.get(key)
            if key in required and not (
                delivery
                and delivery.get("emitted")
                and rows.index(delivery) > rows.index(event)
            ):
                add("contract_not_emitted", event, absence=True)
        for event in deliveries.values():
            if event.get("emitted") and event.get("acknowledged") is not True:
                add("ui_delivery_unconfirmed", event, absence=True)
            if event.get("emitted") is False:
                add("contract_not_emitted", event)

    return [
        {
            "id": key,
            "evidence_count": len(evidence),
            "evidence_event_ids": list(dict.fromkeys(e[0] for e in evidence if e[0]))[
                :32
            ],
            "priority": "P1"
            if key
            in {
                "credit_before_verified_persistence",
                "orphan_application_action",
                "ui_delivery_unconfirmed",
                "contract_recovered_partial",
            }
            else "P0",
            "component": "delivery"
            if "contract" in key or "delivery" in key
            else "artifact",
            "title": key.replace("_", " ").capitalize(),
            "finding": f"Observed {len(evidence)} lineage or delivery evidence gap(s) in the selected trace window.",
            "recommendation": "Inspect the producing action, authoritative input references, parser result and delivery evidence for this trace.",
            "target": "host instrumentation",
            "confidence": "low"
            if incomplete and any(e[1] for e in evidence)
            else "high",
            "effort": "medium",
        }
        for key, evidence in findings.items()
    ]
