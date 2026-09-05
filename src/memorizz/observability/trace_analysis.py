"""Deterministic, read-only analysis of normalized Memorizz trace events.

The analyzer deliberately produces recommendations rather than modifying an
agent.  Operators can inspect the evidence, export the report, and decide
whether a prompt, tool, retrieval policy, or learned workflow should change.
No raw conversation text is copied into the report.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .coverage import trace_coverage
from .diagnostics import diagnose_trace

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}
_TOOL_OUTCOMES = (
    "success",
    "empty",
    "degraded",
    "fallback",
    "provider_error",
    "error",
)
_PROVIDER_ERROR_CODES = {
    "authentication_required",
    "authorization_required",
    "connection_error",
    "oauth_required",
    "provider_error",
    "provider_timeout",
    "provider_unavailable",
    "transport_error",
}
_TOOL_TITLE = re.compile(
    r"^(?:tool\s+(?:call|result)|execution\s+log)\s*(?::|·|-)\s*(.+)$",
    re.IGNORECASE,
)
_MEMORY_CORRECTION = re.compile(
    r"\b(?:"
    r"what\s+i\s+asked\s+earlier|"
    r"you\s+(?:forgot|didn['’]?t\s+remember)|"
    r"(?:earlier|previous)\s+(?:conversation|message|link|url|video|article)|"
    r"unrelated\s+(?:content|conversation|upload)|"
    r"that['’]?s\s+not\s+(?:the|what)|"
    r"try\s+again"
    r")\b",
    re.IGNORECASE,
)
_ASSISTANT_LIMITATION = re.compile(
    r"\b(?:"
    r"i\s+(?:can(?:not|'t)|could(?:\s+not|n't)|don['’]?t)\s+(?:find|identify|access|remember)|"
    r"tell\s+me\s+which|"
    r"paste\s+(?:the|it)|"
    r"still\s+processing|"
    r"not\s+ready\s+yet"
    r")\b",
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = _text(value)
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _tool_name(event: Dict[str, Any]) -> str:
    for key in ("logical_tool_name", "tool_name", "name"):
        value = _text(event.get(key))
        if value:
            return value
    match = _TOOL_TITLE.match(_text(event.get("title")))
    return match.group(1).strip() if match else ""


def _correlation_id(event: Dict[str, Any]) -> str:
    value = _text(event.get("tool_call_id") or event.get("trace_id"))
    for prefix in ("call:", "result:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value


def _failure(event: Dict[str, Any]) -> bool:
    if _tool_outcome(event) in {"provider_error", "error"}:
        return True
    if event.get("success") is False:
        return True
    payload = _payload(event.get("content"))
    if isinstance(payload, dict):
        if payload.get("success") is False or payload.get("ok") is False:
            return True
        if _text(payload.get("status")).lower() in {"failed", "failure", "error"}:
            return True
        if payload.get("error") and payload.get("success") is not True:
            return True
        nested = _payload(payload.get("result"))
        if isinstance(nested, dict) and (
            nested.get("ok") is False or nested.get("success") is False
        ):
            return True
    content = _text(event.get("content")).lower()
    return content.startswith("error:") or content.startswith("error ")


def _tool_outcome(event: Dict[str, Any]) -> str:
    """Return a normalized outcome while retaining legacy trace support."""

    def failed_payload_outcome(value: Any) -> Optional[str]:
        if not isinstance(value, dict):
            return None
        status = _text(value.get("status")).lower()
        failed = (
            value.get("ok") is False
            or value.get("success") is False
            or (bool(value.get("error")) and value.get("ok") is not True)
            or status in {"failed", "failure", "error", "provider_error"}
        )
        if not failed:
            return None
        error_code = _text(value.get("error_code")).lower()
        if (
            status == "provider_error"
            or error_code in _PROVIDER_ERROR_CODES
            or error_code.startswith("provider_")
        ):
            return "provider_error"
        return "error"

    explicit = _text(event.get("outcome")).lower()
    if explicit in _TOOL_OUTCOMES:
        return explicit
    payload = _payload(event.get("content"))
    if isinstance(payload, dict):
        payload_outcome = payload.get("outcome")
        if isinstance(payload_outcome, dict):
            payload_outcome = payload_outcome.get("status")
        normalized = _text(payload_outcome).lower()
        if normalized in _TOOL_OUTCOMES:
            return normalized
        failed = failed_payload_outcome(payload)
        if failed:
            return failed
        nested = _payload(payload.get("result"))
        if isinstance(nested, dict):
            nested_outcome = nested.get("outcome")
            if isinstance(nested_outcome, dict):
                nested_outcome = nested_outcome.get("status")
            normalized = _text(nested_outcome).lower()
            if normalized in _TOOL_OUTCOMES:
                return normalized
            failed = failed_payload_outcome(nested)
            if failed:
                return failed
    status = _text(event.get("status")).lower()
    if status in _TOOL_OUTCOMES:
        return status
    if event.get("success") is False:
        return "error"
    content = _text(event.get("content")).lower()
    if content.startswith("error:") or content.startswith("error "):
        return "error"
    return "success"


def _arguments_fingerprint(event: Dict[str, Any]) -> str:
    content = _text(event.get("content"))
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _confidence(evidence_count: int) -> str:
    if evidence_count >= 3:
        return "high"
    if evidence_count == 2:
        return "medium"
    return "low"


def _insight(
    insight_id: str,
    *,
    priority: str,
    component: str,
    title: str,
    finding: str,
    recommendation: str,
    target: str,
    evidence_count: int,
    effort: str = "medium",
    confidence: Optional[str] = None,
    evidence_event_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": insight_id,
        "priority": priority,
        "component": component,
        "title": title,
        "finding": finding,
        "recommendation": recommendation,
        "target": target,
        "evidence_count": max(0, int(evidence_count)),
        "confidence": confidence or _confidence(evidence_count),
        "effort": effort,
        **({"evidence_event_ids": evidence_event_ids} if evidence_event_ids else {}),
    }


def _result_key(event: Dict[str, Any], index: int) -> Tuple[str, str, str]:
    correlation = _correlation_id(event)
    return (
        _text(event.get("thread_id")) or "—",
        correlation or f"event-{index}",
        _tool_name(event) or "unknown",
    )


def analyze_trace_events(
    events: Iterable[Dict[str, Any]],
    *,
    agent_id: str = "",
    agent_name: str = "",
    scope: str = "agent",
    source_is_virtual: bool = False,
    window_truncated: bool = False,
    signals: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    max_insights: int = 14,
    normalization_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a JSON-safe improvement report for normalized trace events.

    The input is the same event shape used by the local observability timeline.
    Analysis is deterministic and side-effect free so it is safe to run while
    connected to production data.
    """

    metadata = dict(normalization_metadata or getattr(events, "coverage", {}))
    metadata["truncated"] = bool(metadata.get("truncated")) or window_truncated
    normalized = [event for event in events if isinstance(event, dict)]
    coverage = trace_coverage(normalized, metadata)
    threads = {_text(event.get("thread_id")) or "—" for event in normalized}
    role_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    tool_calls: List[Dict[str, Any]] = []
    result_events: List[Tuple[int, Dict[str, Any]]] = []
    execution_events: List[Tuple[int, Dict[str, Any]]] = []
    context_events: List[Dict[str, Any]] = []
    memory_context_events: List[Dict[str, Any]] = []
    memory_reference_events: List[Dict[str, Any]] = []
    cache_decisions: Counter[str] = Counter()
    correction_count = 0
    limitation_count = 0
    signal_payload = signals if isinstance(signals, dict) else {}
    verified_feedback = [
        row
        for row in (signal_payload.get("feedback") or [])
        if isinstance(row, dict) and row.get("verified") is True
    ]
    verified_outcomes = [
        row
        for row in (signal_payload.get("outcomes") or [])
        if isinstance(row, dict) and row.get("verified") is True
    ]
    verified_outcomes.extend(
        event
        for event in normalized
        if event.get("kind") == "verified_outcome" and event.get("verified") is True
    )
    negative_feedback = 0
    for row in verified_feedback:
        label = _text(row.get("label")).lower()
        try:
            rating = float(row.get("rating"))
        except (TypeError, ValueError):
            rating = 0.0
        if rating < 0 or label in {"negative", "thumbs_down", "incorrect", "failed"}:
            negative_feedback += 1
    verified_outcome_failures = sum(
        1 for row in verified_outcomes if _text(row.get("status")).lower() == "failure"
    )
    verified_outcome_successes = sum(
        1 for row in verified_outcomes if _text(row.get("status")).lower() == "success"
    )

    for index, event in enumerate(normalized):
        role = _text(event.get("role")).lower() or "system"
        kind = _text(event.get("kind") or event.get("trace_kind")).lower()
        role_counts[role] += 1
        kind_counts[kind or "unknown"] += 1
        if kind == "tool_call":
            tool_calls.append(event)
        elif kind == "tool_result":
            result_events.append((index, event))
        elif kind == "execution_log":
            execution_events.append((index, event))
        elif kind == "context_provenance":
            structured = _payload(event.get("content"))
            context_event = dict(structured) if isinstance(structured, dict) else {}
            for field in (
                "client_page_type",
                "client_page_id",
                "canonical_page_type",
                "canonical_page_id",
                "thread_binding_status",
                "expected_thread_id",
                "ownership_verified",
                "grounding_status",
                "grounding_source",
            ):
                if event.get(field) is not None:
                    context_event[field] = event.get(field)
            context_event["thread_id"] = event.get("thread_id")
            context_events.append(context_event)
        elif kind == "memory_context":
            structured = _payload(event.get("content"))
            memory_event = dict(structured) if isinstance(structured, dict) else {}
            for field in (
                "memory_history_count",
                "memory_candidate_count",
                "memory_supplied_count",
                "memory_injected_chars",
                "memory_degraded",
                "memory_fallback_used",
                "entity_profile_count",
                "preference_count",
                "conversation_memory_count",
                "writing_sample_count",
            ):
                if event.get(field) is not None:
                    memory_event[field] = event.get(field)
            memory_context_events.append(memory_event)
        elif kind == "memory_reference":
            structured = _payload(event.get("content"))
            memory_event = dict(structured) if isinstance(structured, dict) else {}
            for field in ("memory_supplied_count", "memory_referenced_count"):
                if event.get(field) is not None:
                    memory_event[field] = event.get(field)
            memory_reference_events.append(memory_event)
        elif kind == "cache_decision":
            structured = _payload(event.get("content"))
            decision = _text(event.get("cache_decision"))
            if not decision and isinstance(structured, dict):
                decision = _text(structured.get("cache_decision"))
            cache_decisions[decision.lower() or "unknown"] += 1

        content = _text(event.get("content"))
        if (
            role == "user"
            and kind == "conversation"
            and _MEMORY_CORRECTION.search(content)
        ):
            correction_count += 1
        if (
            role == "assistant"
            and kind == "conversation"
            and _ASSISTANT_LIMITATION.search(content)
        ):
            limitation_count += 1

    # Prefer first-class tool results. Durable execution logs are a fallback
    # for legacy traces and are de-duplicated when a correlation ID is shared.
    results: List[Dict[str, Any]] = []
    seen_results = set()
    for index, event in [*result_events, *execution_events]:
        key = _result_key(event, index)
        if key in seen_results:
            continue
        seen_results.add(key)
        results.append(event)

    tool_stats: Dict[str, Dict[str, Any]] = {}

    def _stats(name: str) -> Dict[str, Any]:
        return tool_stats.setdefault(
            name or "unknown",
            {
                "name": name or "unknown",
                "calls": 0,
                "results": 0,
                "failures": 0,
                "failure_event_ids": [],
                "outcomes": Counter(),
                "durations": [],
            },
        )

    calls_by_thread: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for event in tool_calls:
        name = _tool_name(event) or "unknown"
        _stats(name)["calls"] += 1
        calls_by_thread[_text(event.get("thread_id")) or "—"].append(
            (name, _arguments_fingerprint(event))
        )

    for event in results:
        name = _tool_name(event) or "unknown"
        stat = _stats(name)
        stat["results"] += 1
        stat["outcomes"][_tool_outcome(event)] += 1
        if _failure(event):
            stat["failures"] += 1
            if event.get("event_id"):
                stat["failure_event_ids"].append(event["event_id"])
        duration = event.get("duration_ms")
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            duration_value = -1.0
        if duration_value >= 0:
            stat["durations"].append(duration_value)

    # Legacy result-only traces still represent one observed execution.
    for stat in tool_stats.values():
        if stat["calls"] == 0 and stat["results"]:
            stat["calls"] = stat["results"]

    repeated_calls = 0
    repeated_by_tool: Counter[str] = Counter()
    for calls in calls_by_thread.values():
        for previous, current in zip(calls, calls[1:]):
            if previous == current:
                repeated_calls += 1
                repeated_by_tool[current[0]] += 1

    signature_counts: Counter[Tuple[str, ...]] = Counter()
    for calls in calls_by_thread.values():
        signature = tuple(name for name, _fingerprint in calls)
        if len(signature) >= 2:
            signature_counts[signature] += 1

    insights: List[Dict[str, Any]] = []
    if source_is_virtual:
        insights.append(
            _insight(
                "missing_trace_identity",
                priority="P0",
                component="Observability",
                title="Trace ownership is not durable",
                finding="These events were recovered from an unregistered runtime source rather than a saved agent.",
                recommendation="Automatically upsert the MemAgent on first use and attach agent_id, run_id, turn_id, thread_id, memory_id, and user_id to every event.",
                target="MemAgent lifecycle and trace envelope",
                evidence_count=max(1, len(normalized)),
                effort="medium",
                confidence="high",
            )
        )

    context_mismatches = 0
    ownership_failures = 0
    missing_grounding = 0
    unscoped_content = 0
    for event in context_events:
        client_type = _text(event.get("client_page_type")).lower()
        client_id = _text(event.get("client_page_id"))
        canonical_type = _text(event.get("canonical_page_type")).lower()
        canonical_id = _text(event.get("canonical_page_id"))
        binding_status = _text(event.get("thread_binding_status")).lower()
        expected_thread_id = _text(event.get("expected_thread_id"))
        actual_thread_id = _text(event.get("thread_id"))
        if (
            (client_type and canonical_type and client_type != canonical_type)
            or (client_id and canonical_id and client_id != canonical_id)
            or (
                expected_thread_id
                and actual_thread_id
                and expected_thread_id != actual_thread_id
            )
            or binding_status in {"mismatch", "rejected", "stale", "unverified"}
        ):
            context_mismatches += 1
        if canonical_type in {
            "analysis",
            "group",
            "doc",
            "slides",
            "automation",
        }:
            if event.get("ownership_verified") is not True:
                ownership_failures += 1
            if binding_status == "client_owned_unscoped":
                unscoped_content += 1
        if (
            canonical_type in {"analysis", "group"}
            and _text(event.get("grounding_status")).lower() != "ready"
        ):
            missing_grounding += 1

    if context_mismatches:
        insights.append(
            _insight(
                "page_context_mismatch",
                priority="P0",
                component="Context routing",
                title="The visible page and agent thread disagree",
                finding=f"Detected {context_mismatches} turn(s) with conflicting client, canonical, or thread-bound page identity.",
                recommendation="Reject the turn before model execution, derive content identity from the server-owned thread binding, and require the client to retry after its route transition completes.",
                target="request context canonicalization",
                evidence_count=context_mismatches,
                effort="medium",
                confidence="high",
            )
        )

    if ownership_failures:
        insights.append(
            _insight(
                "unverified_page_ownership",
                priority="P0",
                component="Security",
                title="Content context lacks verified tenant ownership",
                finding=f"Detected {ownership_failures} content-bound turn(s) without a positive ownership/share check.",
                recommendation="Resolve page metadata with an authenticated, tenant-scoped server query and fail closed before retrieval, cache lookup, or tool execution.",
                target="page ownership boundary",
                evidence_count=ownership_failures,
                effort="medium",
                confidence="high",
            )
        )

    if missing_grounding:
        insights.append(
            _insight(
                "missing_current_page_grounding",
                priority="P0",
                component="Retrieval",
                title="Current-page turns ran without authoritative grounding",
                finding=f"Detected {missing_grounding} analysis/group turn(s) whose grounding status was not ready.",
                recommendation="Fail closed for deictic requests and add an owner-scoped stored-text fallback when vector retrieval or reranking has not caught up.",
                target="current-page grounding pipeline",
                evidence_count=missing_grounding,
                effort="medium",
                confidence="high",
            )
        )

    if unscoped_content:
        insights.append(
            _insight(
                "content_on_unscoped_thread",
                priority="P1",
                component="Memory",
                title="Content context is attached to a generic thread",
                finding=f"Detected {unscoped_content} content-bound turn(s) without a matching content thread id.",
                recommendation="Use a stable per-content thread or explicitly disable cross-thread recall for that turn so unrelated memory cannot outrank the current artifact.",
                target="thread and retrieval policy",
                evidence_count=unscoped_content,
                effort="low",
                confidence="high",
            )
        )

    def _count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    memory_source_counts: Counter[str] = Counter()
    memory_candidates = 0
    memory_supplied = 0
    memory_injected_chars = 0
    memory_degraded_turns = 0
    memory_fallback_turns = 0
    for event in memory_context_events:
        source_counts = event.get("source_counts") or {}
        if isinstance(source_counts, dict):
            for source, value in source_counts.items():
                memory_source_counts[str(source)] += _count(value)
        memory_candidates += _count(
            event.get("retrieved_candidate_count")
            or event.get("memory_candidate_count")
        )
        memory_supplied += _count(
            event.get("supplied_count") or event.get("memory_supplied_count")
        )
        memory_injected_chars += _count(
            event.get("injected_char_count") or event.get("memory_injected_chars")
        )
        memory_degraded_turns += int(
            bool(event.get("degraded") or event.get("memory_degraded"))
        )
        memory_fallback_turns += int(
            bool(event.get("fallback_used") or event.get("memory_fallback_used"))
        )

    memory_referenced = 0
    behavioral_sources_not_measured = 0
    referenced_by_source: Counter[str] = Counter()
    for event in memory_reference_events:
        memory_referenced += _count(
            event.get("referenced_count") or event.get("memory_referenced_count")
        )
        behavioral_sources_not_measured += _count(
            event.get("behavioral_sources_not_measured")
        )
        for row in event.get("referenced_refs") or []:
            if isinstance(row, dict):
                referenced_by_source[_text(row.get("source")) or "unknown"] += 1

    model_turn_count = max(
        0,
        kind_counts.get("turn_start", 0) - cache_decisions.get("hit", 0),
    )
    if model_turn_count and len(memory_context_events) < model_turn_count:
        missing_memory_traces = model_turn_count - len(memory_context_events)
        insights.append(
            _insight(
                "missing_memory_supply_trace",
                priority="P1",
                component="Memory observability",
                title="The model's supplied memory is not traceable",
                finding="Agent turns are present, but no content-free memory supply event records history, retrieval, entity, preference, or writing-sample inputs.",
                recommendation="Emit one memory_context event after context assembly and one memory_reference event after completion, using counts and hashed source ids rather than raw memory values.",
                target="MemAgent memory context assembly",
                evidence_count=missing_memory_traces,
                effort="low",
                confidence="high",
            )
        )

    if memory_degraded_turns:
        insights.append(
            _insight(
                "degraded_memory_retrieval",
                priority="P1",
                component="Memory retrieval",
                title="Personalization memory is degrading",
                finding=f"Detected {memory_degraded_turns} turn(s) where a memory source reported degraded retrieval.",
                recommendation="Inspect the per-turn reason and fallback fields, verify vector-index health, and keep the exact tenant-scoped fallback enabled until semantic retrieval is healthy.",
                target="entity and personalization retrieval",
                evidence_count=memory_degraded_turns,
                effort="medium",
                confidence="high",
            )
        )

    referenceable_supplied = sum(
        memory_source_counts.get(source, 0)
        for source in (
            "semantic_memories",
            "entity_attributes",
            "conversation_memories",
        )
    )
    if (
        len(memory_context_events) >= 2
        and referenceable_supplied >= 8
        and memory_referenced == 0
    ):
        insights.append(
            _insight(
                "memory_supplied_without_explicit_reference",
                priority="P2",
                component="Personalization",
                title="Memory supply may be broader than the answer needs",
                finding=f"Supplied {referenceable_supplied} referenceable memory item(s) without a conservatively measurable explicit overlap in the resulting answers.",
                recommendation="Review relevance thresholds and per-source limits against verified outcomes. Do not remove style preferences based on this signal because behavioral application is intentionally not inferred from text matching.",
                target="PersonalizationPolicy limits and relevance threshold",
                evidence_count=referenceable_supplied,
                effort="low",
            )
        )

    if kind_counts.get("turn_start", 0) and not context_events:
        insights.append(
            _insight(
                "missing_context_provenance",
                priority="P1",
                component="Observability",
                title="Request context lineage is not traceable",
                finding="Agent turns are present, but no structured context-provenance event records routing, ownership, or grounding decisions.",
                recommendation="Pass a content-free observability_context on every run and persist only allowlisted ids, status fields, and deterministic fingerprints.",
                target="MemAgent.run observability_context",
                evidence_count=kind_counts.get("turn_start", 0),
                effort="low",
                confidence="high",
            )
        )

    for name, stat in sorted(
        tool_stats.items(),
        key=lambda item: (-item[1]["failures"], -item[1]["calls"], item[0]),
    ):
        executions = max(stat["calls"], stat["results"])
        failures = stat["failures"]
        failure_rate = failures / max(1, stat["results"])
        if failures and (failures >= 2 or failure_rate >= 0.25):
            priority = "P0" if failures >= 2 and failure_rate >= 0.5 else "P1"
            insights.append(
                _insight(
                    f"tool_failure:{name}",
                    priority=priority,
                    component="Tool",
                    title=f"{name} is failing too often",
                    finding=f"Observed {failures} failed result(s) across {executions} execution(s) ({failure_rate:.0%} of recorded results).",
                    recommendation="Review the tool schema, preconditions, timeout/retry policy, error taxonomy, and result contract; add a focused regression eval before changing the prompt.",
                    target=f"tool:{name}",
                    evidence_count=failures,
                    evidence_event_ids=stat["failure_event_ids"][:20],
                )
            )
        fallbacks = stat["outcomes"].get("fallback", 0)
        degraded = stat["outcomes"].get("degraded", 0)
        if fallbacks >= 2:
            insights.append(
                _insight(
                    f"tool_fallback:{name}",
                    priority="P1",
                    component="Tool",
                    title=f"{name} is repeatedly using a fallback",
                    finding=f"Observed {fallbacks} usable fallback result(s) across {executions} execution(s).",
                    recommendation="Inspect primary-provider availability and latency before removing the fallback; keep the alternate path but measure it separately from primary success.",
                    target=f"tool:{name}",
                    evidence_count=fallbacks,
                    effort="low",
                )
            )
        if degraded >= 2:
            insights.append(
                _insight(
                    f"tool_degraded:{name}",
                    priority="P1",
                    component="Tool",
                    title=f"{name} is repeatedly completing with limitations",
                    finding=f"Observed {degraded} degraded result(s) across {executions} execution(s).",
                    recommendation="Review the recorded reason codes and provider capability before treating these executions as clean training or promotion evidence.",
                    target=f"tool:{name}",
                    evidence_count=degraded,
                    effort="low",
                )
            )

    if repeated_calls:
        top = ", ".join(name for name, _count in repeated_by_tool.most_common(3))
        insights.append(
            _insight(
                "repeated_tool_call",
                priority="P1",
                component="Agent policy",
                title="Identical tool calls are repeating within a thread",
                finding=f"Found {repeated_calls} adjacent duplicate call(s){f' involving {top}' if top else ''}.",
                recommendation="Add duplicate-call suppression and tell the agent to inspect the previous result or change its arguments before retrying.",
                target="ContextPolicy and tool router",
                evidence_count=repeated_calls,
                effort="low",
            )
        )

    if correction_count:
        insights.append(
            _insight(
                "user_correction_memory",
                priority="P1",
                component="Prompt and retrieval",
                title="Users are re-grounding the agent in earlier context",
                finding=f"Detected {correction_count} correction/retry signal(s) referring to earlier or unrelated context.",
                recommendation="Strengthen the instruction to resolve follow-ups from the current thread first, retain exact source identifiers, and require explicit user intent before searching other threads or uploads.",
                target="agent.instruction and retrieval_policy",
                evidence_count=correction_count,
                effort="low",
            )
        )

    if limitation_count:
        insights.append(
            _insight(
                "assistant_context_limitation",
                priority="P1",
                component="Prompt and context",
                title="The assistant repeatedly reports missing context",
                finding=f"Detected {limitation_count} response(s) asking the user to restate or identify information.",
                recommendation="Audit context assembly before rewriting the prompt: surface current-thread URLs, ingestion jobs, selected artifacts, and tool outcomes as structured request context.",
                target="context assembly",
                evidence_count=limitation_count,
                effort="medium",
            )
        )

    result_count = len(result_events)
    if result_count:
        latency_count = sum(
            1 for _index, event in result_events if event.get("duration_ms") is not None
        )
        if latency_count / result_count < 0.8:
            insights.append(
                _insight(
                    "missing_tool_latency",
                    priority="P1",
                    component="Observability",
                    title="Tool latency is missing from most results",
                    finding=f"Only {latency_count} of {result_count} tool result(s) include duration_ms.",
                    recommendation="Record monotonic execution duration on every tool result and expose p50/p95 by tool and deployment.",
                    target="tool result trace schema",
                    evidence_count=result_count - latency_count,
                    effort="low",
                    confidence="high",
                )
            )

    if any(
        e.get("kind") in {"model_call", "model_result", "conversation"}
        for e in normalized
    ) and not any(
        any(
            event.get(field) is not None
            for field in (
                "model",
                "provider",
                "input_tokens",
                "output_tokens",
                "cost_usd",
            )
        )
        for event in normalized
    ):
        insights.append(
            _insight(
                "missing_model_usage",
                priority="P2",
                component="Observability",
                title="Model and token economics are not traceable",
                finding="The selected trace window has no model, provider, token, cache-token, or cost metadata.",
                recommendation="Add model-call child spans with token usage, cached tokens, TTFT, finish reason, and estimated cost.",
                target="model call trace schema",
                evidence_count=len(normalized),
                effort="medium",
                confidence="high",
            )
        )

    recurring = [
        (signature, count)
        for signature, count in signature_counts.most_common(4)
        if count >= 2
    ]
    for index, (signature, count) in enumerate(recurring, start=1):
        sequence = " → ".join(signature)
        insights.append(
            _insight(
                f"recurring_workflow:{index}",
                priority="P2",
                component="Continual learning",
                title=f"Recurring workflow candidate: {sequence}",
                finding=f"The same {len(signature)}-step tool sequence appears in {count} thread(s) in this analysis window.",
                recommendation="Confirm outcomes in Workflow Memory, then let the normal diversity, success-rate, recency, and shadow-readiness gates decide whether to distill it into a skill.",
                target="workflow trajectory and Skillbox",
                evidence_count=count,
                effort="low",
            )
        )

    if window_truncated:
        insights.append(
            _insight(
                "bounded_analysis_window",
                priority="P2",
                component="Data quality",
                title="This report analyzes a bounded trace window",
                finding="Older events were omitted by the observability timeline limit.",
                recommendation="Treat counts as directional, then follow the returned cursor or select a narrower agent, thread, or time range before approving a change.",
                target="paginated observability query window",
                evidence_count=len(normalized),
                effort="high",
                confidence="high",
            )
        )

    if negative_feedback:
        insights.append(
            _insight(
                "verified_negative_feedback",
                priority="P0" if negative_feedback >= 2 else "P1",
                component="Outcome",
                title="Verified users reported an unsatisfactory result",
                finding=f"{negative_feedback} verified negative feedback signal(s) are linked to this trace window.",
                recommendation="Build a regression case from the linked trace identity, reproduce it against the current agent version, and require a passing Evalground comparison before rollout.",
                target="feedback-linked regression dataset",
                evidence_count=negative_feedback,
                effort="medium",
                confidence="high",
            )
        )

    if verified_outcome_failures:
        insights.append(
            _insight(
                "verified_task_failure",
                priority="P0" if verified_outcome_failures >= 2 else "P1",
                component="Outcome",
                title="Verified task outcomes are failing",
                finding=f"{verified_outcome_failures} verified task failure(s) are linked to this trace window.",
                recommendation="Use the failed traces as the primary experiment dataset and compare task-success rate, tool failure rate, and negative-feedback rate against the unchanged baseline.",
                target="Evalground outcome-linked experiment",
                evidence_count=verified_outcome_failures,
                effort="medium",
                confidence="high",
            )
        )

    insights.extend(diagnose_trace(normalized, metadata=coverage))
    if coverage["coverage"] != "complete":
        if coverage.get("normalization_errors"):
            coverage_finding = "Some stored events could not be normalized; repair invalid evidence before drawing conclusions."
        elif (
            coverage.get("truncated") or coverage.get("read_completeness") == "partial"
        ):
            coverage_finding = "The trace read is incomplete; narrow the selection or load the remaining events."
        elif coverage.get("instrumentation_coverage") == "unknown":
            coverage_finding = "Capture profiles were not declared; loading all stored events does not establish end-to-end instrumentation."
        else:
            coverage_finding = "Required evidence stages are missing; inspect the declared capture profiles and missing-stage list."
        insights.append(
            _insight(
                "trace_coverage_incomplete",
                priority="P0" if coverage["coverage"] == "untrusted" else "P1",
                component="observability",
                title="Trace evidence is incomplete",
                finding=coverage_finding,
                recommendation="Inspect trace coverage and instrument missing stages before treating absent evidence as success.",
                target="trace capture and normalization",
                evidence_count=max(
                    1,
                    len(coverage.get("missing_stages", []))
                    + int(coverage.get("normalization_errors", 0)),
                ),
            )
        )
    insights.sort(
        key=lambda item: (
            _PRIORITY_ORDER.get(item["priority"], 9),
            -int(item["evidence_count"]),
            item["component"],
            item["title"],
        )
    )
    insights = insights[: max(1, int(max_insights))]

    tool_health = []
    for name, stat in sorted(
        tool_stats.items(),
        key=lambda item: (-item[1]["failures"], -item[1]["calls"], item[0]),
    ):
        durations = stat.pop("durations")
        outcomes = stat.pop("outcomes")
        stat.pop(
            "failure_event_ids", None
        )  # Internal evidence links, not a tool-health schema change.
        results_for_tool = int(stat["results"])
        tool_health.append(
            {
                **stat,
                "outcomes": {
                    outcome: int(outcomes.get(outcome, 0)) for outcome in _TOOL_OUTCOMES
                },
                "failure_rate": stat["failures"] / max(1, results_for_tool),
                "average_duration_ms": (
                    round(sum(durations) / len(durations), 1) if durations else None
                ),
            }
        )

    failure_count = sum(row["failures"] for row in tool_health)
    outcome_counts = Counter()
    for row in tool_health:
        outcome_counts.update(row["outcomes"])
    priority_counts = Counter(item["priority"] for item in insights)
    learning_candidates = [
        item for item in insights if item["component"] == "Continual learning"
    ]
    memory_health_groups = (
        (
            "entity_memory",
            memory_source_counts.get("entity_attributes", 0),
            referenced_by_source.get("entity", 0),
        ),
        (
            "conversation_context",
            sum(
                memory_source_counts.get(source, 0)
                for source in (
                    "history_messages",
                    "semantic_memories",
                    "conversation_memories",
                )
            ),
            sum(
                referenced_by_source.get(source, 0)
                for source in ("conversation", "semantic_memory", "history")
            ),
        ),
        (
            "summaries",
            memory_source_counts.get("summaries", 0),
            referenced_by_source.get("summary", 0),
        ),
        (
            "preferences",
            memory_source_counts.get("preferences", 0),
            0,
        ),
        (
            "writing_samples",
            memory_source_counts.get("writing_samples", 0),
            0,
        ),
    )
    memory_health = [
        {
            "source": source,
            "supplied": int(supplied),
            "explicit_references": int(referenced),
        }
        for source, supplied, referenced in memory_health_groups
        if supplied
    ]
    return {
        "version": 3,
        "agent_id": _text(agent_id),
        "agent_name": _text(agent_name),
        "scope": scope,
        "read_only": True,
        "window_truncated": bool(window_truncated),
        "coverage": coverage,
        "metric_scope": "observed_evidence_window",
        "metric_states": {
            name: {
                "value": value if value else None,
                "state": (
                    "known"
                    if coverage["read_completeness"] == "complete"
                    else "partial"
                )
                if value
                else "no_recorded_evidence"
                if coverage["read_completeness"] == "complete"
                else "unknown",
            }
            for name, value in {
                "tool_calls": len(tool_calls),
                "tool_results": len(results),
                "tool_failures": failure_count,
                "model_results": kind_counts.get("model_result", 0),
                "memory_candidates": memory_candidates,
                "memory_supplied": memory_supplied,
                "artifacts": kind_counts.get("artifact_persisted", 0),
                "output_contracts": kind_counts.get("output_contract", 0),
                "ui_deliveries": kind_counts.get("ui_delivery", 0),
                "verified_outcomes": len(verified_outcomes),
            }.items()
        },
        "summary": {
            "events_analyzed": len(normalized),
            "threads_analyzed": len(threads),
            "conversation_events": kind_counts.get("conversation", 0),
            "tool_calls": len(tool_calls),
            "tool_results": len(results),
            "tool_failures": failure_count,
            "tool_failure_rate": failure_count / max(1, len(results)),
            "tool_successes": outcome_counts.get("success", 0),
            "tool_empty_results": outcome_counts.get("empty", 0),
            "tool_degraded_results": outcome_counts.get("degraded", 0),
            "tool_fallbacks": outcome_counts.get("fallback", 0),
            "tool_provider_errors": outcome_counts.get("provider_error", 0),
            "unique_tools": len(tool_health),
            "user_correction_signals": correction_count,
            "assistant_limitation_signals": limitation_count,
            "context_provenance_events": len(context_events),
            "memory_context_events": len(memory_context_events),
            "memory_reference_events": len(memory_reference_events),
            "memory_candidates": memory_candidates,
            "memory_supplied": memory_supplied,
            "memory_explicit_references": memory_referenced,
            "memory_behavioral_sources_not_measured": behavioral_sources_not_measured,
            "memory_injected_chars": memory_injected_chars,
            "memory_degraded_turns": memory_degraded_turns,
            "memory_fallback_turns": memory_fallback_turns,
            "page_context_mismatches": context_mismatches,
            "unverified_page_ownership": ownership_failures,
            "missing_current_page_grounding": missing_grounding,
            "content_on_unscoped_threads": unscoped_content,
            "cache_hits": cache_decisions.get("hit", 0),
            "cache_misses": cache_decisions.get("miss", 0),
            "cache_bypasses": cache_decisions.get("bypassed", 0),
            "cache_disabled": cache_decisions.get("disabled", 0),
            "cache_rejections": cache_decisions.get("rejected", 0),
            "verified_feedback": len(verified_feedback),
            "verified_negative_feedback": negative_feedback,
            "verified_outcomes": len(verified_outcomes),
            "verified_outcome_successes": verified_outcome_successes,
            "verified_outcome_failures": verified_outcome_failures,
            "learning_candidates": len(learning_candidates),
            "insight_count": len(insights),
            "priority_counts": {
                "P0": priority_counts.get("P0", 0),
                "P1": priority_counts.get("P1", 0),
                "P2": priority_counts.get("P2", 0),
            },
        },
        "tool_health": tool_health,
        "memory_health": memory_health,
        "insights": insights,
        "learning_candidates": learning_candidates,
        "methodology": (
            "Deterministic heuristics over the selected normalized trace window; "
            "no model call, agent mutation, or raw conversation text in the report."
        ),
    }
