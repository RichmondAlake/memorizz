import json

import pytest

from memorizz.observability import analyze_trace_events


def _event(thread_id, kind, title, content="", **extra):
    return {
        "thread_id": thread_id,
        "kind": kind,
        "title": title,
        "content": content,
        **extra,
    }


@pytest.mark.unit
def test_trace_analysis_extracts_actionable_agent_improvements_without_raw_text():
    secret = "Remember my earlier URL secret-source-9281 and try again"
    events = [
        _event("t1", "conversation", "Conversation", secret, role="user"),
        _event(
            "t1",
            "conversation",
            "Conversation",
            "I cannot remember it; tell me which URL.",
            role="assistant",
        ),
        _event("t1", "tool_call", "Tool Call: search", '{"q":"agents"}'),
        _event("t1", "tool_call", "Tool Call: ingest", '{"url":"a"}'),
        _event(
            "t1",
            "tool_result",
            "Tool Result: ingest",
            '{"ok":false,"error":"download failed"}',
            trace_id="result:c1",
            success=False,
        ),
        _event("t2", "tool_call", "Tool Call: search", '{"q":"agents"}'),
        _event("t2", "tool_call", "Tool Call: ingest", '{"url":"b"}'),
        _event(
            "t2",
            "tool_result",
            "Tool Result: ingest",
            '{"ok":false,"error":"timeout"}',
            trace_id="result:c2",
            success=False,
        ),
        _event("t3", "tool_call", "Tool Call: ingest", '{"url":"c"}'),
        _event("t3", "tool_call", "Tool Call: ingest", '{"url":"c"}'),
    ]

    report = analyze_trace_events(
        events,
        agent_id="agent-1",
        source_is_virtual=True,
        window_truncated=True,
    )
    insight_ids = {row["id"] for row in report["insights"]}

    assert report["read_only"] is True
    assert report["summary"]["tool_failures"] == 2
    assert report["summary"]["learning_candidates"] == 1
    assert {
        "missing_trace_identity",
        "tool_failure:ingest",
        "repeated_tool_call",
        "user_correction_memory",
        "assistant_context_limitation",
        "missing_tool_latency",
        "missing_model_usage",
        "recurring_workflow:1",
        "bounded_analysis_window",
    }.issubset(insight_ids)
    serialized = json.dumps(report)
    assert secret not in serialized
    assert "secret-source-9281" not in serialized


@pytest.mark.unit
def test_trace_analysis_uses_structured_success_latency_and_model_metadata():
    report = analyze_trace_events(
        [
            _event(
                "t1",
                "tool_call",
                "Tool Call: lookup",
                '{"id":1}',
                logical_tool_name="lookup",
                model="gpt-test",
                provider="test",
            ),
            _event(
                "t1",
                "tool_result",
                "Tool Result: lookup",
                '{"ok":true}',
                logical_tool_name="lookup",
                success=True,
                duration_ms=25.5,
                model="gpt-test",
            ),
        ]
    )
    insight_ids = {row["id"] for row in report["insights"]}

    assert "tool_failure:lookup" not in insight_ids
    assert "missing_tool_latency" not in insight_ids
    assert "missing_model_usage" not in insight_ids
    assert report["tool_health"] == [
        {
            "name": "lookup",
            "calls": 1,
            "results": 1,
            "failures": 0,
            "outcomes": {
                "success": 1,
                "empty": 0,
                "degraded": 0,
                "fallback": 0,
                "provider_error": 0,
                "error": 0,
            },
            "failure_rate": 0.0,
            "average_duration_ms": 25.5,
        }
    ]


@pytest.mark.unit
def test_trace_analysis_separates_fallback_degraded_and_provider_errors():
    report = analyze_trace_events(
        [
            _event("t1", "tool_result", "Tool Result: search", outcome="fallback"),
            _event("t2", "tool_result", "Tool Result: search", outcome="fallback"),
            _event("t3", "tool_result", "Tool Result: lookup", outcome="degraded"),
            _event(
                "t4",
                "tool_result",
                "Tool Result: calendar",
                outcome="provider_error",
                success=False,
            ),
        ]
    )

    assert report["summary"]["tool_fallbacks"] == 2
    assert report["summary"]["tool_degraded_results"] == 1
    assert report["summary"]["tool_provider_errors"] == 1
    assert report["summary"]["tool_failures"] == 1
    assert "tool_fallback:search" in {row["id"] for row in report["insights"]}


@pytest.mark.unit
def test_trace_analysis_flags_page_binding_and_grounding_failures_without_titles():
    private_title = "Confidential acquisition article"
    report = analyze_trace_events(
        [
            _event(
                "analysis-wrong",
                "context_provenance",
                "Request context provenance",
                json.dumps(
                    {
                        "client_page_type": "analysis",
                        "client_page_id": "visible-page",
                        "canonical_page_type": "analysis",
                        "canonical_page_id": "thread-page",
                        "thread_binding_status": "mismatch",
                        "expected_thread_id": "analysis-thread-page",
                        "ownership_verified": False,
                        "grounding_status": "unavailable",
                        "debug_title": private_title,
                    }
                ),
            ),
            _event(
                "analysis-wrong",
                "cache_decision",
                "Semantic cache · bypassed",
                '{"cache_decision":"bypassed"}',
            ),
        ]
    )

    insight_ids = {row["id"] for row in report["insights"]}
    assert {
        "page_context_mismatch",
        "unverified_page_ownership",
        "missing_current_page_grounding",
    }.issubset(insight_ids)
    assert report["summary"]["page_context_mismatches"] == 1
    assert report["summary"]["unverified_page_ownership"] == 1
    assert report["summary"]["missing_current_page_grounding"] == 1
    assert report["summary"]["content_on_unscoped_threads"] == 0
    assert report["summary"]["cache_bypasses"] == 1
    assert private_title not in json.dumps(report)


@pytest.mark.unit
def test_trace_analysis_summarizes_unscoped_context_and_all_cache_decisions():
    report = analyze_trace_events(
        [
            _event(
                "generic-thread",
                "context_provenance",
                "Request context provenance",
                json.dumps(
                    {
                        "canonical_page_type": "doc",
                        "canonical_page_id": "doc-1",
                        "thread_binding_status": "client_owned_unscoped",
                        "ownership_verified": True,
                    }
                ),
            ),
            _event(
                "generic-thread",
                "cache_decision",
                "Semantic cache disabled",
                '{"cache_decision":"disabled"}',
            ),
            _event(
                "generic-thread",
                "cache_decision",
                "Semantic cache rejected",
                '{"cache_decision":"rejected"}',
            ),
        ]
    )

    assert report["summary"]["content_on_unscoped_threads"] == 1
    assert report["summary"]["unverified_page_ownership"] == 0
    assert report["summary"]["cache_disabled"] == 1
    assert report["summary"]["cache_rejections"] == 1
    assert any(
        insight["id"] == "content_on_unscoped_thread" for insight in report["insights"]
    )


@pytest.mark.unit
def test_trace_analysis_is_memory_first_without_exposing_memory_values():
    private_memory = "private-user-fact-must-never-appear"
    events = [
        _event("t1", "turn_start", "Agent turn started"),
        _event(
            "t1",
            "memory_context",
            "Memory supplied",
            json.dumps(
                {
                    "stage": "supplied",
                    "source_counts": {
                        "history_messages": 2,
                        "semantic_memories": 1,
                        "entity_profiles": 1,
                        "entity_attributes": 2,
                        "preferences": 1,
                        "conversation_memories": 1,
                        "writing_samples": 1,
                    },
                    "retrieved_candidate_count": 6,
                    "supplied_count": 8,
                    "injected_char_count": 1800,
                    "degraded": True,
                    "fallback_used": True,
                    "entity_refs": [
                        {"ref": "sha256:safe", "attribute_names": ["role"]}
                    ],
                }
            ),
        ),
        _event(
            "t1",
            "memory_reference",
            "Memory referenced",
            json.dumps(
                {
                    "stage": "referenced",
                    "supplied_count": 8,
                    "referenced_count": 1,
                    "referenced_refs": [{"source": "entity", "ref": "sha256:safe"}],
                    "behavioral_sources_not_measured": 2,
                }
            ),
        ),
    ]

    report = analyze_trace_events(events)

    assert report["version"] == 3
    assert report["summary"]["memory_context_events"] == 1
    assert report["summary"]["memory_candidates"] == 6
    assert report["summary"]["memory_supplied"] == 8
    assert report["summary"]["memory_explicit_references"] == 1
    assert report["summary"]["memory_degraded_turns"] == 1
    assert report["summary"]["memory_fallback_turns"] == 1
    assert any(row["source"] == "entity_memory" for row in report["memory_health"])
    assert "degraded_memory_retrieval" in {
        insight["id"] for insight in report["insights"]
    }
    assert private_memory not in json.dumps(report)


@pytest.mark.unit
def test_trace_analysis_flags_missing_memory_supply_observability():
    report = analyze_trace_events([_event("t1", "turn_start", "Agent turn started")])

    assert "missing_memory_supply_trace" in {
        insight["id"] for insight in report["insights"]
    }


@pytest.mark.unit
def test_cache_hit_does_not_claim_model_memory_observability_is_missing():
    report = analyze_trace_events(
        [
            _event("t1", "turn_start", "Agent turn started"),
            _event(
                "t1",
                "cache_decision",
                "Semantic cache hit",
                '{"cache_decision":"hit"}',
            ),
        ]
    )

    assert "missing_memory_supply_trace" not in {
        insight["id"] for insight in report["insights"]
    }


@pytest.mark.unit
def test_memory_health_attributes_summary_references_to_summaries():
    report = analyze_trace_events(
        [
            _event("t1", "turn_start", "Agent turn started"),
            _event(
                "t1",
                "memory_context",
                "Memory supplied",
                json.dumps(
                    {
                        "source_counts": {"summaries": 1},
                        "supplied_count": 1,
                    }
                ),
            ),
            _event(
                "t1",
                "memory_reference",
                "Memory referenced",
                json.dumps(
                    {
                        "referenced_count": 1,
                        "referenced_refs": [
                            {"source": "summary", "ref": "sha256:safe"}
                        ],
                    }
                ),
            ),
        ]
    )

    summary_health = next(
        row for row in report["memory_health"] if row["source"] == "summaries"
    )
    assert summary_health["supplied"] == 1
    assert summary_health["explicit_references"] == 1
