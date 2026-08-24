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
            "failure_rate": 0.0,
            "average_duration_ms": 25.5,
        }
    ]


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
