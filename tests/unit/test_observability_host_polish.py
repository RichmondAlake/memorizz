from memorizz.observability import analyze_trace_events


def test_failed_tool_insight_has_supporting_event_ids():
    report = analyze_trace_events(
        [
            {
                "kind": "tool_result",
                "tool_name": "lookup",
                "status": "error",
                "event_id": "failed-event",
                "thread_id": "thread",
                "content": "",
            }
        ]
    )
    insight = next(i for i in report["insights"] if i["id"] == "tool_failure:lookup")
    assert insight["evidence_event_ids"] == ["failed-event"]


def test_unknown_capture_is_not_described_as_normalization_failure():
    report = analyze_trace_events(
        [
            {
                "kind": "tool_result",
                "tool_name": "lookup",
                "status": "success",
                "event_id": "event",
                "thread_id": "thread",
                "content": "",
            }
        ]
    )
    insight = next(
        i for i in report["insights"] if i["id"] == "trace_coverage_incomplete"
    )
    assert "Capture profiles were not declared" in insight["finding"]
    assert "normalization errors" not in insight["finding"]
