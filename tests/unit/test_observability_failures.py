"""The failures panel: what broke, how often, and whether the turn survived."""

import pytest

from memorizz.ui.observability_view import trace_timeline

pytestmark = pytest.mark.unit


def _tool_failure(tool, code, turn, *, second=0):
    return {
        "event_id": f"{turn}-{tool}-{second}",
        "anchor": f"trace-event-{turn}-{second}",
        "kind": "tool_result",
        "tool_name": tool,
        "status": "error",
        "error_code": code,
        "outcome_reason_code": code,
        "turn_id": turn,
        "root_trace_id": f"root-{turn}",
        "thread_id": "thread-1",
        "timestamp": f"2026-11-01T22:00:{second:02d}+00:00",
    }


def _turn_end(turn, *, status="success", error_code=None):
    return {
        "event_id": f"{turn}-end",
        "kind": "turn_result",
        "status": status,
        "error_code": error_code,
        "turn_id": turn,
        "root_trace_id": f"root-{turn}",
        "timestamp": "2026-11-01T22:00:59+00:00",
    }


def _failures(*events):
    return trace_timeline(events)["failures"]


def test_the_same_failure_is_counted_once_with_its_turns():
    failures = _failures(
        _tool_failure("mcp_call_tool", "protocol_error", "a", second=1),
        _turn_end("a"),
        _tool_failure("mcp_call_tool", "protocol_error", "b", second=2),
        _tool_failure("mcp_call_tool", "protocol_error", "b", second=3),
        _turn_end("b"),
    )
    assert failures["total"] == 3 and failures["turns"] == 2
    group = failures["groups"][0]
    assert (group["name"], group["code"], group["count"]) == (
        "mcp_call_tool",
        "protocol_error",
        3,
    )
    assert [turn["turn_id"] for turn in group["turns"]] == ["a", "b"]
    assert group["last_clock"] == "22:00:03"
    assert group["row"]["event"]["event_id"] == "b-mcp_call_tool-3"


def test_a_failure_says_why_in_words():
    group = _failures(
        _tool_failure("mcp_call_tool", "protocol_error", "a"), _turn_end("a")
    )["groups"][0]
    assert "the MCP server rejected the request" in group["why"]


def test_different_failures_of_one_tool_stay_apart():
    failures = _failures(
        _tool_failure("search", "protocol_error", "a", second=1),
        _tool_failure("search", "invalid_arguments", "a", second=2),
        _turn_end("a"),
    )
    assert {group["code"] for group in failures["groups"]} == {
        "protocol_error",
        "invalid_arguments",
    }


def test_a_turn_that_finished_anyway_is_told_apart_from_one_that_stopped():
    failures = _failures(
        _tool_failure("recovered_tool", "protocol_error", "a", second=1),
        _turn_end("a"),
        _tool_failure("fatal_tool", "protocol_error", "b", second=2),
        _turn_end("b", status="error", error_code="tool_budget_exhausted"),
    )
    assert failures["stopped"] == 1
    by_name = {group["name"]: group for group in failures["groups"]}
    assert by_name["recovered_tool"]["stopped"] is False
    assert by_name["fatal_tool"]["stopped"] is True


def test_a_failure_that_stopped_a_turn_comes_first():
    groups = _failures(
        _tool_failure("common", "protocol_error", "a", second=1),
        _tool_failure("common", "protocol_error", "a", second=2),
        _turn_end("a"),
        _tool_failure("rare", "protocol_error", "b", second=3),
        _turn_end("b", status="error", error_code="x"),
    )["groups"]
    assert groups[0]["name"] == "rare"


def test_a_turn_with_no_ending_is_not_called_finished():
    group = _failures(_tool_failure("tool", "protocol_error", "a"))["groups"][0]
    assert group["stopped"] is True


@pytest.mark.parametrize(
    "event",
    [
        {
            "kind": "tool_result",
            "status": "approval_required",
            "error_code": "approval_required",
        },
        {
            "kind": "turn_result",
            "status": "cancelled",
            "error_code": "cancelled",
            "stop_reason": "reader_closed",
        },
    ],
    ids=["waiting for approval", "stopped by the person"],
)
def test_a_pause_or_a_deliberate_stop_is_not_a_failure(event):
    failures = _failures({"turn_id": "a", "timestamp": "2026-11-01T22:00:00", **event})
    assert failures["total"] == 0


def test_a_run_past_its_deadline_is_a_failure():
    failures = _failures(
        {
            "kind": "turn_result",
            "status": "cancelled",
            "error_code": "cancelled",
            "stop_reason": "deadline_exceeded",
            "turn_id": "a",
            "timestamp": "2026-11-01T22:00:00",
        }
    )
    assert failures["total"] == 1


def test_no_failures_is_a_plain_zero():
    failures = _failures(_turn_end("a"))
    assert failures == {"total": 0, "turns": 0, "stopped": 0, "groups": []}
