"""A chat's step link names a step by its span or tool call; Traces finds it."""

import pytest

from memorizz.ui.routers.traces import _step_event_id

EVENTS = [
    {"event_id": "e1", "trace_kind": "model_call", "span_id": "model:1"},
    {"event_id": "e2", "trace_kind": "tool_call", "span_id": "tool:7", "tool_call_id": "call_7"},
    {"event_id": "e3", "trace_kind": "tool_result", "span_id": "tool:7", "tool_call_id": "call_7"},
    {"event_id": "e4", "trace_kind": "tool_result", "trace_id": "result:search:9"},
]  # fmt: skip


@pytest.mark.unit
def test_a_step_is_found_by_span_tool_call_trace_or_event_id():
    # A tool's result says how it went, so it is preferred to its call.
    assert _step_event_id(EVENTS, "tool:7") == "e3"
    assert _step_event_id(EVENTS, "call_7") == "e3"
    assert _step_event_id(EVENTS, "model:1") == "e1"
    assert _step_event_id(EVENTS, "result:search:9") == "e4"
    assert _step_event_id(EVENTS, "e2") == "e2"


@pytest.mark.unit
def test_a_step_that_is_not_loaded_selects_nothing():
    assert _step_event_id(EVENTS, "tool:99") is None
    assert _step_event_id(EVENTS, "") is None
    assert _step_event_id([], "tool:7") is None
