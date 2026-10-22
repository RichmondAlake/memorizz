# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""A turn that fails or is stopped keeps what a reader needs: why it stopped,
the user's message and any reply written so far, and the timeline says so."""

import json
from typing import Any, Dict, List

import pytest

from memorizz.memagent import MemAgent
from memorizz.streaming import CancellationToken, StreamCancelled, current_cancellation
from memorizz.ui.observability_view import trace_timeline
from tests.unit.test_memagent_core_fixes import (  # noqa: F401  (fixtures)
    _fs_provider,
    _offline_embeddings,
    _StreamingStubModel,
    _StubModel,
    _turn_results,
    work_dir,
)


def _events(agent: MemAgent) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    agent.set_stream_event_callback(lambda event: events.append(dict(event)))
    return events


@pytest.mark.unit
def test_a_failed_turn_keeps_the_message_it_was_asked(work_dir):
    agent = MemAgent(
        model=_StubModel(error=RuntimeError("429 Too Many Requests")),
        memory_provider=_fs_provider(work_dir),
        memory_ids=["memory-1"],
        auto_register=False,
    )
    events = _events(agent)
    agent.run("What changed in the Q3 plan?", memory_id="memory-1", thread_id="t")
    result = _turn_results(events)[-1]
    assert result["status"] == "error" and "stop_reason" not in result
    assert json.loads(result["content"]) == {
        "user_message": "What changed in the Q3 plan?"
    }
    assert result["partial_reply_chars"] == 0
    assert "429" not in result["content"], "exception messages are never kept"
    agent.close()


@pytest.mark.unit
def test_a_stopped_stream_keeps_why_the_message_and_the_partial_reply(work_dir):
    token = CancellationToken()

    def stopped_mid_reply():
        yield {"type": "content", "content": "The plan moved the launch"}
        token.cancel("reader_closed")
        raise StreamCancelled()

    model = _StreamingStubModel([])
    model.generate_stream = lambda messages, tools=None, **kwargs: stopped_mid_reply()
    agent = MemAgent(
        model=model,
        memory_provider=_fs_provider(work_dir),
        memory_ids=["memory-1"],
        auto_register=False,
    )
    events = _events(agent)
    context = current_cancellation.set(token)
    try:
        with pytest.raises(StreamCancelled), pytest.warns(DeprecationWarning):
            for _chunk in agent.run_stream(
                "Summarise the plan", memory_id="memory-1", thread_id="t"
            ):
                pass
    finally:
        current_cancellation.reset(context)
    result = _turn_results(events)[-1]
    assert (result["status"], result["error_code"]) == ("cancelled", "cancelled")
    assert result["stop_reason"] == "reader_closed"
    assert result["partial_reply_chars"] == len("The plan moved the launch")
    assert json.loads(result["content"]) == {
        "user_message": "Summarise the plan",
        "partial_reply": "The plan moved the launch",
    }
    from memorizz.enums import MemoryType
    from memorizz.observability.normalization import query_trace_events

    stored = query_trace_events(
        agent.memory_provider,
        MemoryType.SHARED_MEMORY,
        limit=100,
        record_type="observability_trace_bundle",
    )["items"]
    turn = next(e for e in stored if e["kind"] == "turn_result")
    assert turn["stop_reason"] == "reader_closed", "kept when the bundle is read"
    assert turn["partial_reply_chars"] == len("The plan moved the launch")
    agent.close()


def _cancelled_turn(**result):
    base = {"turn_id": "turn-1", "thread_id": "t", "memory_id": "m"}
    return [
        {**base, "kind": "turn_start", "status": "started", "event_id": "1"},
        {**base, "kind": "model_call", "status": "started", "event_id": "2"},
        {
            **base,
            "kind": "tool_call",
            "title": "Tool Call: internet_search",
            "status": "started",
            "event_id": "3",
        },
        {**base, "kind": "tool_result", "status": "success", "event_id": "4"},
        {
            **base,
            "kind": "model_result",
            "status": "success",
            "response_chars": 0,
            "event_id": "5",
        },
        {
            **base,
            "kind": "turn_result",
            "status": "cancelled",
            "error_code": "cancelled",
            "event_id": "6",
            **result,
        },
    ]


@pytest.mark.unit
def test_the_timeline_says_why_a_turn_stopped_and_what_it_did():
    timeline = trace_timeline(
        _cancelled_turn(stop_reason="reader_closed", partial_reply_chars=40),
        content_event_ids={"6"},
    )
    rows = {row["event"]["event_id"]: row for row in timeline["rows"]}
    assert rows["6"]["why"] == (
        "The run was stopped before it finished because the reply stopped being "
        "read (the Stop button, a closed tab or a dropped connection)."
    )
    assert rows["6"]["note"].startswith(
        "Before it stopped, 1 model call wrote no reply text; it called "
        "internet_search. 40 characters of reply had been sent."
    )
    assert "Reveal content shows the message" in rows["6"]["note"]
    assert rows["5"]["note"] == "Wrote no text; asked for internet_search."
    assert not rows["5"]["why"]


@pytest.mark.unit
def test_failed_events_explain_their_codes():
    timeline = trace_timeline(
        [
            {"kind": "model_result", "status": "error", "error_code": "ReadTimeout"},
            {
                "kind": "tool_result",
                "status": "error",
                "outcome": "error",
                "outcome_reason_code": "tool_budget_exhausted",
            },
            *_cancelled_turn()[-1:],
        ]
    )
    why = [row["why"] for row in timeline["rows"]]
    assert why[0] == "Failed: ReadTimeout: the provider stopped responding."
    assert why[1].startswith("The tool call failed: the turn had used all of its")
    assert why[2] == "The run was stopped before it finished."
