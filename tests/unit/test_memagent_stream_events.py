"""Tests for the M3 stream-event taxonomy on MemAgent.run_stream.

After M3, ``set_stream_event_callback`` receives:

  - ``stream_start``  (fired before any text)
  - ``trace`` with ``trace_kind in {reasoning, tool_call, tool_result}``
  - ``error``         (fired on mid-stream exception)
  - ``stream_end``    (fired once, with ``reason in {completed, cache_hit, error}``)

Text chunks remain yielded by the generator, not surfaced via the callback.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

# ---------------------------------------------------------------------------
# Lifecycle: stream_start / stream_end
# ---------------------------------------------------------------------------


def test_stream_start_event_fires_before_any_chunk(memagent_with_mocks):
    """stream_start must be the first event the callback receives."""
    agent = memagent_with_mocks
    events = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    def fake_generate_stream(messages, tools=None, **_):
        yield {"type": "content", "content": "hi"}
        yield {"type": "done", "content": "hi"}

    agent.model.generate_stream = fake_generate_stream

    list(agent.run_stream("hello"))

    assert events, "expected at least one callback event"
    assert events[0]["type"] == "stream_start"
    assert events[0]["agent_id"] == agent.agent_id
    assert events[0]["has_request_context"] is False


def test_stream_start_marks_has_request_context_when_context_present(
    memagent_with_mocks,
):
    agent = memagent_with_mocks
    events = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    agent.model.generate_stream = lambda *_a, **_k: iter(
        [{"type": "content", "content": "x"}, {"type": "done", "content": "x"}]
    )

    list(agent.run_stream("hello", context={"current_page": {"type": "library"}}))

    assert events[0]["type"] == "stream_start"
    assert events[0]["has_request_context"] is True


def test_stream_end_event_fires_on_normal_completion(memagent_with_mocks):
    agent = memagent_with_mocks
    events = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    agent.model.generate_stream = lambda *_a, **_k: iter(
        [{"type": "content", "content": "abc"}, {"type": "done", "content": "abc"}]
    )

    list(agent.run_stream("hello"))

    end_events = [e for e in events if e["type"] == "stream_end"]
    assert len(end_events) == 1
    assert end_events[0]["reason"] == "completed"
    assert end_events[0]["response_length"] == len("abc")


# ---------------------------------------------------------------------------
# error event on mid-stream exception
# ---------------------------------------------------------------------------


def test_error_event_fires_when_stream_raises(memagent_with_mocks):
    agent = memagent_with_mocks
    events = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    def exploding_stream(messages, tools=None, **_):
        yield {"type": "content", "content": "partial"}
        raise RuntimeError("LOB serialisation failed")

    agent.model.generate_stream = exploding_stream

    chunks = list(agent.run_stream("hello"))
    # Generator still yields a friendly error string at the tail.
    assert any("apologize" in c.lower() for c in chunks)

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "LOB serialisation failed" in error_events[0]["message"]
    assert error_events[0]["exception_type"] == "RuntimeError"
    assert error_events[0]["recoverable"] is False

    # And a stream_end with reason=error fires after the error event.
    end_events = [e for e in events if e["type"] == "stream_end"]
    assert end_events and end_events[-1]["reason"] == "error"
    # Order: start -> ... -> error -> stream_end(error)
    types_in_order = [e["type"] for e in events]
    assert types_in_order.index("error") < types_in_order.index("stream_end")


# ---------------------------------------------------------------------------
# Existing trace events still flow (no regressions)
# ---------------------------------------------------------------------------


def test_existing_trace_events_still_emitted(memagent_with_mocks):
    """M3 must not break the existing trace_kind=reasoning event surface."""
    agent = memagent_with_mocks
    events = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    def stream_with_reasoning(messages, tools=None, **_):
        yield {"type": "reasoning", "content": "thinking about the answer"}
        yield {"type": "content", "content": "the answer"}
        yield {"type": "done", "content": "the answer"}

    agent.model.generate_stream = stream_with_reasoning

    list(agent.run_stream("hello"))

    trace_events = [e for e in events if e["type"] == "trace"]
    reasoning_traces = [e for e in trace_events if e.get("trace_kind") == "reasoning"]
    assert reasoning_traces, "reasoning trace should still flow through the callback"


# ---------------------------------------------------------------------------
# Cache-hit path also closes the lifecycle cleanly
# ---------------------------------------------------------------------------


def test_stream_end_reason_cache_hit_when_cache_serves_response(memagent_with_mocks):
    """When semantic_cache returns a hit, stream_start + stream_end(cache_hit) fire."""
    from unittest.mock import MagicMock

    agent = memagent_with_mocks
    # Enable cache and force a hit.
    agent.cache_manager.enabled = True
    agent.cache_manager.get_cached_response = MagicMock(return_value="cached reply")

    events = []
    agent.set_stream_event_callback(lambda e: events.append(dict(e)))

    # Generator should be present but never invoked.
    agent.model.generate_stream = lambda *_a, **_k: iter([])

    chunks = list(agent.run_stream("hello"))
    assert chunks == ["cached reply"]

    types = [e["type"] for e in events]
    assert types[0] == "stream_start"
    assert "stream_end" in types
    assert events[-1]["reason"] == "cache_hit"


# ---------------------------------------------------------------------------
# Callback exceptions never break the stream
# ---------------------------------------------------------------------------


def test_callback_exception_does_not_break_stream(memagent_with_mocks):
    """A buggy consumer callback should be caught — stream completes regardless."""
    agent = memagent_with_mocks

    def bad_callback(_event):
        raise ValueError("oops")

    agent.set_stream_event_callback(bad_callback)

    agent.model.generate_stream = lambda *_a, **_k: iter(
        [{"type": "content", "content": "ok"}, {"type": "done", "content": "ok"}]
    )

    chunks = list(agent.run_stream("hello"))
    # Stream completed; "ok" yielded; no exception bubbled up.
    assert "ok" in "".join(chunks)


def test_per_call_callback_and_execution_ids_survive_worker_thread(
    memagent_with_mocks,
):
    """The UI can stream on a worker without losing trace or resolved IDs."""
    agent = memagent_with_mocks
    parent_events = []
    worker_events = []
    agent.set_stream_event_callback(parent_events.append)
    agent.model.generate_stream = lambda *_a, **_k: iter(
        [{"type": "content", "content": "ok"}, {"type": "done", "content": "ok"}]
    )

    def execute():
        chunks = list(
            agent.run_stream(
                "hello",
                memory_id="memory-worker",
                thread_id="thread-worker",
                event_callback=worker_events.append,
            )
        )
        return chunks, agent.get_current_memory_id(), agent.get_current_thread_id()

    with ThreadPoolExecutor(max_workers=1) as executor:
        chunks, memory_id, thread_id = executor.submit(execute).result(timeout=60)

    assert "ok" in "".join(chunks)
    assert memory_id == "memory-worker"
    assert thread_id == "thread-worker"
    assert worker_events[0]["type"] == "stream_start"
    assert worker_events[-1]["type"] == "stream_end"
    assert parent_events == []
