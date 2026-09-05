"""Ordered public-answer streaming; no network, model calls or host services."""

import asyncio
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memorizz.completion import CompletionPolicy
from memorizz.llms.llm_provider import LLMProvider
from memorizz.llms.streaming import (
    ProviderStreamError,
    responses_events,
    streaming_capabilities,
    tool_response,
)
from memorizz.streaming import CancellationToken


@pytest.fixture
def streaming_agent(memagent_with_mocks, monkeypatch):
    agent = memagent_with_mocks
    monkeypatch.setattr(agent, "_build_llm_tools", lambda *a, **k: [])
    monkeypatch.setattr(agent, "_build_context", lambda *a, **k: {})
    monkeypatch.setattr(agent, "_record_interaction", lambda *a, **k: None)
    return agent


def text_provider(text="hello"):
    yield {"type": "content", "content": text}
    yield {"type": "done", "content": text}


def assert_contract(events):
    assert events[0]["type"] == "run.started"
    assert events[-1]["type"] == "run.done"
    assert sum(e["type"] == "run.done" for e in events) == 1
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert len({e["run_id"] for e in events}) == 1
    json.dumps(events, allow_nan=False)


def test_first_public_delta_precedes_provider_completion(streaming_agent):
    release, completed, closed = threading.Event(), threading.Event(), threading.Event()

    def provider(*a, **k):
        try:
            yield {"type": "content", "content": "first "}
            assert release.wait(5)
            yield {"type": "content", "content": "second"}
            completed.set()
            yield {"type": "done", "content": "first second"}
        finally:
            closed.set()

    streaming_agent.model.generate_stream = provider
    with streaming_agent.run_stream_events("hello") as stream:
        events = []
        for event in stream:
            events.append(event)
            if event["type"] == "answer.delta":
                assert not completed.is_set()
                break
        release.set()
        events.extend(stream)
    assert closed.is_set()
    assert_contract(events)
    assert (
        "".join(e["delta"] for e in events if e["type"] == "answer.delta")
        == "first second"
    )


def test_answer_done_precedes_persistence(streaming_agent, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def persist(*a, **k):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(streaming_agent, "_record_interaction", persist)
    streaming_agent.model.generate_stream = lambda *a, **k: text_provider()
    with streaming_agent.run_stream_events("hello") as stream:
        events = []
        for event in stream:
            events.append(event)
            if event["type"] == "answer.done":
                break
        assert entered.wait(1)
        assert not stream.finished.is_set()
        release.set()
        events.extend(stream)
    assert_contract(events)


def test_error_preserves_partial_answer_without_exception_prose(streaming_agent):
    def provider(*a, **k):
        yield {"type": "content", "content": "partial"}
        raise RuntimeError("PRIVATE API ERROR")

    streaming_agent.model.generate_stream = provider
    events = list(streaming_agent.run_stream_events("hello"))
    assert_contract(events)
    assert events[-1]["status"] == "error"
    assert [e["delta"] for e in events if e["type"] == "answer.delta"] == ["partial"]
    assert "PRIVATE API ERROR" not in json.dumps(events)


def test_cancellation_closes_provider_and_never_caches_partial(streaming_agent):
    closed = threading.Event()
    token = CancellationToken()

    def provider(*a, **k):
        try:
            while True:
                yield {"type": "content", "content": "x"}
        finally:
            closed.set()

    streaming_agent.model.generate_stream = provider
    streaming_agent.cache_manager.cache_response = Mock()
    stream = streaming_agent.run_stream_events(
        "hello", cancellation=token, queue_size=1
    )
    events = []
    for event in stream:
        events.append(event)
        if event["type"] == "answer.delta":
            token.cancel()
    assert closed.wait(1)
    assert_contract(events)
    assert events[-1]["status"] == "cancelled"
    streaming_agent.cache_manager.cache_response.assert_not_called()


def test_slow_consumer_close_unblocks_full_queue(streaming_agent):
    streaming_agent.model.generate_stream = lambda *a, **k: text_provider("x" * 50000)
    stream = streaming_agent.run_stream_events("hello", queue_size=1)
    for event in stream:
        if event["type"] == "answer.delta":
            break
    stream.close()
    assert not stream.worker.is_alive()
    assert stream.queue.qsize() <= 1


def test_cached_answer_precedes_bookkeeping(streaming_agent, monkeypatch):
    release = threading.Event()
    streaming_agent.cache_manager.enabled = True
    streaming_agent.cache_manager.get_cached_response = Mock(return_value="cached")
    monkeypatch.setattr(
        streaming_agent, "_record_interaction", lambda *a, **k: release.wait(5)
    )
    streaming_agent.model.generate_stream = Mock(
        side_effect=AssertionError("cache must not regenerate")
    )
    with streaming_agent.run_stream_events("hello") as stream:
        events = []
        for event in stream:
            events.append(event)
            if event["type"] == "answer.done":
                break
        assert not stream.finished.is_set()
        assert [e["delta"] for e in events if e["type"] == "answer.delta"] == ["cached"]
        release.set()
        events.extend(stream)
    assert_contract(events)
    streaming_agent.model.generate_stream.assert_not_called()


def test_missing_model_is_typed_error(streaming_agent):
    streaming_agent.model = None
    events = list(streaming_agent.run_stream_events("hello"))
    assert_contract(events)
    assert events[-1]["status"] == "error"
    assert events[-1]["error_code"] == "model_not_configured"
    assert not any(e["type"] == "answer.delta" for e in events)


def test_synchronous_provider_fallback_is_reported(streaming_agent):
    streaming_agent.model = SimpleNamespace(generate=lambda *a, **k: "buffered")
    events = list(streaming_agent.run_stream_events("hello"))
    assert_contract(events)
    assert events[0]["delivery_mode"] == "buffered"
    assert events[0]["buffering_reason"] == "provider_has_no_stream"
    assert [e["delta"] for e in events if e["type"] == "answer.delta"] == ["buffered"]


def test_synchronous_provider_failure_does_not_become_answer(streaming_agent):
    streaming_agent.model = SimpleNamespace(
        generate=Mock(side_effect=ValueError("PRIVATE failure"))
    )
    events = list(streaming_agent.run_stream_events("hello"))
    assert events[-1]["status"] == "error"
    assert not any(e["type"] == "answer.delta" for e in events)
    assert "PRIVATE" not in json.dumps(events)


def test_approval_is_not_answer_or_cached(streaming_agent, monkeypatch):
    from memorizz.approval import ApprovalRequired

    proposal = SimpleNamespace(
        proposal_id="proposal",
        tool_name="mutate",
        to_dict=lambda **kw: {"proposal_id": "proposal", "tool_name": "mutate"},
    )
    monkeypatch.setattr(
        streaming_agent,
        "_build_llm_tools",
        lambda *a, **k: [{"type": "function", "function": {"name": "mutate"}}],
    )
    monkeypatch.setattr(
        streaming_agent,
        "_execute_and_record_tool_call",
        Mock(side_effect=ApprovalRequired(proposal)),
    )
    streaming_agent.model.generate_stream = lambda *a, **k: iter(
        [
            {
                "type": "tool_calls",
                "response": tool_response(
                    [{"id": "call", "name": "mutate", "arguments": "{}"}], ""
                ),
            }
        ]
    )
    events = list(streaming_agent.run_stream_events("hello"))
    assert_contract(events)
    assert events[-1]["status"] == "approval_required"
    assert any(e["type"] == "approval.required" for e in events)
    assert not any(e["type"] in {"answer.delta", "answer.done"} for e in events)


def test_iteration_limit_is_not_answer(streaming_agent, monkeypatch):
    monkeypatch.setattr(streaming_agent, "_get_tool_iteration_limit", lambda: 0)
    events = list(streaming_agent.run_stream_events("hello"))
    assert events[-1]["status"] == "error"
    assert events[-1]["error_code"] == "iteration_limit"
    assert not any(e["type"] == "answer.delta" for e in events)


def test_buffered_policy_never_discloses_rejected_deltas(streaming_agent):
    streaming_agent.completion_policy = CompletionPolicy(
        enabled=True, forbidden_response_patterns=["SECRET"], max_rejections=1
    )
    calls = []

    def provider(*a, **k):
        calls.append(1)
        text = "SECRET draft" if len(calls) == 1 else "accepted"
        yield {"type": "content", "content": text[:3]}
        yield {"type": "content", "content": text[3:]}
        yield {"type": "done", "content": text}

    streaming_agent.model.generate_stream = provider
    events = list(streaming_agent.run_stream_events("hello"))
    assert_contract(events)
    assert "SECRET" not in json.dumps(events)
    assert [e["delta"] for e in events if e["type"] == "answer.delta"] == ["accepted"]
    decisions = [e for e in events if e["type"] == "completion.check"]
    assert [e["accepted"] for e in decisions] == [False, True]
    assert decisions[-1]["response_sha256"] == hashlib.sha256(b"accepted").hexdigest()


@pytest.mark.parametrize(
    "policy",
    [
        dict(validator=lambda _: True),
        dict(validator_required=True),
        dict(forbidden_response_patterns=["bad"]),
    ],
)
def test_final_stream_rejects_full_answer_validators(policy):
    with pytest.raises(ValueError, match="incompatible"):
        CompletionPolicy(enabled=True, delivery_mode="final_stream", **policy)


def test_policy_roundtrip_preserves_mode_and_cache_fingerprint():
    policy = CompletionPolicy(
        enabled=True, require_tool_calls=True, delivery_mode="final_stream"
    )
    assert CompletionPolicy.from_value(policy.to_dict()).to_dict() == policy.to_dict()
    assert CompletionPolicy.from_value({"enabled": True}).delivery_mode == "buffered"


def test_tool_phase_never_becomes_public_answer(streaming_agent, monkeypatch):
    monkeypatch.setattr(
        streaming_agent,
        "_build_llm_tools",
        lambda *a, **k: [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    )
    calls = []

    def provider(messages, tools=None, **k):
        calls.append(tools)
        if tools:
            yield {"type": "content", "content": "PRIVATE TOOL PHASE"}
            yield {
                "type": "tool_calls",
                "response": tool_response(
                    [
                        {
                            "id": "finalize",
                            "name": "memorizz_finalize_answer",
                            "arguments": "{}",
                        }
                    ],
                    "PRIVATE TOOL PHASE",
                ),
            }
        else:
            yield from text_provider("public answer")

    streaming_agent.model.generate_stream = provider
    events = list(streaming_agent.run_stream_events("hello"))
    assert_contract(events)
    assert events[-1]["status"] == "completed"
    assert len(calls) == 2 and calls[-1] is None
    assert "PRIVATE TOOL PHASE" not in json.dumps(events)


def test_whitespace_digest_matches_exact_deltas(streaming_agent):
    streaming_agent.model.generate_stream = lambda *a, **k: text_provider(" hello \n")
    events = list(streaming_agent.run_stream_events("hello"))
    done = next(e for e in events if e["type"] == "answer.done")
    assert done["sha256"] == hashlib.sha256(b" hello \n").hexdigest()
    assert done["chars"] == 8


def test_async_stream_uses_one_owned_worker_context(streaming_agent):
    threads = []

    def provider(*a, **k):
        threads.append(threading.get_ident())
        yield {"type": "content", "content": "one"}
        threads.append(threading.get_ident())
        yield {"type": "done", "content": "one"}

    streaming_agent.model.generate_stream = provider

    async def consume():
        return [e async for e in streaming_agent.arun_stream_events("hello")]

    events = asyncio.run(consume())
    assert_contract(events)
    assert len(set(threads)) == 1


def test_concurrent_event_ids_and_callbacks_are_isolated(streaming_agent):
    streaming_agent.model.generate_stream = lambda *a, **k: text_provider()

    def consume(number):
        return list(
            streaming_agent.run_stream_events(
                "hello",
                memory_id=f"m-{number}",
                thread_id=f"t-{number}",
                user_id=f"u-{number}",
            )
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(consume, [1, 2]))
    assert results[0][0]["run_id"] != results[1][0]["run_id"]
    for i, events in enumerate(results, 1):
        assert_contract(events)
        assert events[0]["memory_id"] == events[-1]["memory_id"] == f"m-{i}"


def test_inherited_protocol_stub_is_not_stream_support():
    class Stub(LLMProvider):
        pass

    assert not streaming_capabilities(Stub())["text_deltas"]


def test_responses_native_deltas_tool_arguments_and_closure():
    provider = SimpleNamespace(
        _last_response_metadata={},
        _last_usage=None,
        _extract_responses_usage=lambda _: {"total_tokens": 4},
    )
    closed = threading.Event()

    def upstream():
        try:
            yield SimpleNamespace(type="response.output_text.delta", delta="hello")
            yield SimpleNamespace(
                type="response.output_item.added",
                output_index=1,
                item=SimpleNamespace(
                    type="function_call",
                    id="item",
                    call_id="call",
                    name="lookup",
                    arguments="",
                ),
            )
            for fragment in ['{"x":', "1}"]:
                yield SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    item_id="item",
                    delta=fragment,
                )
            yield SimpleNamespace(
                type="response.function_call_arguments.done",
                item_id="item",
                arguments='{"x":1}',
            )
            yield SimpleNamespace(
                type="response.completed", response=SimpleNamespace(status="completed")
            )
        finally:
            closed.set()

    events = list(responses_events(provider, upstream()))
    assert events[0] == {"type": "content", "content": "hello"}
    call = events[-1]["response"].choices[0].message.tool_calls[0]
    assert call.id == "call" and call.function.arguments == '{"x":1}'
    assert closed.is_set()


@pytest.mark.parametrize(
    "kind",
    ["response.failed", "response.incomplete", "response.refusal.delta", "error", None],
)
def test_responses_incomplete_refusal_and_errors_are_explicit(kind):
    provider = SimpleNamespace(
        _last_response_metadata={},
        _last_usage=None,
        _extract_responses_usage=lambda _: None,
    )
    events = (
        [SimpleNamespace(type=kind, response=SimpleNamespace(status="failed"))]
        if kind
        else []
    )
    with pytest.raises(ProviderStreamError):
        list(responses_events(provider, iter(events)))
