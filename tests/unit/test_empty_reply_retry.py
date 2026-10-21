"""An empty model reply is asked again once, with a nudge that is never stored."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from memorizz.llms.streaming import ProviderStreamError, tool_response
from memorizz.memagent.core import MemAgent

pytestmark = pytest.mark.unit

NUDGE = {"role": "developer", "content": MemAgent.EMPTY_REPLY_NUDGE}


@pytest.fixture
def agent(memagent_with_mocks, monkeypatch):
    agent = memagent_with_mocks
    monkeypatch.setattr(agent, "_build_context", lambda *a, **k: {})
    monkeypatch.setattr(agent, "_build_llm_tools", lambda *a, **k: [])
    agent.cache_manager.cache_response = Mock()
    agent._record_interaction = Mock()
    return agent


def _replying(agent, *replies):
    sent = []
    queue = list(replies)

    def generate(messages, tools=None, **kwargs):
        sent.append([dict(m) for m in messages])
        return queue.pop(0)

    agent.model.generate = generate
    return sent


def test_an_empty_reply_is_asked_again_with_a_nudge(agent):
    sent = _replying(agent, "", "You live in Lisbon.")

    assert agent.run("Where do I live?") == "You live in Lisbon."
    assert len(sent) == 2
    assert NUDGE not in sent[0]
    assert sent[1] == [*sent[0], NUDGE]
    # The question and the answer are stored, never the nudge.
    stored = repr(agent._record_interaction.call_args)
    assert "You live in Lisbon." in stored
    assert MemAgent.EMPTY_REPLY_NUDGE not in stored


def test_the_nudge_goes_with_that_request_only(agent, monkeypatch):
    monkeypatch.setattr(
        agent,
        "_build_llm_tools",
        lambda *a, **k: [{"type": "function", "function": {"name": "lookup"}}],
    )
    monkeypatch.setattr(
        agent,
        "_execute_and_record_tool_call",
        lambda call, messages, *a, **k: messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": "Lisbon"}
        ),
    )
    lookup = tool_response([{"id": "1", "name": "lookup", "arguments": "{}"}], None)
    sent = _replying(agent, "", lookup, "Lisbon.")

    assert agent.run("Where do I live?") == "Lisbon."
    assert [NUDGE in messages for messages in sent] == [False, True, False]


def test_a_second_empty_reply_is_still_an_error(agent):
    sent = _replying(agent, "", "  ")
    with pytest.raises(ProviderStreamError, match="empty_response"):
        agent.run("hello")
    assert len(sent) == 2
    agent._record_interaction.assert_not_called()


def test_other_unfinished_replies_are_not_asked_again(agent):
    agent.model.get_last_response_metadata = lambda: {"finish_reason": "length"}
    sent = _replying(agent, "The first step is", "unused")
    with pytest.raises(ProviderStreamError, match="provider_length"):
        agent.run("hello")
    assert len(sent) == 1


def test_each_attempt_is_its_own_model_call(agent):
    _replying(agent, "", "ok")
    agent._begin_trace_turn(None)
    assert (
        agent._generate_with_trace(
            [{"role": "user", "content": "hi"}], tools=None, iteration=1, stage="t"
        )
        == "ok"
    )
    rows = agent._build_trace_bundle_events(agent._stream_trace_events)
    results = [e for e in rows if e["trace_kind"] == "model_result"]
    assert [r["status"] for r in results] == ["error", "success"]


def test_streaming_asks_again_with_the_nudge(agent):
    sent = []
    replies = [
        # An empty delta is not text the caller saw, so this can be retried.
        [{"type": "content", "content": ""}, {"type": "done", "content": ""}],
        [
            {"type": "content", "content": "Answer."},
            {"type": "done", "content": "Answer."},
        ],
    ]

    def stream(messages, tools=None, **kwargs):
        sent.append([dict(m) for m in messages])
        yield from replies.pop(0)

    agent.model.generate_stream = stream
    with patch("memorizz.memagent.core.time.sleep") as sleep:
        events = list(agent.run_stream_events("hello"))
    sleep.assert_not_called()

    assert events[-1]["status"] == "completed"
    assert "".join(e["delta"] for e in events if e["type"] == "answer.delta") == (
        "Answer."
    )
    assert sent[1] == [*sent[0], NUDGE]


def test_run_stream_asks_again_with_the_nudge(agent):
    sent = []
    replies = [
        [{"type": "done", "content": ""}],
        [{"type": "content", "content": "Hi."}, {"type": "done", "content": "Hi."}],
    ]

    def stream(messages, tools=None, **kwargs):
        sent.append([dict(m) for m in messages])
        yield from replies.pop(0)

    agent.model.generate_stream = stream
    assert "".join(agent.run_stream("hello")) == "Hi."
    assert sent[1] == [*sent[0], NUDGE]


def _attempts(*attempts):
    """make_stream for ``_stream_with_transient_retry``: one attempt per call."""
    calls = iter(attempts)
    nudges = []

    def make_stream(nudge=False):
        nudges.append(nudge)
        events, error = next(calls)

        def stream():
            yield from events
            if error is not None:
                raise error

        return stream()

    return make_stream, nudges


class RemoteProtocolError(Exception):
    """Stands in for httpx.RemoteProtocolError (matched by name)."""


def test_a_dropped_connection_after_the_nudge_keeps_it():
    make_stream, nudges = _attempts(
        ([], ProviderStreamError("empty_response")),
        ([], RemoteProtocolError("closed")),
        ([{"type": "content", "content": "hi"}], None),
    )
    agent = SimpleNamespace(TRANSIENT_STREAM_RETRIES=MemAgent.TRANSIENT_STREAM_RETRIES)
    with patch("memorizz.memagent.core.time.sleep"):
        events = list(
            MemAgent._stream_with_transient_retry(agent, make_stream, public=True)
        )
    assert nudges == [False, True, True]
    assert events[-1] == {"type": "content", "content": "hi"}
