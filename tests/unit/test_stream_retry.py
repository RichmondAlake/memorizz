# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""A dropped provider stream is re-sent without duplicating output."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from memorizz.llms.streaming import ProviderStreamError, is_transient_stream_error
from memorizz.memagent.core import MemAgent


class RemoteProtocolError(Exception):
    """Stands in for httpx.RemoteProtocolError (matched by name)."""


def _agent():
    return SimpleNamespace(TRANSIENT_STREAM_RETRIES=MemAgent.TRANSIENT_STREAM_RETRIES)


def _run(attempts, *, public):
    """attempts: list of (events, error or None) consumed one per call."""
    calls = iter(attempts)

    def make_stream():
        events, error = next(calls)

        def stream():
            yield from events
            if error is not None:
                raise error

        return stream()

    with patch("memorizz.memagent.core.time.sleep"):
        return list(
            MemAgent._stream_with_transient_retry(_agent(), make_stream, public=public)
        )


def _content(text):
    return {"type": "content", "content": text}


@pytest.mark.unit
def test_private_phase_retry_keeps_only_the_successful_attempt():
    events = _run(
        [
            ([_content("partial ")], RemoteProtocolError("peer closed")),
            ([_content("whole"), {"type": "done", "content": "whole"}], None),
        ],
        public=False,
    )
    # Events pass through live; the reset tells the caller to drop "partial ".
    assert [e.get("type") for e in events] == [
        "content",
        "stream_reset",
        "content",
        "done",
    ]
    reset = events.index({"type": "stream_reset"})
    assert [e.get("content") for e in events[reset + 1 :]] == ["whole", "whole"]


@pytest.mark.unit
def test_public_stream_retries_only_before_text_is_sent():
    events = _run(
        [([], RemoteProtocolError("closed")), ([_content("hi")], None)], public=True
    )
    assert events == [{"type": "stream_reset"}, _content("hi")]
    with pytest.raises(RemoteProtocolError):
        _run(
            [([_content("half")], RemoteProtocolError("closed")), ([], None)],
            public=True,
        )


@pytest.mark.unit
def test_no_retry_after_a_terminal_event_so_tools_never_run_twice():
    with pytest.raises(RemoteProtocolError):
        _run(
            [
                ([{"type": "tool_calls", "response": None}], RemoteProtocolError("x")),
                ([], None),
            ],
            public=False,
        )


@pytest.mark.unit
def test_request_errors_and_exhausted_retries_are_raised():
    with pytest.raises(ValueError):
        _run([([], ValueError("bad request")), ([], None)], public=False)
    with pytest.raises(RemoteProtocolError):
        _run([([], RemoteProtocolError("x"))] * 3, public=False)


@pytest.mark.unit
def test_transient_classification():
    assert is_transient_stream_error(RemoteProtocolError())
    assert is_transient_stream_error(ProviderStreamError("provider_stream_error"))
    assert is_transient_stream_error(
        ProviderStreamError(
            "provider_stream_error", {"provider_error_type": "overloaded_error"}
        )
    )
    assert not is_transient_stream_error(
        ProviderStreamError(
            "provider_stream_error", {"provider_error_type": "invalid_request_error"}
        )
    )
    assert not is_transient_stream_error(ProviderStreamError("provider_length"))
    wrapped = RuntimeError("stream failed")
    wrapped.__cause__ = RemoteProtocolError()
    assert is_transient_stream_error(wrapped)
