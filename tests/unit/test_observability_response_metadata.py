from types import SimpleNamespace as NS

import pytest

from memorizz import MemAgent
from memorizz.llms.openai import OpenAI
from memorizz.llms.response_metadata import response_metadata


def openai_model():
    model = OpenAI.__new__(OpenAI)
    model.model = "fixture"
    model.base_url = None
    model._request_options = {"max_completion_tokens": 2200}
    model._prompt_cache_key = model._prompt_cache_retention = None
    return model


def test_nonstreaming_finish_reason_limits_and_usage_survive_compaction():
    model = openai_model()
    model._create_chat_completion = lambda kwargs: NS(
        id="request-1",
        choices=[
            NS(
                finish_reason="length",
                message=NS(content="é<learning_map>", tool_calls=None),
            )
        ],
        usage=NS(prompt_tokens=4, completion_tokens=2200, total_tokens=2204),
    )
    agent = MemAgent(model=model)
    agent._begin_trace_turn(None)
    assert (
        agent._generate_with_trace([], tools=None, iteration=0, stage="test")
        == "é<learning_map>"
    )
    rows = agent._build_trace_bundle_events(agent._stream_trace_events)
    call = next(e for e in rows if e["trace_kind"] == "model_call")
    result = next(e for e in rows if e["trace_kind"] == "model_result")
    assert "input_tokens" not in call
    assert result["finish_reason"] == "length"
    assert result["request_id"] == "request-1"
    assert result["max_output_tokens"] == 2200
    assert result["response_chars"] == 15
    assert result["response_bytes"] == 16
    assert result["input_tokens"] == 4


@pytest.mark.parametrize("empty", [False, True])
def test_stream_finish_and_usage_are_current_and_incomplete_stream_is_explicit(empty):
    model = openai_model()
    chunks = (
        []
        if empty
        else [
            NS(
                id="req",
                usage=None,
                choices=[
                    NS(delta=NS(content="abc", tool_calls=None), finish_reason=None)
                ],
            ),
            NS(
                id="req",
                usage=None,
                choices=[
                    NS(delta=NS(content=None, tool_calls=None), finish_reason="length")
                ],
            ),
            NS(
                id="req",
                choices=[],
                usage=NS(prompt_tokens=4, completion_tokens=3, total_tokens=7),
            ),
        ]
    )
    model._last_usage = {"prompt_tokens": 999}
    model._last_response_metadata = {"finish_reason": "stale"}
    model._create_chat_completion = lambda kwargs: iter(chunks)
    agent = MemAgent(model=model)
    agent._begin_trace_turn(None)
    from memorizz.llms.streaming import ProviderStreamError

    with pytest.raises(ProviderStreamError):
        list(
            agent._generate_stream_with_trace([], tools=None, iteration=0, stage="test")
        )
    result = next(
        e for e in agent._stream_trace_events if e.get("trace_kind") == "model_result"
    )
    assert result["response_chars"] == (0 if empty else 3)
    assert result["stream_duration_ms"] >= 0
    if empty:
        assert "finish_reason" not in result and "input_tokens" not in result
    else:
        assert result["finish_reason"] == "length"
        assert result["input_tokens"] == 4
        assert result["ttft_ms"] >= 0


def test_stream_cancellation_does_not_claim_model_success():
    class Model:
        def generate_stream(self, *args, **kwargs):
            yield {"type": "content", "content": "unfinished"}
            yield {"type": "done", "content": "unfinished"}

    agent = MemAgent(model=Model())
    agent._begin_trace_turn(None)
    stream = agent._generate_stream_with_trace(
        [], tools=None, iteration=0, stage="test"
    )
    next(stream)
    stream.close()
    result = next(
        e for e in agent._stream_trace_events if e.get("trace_kind") == "model_result"
    )
    assert result["status"] == "error"
    assert result["error_code"] == "GeneratorExit"


@pytest.mark.parametrize(
    "response, reason",
    [
        (
            NS(stop_reason="max_tokens", content=[NS(type="text", text="partial")]),
            "max_tokens",
        ),
        ({"done_reason": "length", "message": {"content": "partial"}}, "length"),
        (
            NS(
                incomplete_details=NS(reason="max_output_tokens"), output_text="partial"
            ),
            "max_output_tokens",
        ),
    ],
)
def test_adapter_envelope_supports_stop_reasons_without_raw_content(response, reason):
    metadata = response_metadata(response)
    assert metadata["finish_reason"] == reason
    assert metadata["response_chars"] == 7
    assert "partial" not in str(metadata)
