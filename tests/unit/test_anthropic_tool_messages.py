# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Anthropic provider: OpenAI-style tool messages -> Anthropic content blocks.

Regression tests for the bug where a follow-up request carrying tool results
back to Claude 400'd with ``Unexpected role "tool"`` — memorizz records tool
calls/results in OpenAI's shape, which Anthropic doesn't accept. The provider
now converts them via the (SDK-free) staticmethod ``_convert_messages``.
"""

import pytest

# The anthropic SDK is imported lazily in __init__, so the module + this
# staticmethod import/run without the SDK installed.
from memorizz.llms.anthropic import Anthropic  # noqa: E402

convert = Anthropic._convert_messages


@pytest.mark.unit
def test_tool_round_trip_converts_roles_and_blocks():
    msgs = [
        {"role": "user", "content": "search for X"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "X"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "search", "content": "hits"},
    ]
    out = convert(msgs)

    assert out[0] == {"role": "user", "content": "search for X"}
    assert out[1]["role"] == "assistant"
    assert out[1]["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "X"}}
    ]
    assert out[2]["role"] == "user"
    assert out[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "hits"}
    ]
    # The whole point: no "tool" role survives.
    assert all(m["role"] != "tool" for m in out)


@pytest.mark.unit
def test_assistant_text_and_tool_calls_emit_text_then_tool_use():
    msgs = [
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        }
    ]
    blocks = convert(msgs)[0]["content"]
    assert blocks[0] == {"type": "text", "text": "let me check"}
    assert blocks[1] == {"type": "tool_use", "id": "c1", "name": "f", "input": {}}


@pytest.mark.unit
def test_parallel_tool_results_merge_into_one_user_message():
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "function": {"name": "g", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "ra"},
        {"role": "tool", "tool_call_id": "b", "content": "rb"},
    ]
    out = convert(msgs)
    assert len(out) == 2
    assert out[0]["role"] == "assistant" and len(out[0]["content"]) == 2
    assert out[1]["role"] == "user"
    assert [b["tool_use_id"] for b in out[1]["content"]] == ["a", "b"]
    assert [b["content"] for b in out[1]["content"]] == ["ra", "rb"]


@pytest.mark.unit
def test_malformed_arguments_fall_back_to_empty_input():
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": "not-json"}}
            ],
        }
    ]
    assert convert(msgs)[0]["content"][0]["input"] == {}


@pytest.mark.unit
def test_dict_arguments_passed_through():
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": {"q": "y"}}}
            ],
        }
    ]
    assert convert(msgs)[0]["content"][0]["input"] == {"q": "y"}


@pytest.mark.unit
def test_plain_messages_are_equal_but_request_owned():
    msgs = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
        },
        {"role": "assistant", "content": "hello"},
    ]
    converted = convert(msgs)

    assert converted == msgs
    assert converted[0] is not msgs[0]
    assert converted[0]["content"] is not msgs[0]["content"]
    assert converted[0]["content"][0] is not msgs[0]["content"][0]
    assert converted[1] is not msgs[1]


@pytest.mark.unit
def test_none_tool_content_becomes_empty_string():
    out = convert([{"role": "tool", "tool_call_id": "c1", "content": None}])
    assert out[0]["content"][0]["content"] == ""
