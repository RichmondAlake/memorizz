# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Regression coverage for Anthropic request ownership and cache budgets."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memorizz.llms.anthropic import Anthropic


class _FakeStream:
    def __init__(self, events=None):
        self._events = list(events or []) + [SimpleNamespace(type="message_stop")]

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _make_provider(*, prompt_caching=True):
    provider = Anthropic.__new__(Anthropic)
    provider.model = "claude-sonnet-4-5"
    provider.context_window_tokens = 200_000
    provider._last_usage = None
    provider._max_tokens = 512
    provider._enable_prompt_caching = prompt_caching
    provider._request_options = {}
    provider.client = MagicMock()
    return provider


def _text_response(text="done"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=None,
    )


def _tool_response(call_id):
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id=call_id,
                name="lookup",
                input={"query": call_id},
            )
        ],
        usage=None,
    )


def _tool_definition():
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }


@pytest.mark.parametrize("request_path", ["generate", "generate_stream"])
def test_anthropic_request_paths_do_not_mutate_nested_messages(request_path):
    provider = _make_provider()
    messages = [
        {"role": "system", "content": "STATIC SYSTEM"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "answer"},
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "reference",
                    },
                },
            ],
        },
    ]
    before = deepcopy(messages)

    if request_path == "generate":
        provider.client.messages.create.return_value = _text_response()
        assert provider.generate(messages) == "done"
        request = provider.client.messages.create.call_args.kwargs
    else:
        provider.client.messages.stream.return_value = _FakeStream()
        assert list(provider.generate_stream(messages)) == [
            {"type": "done", "content": ""}
        ]
        request = provider.client.messages.stream.call_args.kwargs

    assert messages == before
    assert request["messages"][0] is not messages[1]
    assert request["messages"][1] is not messages[2]
    assert request["messages"][1]["content"] is not messages[2]["content"]
    assert request["messages"][1]["content"][1] is not messages[2]["content"][1]
    assert provider._count_cache_control_breakpoints(request) == 3


def test_established_tool_loop_keeps_every_outbound_request_within_budget():
    provider = _make_provider()
    provider.client.messages.create.side_effect = [
        _tool_response("call-1"),
        _text_response("final answer"),
    ]
    messages = [
        {"role": "system", "content": "STATIC SYSTEM"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "use the lookup tool"},
    ]
    original_turn = deepcopy(messages)

    tool_result = provider.generate(messages, tools=[_tool_definition()])
    assert messages == original_turn

    tool_call = tool_result.choices[0].message.tool_calls[0]
    messages.extend(
        [
            {
                "role": "assistant",
                "content": tool_result.choices[0].message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "lookup result",
            },
        ]
    )
    post_tool_history = deepcopy(messages)

    assert provider.generate(messages, tools=[_tool_definition()]) == "final answer"
    assert messages == post_tool_history

    requests = [call.kwargs for call in provider.client.messages.create.call_args_list]
    assert len(requests) == 2
    assert [
        provider._count_cache_control_breakpoints(request) for request in requests
    ] == [3, 3]
    assert all(
        provider._count_cache_control_breakpoints(request) <= 4 for request in requests
    )


def test_multiple_tool_iterations_do_not_accumulate_cache_markers():
    provider = _make_provider()
    messages = [
        {"role": "system", "content": "STATIC SYSTEM"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "perform several lookups"},
    ]

    for index in range(5):
        before = deepcopy(messages)
        request = provider._build_request_kwargs(
            messages,
            tools=[_tool_definition()],
        )

        assert messages == before
        assert provider._count_cache_control_breakpoints(request) == 3

        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "working",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": call_id},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"result-{index}",
                },
            ]
        )


def test_caller_markers_are_preserved_and_only_remaining_budget_is_spent():
    provider = _make_provider()
    marker = {"type": "ephemeral"}
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "STATIC SYSTEM",
                    "cache_control": marker,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "caller-cached history",
                    "cache_control": marker,
                }
            ],
        },
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "latest question"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # This is payload/schema data, not an API breakpoint.
                        "cache_control": {"type": "string"},
                    },
                },
                "cache_control": marker,
            },
        }
    ]
    messages_before = deepcopy(messages)
    tools_before = deepcopy(tools)

    request = provider._build_request_kwargs(messages, tools=tools)

    assert messages == messages_before
    assert tools == tools_before
    assert provider._count_cache_control_breakpoints(request) == 4
    assert request["tools"][0]["cache_control"] == marker
    assert request["system"][0]["cache_control"] == marker
    assert request["messages"][0]["content"][0]["cache_control"] == marker
    assert request["messages"][-1]["content"][-1]["cache_control"] == marker
    assert request["messages"][-2]["content"] == "earlier answer"


def test_cache_counter_ignores_nested_user_payload_keys():
    provider = _make_provider()
    request = {
        "tools": [
            {
                "name": "lookup",
                "input_schema": {
                    "type": "object",
                    "properties": {"cache_control": {"type": "string"}},
                },
            }
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "lookup",
                        "input": {"cache_control": "user value"},
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": [
                            {
                                "type": "text",
                                "text": "nested result",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                ],
            }
        ],
    }

    assert provider._count_cache_control_breakpoints(request) == 1


def test_over_budget_caller_markers_fail_before_an_api_call():
    provider = _make_provider()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"cached block {index}",
                    "cache_control": {"type": "ephemeral"},
                }
                for index in range(5)
            ],
        }
    ]
    before = deepcopy(messages)

    with pytest.raises(
        ValueError,
        match=r"at most 4 cache_control breakpoints; the request contains 5",
    ):
        provider.generate(messages)

    assert messages == before
    provider.client.messages.create.assert_not_called()


@pytest.mark.parametrize("request_path", ["generate", "generate_stream"])
def test_disabled_prompt_caching_adds_no_markers_and_keeps_inputs_immutable(
    request_path,
):
    provider = _make_provider(prompt_caching=False)
    messages = [
        {"role": "system", "content": "STATIC SYSTEM"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
        },
    ]
    before = deepcopy(messages)

    if request_path == "generate":
        provider.client.messages.create.return_value = _text_response()
        provider.generate(messages)
        request = provider.client.messages.create.call_args.kwargs
    else:
        provider.client.messages.stream.return_value = _FakeStream()
        list(provider.generate_stream(messages))
        request = provider.client.messages.stream.call_args.kwargs

    assert messages == before
    assert provider._count_cache_control_breakpoints(request) == 0
