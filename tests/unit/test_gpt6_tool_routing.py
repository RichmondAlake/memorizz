"""GPT-6 tool requests preserve reasoning through the Responses endpoint."""

import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import openai
import pytest

from memorizz.llms.llm_factory import create_llm_provider
from memorizz.llms.openai import OpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.mark.parametrize(
    "model", ["gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol", "gpt-6-astra"]
)
def test_gpt6_defaults_to_responses_for_tools_and_preserves_function_results(model):
    requests = []

    def handle(request):
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        requests.append(payload)
        output = (
            [
                {
                    "type": "function_call",
                    "id": "fc1",
                    "call_id": "call1",
                    "name": "lookup",
                    "arguments": "{}",
                    "status": "completed",
                }
            ]
            if len(requests) == 1
            else [
                {
                    "type": "message",
                    "id": "msg1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Checked", "annotations": []}
                    ],
                }
            ]
        )
        return httpx.Response(
            200,
            json={
                "id": "resp" + str(len(requests)),
                "object": "response",
                "created_at": 0,
                "model": model,
                "status": "completed",
                "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            },
        )

    provider = create_llm_provider(
        {
            "provider": "openai",
            "model": model,
            "api_key": "offline-test",
            "additional_config": {
                "reasoning_effort": "low",
                "temperature": 0.5,
                "top_p": 0.9,
            },
            "max_tokens": 64,
        }
    )
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        provider.client = openai.OpenAI(api_key="offline-test", http_client=transport)
        history = [{"role": "user", "content": "Look up a value"}]
        answer = provider.generate(history, TOOLS)
        call = answer.choices[0].message.tool_calls[0]
        assert call.id == "call1" and call.function.name == "lookup"
        history.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call.id, "content": "Found it"},
            ]
        )
        assert provider.generate(history, TOOLS) == "Checked"
    assert (
        provider.api_mode == "responses"
        and provider.get_config()["api_mode"] == "responses"
    )
    assert provider.context_window_tokens == 1_050_000
    assert all(request["reasoning"] == {"effort": "low"} for request in requests)
    assert all(
        request["store"] is False and request["max_output_tokens"] == 64
        for request in requests
    )
    assert all(
        "temperature" not in request and "top_p" not in request for request in requests
    )
    assert requests[0]["tools"][0]["name"] == "lookup"
    assert {
        "type": "function_call_output",
        "call_id": "call1",
        "output": "Found it",
    } in requests[1]["input"]
    assert provider.get_last_usage()["total_tokens"] == 12


def test_gpt6_stream_uses_responses_with_reasoning_and_tools():
    provider = OpenAI(
        model="gpt-6-luna", api_key="offline-test", reasoning_effort="medium"
    )
    create = Mock(
        return_value=iter(
            [
                NS(type="response.output_text.delta", delta="Checked"),
                NS(
                    type="response.completed",
                    response=NS(status="completed", usage=None),
                ),
            ]
        )
    )
    provider.client = NS(responses=NS(create=create))
    events = list(
        provider.generate_stream([{"role": "user", "content": "Check"}], TOOLS)
    )
    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["reasoning"] == {"effort": "medium"}
    assert create.call_args.kwargs["tools"][0]["name"] == "lookup"
    assert events[-1] == {"type": "done", "content": "Checked"}


def test_explicit_chat_completions_with_no_reasoning_is_preserved():
    provider = OpenAI(
        model="gpt-6-luna",
        api_key="offline-test",
        api_mode="chat_completions",
        reasoning_effort="none",
    )
    create = Mock(
        return_value=NS(choices=[NS(message=NS(content="Checked", tool_calls=None))])
    )
    provider.client = NS(chat=NS(completions=NS(create=create)))
    assert provider.generate([{"role": "user", "content": "Check"}], TOOLS) == "Checked"
    assert create.call_args.kwargs["reasoning_effort"] == "none"
    assert create.call_args.kwargs["tools"] == TOOLS


def test_compatible_server_and_legacy_models_keep_chat_completions():
    local = OpenAI(model="gpt-6-luna", base_url="http://127.0.0.1:1234/v1")
    older = OpenAI(model="gpt-4.1", api_key="offline-test")
    assert local.api_mode == older.api_mode == "chat_completions"
    with pytest.raises(ValueError, match="official OpenAI endpoint"):
        OpenAI(
            model="gpt-6-luna",
            base_url="http://127.0.0.1:1234/v1",
            api_mode="responses",
        )
