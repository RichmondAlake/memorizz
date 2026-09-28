from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from jinja2 import Environment, FileSystemLoader

from memorizz.llms.anthropic import Anthropic
from memorizz.llms.openai import OpenAI
from memorizz.llms.prompt_cache import cache_request_metadata
from memorizz.observability.analytics import aggregate_usage


def result(number=0, **kwargs):
    return dict(
        kind="model_result",
        agent_id="agent",
        user_id="alice",
        thread_id="thread",
        span_id=str(number),
        timestamp=f"2026-09-18T00:00:{number:02d}Z",
        status="success",
        provider="anthropic",
        model="claude-opus-4-8",
        input_tokens=2000,
        output_tokens=10,
        **kwargs,
    )


def test_fingerprints_keep_instructions_stable_and_do_not_store_content():
    kwargs = {
        "system": [
            {
                "type": "text",
                "text": "PRIVATE INSTRUCTIONS",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "PRIVATE REQUEST"}],
    }
    before = deepcopy(kwargs)
    first = cache_request_metadata(kwargs, "anthropic")
    kwargs["messages"][0]["content"] = "DIFFERENT REQUEST"
    assert cache_request_metadata(kwargs, "anthropic") == first
    assert first["prompt_cache_enabled"]
    assert "PRIVATE" not in str(first)
    kwargs["system"][0]["text"] = "changed instructions"
    assert (
        cache_request_metadata(kwargs, "anthropic")["prompt_cache_prefix"]
        != first["prompt_cache_prefix"]
    )
    assert before["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_openai_implicit_explicit_disabled_and_compatible_are_distinct():
    request = {
        "input": [{"role": "developer", "content": "stable"}],
        "prompt_cache_key": "tenant-secret",
    }
    implicit = cache_request_metadata(request, "openai")
    assert implicit["prompt_cache_enabled"]
    assert "tenant-secret" not in str(implicit)
    request["prompt_cache_options"] = {"mode": "explicit"}
    assert cache_request_metadata(request, "openai")["prompt_cache_enabled"] is False
    request["input"][0]["content"] = [
        {
            "type": "input_text",
            "text": "stable",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    assert cache_request_metadata(request, "openai")["prompt_cache_enabled"]
    assert cache_request_metadata(request, "openai_compatible") == {}


def test_nested_tool_schema_does_not_enable_caching():
    request = {
        "tools": [
            {
                "input_schema": {
                    "properties": {
                        "cache_control": {"type": "object"},
                        "prompt_cache_breakpoint": {"type": "object"},
                    }
                }
            }
        ],
        "prompt_cache_options": {"mode": "explicit"},
    }
    assert cache_request_metadata(request, "anthropic")["prompt_cache_enabled"] is False
    assert cache_request_metadata(request, "openai")["prompt_cache_enabled"] is False


def test_anthropic_nested_tool_result_boundary_is_recognized():
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call",
                        "content": [
                            {
                                "type": "text",
                                "text": "reference",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
            }
        ]
    }
    assert cache_request_metadata(request, "anthropic")["prompt_cache_enabled"]


@pytest.mark.parametrize(
    "model,marked",
    [
        ("gpt-6-astra", True),
        ("gpt-5.6-sol", True),
        ("gpt-5.5", False),
        ("gpt-4o", False),
    ],
)
def test_responses_cache_boundary_stays_before_changing_questions(model, marked):
    provider = OpenAI.__new__(OpenAI)
    provider.model = model
    provider._prompt_cache_key, provider._prompt_cache_retention = "thread", None
    messages = [
        {"role": "system", "content": "stable"},
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": "more instructions"}],
        },
        {"role": "user", "content": "question"},
    ]
    original = deepcopy(messages)
    first = {"input": messages}
    provider._apply_cache_options(first)
    assert messages == original
    if marked:
        assert first["input"][1]["content"][0]["prompt_cache_breakpoint"] == {
            "mode": "explicit"
        }
        assert first["input"][2] == messages[2]
        second = {"input": [*messages[:2], {"role": "user", "content": "new question"}]}
        provider._apply_cache_options(second)
        assert first["input"][:2] == second["input"][:2]
    else:
        assert first["input"] == original


def test_responses_text_instructions_cached_and_explicit_caller_boundaries_preserved():
    provider = OpenAI.__new__(OpenAI)
    provider.model = "gpt-6-astra"
    kwargs = {"input": "question", "instructions": "stable"}
    provider._apply_explicit_cache_boundaries(kwargs)
    assert "instructions" not in kwargs
    assert kwargs["input"][0] == {
        "role": "developer",
        "content": [
            {
                "type": "input_text",
                "text": "stable",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ],
    }
    caller = {
        "input": [
            {"role": "system", "content": "stable"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "reference",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
        ]
    }
    before = deepcopy(caller)
    provider._apply_explicit_cache_boundaries(caller)
    assert caller == before


def test_new_openai_cache_boundary_reaches_sdk_wire_without_new_top_level_parameters():
    import json

    import httpx
    import openai

    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 0,
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [],
            },
        )

    provider = OpenAI.__new__(OpenAI)
    provider.model, provider.reasoning_effort = "gpt-6-astra", None
    provider._request_options = {"max_completion_tokens": 16}
    provider._prompt_cache_key, provider._prompt_cache_retention = "scope", None
    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        provider.client = openai.OpenAI(api_key="offline-test", http_client=transport)
        provider.generate_text("question one", instructions="static instructions")
        provider.generate_text("question two", instructions="static instructions")
    assert len(requests) == 2
    assert requests[0]["input"][0] == requests[1]["input"][0]
    assert requests[0]["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert requests[0]["input"][1] != requests[1]["input"][1]


@pytest.mark.parametrize("method", ["generate", "generate_text"])
def test_claude_text_and_chat_enable_cache_and_preserve_zero_and_write_counts(method):
    provider = Anthropic.__new__(Anthropic)
    provider.model = "claude-opus-4-8"
    provider._max_tokens, provider._request_options = 64, {}
    provider._enable_prompt_caching = True
    create = Mock(
        return_value=NS(
            content=[NS(type="text", text="ok")],
            usage=NS(
                input_tokens=100,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=2000,
            ),
        )
    )
    provider.client = NS(messages=NS(create=create))
    if method == "generate_text":
        provider.generate_text("question", instructions="static")
    else:
        provider.generate(
            [
                {"role": "system", "content": "static"},
                {"role": "user", "content": "question"},
            ]
        )
    assert create.call_args.kwargs["system"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    metadata = provider.get_last_response_metadata()
    assert metadata["prompt_cache_enabled"]
    assert metadata["cached_tokens"] == 0
    assert metadata["cache_write_tokens"] == 2000
    assert metadata["input_tokens"] == 2100


def test_missing_counters_are_unknown_and_cold_writes_are_not_broken():
    events = [
        result(0),
        result(1, cached_tokens=0, cache_write_tokens=2000),
        result(2, cached_tokens=0, cache_write_tokens=0),
        result(3, cached_tokens=1500),
    ]
    report = aggregate_usage(events)["prompt_cache"]
    assert (
        report["unknown"] == report["write"] == report["no_reuse"] == report["hit"] == 1
    )
    assert report["read_percent"] == 25
    assert not report["attention"]


@pytest.mark.parametrize(
    "change,code",
    [
        ({}, "reuse_dropped"),
        ({"prompt_cache_prefix": "other"}, "prefix_changed"),
        ({"prompt_cache_key": "other"}, "key_changed"),
    ],
)
def test_cache_drop_flags_observed_evidence(change, code):
    before = result(
        0, cached_tokens=1800, prompt_cache_prefix="prefix", prompt_cache_key="key"
    )
    after = {
        **result(
            1, cached_tokens=0, prompt_cache_prefix="prefix", prompt_cache_key="key"
        ),
        **change,
    }
    report = aggregate_usage([after, before, before])["prompt_cache"]
    assert report["calls"] == 2
    assert report["attention"]
    assert report["warnings"][0]["code"] == code


@pytest.mark.parametrize(
    "change",
    [
        {"user_id": "bob"},
        {"thread_id": "other"},
        {"model": "claude-opus-5"},
        {"response_model": "new-provider-snapshot"},
        {"timestamp": "2026-09-18T01:00:00Z"},
        {"status": "error"},
    ],
)
def test_other_scope_expired_or_failed_calls_do_not_create_cache_drop_alert(change):
    before = result(0, cached_tokens=1800, prompt_cache_prefix="prefix")
    after = {**result(1, cached_tokens=0, prompt_cache_prefix="prefix"), **change}
    assert not aggregate_usage([before, after])["prompt_cache"]["attention"]


def test_cache_disabled_and_sdk_fallback_are_visible_in_ui():
    usage = aggregate_usage(
        [
            result(0, prompt_cache_enabled=False),
            result(1, prompt_cache_warning="cache_options_dropped"),
        ]
    )
    templates = Path(__file__).parents[2] / "src/memorizz/ui/templates"
    html = (
        Environment(loader=FileSystemLoader(templates), autoescape=True)
        .get_template("_prompt_cache.html")
        .render(usage=usage)
    )
    assert "Needs attention" in html
    assert "disabled on 1 call." in html
    assert "dropped or unsupported on 1 call." in html
