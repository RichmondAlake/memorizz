# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Trust-aware learned-skill role mapping across LLM providers."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memorizz.llms.anthropic import Anthropic
from memorizz.llms.message_roles import developer_messages_to_system
from memorizz.llms.ollama import OllamaLLM
from memorizz.llms.openai import OpenAI


def _openai_provider(base_url=None):
    provider = OpenAI.__new__(OpenAI)
    provider.model = "test-model"
    provider.base_url = base_url
    provider.context_window_tokens = 128_000
    provider._last_usage = None
    provider._prompt_cache_key = None
    provider._prompt_cache_retention = None
    provider._request_options = {}
    provider.client = MagicMock()
    return provider


def _text_response():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))
        ],
        usage=None,
    )


@pytest.mark.unit
def test_compatibility_mapping_merges_developer_into_system_without_mutation():
    messages = [
        {"role": "system", "content": "platform policy"},
        {"role": "user", "content": "earlier turn"},
        {"role": "developer", "content": "reviewed refund skill"},
        {"role": "user", "content": "refund R-1008"},
    ]
    before = deepcopy(messages)

    mapped = developer_messages_to_system(messages)

    assert messages == before
    assert [message["role"] for message in mapped] == [
        "system",
        "user",
        "user",
    ]
    assert mapped[0]["content"] == ("platform policy\n\nreviewed refund skill")


@pytest.mark.unit
def test_official_openai_keeps_native_developer_message():
    provider = _openai_provider()
    captured = {}
    provider._create_chat_completion = lambda kwargs: (
        captured.update(kwargs) or _text_response()
    )
    messages = [
        {"role": "system", "content": "platform policy"},
        {"role": "developer", "content": "reviewed skill"},
        {"role": "user", "content": "request"},
    ]

    assert provider.generate(messages) == "done"
    assert captured["messages"] is messages
    assert captured["messages"][1]["role"] == "developer"


@pytest.mark.unit
def test_openai_compatible_endpoint_maps_developer_to_system():
    provider = _openai_provider(base_url="http://127.0.0.1:8080/v1")
    captured = {}
    provider._create_chat_completion = lambda kwargs: (
        captured.update(kwargs) or _text_response()
    )
    messages = [
        {"role": "system", "content": "platform policy"},
        {"role": "developer", "content": "reviewed skill"},
        {"role": "user", "content": "request"},
    ]

    assert provider.generate(messages) == "done"
    assert all(message["role"] != "developer" for message in captured["messages"])
    assert "reviewed skill" in captured["messages"][0]["content"]
    assert messages[1]["role"] == "developer"


@pytest.mark.unit
def test_anthropic_maps_developer_to_top_level_system():
    messages = [
        {"role": "system", "content": "platform policy"},
        {"role": "user", "content": "earlier turn"},
        {"role": "developer", "content": "reviewed skill"},
        {"role": "user", "content": "request"},
    ]
    before = deepcopy(messages)

    system, remaining = Anthropic._split_system(messages)

    assert messages == before
    assert system == "platform policy\n\nreviewed skill"
    assert [message["role"] for message in remaining] == ["user", "user"]


@pytest.mark.unit
def test_ollama_maps_developer_to_system():
    messages = [
        {"role": "developer", "content": "reviewed skill"},
        {"role": "user", "content": "request"},
    ]

    mapped = OllamaLLM._normalize_messages(messages)

    assert mapped == [
        {"role": "system", "content": "reviewed skill"},
        {"role": "user", "content": "request"},
    ]
    assert messages[0]["role"] == "developer"
