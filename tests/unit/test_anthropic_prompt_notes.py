"""Mid-turn host notes keep the cached Anthropic prefix intact."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Mock()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from memorizz.llms.anthropic import Anthropic

    return Anthropic(model="claude-sonnet-5")


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def test_a_note_added_mid_turn_extends_the_turn_not_the_system(provider):
    history = [
        {"role": "system", "content": "STABLE"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "new question"},
        {"role": "system", "content": "Use tools, then finalize."},
    ]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    first = provider._build_request_kwargs(history, tools)
    later = provider._build_request_kwargs(
        history + [{"role": "system", "content": "Evidence is complete; answer."}],
        tools,
    )

    # The system prompt and tools are unchanged; the note joins the user turn.
    assert (
        _strip(first["system"])
        == _strip(later["system"])
        == [{"type": "text", "text": "STABLE"}]
    )
    assert first["tools"] == later["tools"]
    assert len(first["messages"]) == len(later["messages"]) == 3
    turn = later["messages"][-1]["content"]
    assert [block["text"] for block in turn][0] == "new question"
    assert "Use tools, then finalize." in turn[1]["text"]
    assert "Evidence is complete" in turn[2]["text"]
    # Everything the first request sent is an unchanged prefix of the next.
    first_turn = _strip(first["messages"][-1]["content"])
    assert _strip(turn)[: len(first_turn)] == first_turn
    # Cross-turn breakpoint still sits on the previous answer.
    assert later["messages"][-2]["content"][-1].get("cache_control")


def test_notes_after_tool_results_follow_the_results(provider):
    messages = [
        {"role": "system", "content": "STABLE"},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "sunny"},
        {"role": "system", "content": "Answer now."},
    ]
    request = provider._build_request_kwargs(messages, None)
    last = request["messages"][-1]
    assert last["role"] == "user"
    assert [block["type"] for block in last["content"]] == ["tool_result", "text"]


def test_without_caching_notes_still_join_the_system(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Mock()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from memorizz.llms.anthropic import Anthropic

    plain = Anthropic(model="claude-sonnet-5", enable_prompt_caching=False)
    request = plain._build_request_kwargs(
        [
            {"role": "system", "content": "STABLE"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "note"},
        ],
        None,
    )
    assert request["system"] == "STABLE\n\nnote"
    assert request["messages"] == [{"role": "user", "content": "hi"}]


def test_skills_before_the_turn_keep_a_system_block_and_retries_stay_in_place(
    provider,
):
    request = provider._build_request_kwargs(
        [
            {"role": "system", "content": "STABLE"},
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "answer"},
            {"role": "developer", "content": "Reviewed skill: be brief."},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "draft"},
            {"role": "developer", "content": "Retry: cite the tool result."},
        ],
        None,
    )
    assert [block["text"] for block in request["system"]] == [
        "STABLE",
        "Reviewed skill: be brief.",
    ]
    retry = request["messages"][-1]
    assert retry["role"] == "user"
    assert "Retry: cite the tool result." in retry["content"][0]["text"]
