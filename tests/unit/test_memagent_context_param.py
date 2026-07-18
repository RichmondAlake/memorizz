"""Tests for the M2 per-call ``context`` parameter on MemAgent.run/run_stream.

The contract:

1. ``run(query, context={...})`` and ``run_stream(query, context={...})``
   accept an optional dict that's rendered for the LLM as an ephemeral
   context block.
2. The dict appears in the volatile context block at the START of the
   FINAL USER message (after the system prompt and conversation history).
   Rationale: the system prompt and history are the stable prompt-cache
   prefix; anything per-turn must live at the very end of the prompt.
3. The dict is NOT persisted to ``conversation_memory`` — only the
   original ``query`` string is recorded by ``_record_interaction``.
4. Omitting the parameter preserves legacy behaviour (no context block,
   callers don't break).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from memorizz.enums import MemoryType, Role

# ---------------------------------------------------------------------------
# _build_prompt_messages — the core wiring point for M2
# ---------------------------------------------------------------------------


def test_build_prompt_messages_injects_request_context_block(memagent_with_mocks):
    """When request_context is provided, it is rendered in the volatile
    context block at the start of the final user message — never as a
    separate system message (that would sit in the prompt-cache prefix and
    invalidate it every turn)."""
    agent = memagent_with_mocks

    request_context = {
        "current_page": {
            "type": "analysis",
            "id": "analysis-abc",
            "title": "Sample Podcast",
        },
        "quoted_text": "the pricing point made on slide 12",
        "internet_search_allowed": False,
    }

    messages = agent._build_prompt_messages(
        "MAIN SYSTEM PROMPT",
        "what did we say about pricing?",
        {"conversation_history": []},
        request_context=request_context,
    )

    # [0] is the (only) system prompt; [-1] is the user turn carrying the
    # volatile block followed by the raw query.
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "MAIN SYSTEM PROMPT"

    assert messages[-1]["role"] == "user"
    body = messages[-1]["content"]
    assert "REQUEST CONTEXT" in body
    # The JSON for each field of the supplied dict must be present.
    assert "analysis-abc" in body
    assert "Sample Podcast" in body
    assert "the pricing point made on slide 12" in body
    assert "internet_search_allowed" in body
    # The raw query ends the message (volatile block precedes it).
    assert body.endswith("what did we say about pricing?")


def test_build_prompt_messages_without_request_context_is_unchanged(
    memagent_with_mocks,
):
    """No request_context => no extra system message; legacy contract intact."""
    agent = memagent_with_mocks

    messages = agent._build_prompt_messages(
        "MAIN SYSTEM PROMPT",
        "hello",
        {"conversation_history": []},
    )

    # Only the main system prompt + user turn (no history); exactly 2 messages.
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "MAIN SYSTEM PROMPT"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "hello"


def test_build_prompt_messages_empty_request_context_skipped(memagent_with_mocks):
    """Empty dict / falsy values should NOT inject a noop context block."""
    agent = memagent_with_mocks

    for value in (None, {}, False, 0, ""):
        messages = agent._build_prompt_messages(
            "SYS", "q", {"conversation_history": []}, request_context=value
        )
        # Only [system, user]; no extra system block.
        assert len(messages) == 2, f"unexpected extra message for value={value!r}"


def test_build_prompt_messages_request_context_serializes_non_json_safely(
    memagent_with_mocks,
):
    """Non-JSON-native values (e.g. datetime) should not crash; default=str fallback."""
    from datetime import datetime

    agent = memagent_with_mocks
    request_context = {
        "current_page": {"type": "analysis", "id": "a-1"},
        "created_at": datetime(2026, 5, 23, 12, 0, 0),  # not JSON-native
    }

    messages = agent._build_prompt_messages(
        "S",
        "q",
        {"conversation_history": []},
        request_context=request_context,
    )

    assert "REQUEST CONTEXT" in messages[-1]["content"]
    assert "a-1" in messages[-1]["content"]


# ---------------------------------------------------------------------------
# run() and run_stream() — public API surface
# ---------------------------------------------------------------------------


def test_run_accepts_context_kwarg_and_forwards_to_messages(memagent_with_mocks):
    """run(query, context={...}) reaches the LLM via _build_prompt_messages."""
    agent = memagent_with_mocks

    captured_messages = {}

    def fake_generate(messages, tools=None, **_):
        captured_messages["last"] = messages
        return "ok"

    agent.model.generate = fake_generate

    response = agent.run(
        "what's on this page?",
        context={
            "current_page": {"type": "analysis", "id": "page-xyz", "title": "Doc"},
            "internet_search_allowed": True,
        },
    )

    assert response == "ok"
    # The context block rides in the final user message (volatile tail).
    user_messages = [m for m in captured_messages["last"] if m["role"] == "user"]
    combined = " ".join(m["content"] for m in user_messages)
    assert "REQUEST CONTEXT" in combined
    assert "page-xyz" in combined
    assert "Doc" in combined


def test_run_without_context_kwarg_works_unchanged(memagent_with_mocks):
    """Legacy callers passing no context still work and don't get a context block."""
    agent = memagent_with_mocks

    captured = {}

    def fake_generate(messages, tools=None, **_):
        captured["last"] = messages
        return "fine"

    agent.model.generate = fake_generate

    response = agent.run("legacy call")

    assert response == "fine"
    combined = " ".join(str(m.get("content")) for m in captured["last"])
    assert "REQUEST CONTEXT" not in combined


def test_run_stream_accepts_context_kwarg(memagent_with_mocks):
    """run_stream(query, context={...}) reaches the LLM via _build_prompt_messages."""
    agent = memagent_with_mocks

    captured = {}

    def fake_generate_stream(messages, tools=None, **_):
        captured["last"] = messages
        yield {"type": "content", "content": "hello"}
        yield {"type": "done", "content": "hello"}

    agent.model.generate_stream = fake_generate_stream

    chunks = list(
        agent.run_stream(
            "streamed query",
            context={"current_page": {"type": "group", "id": "g-1", "title": "G"}},
        )
    )

    assert chunks, "expected at least one chunk yielded"
    user_messages = [m for m in captured["last"] if m["role"] == "user"]
    combined = " ".join(m["content"] for m in user_messages)
    assert "REQUEST CONTEXT" in combined
    assert "g-1" in combined


# ---------------------------------------------------------------------------
# The crucial guarantee: context is NOT persisted to conversation_memory.
# ---------------------------------------------------------------------------


def test_request_context_is_not_persisted_in_conversation_memory(memagent_with_mocks):
    """``_record_interaction`` must store only the original query, not the
    REQUEST CONTEXT block. This is what makes the M2 context ephemeral."""
    agent = memagent_with_mocks

    saved_units = []

    if agent.memory_manager:
        agent.memory_manager.save_memory_unit = MagicMock(
            side_effect=lambda unit, memory_id: saved_units.append(unit)
        )

    def fake_generate(messages, tools=None, **_):
        return "assistant reply"

    agent.model.generate = fake_generate

    secret_context = {
        "internal_admin_note": "DO NOT LEAK INTO HISTORY",
        "current_page": {"type": "analysis", "id": "a-1", "title": "T"},
    }

    agent.run("user question", context=secret_context)

    # Two units recorded: user query + assistant response.
    assert len(saved_units) >= 1
    # Reconstruct the content of every saved unit; the leak string must
    # not appear anywhere.
    rendered = []
    for unit in saved_units:
        content = getattr(unit, "content", None)
        if content is None and isinstance(unit, dict):
            content = unit.get("content")
        rendered.append(str(content))
    blob = "\n".join(rendered)
    assert (
        "DO NOT LEAK INTO HISTORY" not in blob
    ), "request_context leaked into conversation_memory — M2 contract violated"
    assert "internal_admin_note" not in blob


# ---------------------------------------------------------------------------
# Backwards-compatibility for downstream callers that pass our keyword args.
# ---------------------------------------------------------------------------


def test_run_keyword_args_still_compatible(memagent_with_mocks):
    """All existing keyword args still bind correctly when context is added."""
    agent = memagent_with_mocks
    agent.model.generate = lambda messages, tools=None, **_: "ok"

    # Mix of every existing kwarg + the new context kwarg.
    response = agent.run(
        query="hello",
        memory_id="m-1",
        thread_id="t-1",
        user_id="alice",
        context={"current_page": {"type": "library"}},
    )
    assert response == "ok"
