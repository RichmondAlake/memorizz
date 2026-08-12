"""Tests for the M4 per-call ``tool_context`` mechanism.

The contract:

1. ``memorizz.get_tool_context()`` returns ``{}`` by default and never None.
2. ``set_tool_context()`` / ``reset_tool_context()`` round-trip cleanly.
3. ``MemAgent.run(..., tool_context=...)`` makes the dict visible to tool
   functions via ``get_tool_context()`` for the duration of the call.
4. The context is released (back to whatever it was before) when ``run()``
   returns OR raises.
5. The context is NOT sent to the LLM — only the regular ``query`` /
   ``context=`` payload is visible to the model.
6. Concurrent calls from different asyncio tasks / threads see independent
   contexts (contextvars semantics).
"""
from __future__ import annotations

import threading

import pytest

from memorizz import get_tool_context, reset_tool_context, set_tool_context

# ---------------------------------------------------------------------------
# Plain helper API
# ---------------------------------------------------------------------------


def test_default_is_empty_dict():
    assert get_tool_context() == {}


def test_set_then_get_returns_the_value():
    token = set_tool_context({"user_id": "alice", "tenant": "acme"})
    try:
        ctx = get_tool_context()
        assert ctx == {"user_id": "alice", "tenant": "acme"}
    finally:
        reset_tool_context(token)
    assert get_tool_context() == {}


def test_set_none_yields_empty_dict():
    token = set_tool_context(None)
    try:
        assert get_tool_context() == {}
    finally:
        reset_tool_context(token)


def test_get_always_returns_dict_never_none():
    """Tools should be able to .get('x') without guarding."""
    ctx = get_tool_context()
    assert isinstance(ctx, dict)
    # Doesn't raise:
    assert ctx.get("missing") is None


def test_caller_mutation_does_not_leak_into_stored_ctx():
    """``set_tool_context`` shallow-copies; mutating the source after set
    must not affect what tools see."""
    source = {"user_id": "alice"}
    token = set_tool_context(source)
    try:
        source["user_id"] = "EVE"  # caller mutates after set
        ctx = get_tool_context()
        assert (
            ctx["user_id"] == "alice"
        ), "stored ctx must be insulated from source mutation"
    finally:
        reset_tool_context(token)


def test_thread_isolation():
    """Different threads have independent tool_context scopes."""
    seen = {}

    def worker(name):
        token = set_tool_context({"user_id": name})
        try:
            # Give other threads a chance to overwrite if isolation is broken.
            import time as _t

            _t.sleep(0.05)
            seen[name] = get_tool_context().get("user_id")
        finally:
            reset_tool_context(token)

    threads = [
        threading.Thread(target=worker, args=(n,)) for n in ("alice", "bob", "carol")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {"alice": "alice", "bob": "bob", "carol": "carol"}


# ---------------------------------------------------------------------------
# MemAgent.run() / run_stream() integration
# ---------------------------------------------------------------------------


def test_run_makes_tool_context_visible_inside_tool(memagent_with_mocks):
    """A tool invoked during run() sees what was passed in tool_context."""
    agent = memagent_with_mocks

    captured = {}

    def my_tool() -> str:
        ctx = get_tool_context()
        captured["snapshot"] = dict(ctx)
        return "ok"

    agent.tool_manager.add_tool(my_tool)

    # Force the mocked LLM to call my_tool, then return a final text.
    from types import SimpleNamespace

    call_count = {"n": 0}

    def fake_generate(messages, tools=None, **_):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: invoke my_tool
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="my_tool", arguments="{}"
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        # Second call: final response
        return "done"

    agent.model.generate = fake_generate

    result = agent.run(
        "trigger my tool",
        tool_context={"user_id": "alice", "tenant": "acme"},
    )

    assert result == "done"
    assert captured["snapshot"] == {"user_id": "alice", "tenant": "acme"}


def test_run_resets_tool_context_after_completion(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.model.generate = lambda messages, tools=None, **_: "fine"

    before = get_tool_context()
    agent.run("hello", tool_context={"user_id": "alice"})
    after = get_tool_context()
    assert before == after == {}


def test_run_resets_tool_context_after_exception(memagent_with_mocks):
    agent = memagent_with_mocks

    def exploding(messages, tools=None, **_):
        raise RuntimeError("boom")

    agent.model.generate = exploding

    # run() catches and returns an error string; tool_context must still be cleared.
    agent.run("hello", tool_context={"user_id": "alice"})
    assert get_tool_context() == {}


def test_run_stream_makes_tool_context_visible_inside_tool(memagent_with_mocks):
    agent = memagent_with_mocks
    captured = {}

    def my_tool() -> str:
        captured["snapshot"] = dict(get_tool_context())
        return "ok"

    agent.tool_manager.add_tool(my_tool)

    from types import SimpleNamespace

    def fake_generate_stream(messages, tools=None, **_):
        # Emit a tool_calls event, then a done event.
        yield {
            "type": "tool_calls",
            "response": SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="my_tool", arguments="{}"
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
        }
        yield {"type": "content", "content": "ok"}
        yield {"type": "done", "content": "ok"}

    agent.model.generate_stream = fake_generate_stream

    list(
        agent.run_stream(
            "trigger my tool",
            tool_context={"user_id": "bob"},
        )
    )

    assert captured["snapshot"] == {"user_id": "bob"}
    # And the context is released after the stream finishes.
    assert get_tool_context() == {}


def test_tool_context_omitted_yields_empty_dict_inside_tool(memagent_with_mocks):
    """Backwards-compat: tools called without tool_context still see {}."""
    agent = memagent_with_mocks
    captured = {}

    def my_tool() -> str:
        captured["snapshot"] = dict(get_tool_context())
        return "ok"

    agent.tool_manager.add_tool(my_tool)

    from types import SimpleNamespace

    call_count = {"n": 0}

    def fake_generate(messages, tools=None, **_):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="c",
                                    function=SimpleNamespace(
                                        name="my_tool", arguments="{}"
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        return "done"

    agent.model.generate = fake_generate

    # Progressive disclosure must select the tool before a direct invocation;
    # include its intent/name in the turn instead of fabricating a hidden tool.
    agent.run("trigger my tool")
    assert captured["snapshot"] == {}


# ---------------------------------------------------------------------------
# Defense — context never leaks into the LLM prompt
# ---------------------------------------------------------------------------


def test_tool_context_never_appears_in_messages_sent_to_llm(memagent_with_mocks):
    """tool_context is a server-only side channel; the LLM must never see it."""
    agent = memagent_with_mocks
    captured = {}

    def fake_generate(messages, tools=None, **_):
        captured["last"] = messages
        return "done"

    agent.model.generate = fake_generate

    secret = "DO-NOT-LEAK-TO-LLM-12345"
    agent.run(
        "hello",
        tool_context={"internal_user_id": secret, "tenant": "acme"},
    )

    blob = "\n".join(
        m.get("content", "") if isinstance(m, dict) else str(m)
        for m in captured["last"]
    )
    assert secret not in blob, "tool_context leaked into LLM messages!"
    assert "internal_user_id" not in blob
