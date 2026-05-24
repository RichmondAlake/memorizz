# Copyright (c) 2026 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
"""Per-call tool context (M4).

A :class:`contextvars.ContextVar`-backed dict that tool functions can read
from inside ``MemAgent.run()`` / ``MemAgent.run_stream()`` without the LLM
needing to know about it. This is the canonical pattern for passing
per-request facts (e.g. the current tenant's ``user_id`` in a multi-tenant
web app) into tools whose function signatures should NOT expose those
facts to the model — putting tenant identifiers in the model's prompt is
both wasteful and a cross-tenant leakage risk.

Usage::

    from memorizz import MemAgent, get_tool_context

    def list_user_documents() -> list[dict]:
        # Tool reads the per-call context — never receives user_id as an
        # LLM-supplied argument.
        ctx = get_tool_context()
        user_id = ctx.get("user_id")
        return db.find({"user_id": user_id})

    agent = MemAgent(tools=[list_user_documents], ...)
    agent.run(
        "What did I upload yesterday?",
        user_id="alice",
        tool_context={"user_id": "alice", "tenant": "acme"},
    )

The context is set automatically by ``run()`` / ``run_stream()`` before the
tool-calling loop and reset in ``finally`` when the call returns or raises.
Nested calls and concurrent calls from different tasks/threads each see
their own scope because ``contextvars`` is the standard Python primitive
for that.
"""
from __future__ import annotations

import contextvars
from typing import Any, Dict

_tool_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "memorizz_tool_context",
    default={},
)


def get_tool_context() -> Dict[str, Any]:
    """Return the current per-call tool context (or ``{}`` if unset).

    Always returns a dict — never ``None`` — so tools can safely
    ``ctx.get("key")`` without a guard.
    """
    return _tool_context.get()


def set_tool_context(ctx: Dict[str, Any] | None):
    """Set the per-call tool context. Returns a token for :func:`reset_tool_context`.

    A shallow copy of ``ctx`` is stored so that callers mutating the dict
    afterwards don't accidentally change what tools see.
    """
    return _tool_context.set(dict(ctx) if ctx else {})


def reset_tool_context(token: contextvars.Token | None = None) -> None:
    """Reset the per-call tool context.

    Two calling shapes are supported:

    * ``reset_tool_context(token)`` — restore to whatever was active before
      the matching :func:`set_tool_context` call. This is the recommended
      pattern: a ``set_tool_context`` / ``reset_tool_context`` pair behaves
      like a context manager and composes correctly across nested calls.

    * ``reset_tool_context()`` — clear the context unconditionally. Useful
      as a defensive cleanup in fixtures or top-level error handlers
      where the original token is no longer in scope. Note this *drops*
      any outer scope; prefer the token-based call when you have it.

    Prior to 0.0.44 the ``token`` argument was required; 0.0.45 makes it
    optional for backwards compatibility with the 0.0.42-era no-arg
    pattern downstream code still uses.
    """
    if token is None:
        _tool_context.set({})
        return
    _tool_context.reset(token)
