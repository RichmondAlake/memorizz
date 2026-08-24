# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Helpers for turning raw ``conversation_memory`` rows into display-clean
conversation history.

Memorizz versions before 0.5.2 persisted a ``role="tool"`` conversation row
whose content was a ``{"type": "trace_bundle", ...}`` JSON document. Trace
bundles now live in private observability storage, but existing databases can
still contain those legacy rows. They are internal telemetry, not user-facing
tool calls.

External consumers that render conversation history from the raw provider rows
(a custom chat UI, an export, an audit view) previously had to reverse-engineer
this shape to filter it out. Worse, because the trace-bundle row lands *between*
a turn's two persisted writes, a naïve "collapse consecutive duplicates" pass
fails to merge them and one assistant reply renders twice — with an empty,
nameless tool entry wedged between the copies.

These helpers make the shape part of memorizz's public contract so consumers
don't have to. See also the opt-in ``exclude_trace_bundles`` flag on
``MemoryProvider.retrieve_conversation_history_ordered_by_timestamp`` (MongoDB
provider), which applies :func:`strip_trace_bundles` for you.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

__all__ = ["is_trace_bundle_entry", "strip_trace_bundles"]

# Mirrors the ``type`` field used by both legacy and current trace bundles.
_TRACE_BUNDLE_TYPE = "trace_bundle"


def is_trace_bundle_entry(entry: Any) -> bool:
    """Return ``True`` for an internal streamed-trace-bundle conversation row.

    A trace bundle is a ``role="tool"`` ``conversation_memory`` row whose
    ``content`` is a JSON object with ``"type": "trace_bundle"``. These rows are
    legacy telemetry rows and must never surface in user-facing history.

    Safe to call on any row shape — anything that isn't a trace bundle (plain
    ``user``/``assistant`` turns, real ``[Tool '…']`` placeholder rows, malformed
    rows) returns ``False``.
    """
    if not isinstance(entry, dict):
        return False
    role = str(entry.get("role") or "").strip().lower()
    if role != "tool":
        return False
    content = entry.get("content")
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    # Cheap guard before paying for json.loads — every trace bundle is an object.
    if not stripped.startswith("{"):
        return False
    try:
        payload = json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return (
        isinstance(payload, dict)
        and str(payload.get("type") or "").strip().lower() == _TRACE_BUNDLE_TYPE
    )


def strip_trace_bundles(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return ``entries`` with internal trace-bundle rows removed.

    A convenience wrapper over :func:`is_trace_bundle_entry` for the common case
    of cleaning a whole history list before rendering it. Real tool-placeholder
    rows (``[Tool '…']``) and ``user``/``assistant`` turns pass through
    untouched. Order is preserved.
    """
    if not entries:
        return []
    return [e for e in entries if not is_trace_bundle_entry(e)]
