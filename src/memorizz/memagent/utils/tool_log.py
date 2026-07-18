# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tool-log placeholder helpers shared by MemAgent's streaming and
non-streaming execution paths.

Moved verbatim from ``memorizz.memagent.core`` — pure functions with no
MemAgent dependency.
"""

import json
from typing import Any, List, Optional


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce a value into a JSON-serializable primitive.

    Oracle ``oracledb`` surfaces CLOB/BLOB columns as LOB objects with a
    ``.read()`` method rather than as strings. These slip through
    conversation-history, knowledge-base, and summary loads into the
    message list we hand to the LLM provider, where the streaming path
    ultimately does ``json.dumps(...)`` and fails with
    ``Object of type LOB is not JSON serializable``.

    Rather than chase every data path individually, we sanitize at the
    prompt-assembly boundary. One helper, used everywhere we serialize
    model input or tool output.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    reader = getattr(value, "read", None)
    if callable(reader):
        try:
            read_value = reader()
        except Exception:
            return str(value)
        return _to_jsonable(read_value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="ignore")
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


# ---------------------------------------------------------------------------
# Tool-log disambiguation helpers
# ---------------------------------------------------------------------------
# The agent-facing tool placeholder needs enough signal for the LLM to pick
# the right ``tool_log_id`` among multiple calls to the same tool in one
# session. These helpers centralize the three improvements:
#   1) args summary so calls to the same tool can be told apart
#   2) field-aware result summary for common tool output shapes
#   3) a single builder used by both streaming and non-streaming paths
_TOOL_LOG_PLACEHOLDER_PREFIX = "[Tool '"


def _summarize_tool_args(arguments: Any, limit: int = 200) -> str:
    """One-line, length-bounded JSON summary of tool arguments.

    Falls back to ``str(arguments)`` for anything non-JSON. Empty mapping
    returns an empty string so the caller can omit the field cleanly.
    """
    if arguments is None:
        return ""
    try:
        if isinstance(arguments, str):
            # Already serialized — just trim.
            text = arguments
        else:
            text = json.dumps(_to_jsonable(arguments), ensure_ascii=False)
    except Exception:
        text = str(arguments)
    text = text.strip()
    if not text or text in ("{}", "null"):
        return ""
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _summarize_tool_result(result: Any, preview_limit: int = 500) -> str:
    """Field-aware summary of a tool's return value.

    Produces a one-line hint based on common tool output shapes plus a
    bounded ``Preview:`` of the most content-rich field. Unknown shapes
    fall through to ``str(result)[:preview_limit]`` so nothing blows up.
    """
    if result is None:
        return ""
    # Errors first — always useful to surface verbatim
    if isinstance(result, dict):
        if "error" in result and result.get("error"):
            err = str(result["error"])
            return f"error: {err[:preview_limit]}"
        # matches = KB lookup, entity lookup, semantic search
        matches = result.get("matches")
        if isinstance(matches, list):
            hints: List[str] = [f"{len(matches)} match(es)"]
            if matches and isinstance(matches[0], dict):
                top = matches[0]
                ns = top.get("namespace") or top.get("name")
                if ns:
                    hints.append(f"top namespace: {ns}")
                first_content = top.get("content") or top.get("text") or ""
                if isinstance(first_content, str) and first_content.strip():
                    snippet = first_content.strip().replace("\n", " ")
                    if len(snippet) > preview_limit:
                        snippet = snippet[: preview_limit - 1] + "…"
                    hints.append(f"top content: {snippet}")
            return " · ".join(hints)
        # generic list-of-entries return shapes
        entries = result.get("entries") or result.get("logs") or result.get("items")
        if isinstance(entries, list):
            return f"{len(entries)} entr(ies)"
        # single-result wrappers
        if "content" in result and isinstance(result["content"], str):
            content = result["content"].strip().replace("\n", " ")
            if len(content) > preview_limit:
                content = content[: preview_limit - 1] + "…"
            return f"content: {content}"
        if "ok" in result:
            # Tool loops need decision fields such as ``status``, ``accepted``,
            # and ``reason`` on the very next model call. Returning only
            # ``ok=True`` makes a successful lookup unusable even though the
            # full payload was stored correctly in TOOL_LOG.
            scalar_fields = []
            for key, value in result.items():
                if value is not None and not isinstance(value, (str, int, float, bool)):
                    continue
                rendered = str(value).strip().replace("\n", " ")
                pair = f"{key}={rendered}"
                remaining = preview_limit - len(" · ".join(scalar_fields))
                if remaining <= 4:
                    break
                if len(pair) > remaining:
                    pair = pair[: remaining - 1].rstrip() + "…"
                scalar_fields.append(pair)
            if scalar_fields:
                return " · ".join(scalar_fields)
            return f"ok={result['ok']}"

    text = str(result).strip().replace("\n", " ")
    if len(text) > preview_limit:
        text = text[: preview_limit - 1] + "…"
    return text


def _extract_identifiers(result: Any, limit: int = 6, value_limit: int = 80) -> str:
    """Pull identifier handles out of a tool result so they survive the digest.

    ``_summarize_tool_result`` is shape-aware and, for a result like
    ``{"ok": true, "doc_id": "abc"}``, returns only ``ok=true`` — dropping the
    very id the agent needs on the NEXT turn (the whole point of the digest as
    the durable cross-turn channel). This scans the parsed result for id-like
    keys (``id`` / ``*_id``) plus the common ``url`` / ``open_url`` handles and
    returns them as ``key=value`` pairs, so identifiers are carried verbatim no
    matter which summary branch fired. Returns "" for non-dict / id-less
    results.
    """
    if not isinstance(result, dict):
        return ""
    pairs: List[str] = []
    for key, value in result.items():
        if not isinstance(key, str):
            continue
        kl = key.lower()
        if not (kl == "id" or kl.endswith("_id") or kl in ("url", "open_url")):
            continue
        if not isinstance(value, (str, int, float)) or not str(value).strip():
            continue
        text = str(value)
        if len(text) > value_limit:
            text = text[: value_limit - 1] + "…"
        pairs.append(f"{key}={text}")
        if len(pairs) >= limit:
            break
    return " ".join(pairs)


def _build_tool_log_placeholder(
    *,
    tool_name: str,
    tool_log_id: str,
    arguments: Any,
    result: Any,
    error_message: Optional[str] = None,
) -> str:
    """Compose the bracketed placeholder that replaces the raw tool result
    in the LLM message list and in conversation_memory.

    Includes (in order) the tool name, a short args summary, the
    tool_log_id + retrieval-call template, and a field-aware preview.
    That's enough signal for the LLM to pick the right id even among
    several calls to the same tool in one session.
    """
    args_summary = _summarize_tool_args(arguments)
    result_summary = _summarize_tool_result(result)
    args_clause = f" args={args_summary}" if args_summary else ""
    if error_message:
        header = (
            f"{_TOOL_LOG_PLACEHOLDER_PREFIX}{tool_name}' failed: {error_message}."
            f"{args_clause} Full output stored as tool_log:{tool_log_id} — "
            f"use retrieve_tool_log_entry('{tool_log_id}') for complete data]"
        )
    else:
        header = (
            f"{_TOOL_LOG_PLACEHOLDER_PREFIX}{tool_name}' executed successfully."
            f"{args_clause} Full output stored as tool_log:{tool_log_id} — "
            f"use retrieve_tool_log_entry('{tool_log_id}') for complete data]"
        )
    return f"{header}\nSummary: {result_summary}" if result_summary else header


def _is_tool_placeholder_content(text: Any) -> bool:
    """True when a stored conversation row's content is a tool-log placeholder.

    Used to filter these rows out of the message list handed to the LLM
    (they'd otherwise appear as orphan tool-role messages without a
    matching tool_call_id and the OpenAI API would reject the batch).
    The placeholders remain visible in the UI and are surfaced to the
    agent as a structured digest in the system prompt instead.
    """
    if not isinstance(text, str):
        return False
    return text.lstrip().startswith(_TOOL_LOG_PLACEHOLDER_PREFIX)
