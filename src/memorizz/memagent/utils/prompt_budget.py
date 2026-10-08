"""Conservative text budgeting at the complete model-request boundary."""

import json
import logging
import math

from ...llms.streaming import ProviderStreamError

logger = logging.getLogger(__name__)

# Framing tokens a chat provider adds around each message (role, separators)
# and once per request (template), over and above the text itself.
MESSAGE_OVERHEAD_TOKENS = 6
REQUEST_OVERHEAD_TOKENS = 16


def estimate_tokens(value):
    """Estimate text/JSON tokens without downloads or provider network calls.

    UTF-8 bytes account for non-ASCII text better than Python character counts.
    The caller also leaves 20% of the window for generation/template overhead.
    This remains an estimate, not a model-specific tokenizer guarantee.
    """
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return math.ceil(len(value.encode("utf-8")) / 4)


def estimate_message_tokens(message):
    """Estimate one chat message's text content plus its per-message framing.

    Prices only ``content`` (what the history window and auto-compaction
    heuristics budget for). ``fit_prompt`` prices the whole serialized
    message instead, because that is what reaches the provider.
    """
    content = message.get("content") if isinstance(message, dict) else message
    return estimate_tokens(str(content or "")) + MESSAGE_OVERHEAD_TOKENS


def fit_prompt(messages, tools, context_window_tokens):
    """Evict old turns; never cut instructions, the current query, or evidence.

    Called again on every tool-loop iteration, after tools and tool results
    have been added. Returns a new list without changing the caller's history.
    If required input alone exceeds the budget, stop before making a request.
    """
    if type(context_window_tokens) is not int or context_window_tokens <= 0:
        return list(messages)

    budget = int(context_window_tokens * 0.8)
    costs = [estimate_tokens(message) + MESSAGE_OVERHEAD_TOKENS for message in messages]
    total = (
        sum(costs) + (estimate_tokens(tools) if tools else 0) + REQUEST_OVERHEAD_TOKENS
    )
    if total <= budget:
        return list(messages)

    current_user = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "user"
        ),
        0,
    )
    # Drop complete old turns, including their assistant/tool-call pairs.
    # Privileged messages stay pinned even when embedded in older history.
    boundaries = [i for i in range(current_user) if messages[i].get("role") == "user"]
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(current_user)
    removed = set()
    for start, end in zip(boundaries, boundaries[1:]):
        for i in range(start, end):
            if messages[i].get("role") not in {"system", "developer"}:
                total -= costs[i]
                removed.add(i)
        if total <= budget:
            return [message for i, message in enumerate(messages) if i not in removed]

    # Instructions, tools and the current query alone do not fit. Say by how
    # much, so the fix (a larger window) is obvious from the log or trace.
    logger.warning(
        "Request needs about %d tokens but the budget is %d (80%% of a %d-token "
        "context window); raise context_window_tokens (num_ctx for Ollama)",
        total,
        budget,
        context_window_tokens,
    )
    raise ProviderStreamError(
        "context_window_exceeded",
        {
            "required_tokens": total,
            "budget_tokens": budget,
            "context_window_tokens": context_window_tokens,
        },
    )
