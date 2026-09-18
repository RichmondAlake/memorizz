"""Conservative text budgeting at the complete model-request boundary."""

import json
import math

from ...llms.streaming import ProviderStreamError


def estimate_tokens(value):
    """Estimate text/JSON tokens without downloads or provider network calls.

    UTF-8 bytes account for non-ASCII text better than Python character counts.
    The caller also leaves 20% of the window for generation/template overhead.
    This remains an estimate, not a model-specific tokenizer guarantee.
    """
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return math.ceil(len(value.encode("utf-8")) / 4)


def fit_prompt(messages, tools, context_window_tokens):
    """Evict old turns; never cut instructions, the current query, or evidence.

    Called again on every tool-loop iteration, after tools and tool results
    have been added. Returns a new list without changing the caller's history.
    If required input alone exceeds the budget, stop before making a request.
    """
    if type(context_window_tokens) is not int or context_window_tokens <= 0:
        return list(messages)

    budget = int(context_window_tokens * 0.8)
    costs = [estimate_tokens(message) + 6 for message in messages]
    total = sum(costs) + (estimate_tokens(tools) if tools else 0) + 16
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

    raise ProviderStreamError("context_window_exceeded")
