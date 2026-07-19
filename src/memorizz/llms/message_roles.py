"""Provider compatibility helpers for MemoRizz's logical instruction roles."""

from copy import deepcopy
from typing import Any, Dict, List


def developer_messages_to_system(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge logical developer messages into the leading system instruction.

    This is the compatibility path for APIs/templates without a native
    ``developer`` role. Caller-owned messages are never mutated. MemoRizz only
    uses this for reviewed application instructions, never retrieved data.
    """
    copied = deepcopy(messages)
    developer_parts: List[str] = []
    remaining: List[Dict[str, Any]] = []
    for message in copied:
        if not isinstance(message, dict):
            remaining.append(message)
            continue
        if str(message.get("role") or "").lower() != "developer":
            remaining.append(message)
            continue
        content = message.get("content")
        if content:
            developer_parts.append(
                content if isinstance(content, str) else str(content)
            )

    if not developer_parts:
        return copied

    developer_text = "\n\n".join(developer_parts)
    for message in remaining:
        if (
            isinstance(message, dict)
            and str(message.get("role") or "").lower() == "system"
        ):
            existing = message.get("content")
            existing_text = (
                existing if isinstance(existing, str) else str(existing or "")
            )
            message["content"] = (
                f"{existing_text}\n\n{developer_text}"
                if existing_text
                else developer_text
            )
            return remaining

    return [{"role": "system", "content": developer_text}, *remaining]
