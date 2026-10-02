"""Small JSON helpers shared across MemoRizz modules."""

import json
from typing import Any, Dict, Mapping, Optional


def read_json_object(value: Any) -> Optional[Dict[str, Any]]:
    """A stored JSON object as a dict, from a string, bytes, a mapping or a
    readable LOB; None when it isn't a JSON object."""
    if hasattr(value, "read"):
        try:
            value = value.read()
        except Exception:
            return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = ["read_json_object"]
