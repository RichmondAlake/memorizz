"""Small datetime helpers shared across MemoRizz modules."""

from datetime import datetime, timezone
from typing import Any, Optional


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    """An ISO-8601 timestamp as written (naive stays naive); None when blank
    or unparseable."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def as_utc(value: Any) -> Optional[datetime]:
    """A datetime or ISO string as an aware datetime; naive values are taken
    as UTC. None when missing or unparseable."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


__all__ = ["as_utc", "parse_iso_datetime"]
