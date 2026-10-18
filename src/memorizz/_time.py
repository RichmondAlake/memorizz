"""Small datetime helpers shared across MemoRizz modules.

Times are written as UTC with an explicit offset (``utc_now``,
``utc_now_iso``). Values without an offset were written by older code, so a
reader says how that writer meant them: ``naive="utc"`` for
``datetime.utcnow()`` and Oracle ``SYS_EXTRACT_UTC(...)`` values, or
``naive="local"`` for ``datetime.now()`` and Oracle ``CURRENT_TIMESTAMP`` in a
session on the machine's time zone.
"""

from datetime import datetime, timezone
from typing import Any, Optional


def utc_now() -> datetime:
    """The current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """The current time as ISO-8601 with a ``+00:00`` offset."""
    return utc_now().isoformat()


def to_utc(value: Any, *, naive: str = "utc") -> Optional[datetime]:
    """A datetime, ISO-8601 string or epoch seconds as an aware UTC datetime.

    ``naive`` ("utc" or "local") says how a value without an offset was
    written. None when missing or unparseable.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        try:
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            pass
        if text[-1:] in ("Z", "z"):  # fromisoformat reads "Z" only from 3.11
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        if naive == "local":
            return value.astimezone(timezone.utc)
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_epoch(value: Any, *, naive: str = "utc") -> Optional[float]:
    """Epoch seconds for anything ``to_utc`` reads; None when it can't."""
    moment = to_utc(value, naive=naive)
    return moment.timestamp() if moment is not None else None


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


__all__ = [
    "as_utc",
    "parse_iso_datetime",
    "to_epoch",
    "to_utc",
    "utc_now",
    "utc_now_iso",
]
