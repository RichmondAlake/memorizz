# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The UTC convention for the Oracle tables MemoRizz writes.

Every plain ``TIMESTAMP`` column holds UTC wall-clock time. SQL writes use
``UTC_NOW_SQL`` (never ``CURRENT_TIMESTAMP``, ``LOCALTIMESTAMP`` or a bare
``SYSTIMESTAMP``, which are session- or server-local), column defaults are
``UTC_NOW_SQL`` (migration 012), and Python values are bound as naive UTC by
``utc_bind``. Reads return aware UTC datetimes (``utc_read``) or ISO-8601
with ``+00:00`` (``utc_iso``).

python-oracledb never reads a ``datetime``'s ``tzinfo`` when binding (the
4.0 driver drops it), so an aware value must be converted here first, and a
``TIMESTAMP WITH TIME ZONE`` column must be written as
``FROM_TZ(CAST(:x AS TIMESTAMP), 'UTC')`` (``utc_tstz_sql``) or Oracle
labels the bound wall-clock time with the session time zone, which thin mode
sets from the Python process's local offset.

Rows written before this convention hold local time; migrations
``013_utc_timestamps.sql`` converts them once, given their time zone.
"""

from datetime import datetime
from typing import Any, Optional

from ..._time import to_utc, utc_now

UTC_NOW_SQL = "SYS_EXTRACT_UTC(SYSTIMESTAMP)"


def utc_now_bind() -> datetime:
    """The current time as naive UTC, for a plain TIMESTAMP bind."""
    return utc_now().replace(tzinfo=None)


def utc_bind(value: Any, *, naive: str = "utc") -> Optional[datetime]:
    """``value`` as naive UTC for a plain TIMESTAMP bind.

    Accepts datetimes, ISO-8601 strings (``Z`` included) and epoch seconds.
    ``naive`` says how a value without an offset was written: ``"utc"``, or
    ``"local"`` for writers known to use ``datetime.now()``. None when the
    value is missing or unreadable.
    """
    if isinstance(value, str) and value.strip().endswith("Z"):
        value = value.strip()[:-1] + "+00:00"
    moment = to_utc(value, naive=naive)
    return moment.replace(tzinfo=None) if moment is not None else None


def utc_read(value: Any) -> Any:
    """A fetched plain TIMESTAMP (UTC wall clock) as an aware UTC datetime.

    Anything that is not a datetime is returned unchanged.
    """
    if isinstance(value, datetime):
        return to_utc(value, naive="utc")
    return value


def utc_iso(value: Any) -> Any:
    """A fetched plain TIMESTAMP as ISO-8601 with ``+00:00``.

    Anything that is not a datetime is returned unchanged.
    """
    moment = utc_read(value)
    return moment.isoformat() if isinstance(moment, datetime) else moment


def utc_epoch(value: Any) -> Optional[float]:
    """A fetched plain TIMESTAMP as epoch seconds; None when it is not one."""
    moment = utc_read(value)
    return moment.timestamp() if isinstance(moment, datetime) else None


def utc_tstz_sql(bind: str) -> str:
    """SQL writing the naive UTC bind ``:bind`` into a TIMESTAMP WITH TIME
    ZONE column as a UTC instant."""
    return f"FROM_TZ(CAST(:{bind} AS TIMESTAMP), 'UTC')"


__all__ = [
    "UTC_NOW_SQL",
    "utc_bind",
    "utc_epoch",
    "utc_iso",
    "utc_now_bind",
    "utc_read",
    "utc_tstz_sql",
]
