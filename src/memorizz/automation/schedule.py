# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Schedule parsing and next-run computation.

This module intentionally avoids external dependencies.
Cron support is 5-field: minute hour day_of_month month day_of_week.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - very old Python
    ZoneInfo = None  # type: ignore


_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_DOW_NAMES = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def validate_timezone_name(name: str):
    tz_name = str(name or "").strip()
    if not tz_name:
        raise ValueError("timezone is required (IANA name, e.g. America/New_York)")
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo is unavailable; cannot validate timezone")
    try:
        return ZoneInfo(tz_name)
    except Exception as exc:
        raise ValueError(f"Invalid timezone '{tz_name}': {exc}") from exc


def _parse_int(value: str, names: Optional[Dict[str, int]] = None) -> int:
    text = value.strip().lower()
    if names and text in names:
        return int(names[text])
    return int(text)


def _expand_range(
    start: int, end: int, step: int, *, min_value: int, max_value: int
) -> Set[int]:
    if step <= 0:
        raise ValueError("step must be positive")
    if start < min_value or start > max_value or end < min_value or end > max_value:
        raise ValueError("range values out of bounds")
    if start <= end:
        values = range(start, end + 1, step)
    else:
        # Wrap-around (e.g., 22-2 in hours)
        values = list(range(start, max_value + 1, step)) + list(
            range(min_value, end + 1, step)
        )
    return {int(v) for v in values}


def parse_cron_field(
    field: str,
    *,
    min_value: int,
    max_value: int,
    names: Optional[Dict[str, int]] = None,
    allow_wrap: bool = False,
) -> Set[int]:
    text = str(field or "").strip()
    if not text:
        raise ValueError("Empty cron field")
    if text == "*":
        return set(range(min_value, max_value + 1))

    result: Set[int] = set()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for part in parts:
        step = 1
        if "/" in part:
            base, step_text = part.split("/", 1)
            base = base.strip()
            step = int(step_text.strip())
        else:
            base = part

        if base == "*":
            result |= _expand_range(
                min_value, max_value, step, min_value=min_value, max_value=max_value
            )
            continue

        if "-" in base:
            start_text, end_text = base.split("-", 1)
            start = _parse_int(start_text, names=names)
            end = _parse_int(end_text, names=names)
            if not allow_wrap and start > end:
                raise ValueError(f"Invalid range '{base}'")
            result |= _expand_range(
                start, end, step, min_value=min_value, max_value=max_value
            )
            continue

        value = _parse_int(base, names=names)
        if value == 7 and min_value == 0 and max_value == 6:
            # Cron allows 7 as Sunday in some syntaxes; normalize to 0.
            value = 0
        if value < min_value or value > max_value:
            raise ValueError(f"Value {value} out of bounds [{min_value}, {max_value}]")
        result.add(value)

    return result


@dataclass(frozen=True)
class CronSpec:
    minutes: Set[int]
    hours: Set[int]
    dom: Set[int]
    months: Set[int]
    dow: Set[int]
    dom_any: bool
    dow_any: bool


def parse_cron_expr(expr: str) -> CronSpec:
    raw = str(expr or "").strip()
    fields = [f for f in raw.split() if f]
    if len(fields) != 5:
        raise ValueError("cron_expr must have 5 fields: min hour dom month dow")

    minute_f, hour_f, dom_f, month_f, dow_f = fields
    dom_any = dom_f.strip() == "*"
    dow_any = dow_f.strip() == "*"

    minutes = parse_cron_field(minute_f, min_value=0, max_value=59, allow_wrap=False)
    hours = parse_cron_field(hour_f, min_value=0, max_value=23, allow_wrap=False)
    dom = parse_cron_field(dom_f, min_value=1, max_value=31, allow_wrap=False)
    months = parse_cron_field(
        month_f, min_value=1, max_value=12, names=_MONTH_NAMES, allow_wrap=False
    )
    dow = parse_cron_field(
        dow_f, min_value=0, max_value=6, names=_DOW_NAMES, allow_wrap=False
    )

    return CronSpec(
        minutes=minutes,
        hours=hours,
        dom=dom,
        months=months,
        dow=dow,
        dom_any=dom_any,
        dow_any=dow_any,
    )


def _cron_dow(dt: datetime) -> int:
    # Cron: Sunday=0, Monday=1 ... Saturday=6.
    return int(dt.isoweekday() % 7)


def cron_matches(dt: datetime, spec: CronSpec) -> bool:
    if dt.month not in spec.months:
        return False
    if dt.hour not in spec.hours:
        return False
    if dt.minute not in spec.minutes:
        return False

    dom_match = dt.day in spec.dom
    dow_match = _cron_dow(dt) in spec.dow

    # Vixie cron semantics: when both DOM and DOW are restricted, match if either matches.
    if spec.dom_any and spec.dow_any:
        return True
    if spec.dom_any:
        return dow_match
    if spec.dow_any:
        return dom_match
    return dom_match or dow_match


def next_cron_time(expr: str, *, after_utc: datetime, tz_name: str) -> datetime:
    if after_utc.tzinfo is None:
        raise ValueError("after_utc must be timezone-aware")
    tz = validate_timezone_name(tz_name)
    spec = parse_cron_expr(expr)

    cursor = after_utc.astimezone(tz).replace(second=0, microsecond=0) + timedelta(
        minutes=1
    )

    # Scan ahead up to ~1 year, minute by minute.
    for _ in range(0, 60 * 24 * 366):
        if cron_matches(cursor, spec):
            return cursor.astimezone(timezone.utc)
        cursor = cursor + timedelta(minutes=1)

    raise ValueError("cron_expr did not produce a next run time within 366 days")


def compute_next_run_at(
    *,
    schedule_type: str,
    cron_expr: Optional[str],
    interval_seconds: Optional[int],
    tz_name: str,
    after_utc: datetime,
) -> datetime:
    normalized = str(schedule_type or "").strip().lower()
    if after_utc.tzinfo is None:
        raise ValueError("after_utc must be timezone-aware")

    if normalized == "cron":
        if not cron_expr:
            raise ValueError("cron_expr is required for schedule_type='cron'")
        return next_cron_time(cron_expr, after_utc=after_utc, tz_name=tz_name)

    if normalized == "interval":
        seconds = int(interval_seconds or 0)
        if seconds <= 0:
            raise ValueError("interval_seconds must be a positive integer")
        return after_utc + timedelta(seconds=seconds)

    if normalized in ("one_shot", "once"):
        # one_shot/once defaults to immediate execution after creation/run-now
        return after_utc

    raise ValueError(
        f"Unknown schedule_type '{schedule_type}' (valid: cron, interval, one_shot, once)"
    )


def render_query_template(
    template: str, *, scheduled_for_utc: datetime, tz_name: str
) -> str:
    tz = validate_timezone_name(tz_name)
    scheduled_local = scheduled_for_utc.astimezone(tz)
    mapping = {
        "scheduled_for_iso": scheduled_local.isoformat(),
        "today_iso": scheduled_local.date().isoformat(),
        "now_utc_iso": utcnow().isoformat(),
        "timezone": tz_name,
    }

    rendered = str(template or "")
    for key, value in mapping.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered
