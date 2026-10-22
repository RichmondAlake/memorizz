"""Labels on a turn's trace: validation, filters and turn selection.

A trace tag is a short ``key=value`` label a host puts on a turn, such as
``experiment=lme-v2`` or ``question=q17``. Tags are content-free metadata:
bounded identifiers, never prompts, answers or credentials. MemAgent records
them on the turn's ``turn_start`` and ``turn_result`` events and on the trace
bundle; normalization copies a bundle's tags onto each of its events, so the
Traces page can select a turn by any of them.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Mapping, Optional

from .privacy import validate_opaque

MAX_TRACE_TAGS = 16
MAX_TAG_KEY_LENGTH = 64
MAX_TAG_VALUE_LENGTH = 128

# Keys are identifiers; values are labels that may hold single spaces. Neither
# may hold "=" or ",", which separate tags in a filter.
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_VALUE = re.compile(r"[\w.:@+/-]+(?: [\w.:@+/-]+)*\Z")

# The tags of the turn running in this context. Work that turn starts here,
# such as a delegate agent's turn or a harness task, inherits them.
current_trace_tags: ContextVar[Dict[str, str]] = ContextVar(
    "memorizz_current_trace_tags", default={}
)

# Conversation and tool-log rows carry no turn id. They belong to a tagged
# turn when written in its thread between its first and last event, give or
# take this much clock skew between the process and the database.
_ROW_SLACK = timedelta(seconds=1)


def normalize_trace_tags(tags: Any) -> Dict[str, str]:
    """Return ``tags`` validated and sorted by key; ``None`` is no tags.

    Raises ``TypeError`` when ``tags`` is not a mapping of strings and
    ``ValueError`` when a key or value breaks the rules: at most 16 tags; keys
    of 1-64 characters (letters, digits, ``_ . : -``, starting with a letter
    or digit); values of 1-128 characters (letters, digits, ``_ . : @ + / -``
    and single inner spaces); no address, URL or credential in either.
    """
    if tags is None:
        return {}
    if not isinstance(tags, Mapping):
        raise TypeError("trace_tags must be a dict of string keys and values")
    if len(tags) > MAX_TRACE_TAGS:
        raise ValueError(f"trace_tags holds at most {MAX_TRACE_TAGS} tags")
    normalized: Dict[str, str] = {}
    for key, value in tags.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("trace_tags keys and values must be strings")
        if len(key) > MAX_TAG_KEY_LENGTH or not _KEY.match(key):
            raise ValueError(
                f"trace tag key {key[:80]!r} must be 1-{MAX_TAG_KEY_LENGTH} "
                "letters, digits, '_', '.', ':' or '-', starting with a letter "
                "or digit"
            )
        if len(value) > MAX_TAG_VALUE_LENGTH or not _VALUE.match(value):
            raise ValueError(
                f"trace tag {key!r} needs a value of 1-{MAX_TAG_VALUE_LENGTH} "
                "letters, digits, '_', '.', ':', '@', '+', '/', '-' or single "
                "inner spaces"
            )
        validate_opaque(key)
        validate_opaque(value)
        normalized[key] = value
    return dict(sorted(normalized.items()))


def inherited_trace_tags(own: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """``own`` tags plus those of the turn running in this context, own first:
    a delegate's or a harness run's turn keeps its parent's labels."""
    merged = dict(own or {})
    for key, value in current_trace_tags.get().items():
        if key not in merged and len(merged) < MAX_TRACE_TAGS:
            merged[key] = value
    return merged


def stored_trace_tags(value: Any) -> Optional[Dict[str, str]]:
    """Tags read back from storage, or ``None`` when they are malformed."""
    try:
        return normalize_trace_tags(value)
    except (TypeError, ValueError):
        return None


def parse_tag_filter(values: Iterable[Any]) -> Dict[str, str]:
    """Parse ``key=value`` filters; each value may hold several, comma-separated.

    Every tag must match, so one key cannot ask for two values. Raises
    ``ValueError`` for a malformed filter.
    """
    wanted: Dict[str, str] = {}
    for raw in values or ():
        for part in str(raw or "").split(","):
            part = part.strip()
            if not part:
                continue
            key, separator, value = part.partition("=")
            key, value = key.strip(), value.strip()
            if not separator or not key or not value:
                raise ValueError("A tag filter is key=value, such as experiment=lme-v2")
            if wanted.get(key, value) != value:
                raise ValueError(f"Tag {key!r} can be filtered by one value at a time")
            wanted[key] = value
    return normalize_trace_tags(wanted)


def format_tag_filter(tags: Mapping[str, str]) -> str:
    """The filter string that selects ``tags`` again: ``k=v,k2=v2``."""
    return ",".join(f"{key}={value}" for key, value in sorted(tags.items()))


def tags_match(tags: Any, wanted: Mapping[str, str]) -> bool:
    """Whether ``tags`` carries every ``key=value`` in ``wanted``."""
    if not wanted:
        return True
    if not isinstance(tags, Mapping):
        return False
    return all(tags.get(key) == value for key, value in wanted.items())


def _turn_key(event: Mapping[str, Any]) -> str:
    return str(event.get("turn_id") or event.get("root_trace_id") or "")


def _moment(value: Any) -> Optional[datetime]:
    from .normalization import timestamp

    text = timestamp(value)
    return datetime.fromisoformat(text) if text else None


def select_tagged_turns(events, wanted: Mapping[str, str]):
    """Keep the events of turns tagged with every ``key=value`` in ``wanted``.

    A trace event belongs to a turn by its turn id. A conversation or tool-log
    row has none, so it is kept when it was written in the thread of a tagged
    turn while that turn ran. The result keeps the window's coverage metadata
    and records the filter.
    """
    from .normalization import TraceEvents

    if not wanted:
        return events
    rows = list(events)
    spans: Dict[str, Dict[str, Any]] = {}
    for event in rows:
        if not tags_match(event.get("trace_tags"), wanted):
            continue
        key = _turn_key(event)
        if not key:
            continue
        span = spans.setdefault(
            key, {"thread": event.get("thread_id"), "start": None, "end": None}
        )
        moment = _moment(event.get("timestamp"))
        if moment is not None:
            span["start"] = (
                moment if span["start"] is None else min(span["start"], moment)
            )
            span["end"] = moment if span["end"] is None else max(span["end"], moment)
    selected = []
    for event in rows:
        key = _turn_key(event)
        if key:
            if key in spans:
                selected.append(event)
            continue
        moment = _moment(event.get("timestamp"))
        if moment is None:
            continue
        thread = event.get("thread_id")
        if any(
            span["start"] is not None
            and span["thread"] == thread
            and span["start"] - _ROW_SLACK <= moment <= span["end"] + _ROW_SLACK
            for span in spans.values()
        ):
            selected.append(event)
    metadata = dict(getattr(events, "coverage", {}) or {})
    metadata.update(
        normalized_events=len(selected),
        tag_filter=format_tag_filter(wanted),
        tagged_turns=len(spans),
    )
    return TraceEvents(selected, coverage=metadata)


def tag_pairs(events, limit: int = MAX_TRACE_TAGS) -> list:
    """Distinct ``(key, value)`` tags across ``events``, sorted, at most ``limit``."""
    pairs = set()
    for event in events:
        tags = event.get("trace_tags")
        if isinstance(tags, Mapping):
            pairs.update((str(key), str(value)) for key, value in tags.items())
    return sorted(pairs)[:limit]


__all__ = [
    "MAX_TAG_KEY_LENGTH",
    "MAX_TAG_VALUE_LENGTH",
    "MAX_TRACE_TAGS",
    "format_tag_filter",
    "current_trace_tags",
    "inherited_trace_tags",
    "normalize_trace_tags",
    "parse_tag_filter",
    "select_tagged_turns",
    "stored_trace_tags",
    "tag_pairs",
    "tags_match",
]
