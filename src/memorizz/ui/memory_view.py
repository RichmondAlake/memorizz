"""Shape raw memory records for the memory explorer pages.

Records arrive as provider dicts with arbitrary keys. This keeps the page
readable and bounded: newest first, a one-line preview, every stored field for
the detail pane, and no embedding vectors (large and meaningless on screen).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..observability.normalization import timestamp

PAGE_LIMIT = 100
FIELD_CHARS = 1500
PREVIEW_KEYS = ("content", "summary", "description", "text", "goals", "background")
NAME_KEYS = ("name", "title", "entity_name", "tool_name")
TIME_KEYS = ("timestamp", "created_at", "updated_at", "last_accessed")
_SHOWN_ELSEWHERE = {"_id", "id"}


def _is_vector_key(key: str) -> bool:
    lowered = key.lower()
    return "embedding" in lowered or lowered in {"vector", "vectors"}


def _is_vector(value: Any) -> bool:
    if getattr(value, "ndim", None) == 1:  # numpy-style arrays
        return len(value) > 16
    return (
        isinstance(value, (list, tuple))
        and len(value) > 16
        and all(isinstance(item, (int, float)) for item in value[:16])
    )


def _is_empty(value: Any) -> bool:
    # Explicit checks: `value in (...)` would compare arrays element-wise.
    return value is None or (
        isinstance(value, (str, list, tuple, dict)) and len(value) == 0
    )


def _text(value: Any, limit: int = FIELD_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str, indent=1)
        except (TypeError, ValueError):
            rendered = str(value)
    else:
        rendered = str(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def _plain(text: str) -> str:
    """One-line preview: markdown emphasis and code marks removed."""
    for mark in ("**", "__", "`", "##", "#"):
        text = text.replace(mark, "")
    return " ".join(text.split())


def _when(record: Dict[str, Any]) -> tuple[str, Optional[datetime]]:
    for key in TIME_KEYS:
        raw = record.get(key)
        if raw in (None, ""):
            continue
        parsed = timestamp(raw)
        shown = str(raw)[:19].replace("T", " ")
        return shown, datetime.fromisoformat(parsed) if parsed else None
    return "", None


def shape_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """One row: identity, preview, time, and the remaining fields for the pane."""
    record_id = str(record.get("_id") or record.get("id") or "")
    name = next((str(record[k]) for k in NAME_KEYS if not _is_empty(record.get(k))), "")
    body_key = next(
        (
            k
            for k in PREVIEW_KEYS
            if not _is_empty(record.get(k)) and not _is_vector(record.get(k))
        ),
        None,
    )
    body = _text(record.get(body_key), limit=20000) if body_key else ""
    recorded, recorded_at = _when(record)
    fields = []
    for key in sorted(record, key=str):
        value = record[key]
        if (
            key in _SHOWN_ELSEWHERE
            or key == body_key
            or _is_vector_key(str(key))
            or _is_vector(value)
            or _is_empty(value)
        ):
            continue
        fields.append({"key": str(key), "value": _text(value)})
    return {
        "id": record_id,
        "name": name,
        "body": body,
        "body_key": body_key or "",
        "preview": _plain(body)[:240],
        "role": str(record.get("role") or ""),
        "memory_id": str(record.get("memory_id") or ""),
        "agent_id": str(record.get("agent_id") or ""),
        "recorded": recorded,
        "recorded_at": recorded_at,
        "fields": fields,
        "size": len(body),
    }


def build_memory_view(
    items: Iterable[Dict[str, Any]],
    *,
    limit: int = PAGE_LIMIT,
    total: Optional[int] = None,
) -> Dict[str, Any]:
    """Newest records first, plus the summary the page's tape shows.

    ``total`` is the store's full size when ``items`` is only its newest page
    (database providers); summary figures then describe the loaded page.
    """
    shaped = [shape_record(item) for item in items if isinstance(item, dict)]
    # Newest first; the sort is stable, so undated records keep their stored
    # order after every dated one. Parsed times are always timezone-aware.
    undated = datetime.min.replace(tzinfo=timezone.utc)
    shaped.sort(key=lambda row: row["recorded_at"] or undated, reverse=True)
    records = shaped[:limit]
    dated = [row["recorded_at"] for row in shaped if row["recorded_at"]]
    newest = max(dated) if dated else None
    series: List[int] = []
    if newest is not None:
        start = (newest - timedelta(days=13)).date()
        counts = Counter(when.date() for when in dated if when.date() >= start)
        series = [counts.get(start + timedelta(days=i), 0) for i in range(14)]
    roles = Counter(row["role"] for row in shaped if row["role"])
    sizes = [row["size"] for row in shaped if row["size"]]
    return {
        "records": records,
        "total": max(total or 0, len(shaped)),
        "has_more": max(total or 0, len(shaped)) > len(records),
        "sampled": total is not None and total > len(shaped),
        "memory_ids": sorted({row["memory_id"] for row in shaped if row["memory_id"]}),
        "agent_count": len({row["agent_id"] for row in shaped if row["agent_id"]}),
        "roles": roles.most_common(),
        "newest": next((row["recorded"] for row in shaped if row["recorded"]), ""),
        "oldest": next(
            (row["recorded"] for row in reversed(shaped) if row["recorded"]), ""
        ),
        "mean_chars": round(sum(sizes) / len(sizes)) if sizes else 0,
        "series": series,
        "show_name": any(row["name"] for row in records),
        "show_role": bool(roles),
        "show_memory": any(row["memory_id"] for row in records),
    }
