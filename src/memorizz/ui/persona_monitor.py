"""Shape a host's canonical persona profile for the persona evolution monitor.

The host adapter returns one account's profile: the approved persona and its
evolution history, an optional pending suggestion, and reflection attempts.
This module flattens them into one activity timeline (the grid) plus the
summary the page's tape shows. Pure functions over plain dicts; the entries
themselves are passed through unchanged for the detail pane.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .control_plane_monitor import _parse

# Reflection result -> (label, pill modifier, signal).
REVIEW_STATUS = {
    "proposed": ("Proposed", "info", ""),
    "approved": ("Approved", "success", "good"),
    "applied": ("Approved", "success", "good"),
    "no_change": ("No change", "", ""),
    "dismissed": ("Dismissed", "", ""),
    "failed": ("Failed", "danger", "bad"),
    "reviewing": ("Reviewing", "info", ""),
}


def _shown(value: Any) -> str:
    parsed = _parse(value)
    if parsed is None:
        return str(value or "")
    return parsed.strftime("%Y-%m-%d %H:%M")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(*values: Any) -> str:
    return " ".join(str(value) for value in values if value not in (None, ""))


def _row(kind: str, key: str, when: Any, order: int, **fields: Any) -> Dict[str, Any]:
    parsed = _parse(when)
    row = {
        "kind": kind,
        "key": key,
        "timestamp": str(when or ""),
        "recorded": _shown(when),
        "recorded_at": parsed,
        "sort": parsed.timestamp() if parsed else 0.0,
        "order": order,
    }
    row.update(fields)
    row["search"] = _text(
        row.get("kind_label"),
        row.get("summary"),
        row.get("actor"),
        row.get("status_label"),
        row.get("reference"),
        f"v{row['version']}" if row.get("version") is not None else "",
    ).lower()
    return row


def build_persona_view(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Timeline rows (pending first, then newest first) and tape figures."""
    persona = _mapping(profile.get("persona"))
    pending = profile.get("pending")
    pending = pending if isinstance(pending, Mapping) and pending else None
    history = [
        entry
        for entry in _list(persona.get("evolution_history"))
        if isinstance(entry, Mapping)
    ]
    reviews = [
        review
        for review in _list(profile.get("reviews"))
        if isinstance(review, Mapping)
    ]

    rows: List[Dict[str, Any]] = []
    for index, entry in enumerate(history):
        undo = entry.get("action") == "undo"
        trigger = _mapping(entry.get("change_trigger"))
        rows.append(
            _row(
                "undo" if undo else "change",
                f"change-{index}",
                entry.get("timestamp"),
                index,
                kind_label="Change undone" if undo else "Approach updated",
                version=entry.get("version"),
                summary=str(trigger.get("reason") or ""),
                evidence_count=len(_list(entry.get("evidence"))),
                actor=str(entry.get("actor_id") or ""),
                reference=str(entry.get("review_id") or ""),
                status_label="Reverted" if undo else "Applied",
                status_class="info" if undo else "success",
                signal="" if undo else "good",
                tags="change",
                entry=entry,
            )
        )
    for index, review in enumerate(reviews):
        status = str(review.get("status") or "").strip().lower()
        label, pill, signal = REVIEW_STATUS.get(
            status, (status.replace("_", " ").capitalize() or "Unknown", "", "")
        )
        rows.append(
            _row(
                "review",
                f"review-{index}",
                review.get("timestamp"),
                index,
                kind_label="Reflection",
                version=None,
                summary=str(review.get("reason") or ""),
                evidence_count=review.get("evidence_count"),
                actor=str(review.get("trigger") or ""),
                reference="",
                status_label=label,
                status_class=pill,
                signal=signal,
                tags="review failed" if signal == "bad" else "review",
                entry=review,
            )
        )
    # Newest first; equal or missing timestamps keep the newer list entry first.
    rows.sort(key=lambda row: (row["sort"], row["order"]), reverse=True)
    if pending is not None:
        rows.insert(
            0,
            _row(
                "proposal",
                "pending",
                profile.get("last_review_at"),
                len(history),
                kind_label="Suggested change",
                version=None,
                summary=str(pending.get("reason") or ""),
                evidence_count=len(_list(pending.get("evidence"))),
                actor="Reflection",
                reference=str(pending.get("id") or ""),
                status_label="Awaiting approval",
                status_class="warning",
                signal="warn",
                tags="pending",
                entry=pending,
            ),
        )

    changes = sum(row["kind"] == "change" for row in rows)
    undos = sum(row["kind"] == "undo" for row in rows)
    failed = sum(row["kind"] == "review" and row["signal"] == "bad" for row in rows)
    review_state = str(profile.get("review_state") or "")
    if profile.get("paused"):
        status, signal = "Paused", "warn"
    elif pending is not None:
        status, signal = "Awaiting decision", "warn"
    elif review_state == "failed":
        status, signal = "Last reflection failed", "bad"
    elif review_state == "reviewing":
        status, signal = "Reviewing", ""
    else:
        status, signal = "Ready", "good"
    return {
        "rows": rows,
        "pending": pending is not None,
        "changes": changes,
        "undos": undos,
        "reviews": len(reviews),
        "failed_reviews": failed,
        "proposals": sum(
            1
            for review in reviews
            if str(review.get("status") or "").lower()
            in {"proposed", "approved", "applied"}
        ),
        "evidence": len(_list(pending.get("evidence"))) if pending else 0,
        "status": status,
        "status_signal": signal,
        "last_review": _shown(profile.get("last_review_at")),
        "last_review_at": _parse(profile.get("last_review_at")),
        "next_check_at": _parse(profile.get("next_daily_check_at")),
        "chips": [
            (tag, label, count)
            for tag, label, count in (
                ("pending", "Pending", 1 if pending is not None else 0),
                ("change", "Changes", changes + undos),
                ("review", "Reflections", len(reviews)),
                ("failed", "Failed", failed),
            )
            if count
        ],
    }


__all__ = ["REVIEW_STATUS", "build_persona_view"]
