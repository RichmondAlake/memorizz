# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Rows for the Observability waterfall, from ``build_causal_waterfall``.

Pure data shaping. Each turn becomes a tree of rows on one clock: its spans
by recorded parent, runs of identical leaf spans folded into one row, and
stretches where the parent ran but no span was recorded. Every failure marks
the bars of the spans above it, so a collapsed tree still shows where it is.

Nothing is inferred: a span without a recorded end stays open, an event with
no duration is a point, and time no span covers is called unaccounted.
"""

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .observability_view import explain_error, timeline_error, timeline_group

# Lane order for filters and the time breakdown, with the label shown.
LANES = (
    ("model", "Model"),
    ("tool", "Tools"),
    ("memory", "Memory"),
    ("output", "Output"),
    ("run", "Run"),
)
LANE_LABELS = dict(LANES)

KIND_LABELS = {
    "turn_start": "Turn",
    "turn_result": "Turn",
    "model_call": "Model call",
    "model_result": "Model call",
    "tool_call": "Tool call",
    "tool_result": "Tool call",
    "memory_retrieval": "Memory retrieval",
    "memory_supply": "Memory supplied",
    "memory_context": "Memory context",
    "memory_selection": "Memory selection",
    "memory_reference": "Memory referenced",
    "context_provenance": "Request context",
    "cache_decision": "Semantic cache",
    "artifact_persisted": "Artifact saved",
    "output_contract": "Output contract",
    "ui_delivery": "Delivered to the UI",
    "intent_plan": "Intent",
    "candidate_selection": "Source selection",
}

# A run of this many identical leaf siblings folds into one row.
FOLD_AT = 3
# Unaccounted time shorter than this share of the turn is noise, not a gap.
GAP_SHARE = 0.015
GAP_FLOOR_MS = 20.0


def duration_label(ms: Optional[float]) -> str:
    """``0.08 ms``, ``8.6 ms``, ``287 ms``, ``1.65 s``, ``24.9 s``, ``1m 04s``."""
    if ms is None:
        return ""
    if ms < 1:
        return f"{ms:.2f} ms"
    if ms < 100:
        return f"{ms:.1f} ms"
    if ms < 1000:
        return f"{ms:.0f} ms"
    seconds = ms / 1000
    if seconds < 10:
        return f"{seconds:.2f} s"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def _tick_step(span_ms: float, target: int = 6) -> float:
    raw = max(span_ms, 1e-6) / target
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        if multiple * magnitude >= raw:
            return multiple * magnitude
    return 10 * magnitude


def tick_label(ms: float, step: float) -> str:
    """An axis label as precise as its step: ``0``, ``250 ms``, ``2.5 s``."""
    if ms == 0:
        return "0"
    if step < 1:
        decimals = min(3, max(0, -math.floor(math.log10(step))))
        return f"{ms:.{decimals}f} ms"
    if step < 1000:
        return f"{ms:.0f} ms"
    if step < 60000:
        seconds = ms / 1000
        return f"{seconds:g} s" if seconds == round(seconds, 1) else f"{seconds:.1f} s"
    return duration_label(ms)


def axis_ticks(extent_ms: float) -> List[Dict[str, Any]]:
    step = _tick_step(extent_ms)
    ticks, value = [], 0.0
    while value <= extent_ms + 1e-9 and len(ticks) < 50:
        ticks.append(
            {
                "left": round(100 * value / extent_ms, 4),
                "label": tick_label(value, step),
            }
        )
        value += step
    return ticks


def label_side(left: float, width: float) -> str:
    """A duration label sits after its bar, before it near the end, and
    inside it when the bar fills the row."""
    end = left + width
    if end <= 82:
        return "right"
    return "left" if left >= 15 else "inside"


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _humanize(value: Any) -> str:
    return str(value or "").replace("_", " ").strip()


def _label_and_detail(node: Mapping[str, Any], lane: str) -> Tuple[str, str]:
    kind = str(node.get("kind") or "")
    attributes = node.get("attributes") or {}
    operation = str(node.get("operation") or "")
    label = KIND_LABELS.get(kind) or _humanize(kind or operation).capitalize() or "Span"
    if lane == "tool":
        detail = attributes.get("logical_tool_name") or attributes.get("tool_name")
        detail = str(detail or (operation if operation != kind else ""))
    elif lane == "model" and kind.startswith("model"):
        model = attributes.get("response_model") or attributes.get("model")
        step = attributes.get("iteration")
        detail = " · ".join(
            part
            for part in (
                str(model or ""),
                f"step {step}" if step not in (None, "") else "",
            )
            if part
        )
    elif kind == "cache_decision":
        detail = _humanize(attributes.get("cache_decision") or node.get("status"))
    elif attributes.get("memory_type"):
        detail = _humanize(attributes["memory_type"])
    else:
        detail = operation if operation and operation != kind else ""
    return label, detail


def _failed(node: Mapping[str, Any]) -> bool:
    attributes = node.get("attributes") or {}
    event = {**attributes, "status": node.get("status"), "kind": node.get("kind")}
    if not timeline_error(event):
        return False
    # Waiting on a person, or stopped on purpose, is not a failure.
    if "approval_required" in (attributes.get("error_code"), node.get("status")):
        return False
    return not (
        node.get("status") == "cancelled"
        and attributes.get("stop_reason") in ("reader_closed", "cancel_requested")
    )


def _warning(node: Mapping[str, Any]) -> str:
    if node.get("cycle"):
        return "Its parent chain loops"
    if node.get("missing_parent"):
        return "Its parent span isn't in this trace"
    if node.get("missing_cause"):
        return "The event that caused it isn't in this trace"
    return ""


def _union(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[List[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _uncovered(
    frame: Tuple[float, float], covered: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    start, end = frame
    holes, cursor = [], start
    for left, right in _union(
        (max(a, start), min(b, end)) for a, b in covered if b > start and a < end
    ):
        if left > cursor:
            holes.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        holes.append((cursor, end))
    return holes


class _Span:
    __slots__ = (
        "node",
        "index",
        "parent",
        "children",
        "depth",
        "lane",
        "start",
        "end",
        "failed",
        "size",
        "failed_below",
        "covered",
    )

    def __init__(self, node: Mapping[str, Any], index: int):
        self.node = node
        self.index = index
        self.parent: Optional["_Span"] = None
        self.children: List["_Span"] = []
        self.depth = 0
        self.lane = timeline_group(node)
        self.start = _number(node.get("offset_ms"))
        duration = _number(node.get("duration_ms"))
        self.end = (
            self.start + duration
            if self.start is not None and duration is not None
            else None
        )
        self.failed = _failed(node)
        # Filled bottom-up by _tree: spans in this subtree, the failures
        # below this span, and the time its descendants' spans cover.
        self.size = 1
        self.failed_below: List["_Span"] = []
        self.covered: List[Tuple[float, float]] = []

    @property
    def measured(self) -> bool:
        return self.end is not None


def _tree(nodes: Sequence[Mapping[str, Any]]) -> List[_Span]:
    """Roots, children sorted by start. A looping chain is shown flat."""
    spans = [_Span(node, index) for index, node in enumerate(nodes)]
    by_id: Dict[Any, _Span] = {}
    roots: List[_Span] = []
    for span in spans:
        parent = by_id.get(span.node.get("parent_span_id"))
        if parent is not None and not span.node.get("cycle"):
            span.parent = parent
            parent.children.append(span)
        else:
            roots.append(span)
        by_id.setdefault(span.node.get("span_id"), span)

    def order(group: List[_Span]) -> None:
        group.sort(key=lambda s: (s.start is None, s.start or 0.0, s.index))

    order(roots)
    visited: List[_Span] = []
    stack = [(root, 0) for root in reversed(roots)]
    while stack:
        span, depth = stack.pop()
        span.depth = depth
        visited.append(span)
        order(span.children)
        stack.extend((child, depth + 1) for child in reversed(span.children))
    # Children before parents, so each subtree is summed once.
    for span in reversed(visited):
        for child in span.children:
            span.size += child.size
            if child.failed:
                span.failed_below.append(child)
            span.failed_below.extend(child.failed_below)
        span.covered = _union(
            [(c.start, c.end) for c in span.children if c.measured]
            + [interval for c in span.children for interval in c.covered]
        )
    return roots


def _is_turn(span: _Span) -> bool:
    return span.measured and span.parent is None and span.lane == "run"


def _breakdown(
    roots: List[_Span], spans: List[_Span], extent: float
) -> List[Dict[str, Any]]:
    """Where a turn's time went: each moment goes to the deepest span running.

    A turn's own span is the frame, so the time none of its spans cover is
    unaccounted rather than counted as the turn's lane. Any other span's own
    time is its work, and counts in its lane.
    """
    if len(roots) == 1 and _is_turn(roots[0]):
        root = roots[0]
        frame = (root.start, root.end)
        candidates = [s for s in spans if s is not root and s.measured]
    else:
        frame = (0.0, extent)
        candidates = [s for s in spans if s.measured]
    length = frame[1] - frame[0]
    if length <= 0 or not candidates:
        return []

    def clip(value: float) -> float:
        return min(max(value, frame[0]), frame[1])

    starts: Dict[float, List[_Span]] = {}
    ends: Dict[float, List[_Span]] = {}
    for span in candidates:
        start, end = clip(span.start), clip(span.end)
        if end > start:
            starts.setdefault(start, []).append(span)
            ends.setdefault(end, []).append(span)
    bounds = sorted({frame[0], frame[1], *starts, *ends})
    totals: Dict[str, float] = {}
    running: set = set()
    # One sweep: the spans running between two boundaries change only at them.
    for left, right in zip(bounds, bounds[1:]):
        running.difference_update(ends.get(left, ()))
        running.update(starts.get(left, ()))
        if running:
            owner = max(running, key=lambda s: (s.depth, s.start, s.index))
            key = owner.lane
        else:
            key = "unaccounted"
        totals[key] = totals.get(key, 0.0) + (right - left)
    order = [key for key, _label in LANES] + ["unaccounted"]
    return [
        {
            "key": key,
            "label": LANE_LABELS.get(key, "Unaccounted"),
            "ms": round(totals[key], 3),
            "share": round(100 * totals[key] / length, 3),
            "duration": duration_label(totals[key]),
        }
        for key in order
        if totals.get(key, 0) > 0
    ]


def _position(start: float, end: float, extent: float) -> Dict[str, float]:
    return {
        "left": round(100 * start / extent, 4),
        "width": round(100 * max(0.0, end - start) / extent, 4),
    }


def _waterfall_group(
    group: Mapping[str, Any], number: int, selected_event_id: Optional[str]
) -> Dict[str, Any]:
    nodes = group.get("nodes") or []
    roots = _tree(nodes)
    spans: List[_Span] = []
    stack = list(reversed(roots))
    while stack:
        span = stack.pop()
        spans.append(span)
        stack.extend(reversed(span.children))
    extent = max(
        [_number(group.get("extent_ms")) or 0.0]
        + [s.end or s.start or 0.0 for s in spans]
    )
    extent = extent if extent > 0 else 1.0
    gap_floor = max(GAP_FLOOR_MS, extent * GAP_SHARE)
    rows: List[Dict[str, Any]] = []
    prefix = f"wf{number}"

    def failure_marks(span: _Span) -> List[Dict[str, Any]]:
        marks = []
        for below in span.failed_below:
            if below.start is not None:
                marks.append(
                    {
                        "left": round(100 * below.start / extent, 4),
                        "target": f"{prefix}-s{below.index}",
                        "label": "Failed: "
                        + " ".join(
                            part
                            for part in _label_and_detail(below.node, below.lane)
                            if part
                        ),
                    }
                )
        return marks

    def span_row(span: _Span, parent_row: Optional[str], depth: int) -> Dict[str, Any]:
        node = span.node
        label, detail = _label_and_detail(node, span.lane)
        attributes = dict(node.get("attributes") or {})
        state = node.get("timing_state") or "unknown_legacy"
        row: Dict[str, Any] = {
            "id": f"{prefix}-s{span.index}",
            "type": "span",
            "parent": parent_row,
            "depth": depth,
            "lane": span.lane,
            "kind": node.get("kind") or "",
            "status": node.get("status") or "unknown",
            "label": label,
            "detail": detail,
            "failed": span.failed,
            "warning": _warning(node),
            "why": explain_error(
                {**attributes, "status": node.get("status"), "kind": node.get("kind")}
            )
            if span.failed
            else "",
            "gap": bool(
                node.get("missing_parent")
                or node.get("missing_cause")
                or node.get("cycle")
            ),
            "anchor": node.get("anchor"),
            "selected": bool(
                selected_event_id and selected_event_id in (node.get("event_ids") or [])
            ),
            "offset_ms": span.start,
            "start": duration_label(span.start) if span.start else "0 ms",
            "duration_ms": _number(node.get("duration_ms")),
            "duration": duration_label(_number(node.get("duration_ms"))),
            "marks": failure_marks(span),
            "spans": span.size,
        }
        if span.start is None:
            row["shape"] = "untimed"
        elif span.measured:
            row["shape"] = "bar"
            row.update(_position(span.start, span.end, extent))
        elif state == "missing_completion":
            row["shape"] = "open"
            row.update(_position(span.start, extent, extent))
        else:
            row["shape"] = "point"
            row.update(_position(span.start, span.start, extent))
        row["detail_data"] = {
            "span_id": node.get("span_id"),
            "parent_span_id": node.get("parent_span_id"),
            "caused_by_event_id": node.get("caused_by_event_id"),
            "task_id": node.get("task_id"),
            "timestamp": node.get("timestamp"),
            "timing_state": state,
            "events": len(node.get("event_ids") or []),
            "self_ms": round(
                sum(b - a for a, b in _uncovered((span.start, span.end), span.covered)),
                3,
            )
            if span.measured and span.children
            else None,
            "attributes": attributes,
        }
        return row

    def gap_row(
        left: float, right: float, parent_row: str, depth: int
    ) -> Dict[str, Any]:
        return {
            "id": f"{prefix}-g{len(rows)}",
            "type": "gap",
            "parent": parent_row,
            "depth": depth,
            "lane": "gap",
            "kind": "",
            "status": "",
            "label": "Unaccounted",
            "detail": "no span recorded",
            "failed": False,
            "warning": "",
            "why": "",
            "gap": False,
            "anchor": None,
            "selected": False,
            "offset_ms": left,
            "duration_ms": round(right - left, 3),
            "duration": duration_label(right - left),
            "marks": [],
            "spans": 0,
            "shape": "gap",
            **_position(left, right, extent),
        }

    def foldable(span: _Span) -> bool:
        return not span.children and not span.failed and not _warning(span.node)

    def fold_key(span: _Span) -> Tuple[Any, Any]:
        return (span.node.get("kind"), span.node.get("operation"))

    def gap_first(holes: List[Tuple[float, float]], span: Optional[_Span]) -> bool:
        return bool(holes) and (
            span is None or span.start is None or holes[0][0] < span.start
        )

    def frame(
        children: List[_Span],
        parent: Optional[_Span],
        parent_row: Optional[str],
        depth: int,
    ) -> Dict[str, Any]:
        # Unaccounted stretches inside a turn, between its spans. A deeper
        # span's own time is its work (a model generating), not a gap.
        holes: List[Tuple[float, float]] = []
        if parent is not None and _is_turn(parent) and children:
            holes = [
                hole
                for hole in _uncovered((parent.start, parent.end), parent.covered)
                if hole[1] - hole[0] >= gap_floor
            ]
        return {
            "children": children,
            "holes": holes,
            "row": parent_row,
            "depth": depth,
            "index": 0,
        }

    def emit(roots: List[_Span]) -> None:
        # Iterative, so an arbitrarily deep recorded chain can't overflow.
        stack = [frame(roots, None, None, 0)]
        while stack:
            current = stack[-1]
            children, holes, index = (
                current["children"],
                current["holes"],
                current["index"],
            )
            parent_row, depth = current["row"], current["depth"]
            if index >= len(children) and not holes:
                stack.pop()
                continue
            child = children[index] if index < len(children) else None
            if gap_first(holes, child):
                rows.append(gap_row(*holes.pop(0), parent_row, depth))
                continue
            run = [child]
            while (
                foldable(child)
                and index + len(run) < len(children)
                and foldable(children[index + len(run)])
                and fold_key(children[index + len(run)]) == fold_key(child)
                and not gap_first(holes, children[index + len(run)])
            ):
                run.append(children[index + len(run)])
            if len(run) >= FOLD_AT:
                rows.append(fold_row(run, parent_row, depth))
                fold_id = rows[-1]["id"]
                rows.extend(span_row(member, fold_id, depth + 1) for member in run)
                current["index"] += len(run)
                continue
            row = span_row(child, parent_row, depth)
            rows.append(row)
            current["index"] += 1
            if child.children:
                stack.append(frame(child.children, child, row["id"], depth + 1))

    def fold_row(
        run: List[_Span], parent_row: Optional[str], depth: int
    ) -> Dict[str, Any]:
        first = run[0]
        label, _detail = _label_and_detail(first.node, first.lane)
        details = list(
            dict.fromkeys(
                part
                for part in (_label_and_detail(s.node, s.lane)[1] for s in run)
                if part
            )
        )
        starts = [s.start for s in run if s.start is not None]
        ends = [s.end for s in run if s.end is not None]
        row = {
            "id": f"{prefix}-f{first.index}",
            "type": "fold",
            "parent": parent_row,
            "depth": depth,
            "lane": first.lane,
            "kind": first.node.get("kind") or "",
            "status": first.node.get("status") or "",
            "label": label,
            "detail": ", ".join(details[:4])
            + (f" and {len(details) - 4} more" if len(details) > 4 else ""),
            "count": len(run),
            "failed": False,
            "warning": "",
            "why": "",
            "gap": False,
            "anchor": None,
            "selected": False,
            "offset_ms": min(starts) if starts else None,
            "duration_ms": None,
            "duration": "",
            "marks": [],
            "spans": len(run),
        }
        if not starts:
            row["shape"] = "untimed"
        elif ends:
            end = max(ends + starts)
            row["shape"] = "bar"
            row["duration_ms"] = round(end - min(starts), 3)
            row["duration"] = duration_label(end - min(starts))
            row.update(_position(min(starts), end, extent))
        else:
            row["shape"] = "point"
            row.update(_position(min(starts), max(starts), extent))
        return row

    emit(roots)
    child_counts: Dict[str, int] = {}
    for row in rows:
        if row["parent"]:
            child_counts[row["parent"]] = child_counts.get(row["parent"], 0) + 1
    details: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row["children"] = child_counts.get(row["id"], 0)
        row["label_side"] = label_side(row.get("left", 0), row.get("width", 0))
        # What the detail panel shows, read by the page's script on demand.
        details[row["id"]] = {
            key: row.get(key)
            for key in (
                "type",
                "label",
                "detail",
                "lane",
                "kind",
                "status",
                "failed",
                "warning",
                "why",
                "anchor",
                "offset_ms",
                "duration_ms",
                "shape",
                "count",
                "spans",
            )
        } | (row.pop("detail_data", None) or {})

    failed = sum(1 for s in spans if s.failed)
    warnings = sum(1 for s in spans if _warning(s.node))
    lanes: Dict[str, int] = {}
    for span in spans:
        lanes[span.lane] = lanes.get(span.lane, 0) + 1
    turn = group.get("turn_id") or group.get("root_trace_id")
    return {
        "id": prefix,
        "root_trace_id": group.get("root_trace_id"),
        "turn_id": group.get("turn_id"),
        "title": f"Turn {str(turn)[:8]}" if turn else "Spans without a turn",
        "started_at": group.get("started_at"),
        "extent_ms": extent,
        "duration": duration_label(extent if any(s.measured for s in spans) else None),
        "span_count": len(spans),
        "failed": failed,
        "warnings": warnings,
        "lanes": lanes,
        "ticks": axis_ticks(extent),
        "breakdown": _breakdown(roots, spans, extent),
        "rows": rows,
        "details": details,
        "selected": any(row["selected"] for row in rows),
    }


def waterfall_view(
    groups: Iterable[Mapping[str, Any]], *, selected_event_id: Optional[str] = None
) -> Dict[str, Any]:
    """Each turn's rows, and the lane and failure counts the filters show."""
    turns = [
        _waterfall_group(group, number, selected_event_id)
        for number, group in enumerate(groups)
        if group.get("nodes")
    ]
    lanes: Dict[str, int] = {}
    for turn in turns:
        for lane, count in turn["lanes"].items():
            lanes[lane] = lanes.get(lane, 0) + count
    filters = [
        {"key": key, "label": label, "count": lanes[key]}
        for key, label in LANES
        if lanes.get(key)
    ]
    failed = sum(turn["failed"] for turn in turns)
    return {
        "turns": turns,
        "filters": filters,
        "failed": failed,
        "spans": sum(turn["span_count"] for turn in turns),
    }
