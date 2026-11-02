"""The Observability waterfall's rows: tree, folds, gaps, failures and time."""

import json

import pytest
from fastapi.testclient import TestClient

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import ObservabilityStore, build_causal_waterfall
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.waterfall_view import (
    axis_ticks,
    duration_label,
    label_side,
    waterfall_view,
)

T0 = "2026-11-01T22:08:30.000Z"


def at(ms):
    """An ISO timestamp ``ms`` milliseconds after T0."""
    seconds, millis = divmod(int(ms), 1000)
    minutes, seconds = divmod(30 + seconds, 60)
    return f"2026-11-01T22:{8 + minutes:02d}:{seconds:02d}.{millis:03d}Z"


def span(span_id, start, duration=None, parent=None, **fields):
    """One start/result pair, or a single event when it has no duration."""
    base = dict(span_id=span_id, parent_span_id=parent, **fields)
    if duration is None:
        return [dict(base, event_id=f"{span_id}-e", timestamp=at(start))]
    return [
        dict(
            base,
            event_id=f"{span_id}-s",
            phase="start",
            status="started",
            timestamp=at(start),
        ),
        dict(
            base,
            event_id=f"{span_id}-r",
            phase="result",
            status=fields.get("status", "success"),
            timestamp=at(start + duration),
            duration_ms=duration,
        ),
    ]


def turn(*spans, selected=None):
    rows = [event for group in spans for event in group]
    view = waterfall_view(build_causal_waterfall(rows), selected_event_id=selected)
    return view, view["turns"][0]


def row(turn_view, label, detail=None):
    return next(
        r
        for r in turn_view["rows"]
        if r["label"] == label and (detail is None or r["detail"] == detail)
    )


def agent_turn(**model_fields):
    """A turn of 10 s: memory, two model calls, one failing tool call."""
    return turn(
        span("root", 0, 10000, kind="turn_start"),
        span(
            "cache",
            100,
            kind="cache_decision",
            cache_decision="disabled",
            parent="root",
        ),
        span(
            "m1",
            1000,
            2000,
            parent="root",
            kind="model_call",
            model="gpt-x",
            iteration=1,
            **model_fields,
        ),
        span("sup-a", 1000, parent="m1", kind="memory_supply", memory_type="history"),
        span("sup-b", 1000, parent="m1", kind="memory_supply", memory_type="tool_log"),
        span("sup-c", 1000, parent="m1", kind="memory_supply", memory_type="entity"),
        span(
            "tool",
            3000,
            500,
            parent="m1",
            kind="tool_call",
            logical_tool_name="search_docs",
            status="error",
            error_code="tool_timeout",
        ),
        span(
            "m2",
            3600,
            2400,
            parent="root",
            kind="model_call",
            model="gpt-x",
            iteration=2,
        ),
    )


def test_rows_name_what_ran_and_where():
    view, t = agent_turn()
    assert row(t, "Turn")["depth"] == 0
    assert row(t, "Model call", "gpt-x · step 1")["lane"] == "model"
    assert row(t, "Tool call", "search_docs")["lane"] == "tool"
    assert row(t, "Semantic cache", "disabled")["shape"] == "point"
    assert [f["key"] for f in view["filters"]] == ["model", "tool", "memory", "run"]
    assert t["title"] == "Spans without a turn"


def test_identical_leaf_siblings_fold_and_open_on_demand():
    _, t = agent_turn()
    fold = row(t, "Memory supplied", "history, tool log, entity")
    assert fold["type"] == "fold" and fold["count"] == 3 and fold["children"] == 3
    members = [r for r in t["rows"] if r["parent"] == fold["id"]]
    assert [m["detail"] for m in members] == ["history", "tool log", "entity"]
    assert all(m["depth"] == fold["depth"] + 1 for m in members)


def test_two_alike_or_a_failed_sibling_never_fold():
    _, t = turn(
        span("root", 0, 1000),
        span("a", 10, 5, parent="root", kind="tool_call", operation="x"),
        span("b", 20, 5, parent="root", kind="tool_call", operation="x"),
        span(
            "c", 30, 5, parent="root", kind="tool_call", operation="x", status="error"
        ),
        span("d", 40, 5, parent="root", kind="tool_call", operation="x"),
    )
    assert not [r for r in t["rows"] if r["type"] == "fold"]


def test_a_failure_is_flagged_explained_and_marked_on_every_span_above_it():
    view, t = agent_turn()
    tool = row(t, "Tool call", "search_docs")
    assert tool["failed"] and "tool call failed" in tool["why"].lower()
    for above in (row(t, "Turn"), row(t, "Model call", "gpt-x · step 1")):
        assert [mark["target"] for mark in above["marks"]] == [tool["id"]]
        assert above["marks"][0]["left"] == pytest.approx(30.0)
    assert not row(t, "Model call", "gpt-x · step 2")["marks"]
    assert view["failed"] == t["failed"] == 1


def test_approvals_and_deliberate_stops_are_not_failures():
    _, t = turn(
        span("a", 0, 10, status="approval_required", kind="tool_call"),
        span(
            "b",
            20,
            10,
            status="cancelled",
            error_code="cancelled",
            stop_reason="reader_closed",
            kind="tool_call",
        ),
        span(
            "c",
            40,
            10,
            status="cancelled",
            error_code="cancelled",
            stop_reason="deadline_exceeded",
            kind="tool_call",
        ),
    )
    assert [r["failed"] for r in t["rows"]] == [False, False, True]


def test_breakdown_gives_each_moment_to_the_deepest_running_span():
    _, t = agent_turn()
    parts = {part["key"]: part["ms"] for part in t["breakdown"]}
    # The model ran 1000-3000 and 3600-6000; the tool ran 3000-3500, under
    # the first model call. Nothing ran 0-1000, 3500-3600 and 6000-10000.
    assert parts == {"model": 4400, "tool": 500, "unaccounted": 5100}
    assert sum(part["share"] for part in t["breakdown"]) == pytest.approx(100)
    assert t["details"][row(t, "Turn")["id"]]["self_ms"] == 5100


def test_a_lone_model_call_keeps_its_own_time():
    _, t = turn(
        span("m", 0, 2000, kind="model_call", model="gpt-x"),
        span("tool", 1500, 400, parent="m", kind="tool_call", tool_name="x"),
    )
    parts = {part["key"]: part["ms"] for part in t["breakdown"]}
    assert parts == {"model": 1600, "tool": 400}
    assert not [r for r in t["rows"] if r["type"] == "gap"]


def test_unaccounted_rows_sit_in_the_turn_not_inside_a_model_call():
    _, t = agent_turn()
    gaps = [r for r in t["rows"] if r["type"] == "gap"]
    assert [(g["offset_ms"], g["duration_ms"]) for g in gaps] == [
        (0, 1000),
        (6000, 4000),
    ]
    assert all(g["depth"] == 1 and g["parent"] == row(t, "Turn")["id"] for g in gaps)
    # The short 100 ms stretch is below the noise floor (1.5% of 10 s).
    assert all(g["duration_ms"] >= 150 for g in gaps)
    order = [r["label"] for r in t["rows"] if r["depth"] == 1]
    assert order.index("Unaccounted") < order.index("Model call")


def test_shapes_say_what_was_recorded():
    _, t = turn(
        span("root", 0, 1000),
        [
            dict(
                event_id="open",
                span_id="open",
                parent_span_id="root",
                phase="start",
                status="started",
                timestamp=at(200),
            )
        ],
        span("instant", 300, parent="root", kind="memory_context"),
        [dict(event_id="untimed", span_id="untimed", parent_span_id="root")],
    )
    shapes = {r["id"]: r["shape"] for r in t["rows"] if r["type"] == "span"}
    assert sorted(shapes.values()) == ["bar", "open", "point", "untimed"]
    open_row = next(r for r in t["rows"] if r.get("shape") == "open")
    assert open_row["left"] == pytest.approx(20) and open_row["width"] == pytest.approx(
        80
    )


def test_the_selected_event_is_marked():
    _, t = agent_turn()
    assert not t["selected"] and not any(r["selected"] for r in t["rows"])
    _, t = turn(
        span("root", 0, 1000), span("child", 10, 5, parent="root"), selected="child-r"
    )
    selected = [t["details"][r["id"]]["span_id"] for r in t["rows"] if r["selected"]]
    assert selected == ["child"] and t["selected"]


def test_loops_show_flat_with_a_warning_and_deep_chains_do_not_overflow():
    rows = [
        dict(event_id="a", span_id="a", parent_span_id="b"),
        dict(event_id="b", span_id="b", parent_span_id="a"),
    ] + [
        dict(
            event_id=f"e{i}",
            span_id=str(i),
            parent_span_id=str(i - 1) if i else None,
            timestamp=at(i),
            duration_ms=4000 - i,
        )
        for i in range(3000)
    ]
    view = waterfall_view(build_causal_waterfall(rows))
    flat = view["turns"][0]["rows"]
    looped = [r for r in flat if r["warning"] == "Its parent chain loops"]
    assert len(looped) == 2 and all(r["depth"] == 0 for r in looped)
    assert max(r["depth"] for r in flat) == 2999


def test_details_carry_metadata_but_never_content():
    _, t = agent_turn(
        input_tokens=1200, output_tokens=40, cost_usd=0.0021, content="PRIVATE"
    )
    data = t["details"][row(t, "Model call", "gpt-x · step 1")["id"]]
    assert data["attributes"]["input_tokens"] == 1200
    assert data["attributes"]["cost_usd"] == 0.0021
    assert "PRIVATE" not in json.dumps(t)


@pytest.mark.parametrize(
    "ms, text",
    [
        (0.08, "0.08 ms"),
        (8.61, "8.6 ms"),
        (286.9, "287 ms"),
        (1648, "1.65 s"),
        (28483, "28.5 s"),
        (64000, "1m 04s"),
        (7380000, "2h 03m"),
        (None, ""),
    ],
)
def test_duration_labels(ms, text):
    assert duration_label(ms) == text


def test_axis_ticks_are_round_and_inside_the_turn():
    assert [tick["label"] for tick in axis_ticks(34310)] == [
        "0",
        "10 s",
        "20 s",
        "30 s",
    ]
    assert [tick["label"] for tick in axis_ticks(800)][:3] == ["0", "200 ms", "400 ms"]
    assert all(0 <= tick["left"] <= 100 for tick in axis_ticks(0.4))


@pytest.mark.parametrize(
    "left, width, side",
    [(10, 20, "right"), (70, 20, "left"), (0, 100, "inside"), (5, 90, "inside")],
)
def test_duration_label_side(left, width, side):
    assert label_side(left, width) == side


def test_the_traces_page_renders_the_waterfall(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("MEMORIZZ_UI_TRACE_CONTENT_MODE", "metadata")
    provider = FileSystemProvider(
        FileSystemConfig(root_path=str(tmp_path / "memory"), lazy_vector_indexes=True)
    )
    store = ObservabilityStore(provider)
    store.record_trace_bundle(
        trace_context={
            "agent_id": "agent",
            "thread_id": "thread",
            "root_trace_id": "root",
            "turn_id": "turn-1234567890",
        },
        events=[
            {
                "event_id": "tool</script><b>",
                "span_id": "tool",
                "trace_kind": "tool_result",
                "status": "error",
                "error_code": "tool_timeout",
                "logical_tool_name": "search</script>",
                "duration_ms": 120,
                "content": "PRIVATE TOOL OUTPUT",
                "timestamp": T0,
            }
        ],
    )
    agent = dict(agent_id="agent", memory_ids=[])
    monkeypatch.setitem(state._state, "provider", provider)
    monkeypatch.setattr(provider, "list_memagents", lambda *a, **k: [agent])
    monkeypatch.setattr(provider, "retrieve_memagent", lambda *a, **k: agent)
    page = (
        TestClient(create_app()).get("/traces?agent_id=agent&root_trace_id=root").text
    )
    assert 'id="trace-waterfall"' in page and "trace-waterfall.js" in page
    assert 'data-waterfall-kind="tool_result"' in page and 'data-failed="true"' in page
    assert "Turn turn-123" in page
    assert "PRIVATE TOOL OUTPUT" not in page
    # A name from the trace can't close the details script early.
    script = page.split("data-wf-details>", 1)[1].split("</script>", 1)[0]
    assert "search\\u003c/script\\u003e" in script
    provider.close()
