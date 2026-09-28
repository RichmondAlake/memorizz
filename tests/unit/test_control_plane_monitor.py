"""Learning control-plane monitor: record shaping and the rendered page."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from memorizz import FileSystemConfig, FileSystemProvider, LearningControlPlane
from memorizz.ui.control_plane_monitor import (
    build_control_plane_view,
    event_signal,
    latest_forgetting_plan,
    summarize_event,
)


def _event(event_id, event_type, when, payload=None, stream="s-1", **scope):
    return {
        "record_type": "learning_event",
        "record_id": event_id,
        "event_id": event_id,
        "event_type": event_type,
        "timestamp": when,
        "agent_id": "agent-1",
        "stream_id": stream,
        "event_hash": "hash-" + event_id,
        "payload": payload or {},
        **scope,
    }


@pytest.mark.unit
def test_events_are_newest_first_with_compile_status_from_checkpoints():
    records = [
        _event("e1", "run_started", "2026-09-01T10:00:00+00:00", {"query": "Ship it"}),
        _event(
            "e2",
            "tool_executed",
            "2026-09-01T10:00:05+00:00",
            {"tool_name": "deploy", "success": False},
        ),
        _event(
            "e3",
            "run_completed",
            "2026-09-01T10:00:09+00:00",
            {"status": "success"},
            stream="s-2",
        ),
        {
            "record_type": "learning_checkpoint",
            "stream_id": "s-1",
            "processed_event_hashes": ["hash-e1"],
            "timestamp": "2026-09-01T11:00:00+00:00",
            "compiled_event_count": 1,
        },
    ]
    view = build_control_plane_view(records, scope_stream_id="s-1")

    assert [row["event_id"] for row in view["events"]] == ["e3", "e2", "e1"]
    compiled = {row["event_id"]: row["compiled"] for row in view["events"]}
    assert compiled == {"e1": True, "e2": False, "e3": False}
    assert view["pending_count"] == 2
    # Only e2 is uncompiled in the stream the compile action would process.
    assert view["pending_in_scope"] == 1
    assert view["event_count"] == 3 and view["checkpoint_count"] == 1
    assert view["record_count"] == 4
    assert view["last_compile"] == "2026-09-01 11:00:00"
    failed = next(row for row in view["events"] if row["event_id"] == "e2")
    assert failed["signal"] == "bad"
    assert failed["tags"].split() == ["tool", "pending", "failed"]
    assert failed["summary"] == "deploy failed"
    assert '"event_id": "e2"' in failed["record"]
    streams = {item["stream_id"]: item for item in view["streams"]}
    assert streams["s-1"]["pending"] == 1 and streams["s-1"]["last_compile"]
    assert streams["s-2"]["last_compile"] == ""
    # Streams with uncompiled work come first.
    assert [item["pending"] for item in view["streams"]] == [1, 1]


@pytest.mark.unit
def test_tape_figures_cover_runs_evidence_artifacts_and_tombstones():
    records = [
        _event("r1", "run_completed", "2026-09-01T10:00:00Z", {"status": "success"}),
        _event("r2", "run_completed", "2026-09-02T10:00:00Z", {"status": "failed"}),
        _event(
            "p1",
            "evidence_pack_built",
            "2026-09-02T10:00:01Z",
            {
                "tokens_used": 400,
                "token_budget": 1000,
                "selected_count": 2,
                "candidate_count": 9,
            },
        ),
        _event(
            "p2",
            "evidence_pack_built",
            "2026-09-02T10:00:02Z",
            {"tokens_used": 950, "token_budget": 1000, "warnings": []},
        ),
        {
            "record_type": "learning_artifact",
            "artifact_id": "a1",
            "artifact_kind": "run_digest",
            "updated_at": "2026-09-02T10:00:00Z",
            "utility": 0.55,
            "source_event_ids": ["r1", "r2"],
            "content": "Run completed.",
        },
        {"record_type": "learning_tombstone", "target_id": "a1", "plan_id": "plan-1"},
    ]
    view = build_control_plane_view(records, evidence_budget=1000)

    assert view["runs"] == 2 and view["run_failed"] == 1
    assert view["run_success_rate"] == 50.0
    assert view["evidence_packs"] == 2 and view["evidence_mean_tokens"] == 675
    assert view["evidence_use_percent"] == 68
    assert view["tombstone_count"] == 1 and view["artifact_count"] == 1
    artifact = view["artifacts"][0]
    assert artifact["tombstoned"] is True and artifact["sources"] == 2
    assert artifact["kind_label"] == "Run digest"
    # A pack at 95% of its budget is flagged.
    assert next(r for r in view["events"] if r["event_id"] == "p2")["signal"] == "warn"
    assert dict((key, count) for key, _label, count in view["families"]) == {
        "run": 2,
        "evidence": 2,
    }
    assert [day["count"] for day in view["series"]][-2:] == [1, 3]
    assert view["series"][-1]["failed"] == 1


@pytest.mark.unit
def test_empty_and_malformed_records_are_harmless():
    view = build_control_plane_view(
        [
            None,
            {"record_type": "learning_event"},
            {"record_type": "learning_event", "event_id": "x"},
        ]
    )
    assert view["events"] == [] and view["pending_count"] == 0
    assert view["plan"] is None and view["series"] == [] and view["streams"] == []
    assert view["run_success_rate"] is None and view["evidence_mean_tokens"] is None
    assert view["last_compile"] == "" and view["last_compile_at"] is None


@pytest.mark.unit
def test_event_limit_keeps_the_newest():
    records = [
        _event(f"e{i}", "run_started", f"2026-09-01T10:{i:02d}:00Z") for i in range(12)
    ]
    view = build_control_plane_view(records, event_limit=5)
    assert view["has_more_events"] is True
    assert [row["event_id"] for row in view["events"]] == [
        "e11",
        "e10",
        "e9",
        "e8",
        "e7",
    ]


@pytest.mark.unit
def test_latest_forgetting_plan_tracks_approval_and_approver():
    candidate = {
        "target_id": "a-old",
        "target_type": "memory_fact",
        "reason": "duplicate projection superseded by a newer artifact",
        "utility": 0.1,
        "age_days": 3.2,
        "action": "tombstone",
    }
    planned = [
        _event(
            "f0",
            "forgetting_planned",
            "2026-09-01T09:00:00Z",
            {"plan_id": "old", "candidates": []},
        ),
        _event(
            "f1",
            "forgetting_planned",
            "2026-09-02T09:00:00Z",
            {
                "plan_id": "plan-1",
                "candidate_count": 1,
                "candidates": [candidate],
                "retained": 4,
            },
        ),
    ]
    plan = latest_forgetting_plan(planned)
    assert plan["plan_id"] == "plan-1" and plan["state"] == "awaiting"
    assert plan["candidate_count"] == 1 and plan["retained"] == 4
    assert plan["candidates"][0]["kind_label"] == "Memory fact"

    applied = planned + [
        _event(
            "f2",
            "memory_forgotten",
            "2026-09-02T10:00:00Z",
            {"plan_id": "plan-1", "tombstoned": 1},
        )
    ]
    tombstones = [{"plan_id": "plan-1", "approved_by": "ops@example.test"}]
    plan = latest_forgetting_plan(applied, tombstones)
    assert plan["state"] == "applied" and plan["tombstoned"] == 1
    assert plan["approved_by"] == "ops@example.test"
    assert latest_forgetting_plan([planned[0]])["state"] == "clear"
    assert latest_forgetting_plan([]) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "event_type,payload,signal,summary",
    [
        (
            "tool_executed",
            {"tool_name": "t", "success": True, "duration_ms": 1500},
            "good",
            "t succeeded in 1.5 s",
        ),
        (
            "tool_executed",
            {"tool_name": "t", "success": True, "outcome": {"status": "fallback"}},
            "warn",
            "t completed via fallback",
        ),
        (
            "outcome_recorded",
            {"status": "failure", "verified": True, "source": "csat"},
            "bad",
            "Verified failure from csat",
        ),
        (
            "workflow_recorded",
            {"workflow_id": "wf", "outcome": "success", "step_count": 1},
            "good",
            "Workflow wf: success in 1 step",
        ),
        ("cache_hit", {"reason": "similar"}, "good", "Cache hit: similar"),
        ("skill_demoted", {"skill_id": "s1"}, "warn", "Skill s1"),
        (
            "forgetting_planned",
            {"candidate_count": 2, "retained": 5},
            "warn",
            "Dry run: 2 candidates, 5 retained",
        ),
        (
            "memory_forgotten",
            {"tombstoned": 1, "errors": ["x"]},
            "bad",
            "Tombstoned 1 artifact · 1 error",
        ),
        ("run_started", {}, "", "Run started"),
        ("something_new", {}, "", "Something new"),
    ],
)
def test_signal_and_summary_per_event_type(event_type, payload, signal, summary):
    assert event_signal(event_type, payload) == signal
    assert summarize_event(event_type, payload) == summary


# ---------- The rendered page ----------


@pytest.fixture()
def seeded(tmp_path):
    pytest.importorskip("fastapi")
    from memorizz.memagent.models import MemAgentModel

    provider = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "lcp", lazy_vector_indexes=True)
    )
    provider.store_memagent(
        MemAgentModel(
            agent_id="agent-1", name="Support agent", learning_control_plane=True
        )
    )
    plane = LearningControlPlane(
        provider,
        agent_id="agent-1",
        config={"enabled": True, "compile_async": False, "compile_every_n_events": 0},
    )
    scope = {
        "memory_id": "m-1",
        "user_id": "u-1",
        "thread_id": "t-1",
        "run_id": "run-1",
        "trace_id": "trace-1",
    }
    plane.begin_run("Where is my order?", scope=scope)
    plane.record_tool(
        tool_name="lookup_order",
        arguments={},
        result={"ok": True},
        success=True,
        scope=scope,
    )
    plane.complete_run(
        "It ships today.", status="success", tool_call_count=1, scope=scope
    )
    plane.compile(memory_id="m-1", user_id="u-1", thread_id="t-1")
    plane.begin_run("And the invoice?", scope={**scope, "run_id": "run-2"})
    plane.close()
    return provider


def _client(provider):
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    patcher = patch.dict(
        state._state,
        {"provider": provider, "provider_type": "filesystem", "connection_info": {}},
    )
    patcher.start()
    return TestClient(create_app(), follow_redirects=False), patcher


@pytest.mark.unit
def test_page_shows_events_compile_state_and_keeps_actions(seeded):
    client, patcher = _client(seeded)
    try:
        # The scope form submits every field; blank ones mean "all".
        html = client.get(
            "/learning-control-plane?agent_id=agent-1&memory_id=&user_id=&thread_id="
        ).text
    finally:
        patcher.stop()
    assert 'data-sidebar-scope="learning"' in html
    assert 'aria-label="Continual learning views"' in html
    for tab in ("/memory/workflows", "/memory/skills", "/persona-evolution"):
        assert f'href="{tab}"' in html
    assert 'id="lcp-events"' in html and html.count('class="fleet-detail"') == 4
    assert "Where is my order?" in html and "lookup_order succeeded" in html
    assert html.count("lcp-status--pending") == 1  # the second run only
    assert html.count("lcp-status--compiled") == 3
    for action in (
        "/learning-control-plane/forget-plan",
        "/learning-control-plane/compile",
    ):
        assert f'action="{action}"' in html
    for text in (
        "Plan forgetting (dry run)",
        "Compile pending events",
        "Inspect scope",
    ):
        assert text in html
    for field in (
        'name="agent_id"',
        'name="memory_id"',
        'name="user_id"',
        'name="thread_id"',
    ):
        assert field in html
    assert "Support agent · agent-1" in html
    assert "root_trace_id=trace-1" in html
    assert "Compiled artifacts" in html and "Forgetting plan" in html


@pytest.mark.unit
def test_page_offers_forgetting_approval_after_a_dry_run(seeded):
    from memorizz.ui.routers import learning_control_plane as route

    client, patcher = _client(seeded)
    action = {
        "kind": "forget-plan",
        "message": "Dry run found 2 candidate(s)",
        "report": {"plan_id": "plan-9", "candidate_count": 2},
        "at": "2026-09-28T10:00:00",
    }
    try:
        with patch.dict(route._last_actions, {"agent-1": action}):
            html = client.get("/learning-control-plane?agent_id=agent-1").text
    finally:
        patcher.stop()
    assert "Approve reversible forgetting" in html
    assert 'action="/learning-control-plane/forget-apply"' in html
    assert 'name="plan_id" value="plan-9"' in html and 'name="approved_by"' in html
    assert "Apply 2 tombstone(s)" in html and "Dry run found 2 candidate(s)" in html


@pytest.mark.unit
def test_empty_scope_says_what_records_events_and_links_to_settings(seeded):
    client, patcher = _client(seeded)
    try:
        html = client.get(
            "/learning-control-plane?agent_id=agent-1&user_id=nobody"
        ).text
    finally:
        patcher.stop()
    assert "No learning events in this scope" in html
    assert 'href="/agents/agent-1/edit"' in html and "Clear scope" in html
    assert 'id="lcp-events"' not in html
