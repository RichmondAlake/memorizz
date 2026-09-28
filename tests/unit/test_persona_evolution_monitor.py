"""Persona evolution monitor: one activity timeline from the host's profile."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import pytest

from memorizz.ui.persona_monitor import build_persona_view

PROFILE = {
    "revision": 7,
    "paused": False,
    "review_state": "proposed",
    "last_review_at": "2026-09-20T09:00:00Z",
    "history_limit": 50,
    "pending": {
        "id": "review-9",
        "reason": "You asked for page numbers.",
        "changes": {"goals": {"old": "A", "new": "B"}},
        "evidence": [{"id": "m-1", "thread_id": "t-1", "text": "Cite pages."}],
    },
    "persona": {
        "name": "Tutor",
        "version": 3,
        "goals": "A",
        "evolution_history": [
            {
                "version": 2,
                "timestamp": "2026-09-10T10:00:00Z",
                "action": "applied",
                "actor_id": "operator:casey",
                "review_id": "review-1",
                "change_trigger": {"reason": "Examples first."},
                "evidence": [{"id": "m-0"}],
            },
            {
                "version": 3,
                "timestamp": "2026-09-10T10:00:00Z",
                "action": "undo",
                "actor_id": "operator:local",
                "change_trigger": {"reason": "Previous change undone."},
                "evidence": [],
            },
        ],
    },
    "reviews": [
        {
            "status": "no_change",
            "timestamp": "2026-09-05T09:00:00Z",
            "reason": "Nothing durable",
            "evidence_count": 12,
            "trigger": "daily",
        },
        {
            "status": "failed",
            "timestamp": "2026-09-15T09:00:00Z",
            "reason": "Timed out",
            "evidence_count": 3,
            "trigger": "manual",
        },
    ],
}


@pytest.mark.unit
def test_pending_first_then_newest_first_with_ties_by_list_order():
    view = build_persona_view(PROFILE)
    assert [row["key"] for row in view["rows"]] == [
        "pending",
        "review-1",
        "change-1",  # same timestamp as change-0 but recorded after it
        "change-0",
        "review-0",
    ]
    pending, failed, undo, change, quiet = view["rows"]
    assert (
        pending["status_label"] == "Awaiting approval" and pending["tags"] == "pending"
    )
    assert pending["evidence_count"] == 1 and pending["entry"] is PROFILE["pending"]
    assert failed["signal"] == "bad" and failed["tags"] == "review failed"
    assert undo["kind"] == "undo" and undo["status_label"] == "Reverted"
    assert change["version"] == 2 and change["actor"] == "operator:casey"
    assert quiet["status_label"] == "No change" and quiet["evidence_count"] == 12
    assert "examples first." in change["search"] and "v2" in change["search"]


@pytest.mark.unit
def test_tape_status_and_counts():
    view = build_persona_view(PROFILE)
    assert view["status"] == "Awaiting decision" and view["status_signal"] == "warn"
    assert (
        view["changes"],
        view["undos"],
        view["reviews"],
        view["failed_reviews"],
    ) == (1, 1, 2, 1)
    assert view["evidence"] == 1 and view["pending"] is True
    assert view["last_review"] == "2026-09-20 09:00"
    assert [tag for tag, _label, _count in view["chips"]] == [
        "pending",
        "change",
        "review",
        "failed",
    ]

    paused = deepcopy(PROFILE)
    paused.update(paused=True)
    assert build_persona_view(paused)["status"] == "Paused"
    failed = deepcopy(PROFILE)
    failed.update(pending=None, review_state="failed")
    assert build_persona_view(failed)["status_signal"] == "bad"


@pytest.mark.unit
def test_minimal_or_malformed_profiles_do_not_break_the_page():
    view = build_persona_view(
        {"persona": {"version": 1}, "reviews": "bad", "pending": {}}
    )
    assert view["rows"] == [] and view["chips"] == []
    assert view["status"] == "Ready" and view["last_review"] == ""
    odd = build_persona_view(
        {"persona": {"evolution_history": [None, {"timestamp": "now"}]}}
    )
    assert [row["recorded"] for row in odd["rows"]] == ["now"]


@pytest.mark.unit
def test_rendered_page_puts_each_entry_in_one_detail_pane(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memorizz.ui.app import create_app

    monkeypatch.setattr("memorizz.ui.app._load_layered_env", lambda: None)
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "persona-monitor-token")
    monkeypatch.delenv("MEMORIZZ_UI_AUTH_ACCOUNTS", raising=False)
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    adapter = Mock()
    adapter.view.return_value = deepcopy(PROFILE)
    client = TestClient(
        create_app(persona_evolution=adapter, persona_evolution_actions=True)
    )
    html = client.get(
        "/persona-evolution?user_id=account-1",
        headers={"Authorization": "Bearer persona-monitor-token"},
    ).text
    assert 'data-sidebar-scope="learning"' in html and 'id="pe-table"' in html
    assert html.count('class="fleet-detail pe-detail"') == 5
    # The browser suite relies on these being unique on the page.
    assert html.count("Suggested adaptation · not applied") == 1
    assert html.count('class="pe-evidence"') == 1
    assert html.count("Approve for next turn") == 1 and html.count(">Reflect now<") == 1
    assert "Undo last change" not in html  # the latest entry is already an undo
    assert "Version 3" in html and "Version 2" not in html
    assert html.count('data-action="') == 4  # reflect, pause, approve, dismiss
    assert 'href="/memory/skills"' in html and 'href="/learning-control-plane"' in html
