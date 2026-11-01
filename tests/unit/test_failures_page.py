"""The failures view across agents, and what each account may see of it."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider
from memorizz.observability import ObservabilityStore, TraceContext
from memorizz.ui.app import create_app
from memorizz.ui.observability_view import fleet_failures
from memorizz.ui.state import _state

pytestmark = pytest.mark.unit


def _failure(tool, code, agent, turn, *, second=0):
    return {
        "event_id": f"{agent}-{turn}-{second}",
        "anchor": f"trace-event-{agent}-{turn}-{second}",
        "kind": "tool_result",
        "tool_name": tool,
        "status": "error",
        "error_code": code,
        "outcome_reason_code": code,
        "agent_id": agent,
        "turn_id": turn,
        "root_trace_id": f"root-{turn}",
        "timestamp": f"2026-11-01T22:00:{second:02d}+00:00",
    }


def _ending(agent, turn, *, failed=False):
    return {
        "event_id": f"{agent}-{turn}-end",
        "kind": "turn_result",
        "status": "error" if failed else "success",
        "error_code": "boom" if failed else None,
        "agent_id": agent,
        "turn_id": turn,
        "timestamp": "2026-11-01T22:00:59+00:00",
    }


def test_agents_are_counted_and_the_worst_comes_first():
    fleet = fleet_failures(
        [
            _failure("search", "protocol_error", "calm", "t1", second=1),
            _failure("search", "protocol_error", "calm", "t1", second=2),
            _ending("calm", "t1"),
            _failure("fetch", "protocol_error", "broken", "t2", second=3),
            _ending("broken", "t2", failed=True),
            _ending("healthy", "t3"),
        ],
        names={"calm": "Calm agent"},
    )
    assert fleet["agents_seen"] == 3 and fleet["agents_failing"] == 2
    assert [agent["agent_id"] for agent in fleet["agents"]] == ["broken", "calm"]
    assert fleet["agents"][1]["name"] == "Calm agent"
    assert fleet["agents"][0]["name"] == "broken"


def test_the_same_failure_on_two_agents_is_one_line():
    fleet = fleet_failures(
        [
            _failure("search", "protocol_error", "a", "t1", second=1),
            _failure("search", "protocol_error", "b", "t2", second=2),
            _ending("a", "t1"),
            _ending("b", "t2"),
        ]
    )
    assert len(fleet["groups"]) == 1
    assert fleet["groups"][0]["count"] == 2
    assert sorted(fleet["groups"][0]["agents"]) == ["a", "b"]


def test_the_summary_can_be_sent_as_json():
    fleet = fleet_failures([_failure("search", "protocol_error", "a", "t1")])
    assert "search" in json.dumps(fleet)


def test_no_failures_is_a_plain_zero():
    fleet = fleet_failures([_ending("a", "t1")])
    assert (fleet["total"], fleet["agents_seen"], fleet["agents"]) == (0, 1, [])


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "false")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    accounts = {
        "viewer": {
            "token": "viewer-test-token-123456",
            "role": "viewer",
            "application_id": "app-A",
        },
        "admin": {"token": "admin-test-token-123456", "role": "admin"},
    }
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", json.dumps(accounts))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).isoformat()
    for app, agent in (("app-A", "agent-a"), ("app-B", "agent-b")):
        context = TraceContext(
            application_id=app,
            agent_id=agent,
            thread_id="thread",
            root_trace_id=f"root-{app}",
            run_id=f"run-{app}",
            turn_id=f"turn-{app}",
        )
        ObservabilityStore(provider).record_trace_bundle(
            trace_context=context.to_carrier(),
            events=[
                {
                    "event_id": f"failure-{app}",
                    "trace_kind": "tool_result",
                    "title": f"Tool Result: tool-of-{app}",
                    "tool_name": f"tool-of-{app}",
                    "status": "error",
                    "error_code": "protocol_error",
                    "outcome_reason_code": "protocol_error",
                    "timestamp": stamp,
                }
            ],
        )
    with patch.dict(
        _state, provider=provider, provider_type="filesystem", connection_info={}
    ):
        yield TestClient(create_app())


def _as(account):
    return {"Authorization": f"Bearer {account}-test-token-123456"}


@pytest.mark.parametrize(
    "path", ["/traces/failures", "/traces/failures.json", "/traces/failures.csv"]
)
def test_a_scoped_viewer_sees_only_their_own_failures(ui, path):
    response = ui.get(path, headers=_as("viewer"))
    assert response.status_code == 200, response.text[:300]
    assert "tool-of-app-A" in response.text
    assert "tool-of-app-B" not in response.text
    assert "agent-b" not in response.text


def test_an_admin_sees_every_agent(ui):
    page = ui.get("/traces/failures", headers=_as("admin")).text
    assert "tool-of-app-A" in page and "tool-of-app-B" in page


def test_a_range_with_nothing_in_it_says_so(ui):
    response = ui.get(
        "/traces/failures?window=custom&start=2020-01-01T00:00&end=2020-01-02T00:00",
        headers=_as("admin"),
    )
    assert "No failures" in response.text


def test_a_spreadsheet_cannot_run_a_cell_of_the_export(ui, monkeypatch):
    import memorizz.ui.routers.failures as failures

    monkeypatch.setattr(
        failures,
        "fleet_failures",
        lambda events, names=None: {
            "agents": [],
            "total": 1,
            "groups": [
                {
                    "name": "=HYPERLINK(1)",
                    "code": "x",
                    "why": "y",
                    "count": 1,
                    "turns": 1,
                    "stopped": False,
                    "agents": ["a"],
                    "last_date": "",
                    "last_clock": "",
                }
            ],
        },
    )
    text = ui.get("/traces/failures.csv", headers=_as("admin")).text
    assert ",=HYPERLINK" not in text and "'=HYPERLINK" in text


def test_a_scoped_account_cannot_change_rules(ui):
    response = ui.post(
        "/traces/failures/alerts",
        data={"threshold": "1", "window": "1h"},
        headers=_as("viewer"),
        follow_redirects=False,
    )
    assert response.status_code == 403


def test_rules_are_not_shown_to_a_scoped_viewer(ui, monkeypatch):
    from memorizz.ui import failure_alerts

    monkeypatch.setattr(failure_alerts, "_check_url", lambda url: None)
    failure_alerts.save_rule(
        failure_alerts.rule_from_form(
            threshold="1", window="1h", webhook_url="https://hooks.example.com/s3cr3t"
        ),
        base_url="http://127.0.0.1/",
    )
    assert (
        "hooks.example.com"
        not in ui.get("/traces/failures", headers=_as("viewer")).text
    )
    page = ui.get("/traces/failures", headers=_as("admin")).text
    assert "hooks.example.com" in page and "s3cr3t" not in page


def test_the_agent_page_can_narrow_its_failures_to_a_range(ui):
    page = ui.get("/traces?agent_id=agent-a&window=1h", headers=_as("admin")).text
    assert 'id="trace-failures"' in page and "last hour" in page


class _Read(list):
    coverage = {"truncated": True}


@pytest.mark.parametrize(
    ("oldest_hours_ago", "truncated"),
    [(1, True), (30, False)],
    ids=["stopped inside the range", "read past the start of the range"],
)
def test_a_warning_appears_only_when_the_range_was_not_fully_read(
    monkeypatch, oldest_hours_ago, truncated
):
    from datetime import datetime, timedelta, timezone

    import memorizz.ui.routers.failures as failures
    from memorizz.observability.overview import resolve_window

    now = datetime.now(timezone.utc)
    stamp = (now - timedelta(hours=oldest_hours_ago)).isoformat()
    read = _Read([{"agent_id": "a", "kind": "turn_result", "timestamp": stamp}])
    monkeypatch.setattr(failures, "_selection_window", lambda **kw: (None, read))
    monkeypatch.setattr(failures, "_agent_names", dict)
    span = resolve_window("24h", now=now)
    assert failures.read_fleet_failures(span)["truncated"] is truncated
