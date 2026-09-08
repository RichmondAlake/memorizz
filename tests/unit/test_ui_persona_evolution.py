"""Persona UI renders canonical host data and never bypasses operator controls."""
from copy import deepcopy
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from memorizz.ui.app import create_app

PROFILE = {
    "agent_id": "openspeech-chat-assistant",
    "revision": 2,
    "paused": False,
    "pending": {
        "id": "proposal-1",
        "reason": "You asked for practical examples.",
        "changes": {
            "goals": {
                "old": "Learn effectively.",
                "new": "Start with a practical example.",
            }
        },
        "evidence": [
            {
                "id": "memory-1",
                "thread_id": "thread-1",
                "text": "Use concrete examples before definitions.",
                "timestamp": "2026-09-07T09:00:00",
            }
        ],
    },
    "persona": {
        "name": "OpenSpeech",
        "role": "Learning partner",
        "goals": "Learn effectively.",
        "version": 1,
        "evolution_history": [],
    },
    "review_state": "proposed",
    "reviews": [],
    "last_review_at": "2026-09-07T10:00:00",
    "history_limit": 50,
}


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    monkeypatch.setattr("memorizz.ui.app._load_layered_env", lambda: None)
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "test-persona-operator-token")
    monkeypatch.delenv("MEMORIZZ_UI_AUTH_ACCOUNTS", raising=False)
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "persona-audit.jsonl"))

    def make(adapter=None, actions=False):
        return TestClient(
            create_app(persona_evolution=adapter, persona_evolution_actions=actions)
        )

    return make


HEADERS = {"Authorization": "Bearer test-persona-operator-token"}


@pytest.mark.parametrize(
    "values",
    [
        {"action": "pause", "revision": 2, "paused": "false"},
        {"action": "schedule", "revision": 2, "daily_enabled": 1},
        {"action": "pause", "revision": True, "paused": False},
        {"action": "undo", "revision": -1},
        {"action": "approve", "revision": 2, "proposal_id": None},
        {"action": "dismiss", "revision": 2, "proposal_id": ""},
    ],
)
def test_invalid_settings_do_not_reach_host(make_client, values):
    adapter = Mock()
    response = make_client(adapter, actions=True).post(
        "/persona-evolution/action",
        headers=HEADERS,
        json={"user_id": "account-1", **values},
    )
    assert response.status_code == 400
    assert not adapter.method_calls


def test_oversized_streamed_body_is_rejected_before_host_call(make_client):
    adapter = Mock()
    response = make_client(adapter, actions=True).post(
        "/persona-evolution/action",
        headers={**HEADERS, "Content-Type": "application/json"},
        content=iter([b" " * 2048, b" " * 2049]),
    )
    assert response.status_code == 400
    assert not adapter.method_calls


def test_canonical_diff_and_evidence_render_and_escape_html(make_client):
    profile = deepcopy(PROFILE)
    profile["pending"]["reason"] = '<script>alert("private")</script>'
    adapter = Mock()
    adapter.view.return_value = profile
    client = make_client(adapter, actions=True)
    response = client.get("/persona-evolution?user_id=account-1", headers=HEADERS)
    assert response.status_code == 200, response.text
    assert "Version 1" in response.text and "Approve for next turn" in response.text
    assert "Use concrete examples before definitions." in response.text
    assert (
        "&lt;script&gt;" in response.text
        and '<script>alert("private")</script>' not in response.text
    )
    assert "Trace editing stays disabled" in response.text
    assert (
        "/traces?agent_id=openspeech-chat-assistant&user_id=account-1&thread_id=thread-1#trace-timeline"
        in response.text
    )
    assert response.headers["cache-control"] == "no-store"
    adapter.view.assert_called_once_with("account-1")


def test_read_only_default_disables_actions_and_unauthenticated_reads(make_client):
    adapter = Mock()
    adapter.view.return_value = PROFILE
    client = make_client(adapter)
    assert client.get("/persona-evolution?user_id=account-1").status_code == 401
    response = client.post(
        "/persona-evolution/action",
        headers=HEADERS,
        json={"user_id": "account-1", "action": "reflect"},
    )
    assert response.status_code == 403
    adapter.reflect.assert_not_called()


def test_explicit_persona_controls_dont_enable_other_writes_or_cross_origin(
    make_client,
):
    adapter = Mock()
    adapter.reflect.return_value = PROFILE
    client = make_client(adapter, actions=True)
    request = {"user_id": "account-1", "action": "reflect"}
    assert (
        client.post(
            "/persona-evolution/action",
            headers={**HEADERS, "Origin": "https://evil.invalid"},
            json=request,
        ).status_code
        == 403
    )
    assert client.post("/agents/new", headers=HEADERS, data={}).status_code == 403
    response = client.post(
        "/persona-evolution/action",
        headers={**HEADERS, "Origin": "http://testserver"},
        json=request,
    )
    assert response.status_code == 200
    adapter.reflect.assert_called_once_with("account-1", actor_id="operator:local")


def test_trace_only_accounts_cannot_read_or_mutate_personas(make_client, monkeypatch):
    monkeypatch.setenv(
        "MEMORIZZ_UI_AUTH_ACCOUNTS",
        '{"viewer":{"token":"restricted-persona-token","role":"viewer","user_id":"account-1"}}',
    )
    adapter = Mock()
    client = make_client(adapter, actions=True)
    headers = {"Authorization": "Bearer restricted-persona-token"}
    assert (
        client.get("/persona-evolution?user_id=account-1", headers=headers).status_code
        == 403
    )
    assert (
        client.post(
            "/persona-evolution/action",
            headers=headers,
            json={"action": "reflect", "user_id": "account-1"},
        ).status_code
        == 403
    )
    adapter.view.assert_not_called()
    adapter.reflect.assert_not_called()


def test_missing_host_is_honest_and_service_errors_dont_leak(make_client):
    assert (
        "Connect your application's persona adapter"
        in make_client().get("/persona-evolution", headers=HEADERS).text
    )
    adapter = Mock()
    adapter.reflect.side_effect = RuntimeError("secret model key")
    response = make_client(adapter, actions=True).post(
        "/persona-evolution/action",
        headers=HEADERS,
        json={"user_id": "account-1", "action": "reflect"},
    )
    assert response.status_code == 503 and "secret model key" not in response.text


@pytest.mark.parametrize("action", [[], {}, None, 4, True])
def test_malformed_action_is_a_client_error_not_a_server_crash(make_client, action):
    adapter = Mock()
    response = make_client(adapter, actions=True).post(
        "/persona-evolution/action",
        headers=HEADERS,
        json={"user_id": "account-1", "action": action},
    )
    assert response.status_code == 400
    adapter.reflect.assert_not_called()
    adapter.decide.assert_not_called()


def test_unwritable_audit_log_denies_reads_and_writes(make_client, monkeypatch):
    adapter = Mock()
    monkeypatch.setattr(
        "memorizz.ui.routers.persona_evolution.audit_trace_view",
        lambda *_a, **_k: False,
    )
    client = make_client(adapter, actions=True)
    assert (
        client.get("/persona-evolution?user_id=account-1", headers=HEADERS).status_code
        == 503
    )
    assert (
        client.post(
            "/persona-evolution/action",
            headers=HEADERS,
            json={"user_id": "account-1", "action": "reflect"},
        ).status_code
        == 503
    )
    adapter.view.assert_not_called()
    adapter.reflect.assert_not_called()


def test_configuring_private_profiles_requires_auth(make_client, monkeypatch):
    monkeypatch.delenv("MEMORIZZ_UI_AUTH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="authentication"):
        make_client(Mock(), actions=True)


def test_account_access_audit_is_scoped_and_contains_no_raw_account(
    make_client, tmp_path
):
    import json

    adapter = Mock()
    adapter.view.return_value = PROFILE
    client = make_client(adapter, actions=True)
    for user in ["account-one", "account-one", "account-two"]:
        assert (
            client.get(
                "/persona-evolution", params={"user_id": user}, headers=HEADERS
            ).status_code
            == 200
        )
    text = (tmp_path / "persona-audit.jsonl").read_text()
    rows = [
        json.loads(line)
        for line in text.splitlines()
        if json.loads(line)["action"] == "persona_profile_viewed"
    ]
    assert rows[0]["target_account_hash"] == rows[1]["target_account_hash"]
    assert rows[1]["target_account_hash"] != rows[2]["target_account_hash"]
    assert "account-one" not in text and "account-two" not in text


def test_daily_control_and_reflection_trace_link_are_host_scoped(make_client):
    profile = deepcopy(PROFILE)
    profile.update(daily_enabled=True, next_daily_check_at="2026-09-09T12:00:00+00:00")
    profile["reviews"] = [
        {
            "status": "proposed",
            "timestamp": "now",
            "reason": "Example preference",
            "evidence_count": 1,
            "selected_count": 1,
            "trigger": "daily",
            "trace": {
                "thread_id": "persona-review:review-1",
                "root_trace_id": "review-1",
            },
        }
    ]
    adapter = Mock()
    adapter.view.return_value = profile
    adapter.schedule.return_value = profile
    client = make_client(adapter, actions=True)
    html = client.get("/persona-evolution?user_id=account-1", headers=HEADERS).text
    assert "Turn off daily review" in html and "always wait for approval" in html
    assert "root_trace_id=review-1" in html and "persona-review%3Areview-1" in html
    assert (
        client.post(
            "/persona-evolution/action",
            headers=HEADERS,
            json={
                "user_id": "account-1",
                "action": "schedule",
                "revision": 2,
                "daily_enabled": False,
            },
        ).status_code
        == 200
    )
    adapter.schedule.assert_called_once_with(
        "account-1", revision=2, daily_enabled=False
    )


def test_named_operator_is_attributed_without_accepting_client_actor(
    make_client, monkeypatch
):
    monkeypatch.setenv(
        "MEMORIZZ_UI_AUTH_ACCOUNTS",
        '{"casey":{"token":"named-persona-operator-token","role":"admin"}}',
    )
    adapter = Mock()
    adapter.reflect.return_value = PROFILE
    client = make_client(adapter, actions=True)
    headers = {"Authorization": "Bearer named-persona-operator-token"}
    request = {"user_id": "account-1", "action": "reflect"}
    assert (
        client.post(
            "/persona-evolution/action",
            headers=headers,
            json={**request, "actor_id": "spoofed"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/persona-evolution/action", headers=headers, json=request
        ).status_code
        == 200
    )
    adapter.reflect.assert_called_once_with("account-1", actor_id="operator:casey")
