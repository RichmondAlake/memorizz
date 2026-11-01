"""Failure alerts: when they fire, what they send, and what they never show."""

import json
import logging
import stat
from datetime import datetime, timedelta, timezone

import pytest

from memorizz.ui import failure_alerts

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 1, 22, 0, tzinfo=timezone.utc)
SECRET_URL = "https://hooks.example.com/services/T000/B000/s3cr3t-token"


def _failures(total, *, stopped=0):
    return {
        "total": total,
        "turns": total,
        "stopped": stopped,
        "agents": [
            {
                "agent_id": "agent-1",
                "name": "Billing",
                "total": total,
                "stopped": stopped,
            }
        ],
        "groups": [
            {
                "name": "mcp_call_tool",
                "code": "protocol_error",
                "why": "The tool call failed.",
                "count": total,
                "turns": total,
                "stopped": bool(stopped),
                "agents": ["agent-1"],
            }
        ],
    }


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setattr(failure_alerts, "_check_url", lambda url: None)
    return tmp_path


@pytest.fixture
def sent(monkeypatch):
    messages = []
    monkeypatch.setattr(
        failure_alerts, "_post", lambda url, payload: messages.append((url, payload))
    )
    return messages


def _rule(**values):
    form = {"threshold": "5", "window": "1h", "webhook_url": SECRET_URL, **values}
    rule = failure_alerts.rule_from_form(**form)
    failure_alerts.save_rule(rule, base_url="https://memorizz.example.com/")
    return rule


def _check(total, *, stopped=0, at=NOW):
    return failure_alerts.check_rules(
        now=at, read=lambda rule: _failures(total, stopped=stopped)
    )


def test_a_rule_fires_when_failures_reach_its_limit(home, sent):
    _rule()
    _check(4)
    assert sent == []
    _check(5, at=NOW + timedelta(minutes=2))
    assert len(sent) == 1
    assert failure_alerts.statuses()[0]["firing"] is True


def test_a_firing_rule_sends_once_not_every_check(home, sent):
    _rule()
    for minutes in (0, 2, 4, 6):
        _check(9, at=NOW + timedelta(minutes=minutes))
    assert len(sent) == 1


def test_it_says_when_failures_drop_back_under_the_limit(home, sent):
    _rule()
    _check(9)
    _check(0, at=NOW + timedelta(minutes=2))
    assert [payload["memorizz"]["event"] for _, payload in sent] == [
        "failure_alert.firing",
        "failure_alert.resolved",
    ]
    assert failure_alerts.statuses()[0]["firing"] is False


def test_a_rule_is_not_rechecked_inside_its_interval(home, sent):
    _rule()
    reads = []

    def read(rule):
        reads.append(rule)
        return _failures(0)

    failure_alerts.check_rules(now=NOW, read=read)
    failure_alerts.check_rules(now=NOW + timedelta(seconds=5), read=read)
    assert len(reads) == 1


def test_a_rule_can_count_only_turns_that_stopped(home, sent):
    _rule(threshold="2", stopped_only="on")
    _check(50, stopped=1)
    assert sent == []
    _check(50, stopped=2, at=NOW + timedelta(minutes=2))
    assert len(sent) == 1


def test_a_rule_for_one_agent_counts_only_that_agent(home, sent):
    _rule(threshold="1", agent_id="someone-else")
    assert failure_alerts.count_for(failure_alerts.load_rules()[0], _failures(9)) == 0
    _rule(threshold="1", agent_id="agent-1")
    assert failure_alerts.count_for(failure_alerts.load_rules()[1], _failures(9)) == 9


def test_a_message_says_what_failed_and_where_to_look(home, sent):
    _rule()
    _check(7)
    _url, payload = sent[0]
    assert "7 failures" in payload["text"] and "mcp_call_tool" in payload["text"]
    assert payload["memorizz"]["url"].startswith(
        "https://memorizz.example.com/traces/failures"
    )
    assert payload["memorizz"]["top"][0]["code"] == "protocol_error"


def test_a_message_carries_no_trace_content(home, sent):
    _rule()
    data = _failures(7)
    data["groups"][0]["row"] = {"event": {"content": "CUSTOMER SECRET"}}
    failure_alerts.check_rules(now=NOW, read=lambda rule: data)
    assert "CUSTOMER SECRET" not in json.dumps(sent[0][1])


def test_a_failed_delivery_is_retried_and_reported_without_its_address(
    home, monkeypatch, caplog
):
    _rule()

    def refuse(url, payload):
        raise RuntimeError(f"HTTPSConnectionPool(host=x, url={url}) refused")

    monkeypatch.setattr(failure_alerts, "_post", refuse)
    with caplog.at_level(logging.DEBUG):
        _check(9)
    status = failure_alerts.statuses()[0]
    assert status["delivery_error"]
    assert "s3cr3t" not in status["delivery_error"]
    assert "s3cr3t" not in caplog.text

    delivered = []
    monkeypatch.setattr(
        failure_alerts, "_post", lambda url, payload: delivered.append(payload)
    )
    _check(9, at=NOW + timedelta(minutes=2))
    assert len(delivered) == 1
    assert failure_alerts.statuses()[0]["delivery_error"] == ""


def test_the_address_is_shown_only_as_its_host(home, sent):
    _rule()
    status = failure_alerts.statuses()[0]
    assert status["webhook_host"] == "hooks.example.com"
    assert "s3cr3t" not in json.dumps(status, default=str)


def test_rules_are_readable_only_by_their_owner(home, sent):
    _rule()
    mode = stat.S_IMODE(failure_alerts.rules_path().stat().st_mode)
    assert mode == 0o600


def test_a_rule_works_without_a_webhook(home, sent):
    _rule(webhook_url="")
    _check(9)
    assert sent == []
    assert failure_alerts.statuses()[0]["firing"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold", "0"),
        ("threshold", "many"),
        ("threshold", "1000001"),
        ("window", "forever"),
    ],
)
def test_a_bad_rule_is_refused(home, field, value):
    form = {"threshold": "5", "window": "1h", "webhook_url": "", field: value}
    with pytest.raises(ValueError):
        failure_alerts.rule_from_form(**form)


@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.example.com/x",
        "https://127.0.0.1/x",
        "https://169.254.169.254/latest/meta-data",
        "https://user:pass@hooks.example.com/x",
    ],
    ids=["plain http", "loopback", "cloud metadata", "embedded credentials"],
)
def test_a_webhook_cannot_point_inside_the_network(tmp_path, monkeypatch, url):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.delenv("MEMORIZZ_ALERT_ALLOW_PRIVATE_WEBHOOKS", raising=False)
    with pytest.raises(ValueError):
        failure_alerts.rule_from_form(threshold="5", window="1h", webhook_url=url)


def test_a_redirect_is_not_followed(home, monkeypatch):
    calls = {}

    class Response:
        status_code = 302

    def post(url, **options):
        calls.update(options)
        return Response()

    monkeypatch.setattr(failure_alerts.requests, "post", post)
    with pytest.raises(failure_alerts.DeliveryError):
        failure_alerts._post(SECRET_URL, {"text": "x"})
    assert calls["allow_redirects"] is False


def test_the_overview_lists_a_firing_rule(home, sent):
    _rule()
    _check(9)
    items = failure_alerts.failure_attention()
    assert len(items) == 1 and "9 failures" in items[0]["title"]
    assert items[0]["href"].startswith("/traces/failures")


def test_a_scoped_viewer_never_sees_a_firing_rule(home, sent):
    from memorizz.ui.trace_access import TracePrincipal, current_principal

    _rule()
    _check(9)
    token = current_principal.set(TracePrincipal(role="viewer", application_id="app-A"))
    try:
        assert failure_alerts.failure_attention() == []
    finally:
        current_principal.reset(token)


def test_a_message_names_the_agent_not_its_id(home, sent):
    rule = failure_alerts.rule_from_form(
        threshold="1", window="1h", webhook_url=SECRET_URL, agent_id="agent-1"
    )
    rule["agent_name"] = "Billing"
    failure_alerts.save_rule(rule, base_url="https://memorizz.example.com/")
    _check(3)
    assert "on Billing" in sent[0][1]["text"]
    assert failure_alerts.statuses()[0]["agent"] == "Billing"
