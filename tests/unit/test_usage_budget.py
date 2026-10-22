"""The monthly budget: saved in the app, measured against known spend."""

import json
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.observability import ObservabilityStore, TraceContext  # noqa: E402
from memorizz.ui import state, usage_budget  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.state import templates  # noqa: E402
from memorizz.ui.trace_access import TracePrincipal, current_principal  # noqa: E402

# Ten days into a 31-day month.
NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)
CALL = {
    "provider": "ollama",
    "model": "gemma4:latest",
    "service_tier": "default",
    "input_tokens": 1_000_000,
    "output_tokens": 500_000,
    "cached_tokens": 0,
}
RATE_CARD = {
    **{key: CALL[key] for key in ("provider", "model", "service_tier")},
    "input_per_million": "0.50",
    "output_per_million": "1.5",
    "source_url": "https://example.com/pricing",
}


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "{}")
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "false")
    usage_budget._cache.clear()
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    with patch.dict(
        state._state, provider=provider, provider_type="filesystem", connection_info={}
    ):
        yield {
            "client": lambda **kwargs: TestClient(
                create_app(**kwargs), follow_redirects=False
            ),
            "provider": provider,
            "file": tmp_path / "home" / "usage_budget.json",
            "audit": tmp_path / "audit.jsonl",
        }
    provider.close()
    usage_budget._cache.clear()


def request_for(pricing=None):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(usage_pricing=pricing or object()))
    )


def totals(cost_usd="100", unpriced_calls=3, calls=40):
    return {
        "totals": {
            "cost_usd": cost_usd,
            "unpriced_calls": unpriced_calls,
            "calls": calls,
        }
    }


def set_budget(monthly_usd="500", alert_percent=80):
    usage_budget.save_budget(usage_budget.budget_from_form(monthly_usd, alert_percent))


def test_a_budget_is_saved_loaded_and_cleared(ui):
    assert usage_budget.load_budget() is None

    set_budget("1,250.50", "75")

    assert json.loads(ui["file"].read_text()) == {
        "version": 1,
        "monthly_usd": "1250.50",
        "alert_percent": 75,
    }
    assert usage_budget.load_budget() == {
        "monthly_usd": Decimal("1250.50"),
        "alert_percent": 75,
    }
    usage_budget.clear_budget()
    usage_budget.clear_budget()
    assert not ui["file"].exists() and usage_budget.load_budget() is None


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"monthly_usd": "500"}',
        '{"monthly_usd": "-5", "alert_percent": 80}',
        '{"monthly_usd": "500", "alert_percent": 0}',
        '{"monthly_usd": "500", "alert_percent": "8.5"}',
    ],
)
def test_a_corrupt_budget_file_is_ignored_with_a_warning(ui, caplog, content):
    ui["file"].parent.mkdir(parents=True)
    ui["file"].write_text(content)

    with caplog.at_level(logging.WARNING, logger=usage_budget.__name__):
        assert usage_budget.load_budget() is None

    assert "usage_budget.json holds no valid budget" in caplog.text


def test_status_measures_month_to_date_spend_and_projects_month_end(ui):
    set_budget()
    pricing = object()

    with patch.object(usage_budget, "query_usage", return_value=totals()) as query:
        status = usage_budget.budget_status(request_for(pricing), now=NOW)

    (provider,), kwargs = query.call_args
    assert provider is ui["provider"] and kwargs["pricing"] is pricing
    assert kwargs["max_events"] == 10000
    assert kwargs["filters"] == {
        "start_time": "2026-10-01T00:00:00+00:00",
        "end_time": "2026-10-11T00:00:00+00:00",
    }
    assert status == {
        "budget": Decimal("500"),
        "alert_percent": 80,
        "month_start": datetime(2026, 10, 1, tzinfo=timezone.utc),
        "now": NOW,
        "spend": Decimal("100"),
        "unpriced_calls": 3,
        "calls": 40,
        "percent": 20.0,
        # $100 in 10 of October's 31 days.
        "projected": Decimal("310.00"),
        "state": "ok",
        "can_edit": True,
    }


@pytest.mark.parametrize(
    "spend,state_name",
    [("399.99", "ok"), ("400", "warning"), ("499.99", "warning"), ("500", "over")],
)
def test_status_turns_to_warning_at_the_alert_and_over_at_the_budget(
    ui, spend, state_name
):
    set_budget()

    with patch.object(usage_budget, "query_usage", return_value=totals(spend)):
        status = usage_budget.budget_status(request_for(), now=NOW)

    assert status["state"] == state_name


def test_the_first_day_has_no_projection_and_unpriced_spend_counts_as_zero(ui):
    set_budget()
    early = datetime(2026, 10, 1, 20, tzinfo=timezone.utc)

    with patch.object(usage_budget, "query_usage", return_value=totals(None, 5, 5)):
        status = usage_budget.budget_status(request_for(), now=early)

    assert status["spend"] == 0 and status["percent"] == 0
    assert status["projected"] is None and status["unpriced_calls"] == 5


def test_no_budget_or_no_provider_gives_no_status(ui):
    with patch.object(usage_budget, "query_usage") as query:
        assert usage_budget.budget_status(request_for(), now=NOW) is None
        set_budget()
        with patch.dict(state._state, provider=None):
            assert usage_budget.budget_status(request_for(), now=NOW) is None

    query.assert_not_called()


def test_a_failed_query_reports_an_unknown_state_without_raising(ui, caplog):
    set_budget()

    with patch.object(usage_budget, "query_usage", side_effect=RuntimeError("down")):
        with caplog.at_level(logging.WARNING, logger=usage_budget.__name__):
            status = usage_budget.budget_status(request_for(), now=NOW)

    assert status["state"] == "unknown" and status["spend"] is None
    assert status["budget"] == Decimal("500") and status["can_edit"] is True
    assert "Usage budget spend is unavailable" in caplog.text
    assert usage_budget.budget_attention(request_for(), now=NOW) is None


def test_figures_are_cached_for_five_minutes_per_pricing_and_scope(ui):
    set_budget()
    request = request_for()

    with patch.object(usage_budget, "query_usage", return_value=totals()) as query:
        usage_budget.budget_status(request, now=NOW)
        usage_budget.budget_status(request, now=NOW + timedelta(seconds=299))
        assert query.call_count == 1
        usage_budget.budget_status(request, now=NOW + timedelta(seconds=300))
        assert query.call_count == 2
        # Saving the budget, a rate-card change (a new registry) and another
        # tenant's scope each read again.
        set_budget("600")
        status = usage_budget.budget_status(request, now=NOW + timedelta(seconds=301))
        assert query.call_count == 3 and status["percent"] == pytest.approx(100 / 6)
        request.app.state.usage_pricing = object()
        usage_budget.budget_status(request, now=NOW + timedelta(seconds=302))
        assert query.call_count == 4
        token = current_principal.set(TracePrincipal(application_id="tenant-a"))
        try:
            usage_budget.budget_status(request, now=NOW + timedelta(seconds=303))
        finally:
            current_principal.reset(token)
        assert query.call_count == 5
        assert query.call_args.kwargs["filters"]["application_id"] == "tenant-a"
        usage_budget.budget_status(request, now=NOW + timedelta(seconds=304))
        assert query.call_count == 5


def test_a_rate_card_change_reprices_the_budget_at_once(ui):
    context = TraceContext(
        agent_id="agent",
        thread_id="thread",
        root_trace_id="root",
        run_id="run",
        turn_id="turn",
    )
    ObservabilityStore(ui["provider"]).record_trace_bundle(
        trace_context=context.to_carrier(),
        events=[
            {
                "event_id": f"call-{n}",
                "span_id": f"call-{n}",
                "trace_kind": "model_result",
                **CALL,
                "timestamp": "2026-10-10T10:00:00Z",
            }
            for n in range(2)
        ],
    )
    set_budget("3")
    client = ui["client"]()

    before = usage_budget.budget_status(SimpleNamespace(app=client.app), now=NOW)
    assert client.post("/traces/usage/rates", data=RATE_CARD).status_code == 303
    after = usage_budget.budget_status(SimpleNamespace(app=client.app), now=NOW)

    assert (before["spend"], before["unpriced_calls"], before["state"]) == (0, 2, "ok")
    assert (after["spend"], after["unpriced_calls"], after["state"]) == (
        Decimal("2.5"),
        0,
        "warning",
    )


def test_budget_attention_flags_warning_and_over_only(ui):
    assert usage_budget.budget_attention(request_for(), now=NOW) is None
    set_budget()

    def attention(*args):
        usage_budget._cache.clear()
        with patch.object(usage_budget, "query_usage", return_value=totals(*args)):
            return usage_budget.budget_attention(request_for(), now=NOW)

    assert attention("100") is None
    assert attention("420.1", 27) == {
        "severity": "warning",
        "kind": "budget",
        "title": "Spend is at 84% of this month's $500.00 budget",
        "detail": "$420.10 known spend since Oct 1; on pace for $1,302.31 by month"
        " end. 27 unpriced calls aren't counted.",
        "href": "/traces/usage#budget",
        "action": "Open the budget",
    }
    over = attention("1234.5", 1)
    assert over["severity"] == "error"
    assert over["title"] == "Spend is over this month's $500.00 budget"
    assert over["detail"].endswith("1 unpriced call isn't counted.")
    assert attention("450", 0)["detail"] == (
        "$450.00 known spend since Oct 1; on pace for $1,395.00 by month end."
    )


def test_saving_and_clearing_redirect_to_the_budget_panel(ui):
    client = ui["client"]()

    saved = client.post("/traces/usage/budget", data={"monthly_usd": "$2,000"})
    assert saved.status_code == 303
    assert saved.headers["location"] == "/traces/usage?budget_notice=saved#budget"
    assert usage_budget.load_budget() == {
        "monthly_usd": Decimal("2000"),
        "alert_percent": 80,
    }

    cleared = client.post("/traces/usage/budget/clear")
    assert cleared.status_code == 303
    assert cleared.headers["location"] == "/traces/usage?budget_notice=cleared#budget"
    assert not ui["file"].exists()
    audit = ui["audit"].read_text()
    assert "usage_budget_saved" in audit and "usage_budget_cleared" in audit


@pytest.mark.parametrize(
    "form,message",
    [
        ({"monthly_usd": ""}, usage_budget.AMOUNT_ERROR),
        ({"monthly_usd": "five hundred"}, usage_budget.AMOUNT_ERROR),
        ({"monthly_usd": "0"}, usage_budget.RANGE_ERROR),
        ({"monthly_usd": "-10"}, usage_budget.RANGE_ERROR),
        ({"monthly_usd": "NaN"}, usage_budget.RANGE_ERROR),
        ({"monthly_usd": "100000000"}, usage_budget.RANGE_ERROR),
        ({"monthly_usd": "500", "alert_percent": "0"}, usage_budget.ALERT_ERROR),
        ({"monthly_usd": "500", "alert_percent": "101"}, usage_budget.ALERT_ERROR),
        ({"monthly_usd": "500", "alert_percent": "8.5"}, usage_budget.ALERT_ERROR),
    ],
)
def test_invalid_budgets_redirect_with_a_short_reason(ui, form, message):
    response = ui["client"]().post("/traces/usage/budget", data=form)

    assert response.status_code == 303
    location = urlsplit(response.headers["location"])
    assert (location.path, location.fragment) == ("/traces/usage", "budget")
    assert parse_qs(location.query) == {"budget_error": [message]}
    assert usage_budget.budget_messages({"budget_error": message})["error"] == message
    assert not ui["file"].exists()


def test_only_known_messages_are_shown_from_the_query():
    assert usage_budget.budget_messages({"budget_notice": "saved"}) == {
        "notice": "Budget saved.",
        "error": "",
    }
    assert usage_budget.budget_messages(
        {"budget_notice": "Call 555", "budget_error": "Your account is locked"}
    ) == {"notice": "", "error": ""}


def test_cross_site_read_only_and_viewer_posts_are_refused(ui, monkeypatch):
    form = {"monthly_usd": "500"}
    cross_site = ui["client"]().post(
        "/traces/usage/budget", data=form, headers={"Origin": "https://evil.example"}
    )
    assert cross_site.status_code == 403

    token = "viewer-test-token-123456"
    monkeypatch.setenv(
        "MEMORIZZ_UI_AUTH_ACCOUNTS",
        json.dumps({"viewer": {"token": token, "role": "viewer"}}),
    )
    viewer = {"Authorization": f"Bearer {token}"}
    client = ui["client"]()
    saved = client.post("/traces/usage/budget", data=form, headers=viewer)
    assert saved.status_code == 403
    assert client.post("/traces/usage/budget/clear", headers=viewer).status_code == 403

    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "{}")
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    read_only = ui["client"]().post("/traces/usage/budget", data=form)
    assert read_only.status_code == 403
    assert not ui["file"].exists()


def test_only_an_unscoped_admin_outside_read_only_mode_can_edit(ui, monkeypatch):
    set_budget()
    for principal in (
        TracePrincipal(role="viewer"),
        TracePrincipal(application_id="tenant-a"),
    ):
        token = current_principal.set(principal)
        try:
            with patch.object(usage_budget, "query_usage", return_value=totals()):
                status = usage_budget.budget_status(request_for(), now=NOW)
        finally:
            current_principal.reset(token)
        assert status["can_edit"] is False
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    assert usage_budget.can_edit() is False
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "false")
    assert usage_budget.can_edit() is True


def render(status=None, **kwargs):
    return templates.env.from_string(
        "{% from '_usage_budget.html' import budget_panel %}"
        "{{ budget_panel(status, **kwargs) }}"
    ).render(status=status, kwargs=kwargs)


def status_of(spend, state_name, *, unpriced=0, can_edit=True, projected="310.00"):
    budget = Decimal(500)
    return {
        "budget": budget,
        "alert_percent": 80,
        "month_start": datetime(2026, 10, 1, tzinfo=timezone.utc),
        "now": NOW,
        "spend": Decimal(spend),
        "unpriced_calls": unpriced,
        "calls": 1234,
        "percent": float(Decimal(spend) / budget * 100),
        "projected": Decimal(projected) if projected else None,
        "state": state_name,
        "can_edit": can_edit,
    }


def test_the_panel_prompts_for_a_budget_or_says_none_is_set():
    editable = render(can_edit=True, notice="cleared")
    read_only = render()

    assert '<section class="panel usage-budget" id="budget"' in editable
    assert "Set a monthly budget to get an alert before spend runs over" in editable
    assert 'action="/traces/usage/budget"' in editable
    assert 'name="alert_percent" type="number" min="1" max="100"' in editable
    assert 'value="80"' in editable and "Budget removed." in editable
    assert "No monthly budget is set." in read_only
    assert "<form" not in read_only


@pytest.mark.parametrize(
    "spend,state_name,width,shown",
    [
        ("100", "ok", "20.0", "20"),
        ("420.10", "warning", "84.02", "84"),
        ("750", "over", "100", "150"),
    ],
)
def test_the_panel_shows_spend_against_the_budget(spend, state_name, width, shown):
    page = render(status_of(spend, state_name, unpriced=27), error="Budget error")

    assert f'class="panel usage-budget usage-budget--{state_name}" id="budget"' in page
    assert '<h2 class="panel-title" id="budget-title">This month</h2>' in page
    assert f"${Decimal(spend):,.2f}</strong> <span>of $500.00</span>" in page
    assert 'role="progressbar"' in page and 'aria-valuemin="0"' in page
    assert f'aria-valuenow="{min(int(shown), 100)}" aria-valuetext="{shown}%' in page
    assert f"width: {width}" in page
    assert "On pace for $310.00 by month end" in page and "1,234 calls" in page
    assert "27 calls have no price and aren't counted." in page
    assert '<a href="/traces/usage/rates">Set prices</a>' in page
    assert "<summary>Edit budget</summary>" in page and 'value="500"' in page
    assert 'action="/traces/usage/budget/clear"' in page and "Remove budget" in page
    assert 'role="alert">Budget error</p>' in page
    assert '<details class="usage-budget-edit" open>' in page
    assert ("over budget" in page) is (state_name == "over")


def test_read_only_viewers_see_figures_without_forms():
    page = render(status_of("10", "ok", unpriced=1, can_edit=False, projected=None))

    assert "$10.00</strong>" in page and "after the first day" in page
    assert "1 call has no price and isn't counted." in page
    assert "<form" not in page and "Edit budget" not in page
    assert "no price" not in render(status_of("10", "ok"))


def test_an_unreadable_spend_says_so():
    status = {
        **status_of("0", "unknown"),
        "spend": None,
        "percent": None,
        "projected": None,
    }

    page = render(status)

    assert "Spend can't be read right now" in page
    assert 'role="progressbar"' not in page and "Edit budget" in page
