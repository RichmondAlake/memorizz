"""The Rate cards page: prices saved in the app extend the built-in catalog."""

import json
from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.observability import (  # noqa: E402
    DEFAULT_PRICING,
    ObservabilityStore,
    PricingRegistry,
    RateCard,
    TraceContext,
)
from memorizz.observability.pricing import NO_RATE_CARD  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.routers.rate_cards import LOCAL_SOURCE_URL  # noqa: E402

OLLAMA = {"provider": "ollama", "model": "gemma4:latest", "service_tier": "default"}
CALL = {
    **OLLAMA,
    "input_tokens": 1_000_000,
    "output_tokens": 500_000,
    "cached_tokens": 0,
}
PAID = {
    **OLLAMA,
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
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    context = TraceContext(
        agent_id="agent",
        thread_id="thread",
        root_trace_id="root",
        run_id="run",
        turn_id="turn",
    )
    ObservabilityStore(provider).record_trace_bundle(
        trace_context=context.to_carrier(),
        events=[
            {
                "event_id": f"call-{n}",
                "span_id": f"call-{n}",
                "trace_kind": "model_result",
                **CALL,
                "timestamp": "2026-10-21T10:00:00Z",
            }
            for n in range(2)
        ],
    )
    with patch.dict(
        state._state, provider=provider, provider_type="filesystem", connection_info={}
    ):
        yield {
            "client": lambda **kwargs: TestClient(
                create_app(**kwargs), follow_redirects=False
            ),
            "file": tmp_path / "home" / "rate_cards.json",
            "audit": tmp_path / "audit.jsonl",
        }
    provider.close()


def test_query_params_prefill_the_form_and_unpriced_models_are_listed(ui):
    response = ui["client"]().get("/traces/usage/rates", params=OLLAMA)

    assert response.status_code == 200, response.text
    assert "<h1>Rate cards</h1>" in response.text
    assert 'name="provider" value="ollama"' in response.text
    assert 'name="model" value="gemma4:latest"' in response.text
    assert "Models without a price" in response.text
    assert "model=gemma4%3Alatest&amp;service_tier=default#rate-card-form" in (
        response.text
    )
    assert "No custom rate cards yet" in response.text


def test_a_saved_card_is_written_and_prices_calls_without_a_restart(ui):
    client = ui["client"]()
    assert client.app.state.usage_pricing.quote(CALL)["cost_reason"] == NO_RATE_CARD

    response = client.post("/traces/usage/rates", data=PAID)

    assert response.status_code == 303
    assert response.headers["location"] == "/traces/usage/rates?notice=saved"
    today = date.today().isoformat()
    assert json.loads(ui["file"].read_text()) == {
        "version": 1,
        "cards": [
            {
                **OLLAMA,
                "input_per_million": "0.50",
                "cached_input_per_million": "0.50",
                "output_per_million": "1.5",
                "source_url": "https://example.com/pricing",
                "as_of": today,
                "version": f"local-{today}",
            }
        ],
    }
    quote = client.app.state.usage_pricing.quote(CALL)
    assert quote["cost_status"] == "calculated"
    assert Decimal(quote["cost_usd"]) == Decimal("1.25")
    assert Decimal(client.get("/traces/usage.json").json()["totals"]["cost_usd"]) == (
        Decimal("2.5")
    )
    page = client.get("/traces/usage/rates?notice=saved").text
    assert "Rate card saved" in page
    assert "Models without a price" not in page
    assert "$0.50" in page and "$1.5" in page
    # The built-in catalog still applies, and a new app loads the saved card.
    gpt = {**CALL, "provider": "openai", "model": "gpt-4o-mini"}
    assert client.app.state.usage_pricing.quote(gpt)["cost_status"] == "calculated"
    restarted = ui["client"]().app.state.usage_pricing.quote(CALL)
    assert restarted["pricing_version"] == f"local-{today}"
    assert "usage_rate_card_saved" in ui["audit"].read_text()


def test_a_saved_card_replaces_the_built_in_price_for_its_key(ui):
    client = ui["client"]()
    card = {**PAID, "provider": "OpenAI", "model": "gpt-4o-mini"}

    assert client.post("/traces/usage/rates", data=card).status_code == 303

    quote = client.app.state.usage_pricing.quote(
        {**CALL, "provider": "openai", "model": "gpt-4o-mini"}
    )
    assert quote["pricing_version"].startswith("local-")
    assert Decimal(quote["cost_usd"]) == Decimal("1.25")


def test_a_local_model_card_charges_nothing(ui):
    client = ui["client"]()

    response = client.post("/traces/usage/rates", data={**OLLAMA, "local": "1"})

    assert response.status_code == 303
    (saved,) = json.loads(ui["file"].read_text())["cards"]
    assert saved["source_url"] == LOCAL_SOURCE_URL
    assert {saved[key] for key in saved if key.endswith("_per_million")} == {"0"}
    quote = client.app.state.usage_pricing.quote(CALL)
    assert quote["cost_status"] == "calculated"
    assert Decimal(quote["cost_usd"]) == 0


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"input_per_million": "-1"}, "Input must be zero or more"),
        ({"output_per_million": "ten"}, "Output must be a price in USD per 1M tokens"),
        ({"source_url": ""}, "Add the public HTTPS page that lists this price"),
        ({"source_url": "http://example.com/pricing"}, "Pricing sources must be"),
        ({"model": ""}, "Rate cards require identity"),
    ],
)
def test_invalid_cards_show_the_reason_inline(ui, changes, message):
    response = ui["client"]().post("/traces/usage/rates", data={**PAID, **changes})

    assert response.status_code == 400
    assert 'role="alert"' in response.text and message in response.text
    assert 'name="provider" value="ollama"' in response.text
    assert not ui["file"].exists()


def test_removing_a_card_restores_the_unpriced_state(ui):
    client = ui["client"]()
    client.post("/traces/usage/rates", data={**OLLAMA, "local": "1"})

    response = client.post("/traces/usage/rates/remove", data=OLLAMA)

    assert response.status_code == 303
    assert json.loads(ui["file"].read_text())["cards"] == []
    assert client.app.state.usage_pricing.quote(CALL)["cost_reason"] == NO_RATE_CARD
    missing = client.post("/traces/usage/rates/remove", data=OLLAMA)
    assert missing.status_code == 400
    assert "No saved rate card matches" in missing.text
    audit = ui["audit"].read_text()
    assert "usage_rate_card_removed" in audit and "gemma4" not in audit


def test_prices_set_in_code_are_shown_read_only(ui):
    custom = RateCard(
        provider="private",
        model="reader-v1",
        input_per_million="1",
        cached_input_per_million="0.1",
        output_per_million="3",
        source_url="https://example.com/public-pricing",
        as_of="2026-09-07",
        version="reader-contract-v1",
    )
    client = ui["client"](
        usage_pricing=PricingRegistry([*DEFAULT_PRICING.cards.values(), custom])
    )

    page = client.get("/traces/usage/rates", params=OLLAMA).text
    response = client.post("/traces/usage/rates", data=PAID)

    assert "prices are set in code" in page and "reader-v1" in page
    assert 'action="/traces/usage/rates"' not in page
    assert "gpt-4o-mini" not in page
    assert response.status_code == 409
    assert not ui["file"].exists()


def test_cross_site_posts_and_accounts_without_write_access_are_refused(
    ui, monkeypatch
):
    cross_site = ui["client"]().post(
        "/traces/usage/rates", data=PAID, headers={"Origin": "https://evil.example"}
    )
    assert cross_site.status_code == 403

    token = "viewer-test-token-123456"
    monkeypatch.setenv(
        "MEMORIZZ_UI_AUTH_ACCOUNTS",
        json.dumps({"viewer": {"token": token, "role": "viewer"}}),
    )
    client = ui["client"]()
    viewer = {"Authorization": f"Bearer {token}"}
    page = client.get("/traces/usage/rates", headers=viewer)
    refused = client.post("/traces/usage/rates", data=PAID, headers=viewer)

    assert page.status_code == 200
    assert "can view rate cards but not change them" in page.text
    assert refused.status_code == 403
    assert not ui["file"].exists()
