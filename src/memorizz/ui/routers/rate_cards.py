"""Rate cards saved in the app: prices for calls the built-in catalog cannot price.

Cards live in ``memorizz_home() / "rate_cards.json"`` and extend the built-in
list prices unless the host set ``create_app(usage_pricing=...)`` in code.
"""

import json
import logging
import os
import threading
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..._env_io import memorizz_home
from ...observability.analytics import _quote
from ...observability.pricing import (
    DEFAULT_PRICING,
    NO_RATE_CARD,
    PricingRegistry,
    RateCard,
)
from ..security import audit_trace_view, ui_read_only
from ..state import _state, templates
from ..trace_access import current_principal, require_trace_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["usage"])

CARD_FIELDS = (
    "provider",
    "model",
    "service_tier",
    "input_per_million",
    "cached_input_per_million",
    "output_per_million",
    "source_url",
    "as_of",
    "version",
)
# A local model has no price list to cite, so its card cites the docs that
# describe local cards; RateCard requires a public HTTPS source.
LOCAL_SOURCE_URL = "https://richmondalake.github.io/memorizz/usage-analytics/"
# Far above any list price; it keeps a typo from overflowing spend totals.
MAX_RATE = Decimal(100_000)
NOTICES = {
    "saved": "Rate card saved. Spend on the Overview and Usage pages now uses it.",
    "removed": "Rate card removed.",
}

_lock = threading.Lock()


def rate_cards_path():
    return memorizz_home() / "rate_cards.json"


def card_key(card):
    return (card.provider.lower(), card.model, card.service_tier)


def load_rate_cards():
    """The saved cards. Raises ValueError when the file cannot be used."""
    path = rate_cards_path()
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))["cards"]
        return [RateCard(**{key: row[key] for key in CARD_FIELDS}) for row in rows]
    except FileNotFoundError:
        return []
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path.name} holds no valid rate cards") from exc


def save_rate_cards(cards):
    path = rate_cards_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{key: str(getattr(card, key)) for key in CARD_FIELDS} for card in cards]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "cards": rows}, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def effective_pricing(cards=None):
    """Built-in list prices plus saved cards; a saved card replaces a built-in
    card with the same provider, model and tier."""
    if cards is None:
        try:
            cards = load_rate_cards()
        except ValueError as exc:
            logger.warning("Saved rate cards ignored: %s", exc)
            cards = []
    return PricingRegistry([*DEFAULT_PRICING.cards.values(), *cards])


def _rate(value, label):
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"{label} must be a price in USD per 1M tokens") from None
    if not number.is_finite() or number < 0:
        raise ValueError(f"{label} must be zero or more")
    if number > MAX_RATE:
        raise ValueError(f"{label} must be at most ${MAX_RATE:,} per 1M tokens")
    return number


def rate_card_from_form(form):
    """A validated card from the add form; ValueError messages suit the form."""
    if form["local"]:
        inputs = cached = outputs = Decimal(0)
    else:
        inputs = _rate(form["input_per_million"], "Input")
        cached = (
            _rate(form["cached_input_per_million"], "Cached input")
            if form["cached_input_per_million"]
            else inputs
        )
        outputs = _rate(form["output_per_million"], "Output")
        if not form["source_url"]:
            raise ValueError(
                "Add the public HTTPS page that lists this price, or tick "
                "Local model, no charge"
            )
    as_of = date.today().isoformat()
    return RateCard(
        provider=form["provider"].lower(),
        model=form["model"],
        service_tier=form["service_tier"] or "default",
        input_per_million=inputs,
        cached_input_per_million=cached,
        output_per_million=outputs,
        source_url=form["source_url"] or LOCAL_SOURCE_URL,
        as_of=as_of,
        version=f"local-{as_of}",
    )


def unpriced_models(pricing, limit=20):
    """Models in recent traces that no rate card prices, most calls first."""
    from ..dashboard import DEFERRED_PROVIDERS
    from .traces import _selection_window

    # Remote-API providers read traces only on request, as on the Overview.
    if _state.get("provider") is None or (
        _state.get("provider_type") in DEFERRED_PROVIDERS
    ):
        return []
    try:
        _, events = _selection_window(request=None, limit=500, event_limit=5000)
    except Exception:
        logger.warning("Rate cards trace read failed", exc_info=True)
        return []
    counts = Counter(
        (
            str(event.get("provider") or "").lower(),
            str(event.get("model") or ""),
            str(event.get("service_tier") or "default"),
        )
        for event in events
        if (event.get("trace_kind") or event.get("kind")) == "model_result"
        and _quote(event, pricing).get("cost_reason") == NO_RATE_CARD
    )
    return [
        {"provider": provider, "model": model, "service_tier": tier, "calls": calls}
        for (provider, model, tier), calls in counts.most_common(limit)
        if provider and model
    ]


def _page(request, *, form=None, error=None, status_code=200):
    principal = current_principal.get()
    pricing = request.app.state.usage_pricing
    in_code = request.app.state.usage_pricing_in_code
    if in_code:
        cards = [
            card
            for key, card in pricing.cards.items()
            if DEFAULT_PRICING.cards.get(key) != card
        ]
    else:
        try:
            cards = load_rate_cards()
        except ValueError as exc:
            cards, error = [], error or str(exc)
    locked = (
        "This app's prices are set in code with create_app(usage_pricing=...). "
        "Change them there."
        if in_code
        else "This UI is read-only, so rate cards can't be changed here."
        if ui_read_only()
        else "Your account can view rate cards but not change them."
        if principal.restricted or not principal.allows("trace.write")
        else None
    )
    unpriced = unpriced_models(pricing)
    for row in unpriced:
        row["url"] = (
            "/traces/usage/rates?"
            + urlencode(
                {key: row[key] for key in ("provider", "model", "service_tier")}
            )
            + "#rate-card-form"
        )
    audit_trace_view(
        request, "usage_rate_cards", result_count=len(unpriced), content_mode="metadata"
    )
    query = request.query_params
    return templates.TemplateResponse(
        "rate_cards.html",
        {
            "request": request,
            "active_page": "traces",
            "agents_nav": [],
            "provider_type": _state.get("provider_type"),
            "connection_info": _state.get("connection_info"),
            "cards": sorted(cards, key=card_key),
            "unpriced": unpriced,
            "form": form
            or {
                "provider": query.get("provider", "")[:240],
                "model": query.get("model", "")[:240],
                "service_tier": query.get("service_tier", "")[:240] or "default",
            },
            "error": error,
            "notice": NOTICES.get(query.get("notice", "")),
            "locked": locked,
        },
        status_code=status_code,
    )


def _require_editable(request):
    require_trace_permission("trace.write")
    if request.app.state.usage_pricing_in_code:
        raise HTTPException(status_code=409, detail="This app's prices are set in code")


def _change_cards(request, change):
    """Load, change and save the cards under one lock, then price with them."""
    with _lock:
        cards = change(load_rate_cards())
        save_rate_cards(cards)
        request.app.state.usage_pricing = effective_pricing(cards)


@router.get("/traces/usage/rates", response_class=HTMLResponse)
def rate_cards_page(request: Request):
    require_trace_permission("trace.read")
    return _page(request)


@router.post("/traces/usage/rates", response_class=HTMLResponse)
def save_rate_card(
    request: Request,
    provider: str = Form(""),
    model: str = Form(""),
    service_tier: str = Form("default"),
    input_per_million: str = Form(""),
    cached_input_per_million: str = Form(""),
    output_per_million: str = Form(""),
    source_url: str = Form(""),
    local: str = Form(""),
):
    _require_editable(request)
    form = {
        key: value.strip()
        for key, value in {
            "provider": provider,
            "model": model,
            "service_tier": service_tier,
            "input_per_million": input_per_million,
            "cached_input_per_million": cached_input_per_million,
            "output_per_million": output_per_million,
            "source_url": source_url,
            "local": local,
        }.items()
    }
    try:
        card = rate_card_from_form(form)
        _change_cards(
            request,
            lambda cards: [
                *(saved for saved in cards if card_key(saved) != card_key(card)),
                card,
            ],
        )
    except (OSError, ValueError) as exc:
        return _page(request, form=form, error=str(exc), status_code=400)
    audit_trace_view(
        request, "usage_rate_card_saved", result_count=1, content_mode="metadata"
    )
    return RedirectResponse("/traces/usage/rates?notice=saved", status_code=303)


@router.post("/traces/usage/rates/remove", response_class=HTMLResponse)
def remove_rate_card(
    request: Request,
    provider: str = Form(""),
    model: str = Form(""),
    service_tier: str = Form("default"),
):
    _require_editable(request)
    key = (provider.strip().lower(), model.strip(), service_tier.strip() or "default")

    def remove(cards):
        kept = [card for card in cards if card_key(card) != key]
        if len(kept) == len(cards):
            raise ValueError("No saved rate card matches that provider, model and tier")
        return kept

    try:
        _change_cards(request, remove)
    except (OSError, ValueError) as exc:
        return _page(request, error=str(exc), status_code=400)
    audit_trace_view(
        request, "usage_rate_card_removed", result_count=1, content_mode="metadata"
    )
    return RedirectResponse("/traces/usage/rates?notice=removed", status_code=303)
