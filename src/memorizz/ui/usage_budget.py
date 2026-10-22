"""A monthly spend budget: progress on the Usage page, an alert on the Overview.

The budget lives in ``memorizz_home() / "usage_budget.json"``. Spend is the
known spend since the first of the month in UTC, priced with the app's rate
cards; calls without a price are counted apart, never guessed.
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from .._env_io import memorizz_home
from ..observability.usage_query import query_usage
from .security import audit_trace_view, ui_read_only
from .state import _state
from .trace_access import (
    current_principal,
    require_trace_permission,
    scoped_trace_filters,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["usage"])

# Far above any real budget; it keeps a typo out of the spend arithmetic.
MAX_BUDGET = Decimal(100_000_000)
CACHE_TTL = timedelta(seconds=300)
CENT = Decimal("0.01")
NOTICES = {"saved": "Budget saved.", "cleared": "Budget removed."}
AMOUNT_ERROR = "Enter the monthly budget in US dollars, such as 500"
RANGE_ERROR = "The budget must be more than $0 and under $100,000,000"
ALERT_ERROR = "The alert must be a whole percent from 1 to 100"
SAVE_ERROR = "The budget could not be saved"
REMOVE_ERROR = "The budget could not be removed"
# Errors travel in the redirect URL, so only these texts are ever shown.
ERRORS = frozenset({AMOUNT_ERROR, RANGE_ERROR, ALERT_ERROR, SAVE_ERROR, REMOVE_ERROR})

_lock = threading.Lock()
_cache_lock = threading.Lock()
# (month start, scope, id(pricing)) -> (pricing, figures). A rate-card change
# replaces the registry, so its id keys the prices; holding the registry keeps
# that id from being reused by a later one while the entry lives.
_cache = {}


def budget_path():
    return memorizz_home() / "usage_budget.json"


def budget_from_form(monthly_usd, alert_percent="80"):
    """A validated budget; ValueError messages are short enough for a URL."""
    try:
        amount = Decimal(str(monthly_usd).strip().lstrip("$").replace(",", ""))
    except InvalidOperation:
        raise ValueError(AMOUNT_ERROR) from None
    if not amount.is_finite() or not 0 < amount < MAX_BUDGET:
        raise ValueError(RANGE_ERROR)
    try:
        percent = int(str(alert_percent).strip() or 80)
    except ValueError:
        raise ValueError(ALERT_ERROR) from None
    if not 1 <= percent <= 100:
        raise ValueError(ALERT_ERROR)
    return {"monthly_usd": amount, "alert_percent": percent}


def load_budget() -> Optional[dict]:
    """The saved budget, or None when none is set or the file cannot be used."""
    path = budget_path()
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        return budget_from_form(row["monthly_usd"], row["alert_percent"])
    except FileNotFoundError:
        return None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "Saved usage budget ignored: %s holds no valid budget (%s)",
            path.name,
            type(exc).__name__,
        )
        return None


def _forget_figures():
    with _cache_lock:
        _cache.clear()


def save_budget(budget):
    path = budget_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        row = {
            "version": 1,
            "monthly_usd": str(budget["monthly_usd"]),
            "alert_percent": budget["alert_percent"],
        }
        temporary.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    _forget_figures()


def clear_budget():
    with _lock:
        budget_path().unlink(missing_ok=True)
    _forget_figures()


def can_edit():
    """Whether this viewer would pass the POST checks; Rate cards uses the same."""
    principal = current_principal.get()
    return not (
        ui_read_only() or principal.restricted or not principal.allows("trace.write")
    )


def money(value):
    return f"${value:,.2f}"


def _month_figures(request, provider, month_start, now):
    """Known spend and call counts this month, cached for five minutes."""
    pricing = request.app.state.usage_pricing
    scope = scoped_trace_filters()
    key = (month_start, tuple(sorted(scope.items())), id(pricing))
    with _cache_lock:
        cached = _cache.get(key)
    if cached and timedelta(0) <= now - cached[1]["measured_at"] < CACHE_TTL:
        return cached[1]
    totals = query_usage(
        provider,
        filters={
            **scope,
            "start_time": month_start.isoformat(),
            "end_time": now.isoformat(),
        },
        pricing=pricing,
        max_events=10000,
    )["totals"]
    figures = {
        "spend": Decimal(totals.get("cost_usd") or 0),
        "unpriced_calls": int(totals.get("unpriced_calls") or 0),
        "calls": int(totals.get("calls") or 0),
        "measured_at": now,
    }
    with _cache_lock:
        for old in [
            old
            for old, (_, seen) in _cache.items()
            if now - seen["measured_at"] >= CACHE_TTL
        ]:
            del _cache[old]
        _cache[key] = (pricing, figures)
    return figures


def _against(budget, figures, month_start):
    spend = figures["spend"]
    percent = float(spend / budget["monthly_usd"] * 100)
    if percent >= 100:
        state = "over"
    elif percent >= budget["alert_percent"]:
        state = "warning"
    else:
        state = "ok"
    # A straight line from the spend so far; too noisy before a full day.
    projected = None
    elapsed = (figures["measured_at"] - month_start).total_seconds()
    if elapsed >= 86400:
        month = (month_start + timedelta(days=32)).replace(day=1) - month_start
        projected = (
            spend * Decimal(month.total_seconds()) / Decimal(elapsed)
        ).quantize(CENT)
    return {
        "spend": spend,
        "unpriced_calls": figures["unpriced_calls"],
        "calls": figures["calls"],
        "percent": percent,
        "projected": projected,
        "state": state,
    }


def budget_status(request, *, now: datetime) -> Optional[dict]:
    """This month's spend against the budget, or None without one. Never raises."""
    budget = load_budget()
    provider = _state.get("provider")
    if budget is None or provider is None:
        return None
    now = (
        now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
    )
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    status = {
        "budget": budget["monthly_usd"],
        "alert_percent": budget["alert_percent"],
        "month_start": month_start,
        "now": now,
        "can_edit": can_edit(),
    }
    try:
        figures = _month_figures(request, provider, month_start, now)
        return {**status, **_against(budget, figures, month_start)}
    except Exception:
        logger.warning("Usage budget spend is unavailable", exc_info=True)
        return {
            **status,
            "spend": None,
            "unpriced_calls": 0,
            "calls": 0,
            "percent": None,
            "projected": None,
            "state": "unknown",
        }


def budget_attention(request, *, now: datetime) -> Optional[dict]:
    """A Needs-attention item once spend reaches the alert, else None."""
    status = budget_status(request, now=now)
    if status is None or status["state"] not in {"warning", "over"}:
        return None
    budget = money(status["budget"])
    over = status["state"] == "over"
    start = status["month_start"]
    detail = f"{money(status['spend'])} known spend since {start:%b} {start.day}"
    if status["projected"] is not None:
        detail += f"; on pace for {money(status['projected'])} by month end"
    detail += "."
    unpriced = status["unpriced_calls"]
    if unpriced:
        detail += (
            f" {unpriced:,} unpriced call{'s' if unpriced != 1 else ''}"
            f" {'aren' if unpriced != 1 else 'isn'}'t counted."
        )
    return {
        "severity": "error" if over else "warning",
        "kind": "budget",
        "title": f"Spend is over this month's {budget} budget"
        if over
        else f"Spend is at {int(status['percent'])}% of this month's {budget} budget",
        "detail": detail,
        "href": "/traces/usage#budget",
        "action": "Open the budget",
    }


def budget_messages(query_params):
    """The notice and error for ``budget_panel`` from the redirect's query."""
    error = query_params.get("budget_error", "")
    return {
        "notice": NOTICES.get(query_params.get("budget_notice", ""), ""),
        "error": error if error in ERRORS else "",
    }


def _back(**params):
    return RedirectResponse(
        "/traces/usage?" + urlencode(params) + "#budget", status_code=303
    )


@router.post("/traces/usage/budget")
def save_usage_budget(
    request: Request, monthly_usd: str = Form(""), alert_percent: str = Form("80")
):
    require_trace_permission("trace.write")
    try:
        save_budget(budget_from_form(monthly_usd, alert_percent))
    except ValueError as exc:
        return _back(budget_error=str(exc))
    except OSError:
        logger.warning("Usage budget save failed", exc_info=True)
        return _back(budget_error=SAVE_ERROR)
    audit_trace_view(
        request, "usage_budget_saved", result_count=1, content_mode="metadata"
    )
    return _back(budget_notice="saved")


@router.post("/traces/usage/budget/clear")
def clear_usage_budget(request: Request):
    require_trace_permission("trace.write")
    try:
        clear_budget()
    except OSError:
        logger.warning("Usage budget removal failed", exc_info=True)
        return _back(budget_error=REMOVE_ERROR)
    audit_trace_view(
        request, "usage_budget_cleared", result_count=1, content_mode="metadata"
    )
    return _back(budget_notice="cleared")
