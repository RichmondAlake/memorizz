"""Failure alerts: tell someone when an agent keeps failing.

A rule is a limit on failures in a time window, for one agent or all of them.
A background thread in the UI process checks the rules; a rule that crosses its
limit shows on the Overview and, with a webhook, sends one message when it fires
and one when it clears. Rules live in ``memorizz_home() / "failure_alerts.json"``,
readable only by their owner, because a webhook address is a credential: it never
appears on a page or in a log, and delivery errors are written here, not copied
from the HTTP library, whose messages include it.

Rules cover the whole deployment, so only an unrestricted admin can see or change
them. Messages carry names, codes and counts, never trace content.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import requests
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from .._env_io import env_bool, memorizz_home
from ..observability.overview import resolve_window
from .observability_view import window_phrase
from .security import audit_trace_view
from .state import _state
from .trace_access import TracePrincipal, current_principal, require_trace_permission
from .usage_budget import can_edit

logger = logging.getLogger(__name__)

router = APIRouter(tags=["traces"])

ALERT_WINDOWS = ("1h", "24h", "7d")
CHECK_INTERVAL = timedelta(seconds=60)
MAX_RULES = 50
MAX_THRESHOLD = 1_000_000
TIMEOUT = (3, 5)

THRESHOLD_ERROR = "The limit must be a whole number from 1 to 1,000,000"
WINDOW_ERROR = "Choose 1 hour, 24 hours or 7 days"
WEBHOOK_ERROR = "Use a public https address for the webhook"
LIMIT_ERROR = "Remove a rule before adding another"
SAVE_ERROR = "The rule could not be saved"
REMOVE_ERROR = "The rule could not be removed"
TEST_ERROR = "The webhook did not accept the test"
NOTICES = {
    "saved": "Rule saved.",
    "removed": "Rule removed.",
    "tested": "Test sent.",
}
# Errors travel in the redirect URL, so only these texts are ever shown.
ERRORS = frozenset(
    {
        THRESHOLD_ERROR,
        WINDOW_ERROR,
        WEBHOOK_ERROR,
        LIMIT_ERROR,
        SAVE_ERROR,
        REMOVE_ERROR,
        TEST_ERROR,
    }
)

_lock = threading.Lock()


class DeliveryError(Exception):
    """A webhook refused or missed a message; its text is safe to show."""


def rules_path():
    return memorizz_home() / "failure_alerts.json"


def _load() -> Dict[str, Any]:
    empty: Dict[str, Any] = {"version": 1, "rules": [], "state": {}}
    try:
        data = json.loads(rules_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return empty
    except (OSError, ValueError) as exc:
        logger.warning("Failure alert rules ignored (%s)", type(exc).__name__)
        return empty
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        return empty
    data.setdefault("state", {})
    return data


def _store(data: Dict[str, Any]) -> None:
    path = rules_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def _check_url(url: str) -> None:
    """Refuse an address the server shouldn't call: this is a request a form
    sets up, so it must not reach this machine or its private network."""
    from ..mcp.errors import MCPConfigurationError
    from ..mcp.security import validate_remote_url

    try:
        validate_remote_url(
            url,
            allow_private_network=env_bool("MEMORIZZ_ALERT_ALLOW_PRIVATE_WEBHOOKS"),
        )
    except MCPConfigurationError:
        raise ValueError(WEBHOOK_ERROR) from None


def rule_from_form(
    threshold: Any,
    window: Any,
    webhook_url: Any = "",
    agent_id: Any = "",
    stopped_only: Any = "",
) -> Dict[str, Any]:
    """A validated rule; ValueError messages are short enough for a URL."""
    try:
        limit = int(str(threshold).strip())
    except ValueError:
        raise ValueError(THRESHOLD_ERROR) from None
    if not 1 <= limit <= MAX_THRESHOLD:
        raise ValueError(THRESHOLD_ERROR)
    if window not in ALERT_WINDOWS:
        raise ValueError(WINDOW_ERROR)
    url = str(webhook_url or "").strip()
    if url:
        _check_url(url)
    return {
        "threshold": limit,
        "window": window,
        "webhook_url": url,
        "agent_id": str(agent_id or "").strip(),
        "stopped_only": bool(stopped_only),
    }


def save_rule(rule: Dict[str, Any], *, base_url: str) -> Dict[str, Any]:
    with _lock:
        data = _load()
        if len(data["rules"]) >= MAX_RULES:
            raise ValueError(LIMIT_ERROR)
        saved = {
            **rule,
            "id": uuid.uuid4().hex,
            "base_url": base_url,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        data["rules"].append(saved)
        _store(data)
    return saved


def delete_rule(rule_id: str) -> None:
    with _lock:
        data = _load()
        data["rules"] = [rule for rule in data["rules"] if rule.get("id") != rule_id]
        data["state"].pop(rule_id, None)
        _store(data)


def load_rules() -> List[Dict[str, Any]]:
    return _load()["rules"]


def count_for(rule: Dict[str, Any], failures: Dict[str, Any]) -> int:
    """What the rule counts: failures, or turns that stopped, for its scope."""
    key = "stopped" if rule.get("stopped_only") else "total"
    if not rule.get("agent_id"):
        return int(failures.get(key) or 0)
    for agent in failures.get("agents") or []:
        if agent.get("agent_id") == rule["agent_id"]:
            return int(agent.get(key) or 0)
    return 0


def _top(rule: Dict[str, Any], failures: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The commonest failures in the rule's scope: names and codes only."""
    groups = [
        group
        for group in failures.get("groups") or []
        if not rule.get("agent_id") or rule["agent_id"] in (group.get("agents") or [])
    ]
    return [
        {"name": group["name"], "code": group["code"], "count": group["count"]}
        for group in groups[:3]
    ]


def _describe(rule: Dict[str, Any], count: int) -> str:
    noun = "stopped turn" if rule.get("stopped_only") else "failure"
    target = rule.get("agent_name") or rule.get("agent_id") or "any agent"
    return (
        f"{count:,} {noun}{'' if count == 1 else 's'} in "
        f"{window_phrase(rule['window'])} on {target}"
    )


def _link(rule: Dict[str, Any]) -> str:
    base = str(rule.get("base_url") or "").rstrip("/")
    if rule.get("agent_id"):
        query = urlencode({"agent_id": rule["agent_id"], "window": rule["window"]})
        return f"{base}/traces?{query}#trace-failures"
    return f"{base}/traces/failures?{urlencode({'window': rule['window']})}"


def _message(
    rule: Dict[str, Any], state: Dict[str, Any], event: str, now: datetime
) -> Dict[str, Any]:
    count = int(state.get("count") or 0)
    top = state.get("top") or []
    link = _link(rule)
    if event == "firing":
        text = f"MemoRizz: {_describe(rule, count)}, reaching your limit of {rule['threshold']:,}."
        if top:
            text += f" Most often: {top[0]['name']} ({top[0]['code']})."
    elif event == "resolved":
        text = f"MemoRizz: back under your limit of {rule['threshold']:,}: {_describe(rule, count)}."
    else:
        text = "MemoRizz: this is a test of your failure alert."
    return {
        "text": f"{text} {link}",
        "memorizz": {
            "event": f"failure_alert.{event}",
            "rule_id": rule["id"],
            "agent_id": rule.get("agent_id") or None,
            "window": rule["window"],
            "threshold": rule["threshold"],
            "count": count,
            "top": top,
            "url": link,
            "at": now.isoformat(),
        },
    }


def _post(url: str, payload: Dict[str, Any]) -> None:
    """Send one message. The address is checked again here, because what it
    resolves to can change after the rule was saved, and redirects aren't
    followed, which would otherwise lead past that check."""
    _check_url(url)
    try:
        response = requests.post(
            url, json=payload, timeout=TIMEOUT, allow_redirects=False
        )
    except requests.RequestException:
        raise DeliveryError("The webhook could not be reached") from None
    if not 200 <= response.status_code < 300:
        raise DeliveryError(f"The webhook answered {response.status_code}")


def _deliver(rule: Dict[str, Any], payload: Dict[str, Any]) -> str:
    """Send a message and return what went wrong, in words safe to keep."""
    try:
        _post(rule["webhook_url"], payload)
    except DeliveryError as exc:
        return str(exc)
    except Exception as exc:
        # Only the type: an HTTP error's text would hold the address.
        logger.warning("Failure alert delivery failed (%s)", type(exc).__name__)
        return "The webhook could not be reached"
    return ""


def _read_all(window: str, now: datetime) -> Dict[str, Any]:
    from .routers.failures import read_fleet_failures

    return read_fleet_failures(resolve_window(window, now=now))["fleet"]


def check_rules(
    *,
    now: Optional[datetime] = None,
    read: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> None:
    """Check every rule that is due and send what changed.

    Runs as the deployment, not as whoever last used a page: rules are
    deployment-wide, and a thread has no request to take a scope from.
    """
    now = now or datetime.now(timezone.utc)
    token = current_principal.set(TracePrincipal())
    try:
        rules = load_rules()
        known = _load()["state"]
        window_reads: Dict[str, Dict[str, Any]] = {}

        def read_window(rule: Dict[str, Any]) -> Dict[str, Any]:
            window = rule["window"]
            if window not in window_reads:
                window_reads[window] = _read_all(window, now)
            return window_reads[window]

        updates: Dict[str, Dict[str, Any]] = {}
        for rule in rules:
            state = dict(known.get(rule["id"]) or {})
            checked = state.get("checked_at")
            if checked and now - datetime.fromisoformat(checked) < CHECK_INTERVAL:
                continue
            failures = (read or read_window)(rule)
            count = count_for(rule, failures)
            firing = count >= rule["threshold"]
            event = state.get("pending") or ""
            if firing != bool(state.get("firing")):
                event = "firing" if firing else "resolved"
                state["since"] = now.isoformat() if firing else None
            state.update(
                firing=firing,
                count=count,
                top=_top(rule, failures),
                checked_at=now.isoformat(),
            )
            if event and rule.get("webhook_url"):
                error = _deliver(rule, _message(rule, state, event, now))
                state["delivery_error"] = error
                # Kept until it arrives, so an outage doesn't lose the alert.
                state["pending"] = event if error else ""
            updates[rule["id"]] = state
        if updates:
            with _lock:
                data = _load()
                current = {rule.get("id") for rule in data["rules"]}
                data["state"].update(
                    {key: value for key, value in updates.items() if key in current}
                )
                _store(data)
    finally:
        current_principal.reset(token)


def watch(stop: threading.Event) -> None:
    """Check the rules until the UI stops. A check that fails waits for the next."""
    while not stop.wait(CHECK_INTERVAL.total_seconds()):
        if not _state.get("provider") or not load_rules():
            continue
        try:
            check_rules()
        except Exception as exc:
            logger.warning("Failure alert check failed (%s)", type(exc).__name__)


def statuses() -> List[Dict[str, Any]]:
    """Rules as the page shows them. The address is reduced to its host."""
    data = _load()
    rows = []
    for rule in data["rules"]:
        state = data["state"].get(rule.get("id")) or {}
        rows.append(
            {
                "id": rule["id"],
                "threshold": rule["threshold"],
                "window": rule["window"],
                "window_phrase": window_phrase(rule["window"]),
                "agent": rule.get("agent_name") or rule.get("agent_id") or "any agent",
                "stopped_only": bool(rule.get("stopped_only")),
                "webhook_host": urlparse(rule.get("webhook_url") or "").hostname or "",
                "firing": bool(state.get("firing")),
                "count": int(state.get("count") or 0),
                "since": state.get("since"),
                "checked_at": state.get("checked_at"),
                "delivery_error": state.get("delivery_error") or "",
            }
        )
    return rows


def failure_attention() -> List[Dict[str, Any]]:
    """Needs-attention items for rules that are firing. These cover every
    tenant, so a scoped account gets none."""
    if current_principal.get().restricted:
        return []
    data = _load()
    items = []
    for rule in data["rules"]:
        state = data["state"].get(rule.get("id")) or {}
        if state.get("firing"):
            items.append(
                {
                    "severity": "error",
                    "kind": "failures",
                    "title": _describe(rule, int(state.get("count") or 0)),
                    "detail": f"Your limit is {rule['threshold']:,}.",
                    "href": f"/traces/failures?{urlencode({'window': rule['window']})}#alerts",
                    "action": "Open failures",
                }
            )
    return items


def alert_messages(query_params) -> Dict[str, str]:
    error = query_params.get("alert_error", "")
    return {
        "notice": NOTICES.get(query_params.get("alert_notice", ""), ""),
        "error": error if error in ERRORS else "",
    }


def _back(**params: str) -> RedirectResponse:
    return RedirectResponse(
        f"/traces/failures?{urlencode(params)}#alerts", status_code=303
    )


def _agent_names() -> Dict[str, str]:
    from .dashboard import _agent_names as names

    return names()


def _require_edit() -> None:
    require_trace_permission("trace.write")
    if not can_edit():
        raise HTTPException(status_code=403, detail="Trace permission denied")


@router.post("/traces/failures/alerts")
def create_failure_alert(
    request: Request,
    threshold: str = Form(""),
    window: str = Form(""),
    webhook_url: str = Form(""),
    agent_id: str = Form(""),
    stopped_only: str = Form(""),
):
    _require_edit()
    try:
        rule = rule_from_form(threshold, window, webhook_url, agent_id, stopped_only)
        rule["agent_name"] = _agent_names().get(rule["agent_id"], "")
        save_rule(rule, base_url=str(request.base_url))
    except ValueError as exc:
        return _back(alert_error=str(exc))
    except OSError:
        logger.warning("Failure alert rule could not be saved")
        return _back(alert_error=SAVE_ERROR)
    audit_trace_view(
        request, "failure_alert_saved", result_count=1, content_mode="metadata"
    )
    return _back(alert_notice="saved")


@router.post("/traces/failures/alerts/{rule_id}/delete")
def delete_failure_alert(request: Request, rule_id: str):
    _require_edit()
    try:
        delete_rule(rule_id)
    except OSError:
        logger.warning("Failure alert rule could not be removed")
        return _back(alert_error=REMOVE_ERROR)
    audit_trace_view(
        request, "failure_alert_removed", result_count=1, content_mode="metadata"
    )
    return _back(alert_notice="removed")


@router.post("/traces/failures/alerts/{rule_id}/test")
def test_failure_alert(request: Request, rule_id: str):
    _require_edit()
    rule = next((row for row in load_rules() if row.get("id") == rule_id), None)
    if rule is None or not rule.get("webhook_url"):
        return _back(alert_error=TEST_ERROR)
    now = datetime.now(timezone.utc)
    error = _deliver(rule, _message(rule, {}, "test", now))
    audit_trace_view(
        request, "failure_alert_tested", result_count=1, content_mode="metadata"
    )
    return _back(alert_error=TEST_ERROR) if error else _back(alert_notice="tested")


__all__ = [
    "ALERT_WINDOWS",
    "alert_messages",
    "check_rules",
    "count_for",
    "delete_rule",
    "failure_attention",
    "load_rules",
    "router",
    "rule_from_form",
    "rules_path",
    "save_rule",
    "statuses",
    "watch",
]
