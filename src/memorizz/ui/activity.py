# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Who changed what through this UI.

Every change a person makes from the UI (a successful POST, PUT, PATCH or
DELETE) is appended to ``<MEMORIZZ_HOME>/audit/ui_activity.jsonl`` as one
line: when, who, what and which record. It holds no message text, form
values or secrets, only the route and the IDs in it. Chat messages and
other high-volume or read-only posts are left out.

``who`` is the signed-in principal when the UI runs with authentication,
else the name the operator set in the top bar (the ``memorizz_operator``
cookie), else "Local operator".
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from .._env_io import memorizz_home

logger = logging.getLogger(__name__)

OPERATOR_COOKIE = "memorizz_operator"
MAX_READ = 2000
_ROTATE_BYTES = 5 * 1024 * 1024
_lock = threading.Lock()

# Posts that are not a change worth a line: chat turns, searches, reveals,
# polling, sign-in and the webhooks other systems call.
_SKIP = re.compile(
    r"^/(login|create|pull|start|runtime/start|webhook/|traces/reveal|"
    r"traces/account/resolve|traces/replays|models/)"
    r"|/playground/stream$|/knowledge-base/ingest$|/prompts/get$"
    r"|/resources/read$|/servers/[^/]+/test$|/api/mcp/catalog/auth$"
    r"|/api/timers/|/api/harness-conversations/[^/]+/turns$"
    r"|/api/finetune/evaluate$|/api/finetune/runs/[^/]+/(rank|speed)$"
)

# (method, path pattern, area, what happened). ``{id}`` is the record named.
_RULES: List[Tuple[str, str, str, str]] = [
    ("POST", r"/connect", "Store", "Connected to a store"),
    ("POST", r"/settings", "Settings", "Saved settings"),
    ("POST", r"/agents/new", "Agents", "Created an agent"),
    ("POST", r"/agents/(?P<id>[^/]+)/edit", "Agents", "Edited an agent"),
    ("POST", r"/agents/(?P<id>[^/]+)/favorite", "Agents", "Changed a favorite"),
    (
        "POST",
        r"/agents/(?P<id>[^/]+)/playground/config",
        "Agents",
        "Changed an agent's settings",
    ),
    (
        "POST",
        r"/agents/(?P<id>[^/]+)/playground/compact",
        "Agents",
        "Compacted a conversation",
    ),
    ("POST", r"/api/agents/bulk-delete", "Agents", "Deleted agents"),
    ("POST", r"/api/agents/bulk-favorite", "Agents", "Changed favorites"),
    ("POST", r"/api/agents/(?P<id>[^/]+)/name", "Agents", "Renamed an agent"),
    (
        "DELETE",
        r"/api/agents/(?P<id>[^/]+)/threads/[^/]+",
        "Agents",
        "Deleted a conversation",
    ),
    (
        "POST",
        r"/api/agents/(?P<id>[^/]+)/threads/[^/]+",
        "Agents",
        "Renamed a conversation",
    ),
    (
        "DELETE",
        r"/api/agents/(?P<id>[^/]+)/knowledge-base/[^/]+",
        "Memory",
        "Removed a knowledge-base document",
    ),
    (
        "*",
        r"/api/agents/(?P<id>[^/]+)/(skills|capabilities/.+|harness-.+|context-policy)",
        "Agents",
        "Changed an agent's capabilities",
    ),
    ("POST", r"/agents/templates/save", "Agents", "Saved an agent template"),
    (
        "POST",
        r"/agents/templates/(?P<id>[^/]+)/delete",
        "Agents",
        "Deleted an agent template",
    ),
    (
        "POST",
        r".*/approvals/(?P<id>[^/]+)/approve",
        "Approvals",
        "Approved a request",
    ),
    ("POST", r".*/approvals/(?P<id>[^/]+)/reject", "Approvals", "Rejected a request"),
    (
        "POST",
        r".*/approvals/(?P<id>[^/]+)/(cancel|resume)",
        "Approvals",
        "Resumed or cancelled a request",
    ),
    (
        "POST",
        r"/api/harness-approvals/(?P<id>[^/]+)/approve",
        "Approvals",
        "Approved a harness run",
    ),
    (
        "POST",
        r"/api/harness-approvals/(?P<id>[^/]+)/reject",
        "Approvals",
        "Rejected a harness run",
    ),
    ("POST", r"/api/harness-runs/delete", "Harnesses", "Deleted harness runs"),
    ("POST", r"/api/harness-runs", "Harnesses", "Started a harness run"),
    (
        "POST",
        r"/api/harness-runs/(?P<id>[^/]+)/(cancel|retry)",
        "Harnesses",
        "Cancelled or retried a harness run",
    ),
    ("*", r"/api/harness-.+", "Harnesses", "Changed harness settings"),
    ("POST", r"/automations", "Automations", "Created an automation"),
    (
        "POST",
        r"/automations/(?P<id>[^/]+)/delete",
        "Automations",
        "Deleted an automation",
    ),
    ("POST", r"/automations/(?P<id>[^/]+)/edit", "Automations", "Edited an automation"),
    (
        "POST",
        r"/automations/(?P<id>[^/]+)/(pause|resume)",
        "Automations",
        "Paused or resumed an automation",
    ),
    (
        "POST",
        r"/automations/(?P<id>[^/]+)/run-now",
        "Automations",
        "Ran an automation now",
    ),
    (
        "POST",
        r"/learning-control-plane/(forget|consolidation|persona)-apply",
        "Memory",
        "Applied a reviewed memory plan",
    ),
    (
        "POST",
        r"/learning-control-plane/(consolidation-restore|persona-undo|unsuppress)",
        "Memory",
        "Undid a memory change",
    ),
    (
        "POST",
        r"/learning-control-plane/persona-dismiss",
        "Memory",
        "Turned down a persona change",
    ),
    (
        "POST",
        r"/learning-control-plane/(compile|.+-plan|.+-review)",
        "Memory",
        "Planned or compiled learning",
    ),
    (
        "POST",
        r"/memory/personas/(?P<id>[^/]+)/guardrails",
        "Memory",
        "Changed a persona's guardrails",
    ),
    (
        "POST",
        r"/memory/entity/(?P<id>[^/]+)/aliases",
        "Memory",
        "Changed an entity's other names",
    ),
    (
        "POST",
        r"/memory/entity/(?P<id>[^/]+)/undo-merge",
        "Memory",
        "Undid an entity merge",
    ),
    (
        "POST",
        r"/learning-control-plane/refine",
        "Memory",
        "Ran a refine cycle",
    ),
    (
        "POST",
        r"/continual-learning/agents/(?P<id>[^/]+)/holdout",
        "Memory",
        "Changed an agent's holdout settings",
    ),
    (
        "POST",
        r"/continual-learning/skills/(?P<id>[^/]+)/holdout-exempt",
        "Memory",
        "Changed a skill's holdout exemption",
    ),
    ("POST", r"/api/memory-transfer/import", "Memory", "Imported memory"),
    ("POST", r"/persona-evolution/action", "Memory", "Decided a persona change"),
    (
        "POST",
        r"/continual-learning/skills/(?P<id>[^/]+)/(activate|demote)",
        "Memory",
        "Activated or demoted a skill",
    ),
    ("POST", r"/continual-learning/.+", "Memory", "Ran a learning step"),
    (
        "DELETE",
        r"/api/mcp/agents/(?P<id>[^/]+)/servers/[^/]+(/credentials)?",
        "MCP",
        "Removed an MCP server or its credentials",
    ),
    (
        "POST",
        r"/api/mcp/agents/(?P<id>[^/]+)/servers/[^/]+/call",
        "MCP",
        "Called an MCP tool",
    ),
    ("*", r"/api/mcp/agents/(?P<id>[^/]+)/.+", "MCP", "Changed MCP connections"),
    ("POST", r"/api/connected-apps/.+", "MCP", "Changed connected apps"),
    ("*", r"/api/finetune/.+", "Fine-tuning", "Changed fine-tuning data or models"),
    ("POST", r"/evalground/.+", "Evalground", "Started or stopped an evaluation"),
    ("POST", r"/traces/usage/budget(/clear)?", "Usage", "Changed the monthly budget"),
    ("POST", r"/traces/usage/rates(/remove)?", "Usage", "Changed a rate card"),
    ("POST", r"/traces/failures/alerts.*", "Usage", "Changed a failure alert"),
    (
        "POST",
        r"/traces/recommendations.*",
        "Observability",
        "Reviewed a recommendation",
    ),
]
_COMPILED = [
    (method, re.compile("^" + pattern + "$"), area, label)
    for method, pattern, area, label in _RULES
]


def log_path() -> Path:
    configured = os.getenv("MEMORIZZ_UI_ACTIVITY_LOG", "").strip()
    if configured:
        return Path(configured).expanduser()
    return memorizz_home() / "audit" / "ui_activity.jsonl"


def describe(method: str, path: str) -> Optional[Dict[str, str]]:
    """``{area, action, target}`` for a change, or None for a request that
    is not one (or is too frequent to be worth a line)."""
    method = method.upper()
    if method not in {"POST", "PUT", "PATCH", "DELETE"} or _SKIP.search(path):
        return None
    for wanted, pattern, area, label in _COMPILED:
        if wanted not in ("*", method):
            continue
        match = pattern.match(path)
        if match:
            target = (match.groupdict().get("id") or "")[:160]
            return {"area": area, "action": label, "target": unquote(target)}
    return {"area": "Other", "action": "Changed " + path[:120], "target": ""}


def operator_name(request: Any) -> str:
    """The name the operator set in the top bar, cleaned for display."""
    raw = unquote(str(request.cookies.get(OPERATOR_COOKIE) or ""))
    return re.sub(r"[\x00-\x1f\x7f]", "", raw).strip()[:80]


def record(request: Any, status_code: int, principal: Any = None) -> None:
    """Append one line for a change that succeeded. Never raises."""
    try:
        if status_code >= 400:
            return
        entry = describe(request.method, request.url.path)
        if entry is None:
            return
        # Without sign-in every request is the built-in "local" principal,
        # which names nobody: the operator's own name says more.
        signed_in = str(getattr(principal, "principal_id", "") or "")
        if signed_in == "local":
            signed_in = ""
        row = {
            "schema_version": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "actor": signed_in or operator_name(request),
            "method": request.method.upper(),
            "path": request.url.path[:240],
            **entry,
        }
        path = log_path()
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                if path.stat().st_size > _ROTATE_BYTES:
                    path.replace(path.with_suffix(".jsonl.1"))
            except OSError:
                pass
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            try:
                path.chmod(0o600)
            except OSError:
                pass
    except Exception as exc:
        logger.warning("Could not record UI activity (%s)", type(exc).__name__)


def recent(limit: int = 200) -> List[Dict[str, Any]]:
    """The newest entries first. A line that does not parse is skipped."""
    limit = max(1, min(int(limit), MAX_READ))
    try:
        with log_path().open("r", encoding="utf-8") as handle:
            lines = handle.readlines()[-limit:]
    except OSError:
        return []
    rows = []
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("timestamp"):
            if row.get("actor") in (None, "", "local"):
                row["actor"] = "Local operator"
            rows.append(row)
    return rows


__all__ = [
    "OPERATOR_COOKIE",
    "describe",
    "log_path",
    "operator_name",
    "record",
    "recent",
]
