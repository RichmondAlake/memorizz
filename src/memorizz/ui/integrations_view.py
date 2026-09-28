"""Shape data for the integration and configuration monitor pages.

MCP connections, Vercel skills and Settings each open with a summary tape and,
for MCP, a server grid. The shaping lives here so it stays pure and tested:
routes gather raw inputs (server configs, connection status, the MCP audit
log, the last tool list each server returned, settings sections) and templates
only render what these functions return.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

AUDIT_TAIL_BYTES = 1_000_000
WINDOW_DAYS = 7
CACHED_TOOLS_LIMIT = 200
TOOL_DESCRIPTION_CHARS = 400
RECENT_ERRORS = 5

TRANSPORT_LABELS = {
    "streamable_http": "Streamable HTTP",
    "sse": "SSE",
    "stdio": "stdio",
}
AUTH_LABELS = {"none": "None", "oauth": "OAuth", "bearer": "Bearer token"}

# Server state -> (label, fleet-health modifier, sort rank: worst first).
STATES = {
    "failing": ("Failing", "failing", 0),
    "needs_auth": ("Needs sign-in", "degraded", 1),
    "unchecked": ("Not checked", "idle", 2),
    "ready": ("Ready", "healthy", 3),
    "disabled": ("Disabled", "idle", 4),
}

# What to do about each MCP error code the client reports.
ERROR_HINTS = {
    "authorization_required": (
        "Sign-in expired or was never completed. Select Authorize for OAuth "
        "servers, or Edit to paste a new bearer token, then select Test."
    ),
    "connection_error": (
        "The server did not answer in time. Check the URL or command and your "
        "network, then select Test. Raise the request timeout with Edit if it is slow."
    ),
    "protocol_error": (
        "The server answered, but not as an MCP endpoint. Check that the URL "
        "points at the MCP path (often ending in /mcp) or that the stdio command "
        "starts an MCP server, then select Test."
    ),
    "configuration_error": (
        "The saved connection is incomplete or invalid. Select Edit, correct the "
        "highlighted fields and save."
    ),
    "response_too_large": (
        "The server returned more data than the limit allows. Narrow the request, "
        "or block the tool that returns large results."
    ),
    "policy_denied": (
        "A tool policy blocked this call. Adjust allowed, blocked or mutation "
        "tools with Edit."
    ),
    "oauth_start_timeout": (
        "OAuth did not start in time. Check the client ID and redirect URI with "
        "Edit, then select Authorize again."
    ),
    "approval_required": (
        "This mutating tool needs approval. Approve the proposal to run it."
    ),
}
DEFAULT_HINT = "Select Test to retry. If it keeps failing, check the server's own logs."


def error_hint(code: Optional[str]) -> str:
    """One actionable sentence for an MCP error code."""
    return ERROR_HINTS.get(str(code or ""), DEFAULT_HINT)


def _parse_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------- files


def read_jsonl_tail(path: Path, max_bytes: int = AUDIT_TAIL_BYTES) -> List[Dict]:
    """Parse the newest JSON lines of a log, bounded by ``max_bytes``.

    A missing or unreadable file reads as empty. When the file is larger than
    the bound, the first (partial) line of the tail is dropped.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()
            raw = handle.read()
    except OSError:
        return []
    entries = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def load_tool_cache(path: Path) -> Dict[str, Any]:
    """Last tool list per owner and server: ``{owner: {server: entry}}``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def server_endpoint(server: Dict[str, Any]) -> str:
    """The URL, or the stdio command line, that identifies a server."""
    if server.get("url"):
        return str(server["url"])
    parts = [server.get("command")] + list(server.get("args") or [])
    return " ".join(str(part) for part in parts if part)


def tool_cache_entry(
    server: Dict[str, Any], result: Dict[str, Any], now: datetime
) -> Dict[str, Any]:
    """Compact, bounded record of a successful tools/list result."""
    tools = []
    for tool in result.get("tools") or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        description = " ".join(str(tool.get("description") or "").split())
        if len(description) > TOOL_DESCRIPTION_CHARS:
            description = description[: TOOL_DESCRIPTION_CHARS - 1] + "…"
        tools.append({"name": str(tool["name"]), "description": description})
    info = (
        result.get("server_info") if isinstance(result.get("server_info"), dict) else {}
    )
    return {
        "endpoint": server_endpoint(server),
        "checked_at": now.isoformat(),
        "tool_count": len(tools),
        "tools": tools[:CACHED_TOOLS_LIMIT],
        "server_name": str(info.get("name") or ""),
        "server_version": str(info.get("version") or ""),
        "protocol_version": str(result.get("protocol_version") or ""),
    }


def store_tool_list(
    path: Path,
    owner_id: str,
    server: Dict[str, Any],
    result: Dict[str, Any],
    now: Optional[datetime] = None,
) -> None:
    """Remember a server's tool list so the grid can show it on the next load."""
    if not result.get("ok"):
        return
    now = now or datetime.now(timezone.utc)
    cache = load_tool_cache(path)
    owner = cache.setdefault(str(owner_id), {})
    if not isinstance(owner, dict):
        owner = cache[str(owner_id)] = {}
    owner[str(server.get("name") or "")] = tool_cache_entry(server, result, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


# ----------------------------------------------------------------------- MCP


def _config_rows(server: Dict[str, Any]) -> List[Dict[str, str]]:
    auth = server.get("auth") or {}
    rows = [
        (
            "Transport",
            TRANSPORT_LABELS.get(server.get("transport"), server.get("transport")),
        ),
        ("Endpoint", server_endpoint(server)),
        (
            "Authentication",
            AUTH_LABELS.get(auth.get("type") or "none", auth.get("type")),
        ),
        ("OAuth client ID", auth.get("client_id")),
        ("OAuth scopes", " ".join(auth.get("scopes") or [])),
        ("Redirect URI", auth.get("redirect_uri")),
        ("Request timeout", f"{server.get('timeout', 30)} s"),
        (
            "Approval for mutations",
            "Required" if server.get("require_approval", True) else "Off",
        ),
        (
            "Private network",
            "Allowed" if server.get("allow_private_network") else "Blocked",
        ),
        ("Allowed tools", ", ".join(server.get("allowed_tools") or []) or "All"),
        ("Blocked tools", ", ".join(server.get("blocked_tools") or [])),
        ("Read-only tools", ", ".join(server.get("read_only_tools") or [])),
        ("Mutation tools", ", ".join(server.get("mutation_tools") or [])),
        ("Environment keys", ", ".join(server.get("env_keys") or [])),
        ("Header names", ", ".join(server.get("header_names") or [])),
        ("Working directory", server.get("cwd")),
    ]
    return [{"label": label, "value": str(value)} for label, value in rows if value]


def _server_state(
    server: Dict[str, Any], status: Dict[str, Any], last_check: Optional[Dict]
) -> str:
    if server.get("enabled") is False:
        return "disabled"
    if status.get("authenticated") is False or status.get("state") == "reauth_required":
        return "needs_auth"
    if last_check is not None and not last_check.get("ok"):
        if last_check.get("error_code") == "authorization_required":
            return "needs_auth"
        return "failing"
    if last_check is not None:
        return "ready"
    return "unchecked"


def build_mcp_view(
    servers: Iterable[Dict[str, Any]],
    statuses: Iterable[Dict[str, Any]],
    audit: Iterable[Dict[str, Any]],
    *,
    owner_id: str,
    tool_cache: Optional[Dict[str, Any]] = None,
    agents: Optional[Iterable[Dict[str, Any]]] = None,
    now: Optional[datetime] = None,
    window_days: int = WINDOW_DAYS,
) -> Dict[str, Any]:
    """Rows for the server grid and totals for the tape, worst state first.

    ``audit`` holds MCP audit-log entries (any owner); ``tool_cache`` the
    stored tool lists; ``agents`` rows of ``{agent_id, name, servers}`` for
    every agent, used to count MCP adoption and find shared servers.
    """
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)
    status_by_name = {
        str(item.get("name") or item.get("server_name") or ""): item
        for item in statuses
        if isinstance(item, dict)
    }
    owned_cache = (tool_cache or {}).get(str(owner_id)) or {}
    agent_rows = [row for row in agents or [] if isinstance(row, dict)]

    by_server: Dict[str, List[Dict[str, Any]]] = {}
    for entry in audit:
        if not isinstance(entry, dict) or str(entry.get("owner_id")) != str(owner_id):
            continue
        when = _parse_time(entry.get("timestamp"))
        if when is None:
            continue
        by_server.setdefault(str(entry.get("server_name") or ""), []).append(
            {**entry, "when": when}
        )

    rows = []
    for server in servers:
        name = str(server.get("name") or "")
        entries = sorted(by_server.get(name, []), key=lambda item: item["when"])
        checks = [item for item in entries if item.get("operation") == "tools/list"]
        last_check = checks[-1] if checks else None
        recent = [item for item in entries if item["when"] >= since]
        calls = [item for item in recent if item.get("operation") == "tools/call"]
        failures = [item for item in recent if not item.get("ok")]
        endpoint = server_endpoint(server)
        cached = (
            owned_cache.get(name) if isinstance(owned_cache.get(name), dict) else None
        )
        if cached and cached.get("endpoint") != endpoint:
            cached = None  # the server moved since its tools were listed
        state = _server_state(server, status_by_name.get(name, {}), last_check)
        label, health, rank = STATES[state]
        errors = [
            {
                "when": item["when"],
                "operation": item.get("operation") or "",
                "tool": item.get("tool_name") or "",
                "code": item.get("error_code") or "error",
                "hint": error_hint(item.get("error_code")),
            }
            for item in reversed(failures[-RECENT_ERRORS:])
        ]
        used = Counter(item.get("tool_name") for item in calls if item.get("tool_name"))
        failed_by_tool = Counter(
            item.get("tool_name") for item in calls if not item.get("ok")
        )
        shared = [
            {
                "agent_id": row.get("agent_id"),
                "name": row.get("name") or row.get("agent_id"),
            }
            for row in agent_rows
            if str(row.get("agent_id")) != str(owner_id)
            and any(
                server_endpoint(other) == endpoint for other in row.get("servers") or []
            )
        ]
        check_time = last_check["when"] if last_check else None
        cached_time = _parse_time((cached or {}).get("checked_at"))
        if cached_time and (check_time is None or cached_time > check_time):
            check_time = cached_time
        auth_type = (server.get("auth") or {}).get("type") or "none"
        transport = server.get("transport") or "stdio"
        tags = [state]
        if auth_type == "oauth":
            tags.append("oauth")
        rows.append(
            {
                "name": name,
                "server": server,
                "transport": transport,
                "transport_label": TRANSPORT_LABELS.get(transport, transport),
                "endpoint": endpoint,
                "auth_type": auth_type,
                "auth_label": AUTH_LABELS.get(auth_type, auth_type),
                "state": state,
                "state_label": label,
                "health": health,
                "rank": rank,
                "tool_count": cached.get("tool_count") if cached else None,
                "tools": (cached or {}).get("tools") or [],
                "server_info": " ".join(
                    part
                    for part in (
                        (cached or {}).get("server_name"),
                        (cached or {}).get("server_version"),
                    )
                    if part
                ),
                "last_check": check_time,
                "last_check_ok": bool(last_check.get("ok")) if last_check else None,
                "calls": len(calls),
                "failed_calls": sum(1 for item in calls if not item.get("ok")),
                "error_count": len(failures),
                "errors": errors,
                "tools_used": [
                    {
                        "name": tool,
                        "calls": count,
                        "failed": failed_by_tool.get(tool, 0),
                    }
                    for tool, count in used.most_common(8)
                ],
                "shared_with": shared,
                "config": _config_rows(server),
                "tags": " ".join(tags),
                "search": " ".join(
                    (name, transport, endpoint, auth_type, label)
                ).lower(),
            }
        )
    rows.sort(key=lambda row: (row["rank"], row["name"]))

    known_tools = [row["tool_count"] for row in rows if row["tool_count"] is not None]
    checks = [row["last_check"] for row in rows if row["last_check"]]
    counts = Counter(row["state"] for row in rows)
    return {
        "rows": rows,
        "window_days": window_days,
        "totals": {
            "servers": len(rows),
            "ready": counts.get("ready", 0),
            "needs_auth": counts.get("needs_auth", 0),
            "failing": counts.get("failing", 0),
            "unchecked": counts.get("unchecked", 0),
            "disabled": counts.get("disabled", 0),
            "tools": sum(known_tools) if known_tools else None,
            "tools_known": len(known_tools),
            "calls": sum(row["calls"] for row in rows),
            "failed_calls": sum(row["failed_calls"] for row in rows),
            "errors": sum(row["error_count"] for row in rows),
            "last_check": max(checks) if checks else None,
            "agents_with_mcp": sum(1 for row in agent_rows if row.get("servers")),
            "agents": len(agent_rows),
            "transports": sorted({row["transport"] for row in rows}),
        },
    }


# -------------------------------------------------------------- Vercel skills


def _attr(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def skills_marketplace_agents(
    agents: Iterable[Any], provider: str = "vercel"
) -> List[Dict[str, str]]:
    """Agents whose skills marketplace is ``provider``, as ``{agent_id, name}``."""
    rows = []
    for agent in agents:
        value = _attr(agent, "skills_marketplace_provider")
        if isinstance(value, dict):  # same normalization as the agent form
            value = value.get("provider") or value.get("name")
        if str(value or "").strip().lower() != provider:
            continue
        agent_id = str(_attr(agent, "agent_id") or _attr(agent, "_id") or "")
        persona = _attr(agent, "persona")
        name = _attr(agent, "name") or (_attr(persona, "name") if persona else None)
        rows.append(
            {"agent_id": agent_id, "name": str(name or agent_id[:8] or "Agent")}
        )
    return rows


# ------------------------------------------------------------------ Settings


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "section"


def _option_label(field: Optional[Dict[str, Any]]) -> str:
    if not field:
        return ""
    value = str(field.get("current_value") or "")
    for option in field.get("options") or []:
        if str(option.get("value")) == value and value:
            return str(option.get("label") or value)
    return value


def settings_summary(sections: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Tape figures and the section index for the settings console.

    Secret fields (no ``field_type``) are the keys; a key counts as configured
    when the process environment has a value for it.
    """
    sections = list(sections)
    fields = {
        field["env"]: field
        for section in sections
        for field in section.get("fields") or []
        if field.get("env")
    }
    secrets = [field for field in fields.values() if not field.get("field_type")]
    index = []
    seen = Counter()
    for section in sections:
        slug = _slug(section.get("title") or "")
        seen[slug] += 1
        section_fields = section.get("fields") or []
        index.append(
            {
                "id": "settings-"
                + slug
                + ("" if seen[slug] == 1 else f"-{seen[slug]}"),
                "title": section.get("title") or "",
                "set": sum(1 for field in section_fields if field.get("is_set")),
                "total": len(section_fields),
            }
        )

    def value(env: str) -> str:
        return str((fields.get(env) or {}).get("current_value") or "")

    llm_field = fields.get("MEMORIZZ_DEFAULT_LLM_PROVIDER")
    embedding_field = fields.get("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER")
    return {
        "keys_set": sum(1 for field in secrets if field.get("is_set")),
        "keys_total": len(secrets),
        "saved_backend": _option_label(fields.get("MEMORIZZ_BACKEND")),
        "llm_provider": _option_label(llm_field),
        "llm_model": value("MEMORIZZ_DEFAULT_LLM_MODEL"),
        "llm_set": bool((llm_field or {}).get("is_set")),
        "embedding_provider": _option_label(embedding_field),
        "embedding_model": value("MEMORIZZ_DEFAULT_EMBEDDING_MODEL"),
        "sandbox": _option_label(fields.get("MEMORIZZ_DEFAULT_SANDBOX_PROVIDER")),
        "browser": _option_label(fields.get("MEMORIZZ_BROWSER_CONTROL_PROVIDER")),
        "sections": index,
    }
