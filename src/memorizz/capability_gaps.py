# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""What an agent can't do yet, and the concrete ways to enable it.

State is always computed from the agent's current configuration and the
environment (API keys set, MCP servers attached and signed in), never cached,
so a host can show an up-to-date "enable" card whenever a request needs a
capability the agent lacks. Nothing here changes configuration.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .mcp import catalog


@dataclass(frozen=True)
class Capability:
    id: str
    title: str
    summary: str
    # A request that matches one of these probably needs the capability.
    patterns: Tuple[str, ...]
    # An attached MCP server whose name or URL contains one of these provides it.
    server_terms: Tuple[str, ...] = ()
    preset: Optional[str] = None


CAPABILITIES: Tuple[Capability, ...] = (
    Capability(
        id="web_search",
        title="Web search",
        summary="Search the web and read pages",
        patterns=(
            r"\b(search|browse|look (it |this |that )?up|google)\b[^.?!\n]{0,40}"
            r"\b(internet|web|online)\b",
            r"\bweb ?search\b",
            r"\b(latest|breaking|today'?s) news\b",
            r"\bstock price\b",
            r"\bweather (in|for|today|tomorrow)\b",
            r"https?://\S+",
        ),
        server_terms=("search", "brave", "exa", "tavily", "firecrawl", "fetch"),
    ),
    Capability(
        id="email",
        title="Email",
        summary="Read and draft email",
        patterns=(r"\b(e-?mails?|inbox|gmail|outlook)\b",),
        server_terms=("gmail", "mail", "outlook"),
        preset="gmail",
    ),
    Capability(
        id="calendar",
        title="Calendar",
        summary="Read calendars and free/busy times",
        patterns=(
            r"\bcalendar\b",
            r"\bmeetings?\b",
            r"\b(schedule|book) (a |an )?(call|meeting)\b",
            r"\bfree (slots?|time)\b",
        ),
        server_terms=("calendar",),
        preset="google-calendar",
    ),
    Capability(
        id="notes",
        title="Notes and docs",
        summary="Search and update Notion pages",
        patterns=(r"\bnotion\b", r"\bmy notes\b", r"\bwiki\b"),
        server_terms=("notion", "confluence", "obsidian"),
        preset="notion",
    ),
    Capability(
        id="code_execution",
        title="Run code",
        summary="Run code in an isolated sandbox",
        patterns=(
            r"\b(run|execute) (this |the |some |my )?(code|script|python|snippet)\b",
        ),
    ),
    Capability(
        id="browser",
        title="Browse websites",
        summary="Open pages and fill in forms in a browser",
        patterns=(
            r"\b(open|visit|go to|click)\b[^.?!\n]{0,30}\b(website|site|web ?page|browser)\b",
            r"\bfill (in|out) (the |a |this )?form\b",
        ),
    ),
)
BY_ID = {capability.id: capability for capability in CAPABILITIES}

INTERNET_PROVIDERS = (
    {
        "provider": "tavily",
        "title": "Tavily",
        "key_env": "TAVILY_API_KEY",
        "signup_url": "https://app.tavily.com/",
    },
    {
        "provider": "firecrawl",
        "title": "Firecrawl",
        "key_env": "FIRECRAWL_API_KEY",
        "signup_url": "https://www.firecrawl.dev/app/api-keys",
    },
)


def _server_provides(server: Dict[str, Any], capability: Capability) -> bool:
    preset = catalog.PRESETS.get(capability.preset or "") if capability.preset else None
    url = str(server.get("url") or "").rstrip("/")
    if preset and url == preset["url"]:
        return True
    haystack = " ".join(
        str(server.get(key) or "") for key in ("name", "url", "command")
    ).lower()
    haystack += " " + " ".join(str(arg) for arg in server.get("args") or []).lower()
    return any(term in haystack for term in capability.server_terms)


def capability_state(
    *,
    internet_provider: Optional[str] = None,
    mcp_servers: Sequence[Dict[str, Any]] = (),
    signed_in: Optional[Dict[str, bool]] = None,
    sandbox_provider: Optional[str] = None,
    browser_provider: Optional[str] = None,
    environ: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Each capability's status and the ways to enable it, from live config.

    ``status`` is ``enabled``, ``needs_sign_in`` (an MCP server provides it
    but is not signed in) or ``missing``.
    """
    env = os.environ if environ is None else environ
    signed_in = dict(signed_in or {})
    servers = [dict(server) for server in mcp_servers if isinstance(server, dict)]
    rows: List[Dict[str, Any]] = []
    for capability in CAPABILITIES:
        providing = [
            server for server in servers if _server_provides(server, capability)
        ]
        ready = [
            server
            for server in providing
            if signed_in.get(str(server.get("name")), False)
        ]
        row: Dict[str, Any] = {
            "id": capability.id,
            "title": capability.title,
            "summary": capability.summary,
            "status": "missing",
            "via": None,
            "offers": [],
        }
        if capability.id == "web_search" and str(internet_provider or "") not in {
            "",
            "offline",
            "none",
        }:
            row.update(status="enabled", via=str(internet_provider))
        elif capability.id == "code_execution" and sandbox_provider:
            row.update(status="enabled", via=str(sandbox_provider))
        elif capability.id == "browser" and browser_provider:
            row.update(status="enabled", via=str(browser_provider))
        elif ready:
            row.update(status="enabled", via=str(ready[0].get("name")))
        elif providing:
            server = providing[0]
            auth = server.get("auth") or {}
            row.update(status="needs_sign_in", via=str(server.get("name")))
            row["offers"].append(
                {
                    "kind": "mcp_sign_in",
                    "server": str(server.get("name")),
                    "title": f"Sign in to {server.get('name')}",
                    "needs_client": auth.get("type") == "oauth"
                    and not auth.get("client_id")
                    and _preset_needs_client(server),
                }
            )
        if row["status"] == "missing":
            row["offers"] = _offers(capability, env)
        rows.append(row)
    return rows


def _preset_needs_client(server: Dict[str, Any]) -> bool:
    url = str(server.get("url") or "").rstrip("/")
    return any(
        preset["needs_client"] and preset["url"] == url
        for preset in catalog.PRESETS.values()
    )


def _offers(capability: Capability, env: Any) -> List[Dict[str, Any]]:
    offers: List[Dict[str, Any]] = []
    if capability.id == "web_search":
        for provider in INTERNET_PROVIDERS:
            offers.append(
                {
                    "kind": "internet_provider",
                    **provider,
                    "key_set": bool(str(env.get(provider["key_env"], "")).strip()),
                }
            )
    if capability.preset:
        preset = catalog.PRESETS[catalog.preset_key(capability.preset)]
        offers.append(
            {
                "kind": "mcp_preset",
                "preset": catalog.preset_key(capability.preset),
                "title": f"Connect {preset['title']}",
                "needs_client": bool(preset["needs_client"]),
                "docs": preset["docs"],
            }
        )
    if capability.id in {"code_execution", "browser"}:
        section = "sandbox" if capability.id == "code_execution" else "browser"
        offers.append(
            {
                "kind": "setting",
                "section": section,
                "title": (
                    "Choose a sandbox"
                    if section == "sandbox"
                    else "Turn on browser control"
                ),
            }
        )
    else:
        offers.append(
            {
                "kind": "mcp_search",
                "query": {"web_search": "web search", "notes": "notes"}.get(
                    capability.id, capability.id
                ),
                "title": "Find an MCP server",
            }
        )
    return offers


def state_for_agent(agent: Any) -> List[Dict[str, Any]]:
    """Capability state for a live ``MemAgent``."""
    manager = getattr(agent, "mcp_manager", None)
    servers = manager.server_dicts() if manager is not None else []
    statuses = (
        manager.connection_status().get("servers") or [] if manager is not None else []
    )
    internet = None
    if getattr(agent, "has_internet_access", None) and agent.has_internet_access():
        internet = agent.get_internet_access_provider_name()
    sandbox = None
    if getattr(agent, "has_sandbox", None) and agent.has_sandbox():
        sandbox = agent.get_sandbox_provider_name() or "sandbox"
    browser = None
    if getattr(agent, "has_browser_control", None) and agent.has_browser_control():
        browser = "browser"
    return capability_state(
        internet_provider=internet,
        mcp_servers=servers,
        signed_in={
            str(row.get("name")): bool(row.get("authenticated"))
            for row in statuses
            if isinstance(row, dict)
        },
        sandbox_provider=sandbox,
        browser_provider=browser,
    )


def missing(state: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in state if row["status"] != "enabled"]


def suggest(query: str, state: Iterable[Dict[str, Any]]) -> List[str]:
    """IDs of capabilities the request seems to need but the agent lacks."""
    text = str(query or "").lower()
    ids = []
    for row in missing(state):
        capability = BY_ID.get(row["id"])
        if capability and any(re.search(p, text) for p in capability.patterns):
            ids.append(capability.id)
    return ids


def manifest(state: Iterable[Dict[str, Any]]) -> str:
    """The system-prompt note listing what this agent cannot do yet."""
    lines = []
    for row in missing(state):
        note = (
            " (connected but not signed in)" if row["status"] == "needs_sign_in" else ""
        )
        lines.append(f"- {row['title']}: {row['summary']}{note}")
    if not lines:
        return ""
    return (
        "Capabilities not enabled for this agent:\n"
        + "\n".join(lines)
        + "\nIf a request needs one of these, call request_capability with its "
        "name instead of improvising, pretending, or offering a workaround. The "
        "user can enable it from the app."
    )


__all__ = [
    "BY_ID",
    "CAPABILITIES",
    "Capability",
    "capability_state",
    "manifest",
    "missing",
    "state_for_agent",
    "suggest",
]
