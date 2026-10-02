# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Find MCP servers to connect: built-in presets and the official MCP Registry.

Every result carries ready-to-save server configurations with secrets left
blank. Nothing is saved or started here; callers pass a configuration to
``MCPClientManager.upsert_server`` once the user has reviewed it.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .security import validate_remote_url

REGISTRY_URL = "https://registry.modelcontextprotocol.io"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8765/api/mcp/oauth/callback"
USER_AGENT = "Memorizz-MCP-Catalog/1.0"

NOTION_URL = "https://mcp.notion.com/mcp"
GOOGLE_CALENDAR_URL = "https://calendarmcp.googleapis.com/mcp/v1"
GMAIL_URL = "https://gmailmcp.googleapis.com/mcp/v1"
GOOGLE_CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    "https://www.googleapis.com/auth/calendar.events.freebusy",
    "https://www.googleapis.com/auth/calendar.events.readonly",
]
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "notion": {
        "title": "Notion",
        "description": "Search, read and update pages and databases in your Notion workspace.",
        "url": NOTION_URL,
        "scopes": [],
        "needs_client": False,
        "setup": "Sign in with Notion. No OAuth client is needed.",
        "docs": "https://developers.notion.com/guides/mcp/get-started-with-mcp",
    },
    "google-calendar": {
        "name": "calendar",
        "title": "Google Calendar",
        "description": "List calendars, read events and check free/busy times (read-only scopes).",
        "url": GOOGLE_CALENDAR_URL,
        "scopes": GOOGLE_CALENDAR_SCOPES,
        "needs_client": True,
        "setup": (
            "Google Developer Preview. Enable the Calendar API and its MCP "
            "server, then create a Web OAuth client with this app's redirect URI."
        ),
        "docs": "https://developers.google.com/workspace/calendar/api/guides/configure-mcp-server",
    },
    "gmail": {
        "title": "Gmail",
        "description": "Search and read threads, manage labels and create drafts. It cannot send mail.",
        "url": GMAIL_URL,
        "scopes": GMAIL_SCOPES,
        "needs_client": True,
        "setup": (
            "Google Developer Preview. Enable gmail.googleapis.com and "
            "gmailmcp.googleapis.com, then create a Web OAuth client with this "
            "app's redirect URI. One client can serve Gmail and Calendar."
        ),
        "docs": "https://developers.google.com/workspace/gmail/api/guides/configure-mcp-server",
    },
}
PRESET_ALIASES = {"calendar": "google-calendar", "google": "google-calendar"}

_PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_.-]+)\}")
_RUNNERS = {"npm": "npx", "pypi": "uvx", "oci": "docker"}
_REMOTE_TRANSPORTS = {"streamable-http": "streamable_http", "sse": "sse"}


def preset_key(value: str) -> str:
    key = str(value or "").strip().lower().replace("_", "-")
    key = PRESET_ALIASES.get(key, key)
    if key not in PRESETS:
        raise ValueError(
            f"Unknown MCP preset '{value}'. Use one of: {', '.join(PRESETS)}"
        )
    return key


def preset_config(
    key: str,
    *,
    name: Optional[str] = None,
    redirect_uri: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> Dict[str, Any]:
    """A server configuration for a built-in preset, ready for ``upsert_server``."""
    key = preset_key(key)
    preset = PRESETS[key]
    auth: Dict[str, Any] = {
        "type": "oauth",
        "redirect_uri": redirect_uri or DEFAULT_REDIRECT_URI,
        "scopes": list(preset["scopes"]),
    }
    if client_id:
        auth["client_id"] = client_id
    if client_secret:
        auth["client_secret"] = client_secret
    return {
        "name": name or preset.get("name") or key,
        "transport": "streamable_http",
        "url": preset["url"],
        "require_approval": True,
        "auth": auth,
    }


def presets(query: str = "") -> List[Dict[str, Any]]:
    """Built-in presets, optionally filtered by a search query."""
    needle = str(query or "").strip().lower()
    rows = []
    for key, preset in PRESETS.items():
        haystack = f"{key} {preset['title']} {preset['description']}".lower()
        if needle and not all(word in haystack for word in needle.split()):
            continue
        rows.append(
            {
                "key": key,
                "name": preset.get("name") or key,
                "title": preset["title"],
                "description": preset["description"],
                "url": preset["url"],
                "scopes": list(preset["scopes"]),
                "needs_client": preset["needs_client"],
                "setup": preset["setup"],
                "docs": preset["docs"],
            }
        )
    return rows


def _registry_base() -> str:
    return os.environ.get("MEMORIZZ_MCP_REGISTRY_URL", REGISTRY_URL).rstrip("/")


def _get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def server_slug(registry_name: str) -> str:
    """A connection name from a registry name such as ``io.github.acme/linear-mcp``."""
    tail = str(registry_name or "").rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", tail).strip("-._")
    slug = re.sub(r"(?:^|-)mcp(?:-server)?$|^mcp-", "", slug).strip("-._") or slug
    return (slug or "server")[:64]


def _input_row(item: Dict[str, Any], kind: str, name: str) -> Dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "description": str(item.get("description") or "")[:240],
        "required": bool(item.get("isRequired")),
        "secret": bool(item.get("isSecret")),
    }


def _remote_option(remote: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    transport = _REMOTE_TRANSPORTS.get(str(remote.get("type") or ""))
    url = str(remote.get("url") or "").strip()
    if not transport or not url.startswith(("https://", "http://")):
        return None
    auth: Dict[str, Any] = {"type": "none"}
    headers: Dict[str, str] = {}
    inputs: List[Dict[str, Any]] = []
    for header in remote.get("headers") or []:
        if not isinstance(header, dict) or not header.get("name"):
            continue
        name = str(header["name"])
        value = str(header.get("value") or "")
        if name.lower() == "authorization" and "bearer" in value.lower():
            auth = {"type": "bearer"}
            inputs.append({**_input_row(header, "token", "token"), "secret": True})
        elif value and not _PLACEHOLDER.search(value) and not header.get("isSecret"):
            headers[name] = value
        else:
            headers[name] = ""
            inputs.append(_input_row(header, "header", name))
    config: Dict[str, Any] = {"transport": transport, "url": url, "auth": auth}
    if headers:
        config["headers"] = headers
    return {
        "kind": "remote",
        "label": "Hosted" if transport == "streamable_http" else "Hosted (SSE)",
        "config": config,
        "inputs": inputs,
        "placeholders": sorted(set(_PLACEHOLDER.findall(url))),
        "auth_known": auth["type"] != "none",
    }


def _argument_values(arguments: List[Any], placeholders: List[str]) -> List[str]:
    values: List[str] = []
    for argument in arguments or []:
        if not isinstance(argument, dict):
            continue
        value = argument.get("value") or argument.get("default")
        if not value:
            if not argument.get("isRequired"):
                continue
            value = (
                "{"
                + str(argument.get("name") or argument.get("valueHint") or "value")
                + "}"
            )
            placeholders.append(value.strip("{}"))
        if argument.get("type") == "named" and argument.get("name"):
            values.extend([str(argument["name"]), str(value)])
        else:
            values.append(str(value))
    return values


def _package_option(package: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    registry_type = str(package.get("registryType") or "")
    identifier = str(package.get("identifier") or "").strip()
    transport = (package.get("transport") or {}).get("type")
    if registry_type not in _RUNNERS or not identifier or transport != "stdio":
        return None
    version = str(package.get("version") or "").strip()
    env_rows = [
        item
        for item in package.get("environmentVariables") or []
        if isinstance(item, dict) and item.get("name")
    ]
    placeholders: List[str] = []
    if registry_type == "npm":
        args = ["-y", f"{identifier}@{version}" if version else identifier]
    elif registry_type == "pypi":
        args = [f"{identifier}=={version}" if version else identifier]
    else:
        args = ["run", "-i", "--rm"]
        for item in env_rows:
            args.extend(["-e", str(item["name"])])
        args.append(identifier)
    args.extend(_argument_values(package.get("packageArguments") or [], placeholders))
    env = {
        str(item["name"]): (
            ""
            if item.get("isSecret") or not item.get("default")
            else str(item["default"])
        )
        for item in env_rows
    }
    inputs = [
        _input_row(item, "env", str(item["name"]))
        for item in env_rows
        if item.get("isSecret") or not item.get("default")
    ]
    labels = {"npm": "npm", "pypi": "PyPI", "oci": "Docker"}
    return {
        "kind": registry_type,
        "label": f"Local ({labels[registry_type]})",
        "config": {
            "transport": "stdio",
            "command": _RUNNERS[registry_type],
            "args": args,
            "env": env,
            "auth": {"type": "none"},
        },
        "inputs": inputs,
        "placeholders": sorted(set(placeholders)),
        "auth_known": True,
    }


def registry_entry(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One registry server as a catalog row with its connection options."""
    server = raw.get("server") if isinstance(raw.get("server"), dict) else raw
    name = str(server.get("name") or "").strip()
    if not name:
        return None
    meta = (raw.get("_meta") or {}).get("io.modelcontextprotocol.registry/official")
    if isinstance(meta, dict) and meta.get("status") not in (None, "active"):
        return None
    options = [
        option
        for option in (
            *(_remote_option(item) for item in server.get("remotes") or []),
            *(_package_option(item) for item in server.get("packages") or []),
        )
        if option
    ]
    repository = (
        server.get("repository") if isinstance(server.get("repository"), dict) else {}
    )
    slug = server_slug(name)
    for option in options:
        option["config"]["name"] = slug
    return {
        "registry_name": name,
        "name": slug,
        "title": str(server.get("title") or slug),
        "description": str(server.get("description") or "")[:300],
        "version": str(server.get("version") or ""),
        "repository": str(repository.get("url") or ""),
        "website": str(server.get("websiteUrl") or ""),
        "options": options,
    }


_SEARCH_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
SEARCH_CACHE_SECONDS = 600
_UNAVAILABLE = (
    "The MCP Registry did not answer ({reason}). It is sometimes slow; try "
    "again in a minute, or add the server by URL with Add connection."
)


def search_registry(
    query: str,
    *,
    limit: int = 20,
    cursor: Optional[str] = None,
    timeout: float = 45.0,
    attempts: int = 2,
) -> Dict[str, Any]:
    """Search the MCP Registry; servers with no usable option are left out.

    The public registry can take 30 seconds or more for an uncached query and
    sometimes returns 5xx. A 5xx is retried once, a timeout is not, and
    answers are cached for ten minutes.
    """
    params = {"limit": str(max(1, min(int(limit or 20), 100))), "version": "latest"}
    if str(query or "").strip():
        params["search"] = str(query).strip()
    if cursor:
        params["cursor"] = str(cursor)
    url = f"{_registry_base()}/v0.1/servers?{urllib.parse.urlencode(params)}"
    cached = _SEARCH_CACHE.get(url)
    if cached and time.monotonic() - cached[0] < SEARCH_CACHE_SECONDS:
        return cached[1]
    reason = "no response"
    data: Any = None
    for _ in range(max(1, attempts)):
        try:
            data = _get_json(url, timeout)
            break
        except urllib.error.HTTPError as exc:
            reason = f"HTTP {exc.code}"
            if exc.code < 500:
                break
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            break
    if data is None:
        return {"ok": False, "error": _UNAVAILABLE.format(reason=reason)}
    servers: List[Dict[str, Any]] = []
    seen = set()
    for raw in data.get("servers") or [] if isinstance(data, dict) else []:
        entry = registry_entry(raw) if isinstance(raw, dict) else None
        if not entry or not entry["options"] or entry["registry_name"] in seen:
            continue
        seen.add(entry["registry_name"])
        servers.append(entry)
    metadata = data.get("metadata") if isinstance(data, dict) else {}
    result = {
        "ok": True,
        "query": str(query or ""),
        "servers": servers,
        "next_cursor": (metadata or {}).get("nextCursor"),
    }
    if len(_SEARCH_CACHE) > 200:
        _SEARCH_CACHE.clear()
    _SEARCH_CACHE[url] = (time.monotonic(), result)
    return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - urllib hook
        return None


def detect_remote_auth(url: str, *, timeout: float = 6.0) -> str:
    """Guess how a hosted server authenticates: ``oauth``, ``bearer`` or ``none``.

    Follows the MCP authorization spec: protected-resource metadata means
    OAuth, and so does a 401/403 challenge that names it. Any other refusal of
    an anonymous ``initialize`` means a token. Redirects are not followed.
    """
    validate_remote_url(url)
    parsed = urllib.parse.urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    opener = urllib.request.build_opener(_NoRedirect)
    path = parsed.path.rstrip("/")
    for candidate in dict.fromkeys(
        (
            f"{origin}/.well-known/oauth-protected-resource{path}",
            f"{origin}/.well-known/oauth-protected-resource",
        )
    ):
        request = urllib.request.Request(
            candidate, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                data = json.loads(
                    response.read(65536).decode("utf-8", errors="replace")
                )
            if isinstance(data, dict) and data.get("authorization_servers"):
                return "oauth"
        except Exception:
            continue
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "memorizz-catalog", "version": "1"},
            },
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with opener.open(request, timeout=timeout):
            return "none"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            # OAuth servers name their metadata in the challenge; anything
            # else that refuses an anonymous client wants a static token.
            challenge = str(exc.headers.get("WWW-Authenticate") or "").lower()
            return "oauth" if "resource_metadata" in challenge else "bearer"
        return "none"
    except Exception:
        return "none"


__all__ = [
    "GMAIL_SCOPES",
    "GOOGLE_CALENDAR_SCOPES",
    "PRESETS",
    "detect_remote_auth",
    "preset_config",
    "preset_key",
    "presets",
    "registry_entry",
    "search_registry",
    "server_slug",
]
