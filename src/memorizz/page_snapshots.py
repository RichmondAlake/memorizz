# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""A snapshot of a page an agent made, shown to the person in the reply.

An agent that creates or changes a page (a Notion page, a board) calls
``send_page_snapshot`` with its address. A Notion page is read as it is now,
through the agent's own Notion connection or, where one is set, a Notion
integration token, and sent to the app as a small ``display``: its title,
properties and first blocks. The snapshot goes to the person, never to the
model, so it shows what is on the page rather than what the model says is.
Any other page becomes a link card with the title and summary the agent gave.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from .tool_outcomes import MAX_DISPLAY_CHARS, ToolOutcome, ToolOutcomeStatus, ToolResult

logger = logging.getLogger(__name__)

KIND = "page_snapshot"
MAX_BLOCKS = 40
MAX_BLOCK_CHARS = 280
MAX_PROPERTIES = 8
# Room left in a display for its other fields.
_BLOCK_BUDGET = MAX_DISPLAY_CHARS - 1500

_NOTION_HOSTS = ("notion.so", "notion.com", "notion.site")
_ID_IN_PATH = re.compile(
    r"([0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)


def _clip(value: Any, limit: int = MAX_BLOCK_CHARS) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def web_address(url: Any) -> Optional[str]:
    """``url`` when it is a plain web address, else None."""
    text = str(url or "").strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    if parts.username or parts.password:
        return None
    return text[:600]


def notion_page_id(url: Any) -> Optional[str]:
    """The page (or database) ID in a Notion address, as a dashed UUID."""
    address = web_address(url)
    if not address:
        return None
    parts = urlsplit(address)
    host = (parts.hostname or "").lower()
    if not any(host == name or host.endswith("." + name) for name in _NOTION_HOSTS):
        return None
    # The ID ends the last path segment ("Title-<32 hex>"); a "p=" query
    # names the page open in a database's side peek.
    for candidate in (parts.query, parts.path.rstrip("/").rsplit("/", 1)[-1]):
        found = _ID_IN_PATH.findall(candidate or "")
        if found:
            try:
                return str(uuid.UUID(found[-1]))
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Notion-flavoured Markdown (what Notion's MCP server returns) -> blocks
# ---------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_INLINE_MARKS = re.compile(r"(\*\*|__|~~|`)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_PAGE_REF = re.compile(
    r"<(page|database)\b[^>]*?(?:url=\"([^\"]*)\")?[^>]*>(.*?)</\1>", re.IGNORECASE
)


def _inline(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _TAG.sub("", text)
    return _clip(_INLINE_MARKS.sub("", text))


def blocks_from_markdown(
    markdown: Any, limit: int = MAX_BLOCKS
) -> List[Dict[str, Any]]:
    """The first ``limit`` blocks of a page written in Notion's Markdown."""
    blocks: List[Dict[str, Any]] = []
    fence: Optional[List[str]] = None
    for raw in str(markdown or "").splitlines():
        if len(blocks) >= limit:
            break
        line = raw.rstrip()
        stripped = line.strip()
        if fence is not None:
            if stripped.startswith("```"):
                blocks.append({"type": "code", "text": "\n".join(fence)[:600]})
                fence = None
            elif len(fence) < 12:
                fence.append(line[:160])
            continue
        if stripped.startswith("```"):
            fence = []
            continue
        if not stripped:
            continue
        reference = _PAGE_REF.fullmatch(stripped)
        if reference:
            blocks.append(
                {
                    "type": reference.group(1).lower(),
                    "text": _inline(reference.group(3)) or "Untitled",
                }
            )
            continue
        if re.fullmatch(r"(-{3,}|\*{3,}|<divider\s*/?>)", stripped):
            blocks.append({"type": "divider"})
            continue
        heading = re.match(r"(#{1,3})\s+(.*)", stripped)
        if heading:
            blocks.append(
                {
                    "type": "heading",
                    "level": len(heading.group(1)),
                    "text": _inline(heading.group(2)),
                }
            )
            continue
        todo = re.match(r"[-*]\s+\[( |x|X)\]\s+(.*)", stripped)
        if todo:
            blocks.append(
                {
                    "type": "todo",
                    "checked": todo.group(1) != " ",
                    "text": _inline(todo.group(2)),
                }
            )
            continue
        bullet = re.match(r"[-*+]\s+(.*)", stripped)
        if bullet:
            blocks.append({"type": "bullet", "text": _inline(bullet.group(1))})
            continue
        numbered = re.match(r"\d+[.)]\s+(.*)", stripped)
        if numbered:
            blocks.append({"type": "number", "text": _inline(numbered.group(1))})
            continue
        if stripped.startswith(">"):
            blocks.append({"type": "quote", "text": _inline(stripped.lstrip("> "))})
            continue
        if stripped.startswith("|"):
            cells = [_inline(cell) for cell in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", cell or "--") for cell in cells):
                continue  # the rule under a table's header
            blocks.append({"type": "row", "cells": [cell[:60] for cell in cells[:6]]})
            continue
        callout = re.match(r"<callout\b[^>]*>(.*)", stripped, re.IGNORECASE)
        if callout:
            text = _inline(callout.group(1))
            if text:
                blocks.append({"type": "callout", "text": text})
            continue
        text = _inline(stripped)
        if text:
            blocks.append({"type": "paragraph", "text": text})
    if fence:
        blocks.append({"type": "code", "text": "\n".join(fence)[:600]})
    return blocks[:limit]


def _fetched_text(result: Any) -> str:
    """The text of an MCP tool result (``content`` items, or a plain value)."""
    payload = result.get("result") if isinstance(result, dict) else result
    if isinstance(payload, dict):
        parts = [
            str(item.get("text") or "")
            for item in payload.get("content") or []
            if isinstance(item, dict) and item.get("type", "text") == "text"
        ]
        if parts:
            return "\n".join(parts)
        structured = payload.get("structuredContent")
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False)
    return str(payload or "")


def page_from_notion_fetch(text: str) -> Dict[str, Any]:
    """Title, properties and blocks from what Notion's ``fetch`` tool returns:
    JSON holding a ``<page>`` document, or that document itself."""
    title = ""
    document = text
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        title = str(parsed.get("title") or "")
        document = str(parsed.get("text") or parsed.get("content") or "")
    properties: List[Dict[str, str]] = []
    listed = re.search(r"<properties>\s*(\{.*?\})\s*</properties>", document, re.S)
    if listed:
        try:
            values = json.loads(listed.group(1))
        except ValueError:
            values = {}
        for name, value in values.items() if isinstance(values, dict) else ():
            if str(name).lower() in {"title", "name"}:
                title = title or str(value or "")
                continue
            if value in (None, "", [], {}):
                continue
            shown = (
                ", ".join(str(item) for item in value)
                if isinstance(value, list)
                else str(value)
            )
            properties.append({"name": _clip(name, 40), "value": _clip(shown, 80)})
    content = re.search(r"<content>(.*?)</content>", document, re.S)
    body = (
        content.group(1)
        if content
        else re.sub(r"<(properties|ancestor-path)>.*?</\1>", "", document, flags=re.S)
    )
    return {
        "title": _clip(title, 160),
        "properties": properties[:MAX_PROPERTIES],
        "blocks": blocks_from_markdown(body),
    }


# ---------------------------------------------------------------------------
# Notion's REST API (an integration token) -> blocks
# ---------------------------------------------------------------------------


def _plain(rich: Any) -> str:
    return "".join(
        str(item.get("plain_text") or (item.get("text") or {}).get("content") or "")
        for item in rich or []
        if isinstance(item, dict)
    )


def _property_text(prop: Dict[str, Any]) -> str:
    kind = prop.get("type")
    value = prop.get(kind)
    if kind in {"rich_text", "title"}:
        return _plain(value)
    if kind in {"select", "status"}:
        return str((value or {}).get("name") or "")
    if kind == "multi_select":
        return ", ".join(str(item.get("name") or "") for item in value or [])
    if kind == "date":
        start, end = (value or {}).get("start"), (value or {}).get("end")
        return " → ".join(str(part) for part in (start, end) if part)
    if kind == "people":
        return ", ".join(str(item.get("name") or "") for item in value or [])
    if kind == "checkbox":
        return "Yes" if value else "No"
    if kind in {"number", "url", "email", "phone_number"}:
        return "" if value is None else str(value)
    return ""


def blocks_from_notion_api(children: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for child in children or []:
        if len(blocks) >= MAX_BLOCKS or not isinstance(child, dict):
            break
        kind = str(child.get("type") or "")
        data = child.get(kind) or {}
        text = _clip(_plain(data.get("rich_text")))
        if kind.startswith("heading_"):
            blocks.append({"type": "heading", "level": int(kind[-1]), "text": text})
        elif kind == "bulleted_list_item":
            blocks.append({"type": "bullet", "text": text})
        elif kind == "numbered_list_item":
            blocks.append({"type": "number", "text": text})
        elif kind == "to_do":
            blocks.append(
                {"type": "todo", "checked": bool(data.get("checked")), "text": text}
            )
        elif kind == "quote":
            blocks.append({"type": "quote", "text": text})
        elif kind == "callout":
            blocks.append({"type": "callout", "text": text})
        elif kind == "code":
            blocks.append({"type": "code", "text": _plain(data.get("rich_text"))[:600]})
        elif kind == "divider":
            blocks.append({"type": "divider"})
        elif kind in {"child_page", "child_database"}:
            blocks.append(
                {
                    "type": "page" if kind == "child_page" else "database",
                    "text": _clip(data.get("title")) or "Untitled",
                }
            )
        elif text:
            blocks.append({"type": "paragraph", "text": text})
    return blocks


def _page_from_notion_api(client: Any, page_id: str) -> Dict[str, Any]:
    page = client.request("GET", "/pages/" + page_id)
    title, properties = "", []
    for name, prop in (page.get("properties") or {}).items():
        if not isinstance(prop, dict):
            continue
        if prop.get("type") == "title":
            title = _plain(prop.get("title"))
            continue
        shown = _property_text(prop)
        if shown and not str(name).startswith("Memorizz"):
            properties.append({"name": _clip(name, 40), "value": _clip(shown, 80)})
    children = client.request(
        "GET", "/blocks/" + page_id + "/children", params={"page_size": 50}
    )
    icon = page.get("icon") or {}
    return {
        "title": _clip(title, 160),
        "icon": str(icon.get("emoji") or "")[:8],
        "url": str(page.get("url") or ""),
        "properties": properties[:MAX_PROPERTIES],
        "blocks": blocks_from_notion_api(children.get("results")),
    }


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------


def _notion_server(mcp_manager: Any) -> Optional[str]:
    try:
        servers = mcp_manager.server_dicts()
    except Exception:
        return None
    for server in servers:
        if not server.get("enabled", True):
            continue
        url = str(server.get("url") or "").lower()
        name = str(server.get("name") or "")
        if "notion" in url or "notion" in name.lower():
            return name
    return None


def _read_notion_page(
    url: str, page_id: str, mcp_manager: Any, notion_client: Any
) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(page, how it was read)``; ``(None, why not)`` when it was not."""
    why = "no Notion connection to read it with"
    server = _notion_server(mcp_manager) if mcp_manager is not None else None
    if server:
        try:
            result = mcp_manager.call_tool(
                server_name=server, tool_name="notion-fetch", arguments={"id": url}
            )
        except Exception as exc:  # a refused call is a reason, not a crash
            result = {"ok": False, "error": type(exc).__name__}
        if isinstance(result, dict) and result.get("ok"):
            page = page_from_notion_fetch(_fetched_text(result))
            if page["title"] or page["blocks"]:
                return page, "notion"
            why = "Notion returned an empty page"
        elif isinstance(result, dict) and result.get("error_code") == (
            "authorization_required"
        ):
            why = f"{server} needs signing in again"
        else:
            why = "Notion did not return the page"
    client, owned = notion_client, False
    if client is None and os.getenv("NOTION_TOKEN"):
        try:
            from .memory_provider.notion.client import NotionClient

            client, owned = NotionClient(os.environ["NOTION_TOKEN"]), True
        except Exception:
            client = None
    if client is not None:
        try:
            return _page_from_notion_api(client, page_id), "notion"
        except Exception as exc:
            logger.debug("Notion page %s was not read: %s", page_id, exc)
            why = "the Notion token cannot read this page"
        finally:
            if owned:
                try:
                    client.close()
                except Exception:
                    pass
    return None, why


def snapshot_display(
    url: str,
    *,
    title: str = "",
    summary: str = "",
    source: str = "web",
    icon: str = "",
    properties: Optional[List[Dict[str, str]]] = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
    read: bool = False,
    unread_because: str = "",
) -> Dict[str, Any]:
    """The display for one page, cut to what a trace event may carry."""
    host = (urlsplit(url).hostname or "").removeprefix("www.")
    display: Dict[str, Any] = {
        "kind": KIND,
        "id": url,
        "url": url,
        "title": _clip(title, 160) or host or url,
        "summary": _clip(summary, 400),
        "source": source,
        "site": host,
        "icon": icon,
        "properties": list(properties or [])[:MAX_PROPERTIES],
        "blocks": [],
        "read": bool(read),
        "unread_because": _clip(unread_because, 160),
        "truncated": False,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    used = 0
    for block in list(blocks or [])[:MAX_BLOCKS]:
        size = len(json.dumps(block, ensure_ascii=False))
        if used + size > _BLOCK_BUDGET:
            display["truncated"] = True
            break
        display["blocks"].append(block)
        used += size
    if len(blocks or []) > len(display["blocks"]):
        display["truncated"] = True
    return display


def take_snapshot(
    url: Any,
    *,
    title: str = "",
    summary: str = "",
    mcp_manager: Any = None,
    notion_client: Any = None,
) -> ToolResult:
    """What ``send_page_snapshot`` returns: a short note for the model and the
    snapshot for the person."""
    address = web_address(url)
    if not address:
        return ToolResult(
            value={
                "ok": False,
                "error": "Give the page's web address (https://…).",
                "error_code": "invalid_arguments",
            },
            outcome=ToolOutcome(
                status=ToolOutcomeStatus.ERROR, reason_code="invalid_arguments"
            ),
        )
    page_id = notion_page_id(address)
    page, how = (None, "")
    if page_id:
        page, how = _read_notion_page(address, page_id, mcp_manager, notion_client)
    if page is not None:
        display = snapshot_display(
            address,
            title=page.get("title") or title,
            summary=summary,
            source="notion",
            icon=page.get("icon") or "",
            properties=page.get("properties"),
            blocks=page.get("blocks"),
            read=True,
        )
    else:
        display = snapshot_display(
            address,
            title=title,
            summary=summary,
            source="notion" if page_id else "web",
            unread_because=how if page_id else "",
        )
    value: Dict[str, Any] = {
        "ok": True,
        "shown": True,
        "title": display["title"],
        "read_from_page": display["read"],
        "next": (
            "The user can see the snapshot. Give the link and one line on what "
            "you did; do not paste the page's content."
        ),
    }
    if page_id and not display["read"]:
        value["note"] = (
            f"The page itself could not be read ({how}), so the user sees a "
            "link with your title and summary."
        )
    return ToolResult(
        value=value,
        outcome=ToolOutcome(status=ToolOutcomeStatus.SUCCESS, result_count=1),
        display=display,
    )


__all__ = [
    "KIND",
    "blocks_from_markdown",
    "blocks_from_notion_api",
    "notion_page_id",
    "page_from_notion_fetch",
    "snapshot_display",
    "take_snapshot",
    "web_address",
]
