"""A page an agent made is shown to the person as it reads now."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memorizz.page_snapshots import (
    KIND,
    blocks_from_markdown,
    blocks_from_notion_api,
    notion_page_id,
    page_from_notion_fetch,
    snapshot_display,
    take_snapshot,
    web_address,
)
from memorizz.tool_outcomes import MAX_DISPLAY_CHARS, result_display

pytestmark = pytest.mark.unit

PAGE = "https://www.notion.so/acme/Launch-plan-4230b4b34fc881a0a0aef689c1ed7c5f?pvs=4"
PAGE_ID = "4230b4b3-4fc8-81a0-a0ae-f689c1ed7c5f"
FETCHED = json.dumps(
    {
        "title": "Launch plan",
        "url": PAGE,
        "text": (
            'Here is the result of "view" for the Page:\n'
            f'<page url="{PAGE}">\n<ancestor-path>Projects</ancestor-path>\n'
            '<properties>\n{"title":"Launch plan","Status":"In progress",'
            '"Due":"2026-12-01","Tags":["web","q4"],"Notes":""}\n</properties>\n'
            "<content>\n# Goal\nShip **v2** by [Friday](https://x.example).\n"
            "- [x] Draft the brief\n- [ ] Book the review\n"
            '<page url="https://www.notion.so/s12">Sprint 12</page>\n'
            "</content>\n</page>"
        ),
    }
)


class NotionMCP:
    """An agent's MCP manager with one Notion server."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def server_dicts(self):
        return [
            {"name": "calendar", "url": "https://calendarmcp.googleapis.com/mcp/v1"},
            {"name": "notion", "url": "https://mcp.notion.com/mcp", "enabled": True},
        ]

    def call_tool(self, **call):
        self.calls.append(call)
        return self.answer


def test_only_plain_web_and_notion_addresses_are_taken():
    assert notion_page_id(PAGE) == PAGE_ID
    assert notion_page_id("https://app.notion.com/p/4230b4b34fc881ef8152c6051e81f7dc")
    assert notion_page_id(f"https://acme.notion.site/{PAGE_ID}") == PAGE_ID
    # Another site's address that happens to hold an ID is not Notion.
    assert notion_page_id("https://notion.so.example.com/" + PAGE_ID) is None
    assert notion_page_id("https://www.notion.so/acme/No-id-here") is None
    for bad in ("javascript:alert(1)", "file:///etc/passwd", "notion.so/x", "", None):
        assert web_address(bad) is None
    assert web_address("https://user:secret@example.com/x") is None


def test_notions_markdown_becomes_the_blocks_of_a_page():
    blocks = blocks_from_markdown(
        "# Plan\nShip **v2** by [Friday](https://x.example).\n## Tasks\n"
        "- [x] Brief\n- [ ] Review\n- A bullet\n1. First\n> Quoted\n"
        "| Task | Owner |\n|---|---|\n| Brief | Ada |\n---\n"
        '<database url="https://www.notion.so/d">Tasks</database>\n'
        '```python\nprint(1)\n```\n<callout icon="!">Mind the date</callout>'
    )
    assert blocks == [
        {"type": "heading", "level": 1, "text": "Plan"},
        {"type": "paragraph", "text": "Ship v2 by Friday."},
        {"type": "heading", "level": 2, "text": "Tasks"},
        {"type": "todo", "checked": True, "text": "Brief"},
        {"type": "todo", "checked": False, "text": "Review"},
        {"type": "bullet", "text": "A bullet"},
        {"type": "number", "text": "First"},
        {"type": "quote", "text": "Quoted"},
        {"type": "row", "cells": ["Task", "Owner"]},
        {"type": "row", "cells": ["Brief", "Ada"]},
        {"type": "divider"},
        {"type": "database", "text": "Tasks"},
        {"type": "code", "text": "print(1)"},
        {"type": "callout", "text": "Mind the date"},
    ]
    # A long page is cut, not sent whole.
    assert len(blocks_from_markdown("\n".join(f"- item {n}" for n in range(200)))) == 40


def test_a_notion_page_is_read_through_the_agents_own_connection():
    manager = NotionMCP(
        {"ok": True, "result": {"content": [{"type": "text", "text": FETCHED}]}}
    )

    result = take_snapshot(PAGE, title="what the model said", mcp_manager=manager)

    assert manager.calls == [
        {
            "server_name": "notion",
            "tool_name": "notion-fetch",
            "arguments": {"id": PAGE},
        }
    ]
    display = result_display(result)
    assert display["kind"] == KIND and display["source"] == "notion"
    # The title and content are the page's, not the model's account of them.
    assert display["title"] == "Launch plan" and display["read"] is True
    assert display["properties"] == [
        {"name": "Status", "value": "In progress"},
        {"name": "Due", "value": "2026-12-01"},
        {"name": "Tags", "value": "web, q4"},
    ]
    assert display["blocks"] == [
        {"type": "heading", "level": 1, "text": "Goal"},
        {"type": "paragraph", "text": "Ship v2 by Friday."},
        {"type": "todo", "checked": True, "text": "Draft the brief"},
        {"type": "todo", "checked": False, "text": "Book the review"},
        {"type": "page", "text": "Sprint 12"},
    ]
    # The model is told it was shown, and is not handed the page.
    assert result.value["ok"] and result.value["read_from_page"] is True
    assert "Draft the brief" not in json.dumps(result.value)
    assert page_from_notion_fetch(FETCHED)["title"] == "Launch plan"


def test_a_page_that_cannot_be_read_is_shown_as_a_link_and_says_why(monkeypatch):
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    refused = NotionMCP({"ok": False, "error_code": "authorization_required"})

    result = take_snapshot(
        PAGE, title="Launch plan", summary="The plan.", mcp_manager=refused
    )

    display = result.display
    assert display["read"] is False and display["blocks"] == []
    assert (display["title"], display["summary"]) == ("Launch plan", "The plan.")
    assert display["unread_because"] == "notion needs signing in again"
    assert "could not be read" in result.value["note"]

    # A page that is not Notion's is never fetched: a link card.
    other = take_snapshot("https://example.com/spec", title="Spec", mcp_manager=refused)
    assert other.display["source"] == "web" and other.display["site"] == "example.com"
    assert len(refused.calls) == 1 and "note" not in other.value

    bad = take_snapshot("javascript:alert(1)")
    assert bad.value["ok"] is False and bad.display is None


def test_a_notion_token_reads_the_page_when_no_connection_can():
    asked = []

    def request(method, path, **kwargs):
        asked.append((method, path))
        if path.startswith("/pages/"):
            return {
                "url": "https://www.notion.so/" + PAGE_ID.replace("-", ""),
                "icon": {"type": "emoji", "emoji": "🚀"},
                "properties": {
                    "Name": {"type": "title", "title": [{"plain_text": "Launch plan"}]},
                    "Status": {"type": "status", "status": {"name": "In progress"}},
                    "Owner": {"type": "people", "people": [{"name": "Ada"}]},
                    "Done": {"type": "checkbox", "checkbox": False},
                    "Memorizz ID": {
                        "type": "rich_text",
                        "rich_text": [{"plain_text": "x"}],
                    },
                },
            }
        return {
            "results": [
                {
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"plain_text": "Tasks"}]},
                },
                {
                    "type": "to_do",
                    "to_do": {"checked": True, "rich_text": [{"plain_text": "Brief"}]},
                },
                {"type": "child_page", "child_page": {"title": "Sprint 12"}},
                {"type": "divider", "divider": {}},
            ]
        }

    result = take_snapshot(PAGE, notion_client=SimpleNamespace(request=request))

    assert asked == [
        ("GET", "/pages/" + PAGE_ID),
        ("GET", f"/blocks/{PAGE_ID}/children"),
    ]
    display = result.display
    assert display["read"] is True and display["icon"] == "🚀"
    assert display["title"] == "Launch plan"
    assert display["properties"] == [
        {"name": "Status", "value": "In progress"},
        {"name": "Owner", "value": "Ada"},
        {"name": "Done", "value": "No"},
    ]
    assert display["blocks"] == [
        {"type": "heading", "level": 2, "text": "Tasks"},
        {"type": "todo", "checked": True, "text": "Brief"},
        {"type": "page", "text": "Sprint 12"},
        {"type": "divider"},
    ]
    assert blocks_from_notion_api(None) == []


def test_a_snapshot_always_fits_what_a_reply_may_carry():
    long = [{"type": "paragraph", "text": "x" * 280} for _ in range(40)]
    display = snapshot_display(PAGE, title="Big", blocks=long, read=True)
    assert display["truncated"] is True and 0 < len(display["blocks"]) < 40
    assert len(json.dumps(display)) < MAX_DISPLAY_CHARS


def test_an_agent_with_a_connection_can_send_a_snapshot(monkeypatch):
    pytest.importorskip("mcp")
    from memorizz import MemAgent

    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    agent = MemAgent(
        memory_provider=False,
        auto_register=False,
        mcp_servers=[
            {
                "name": "notion",
                "transport": "streamable_http",
                "url": "https://mcp.notion.com/mcp",
            }
        ],
    )
    try:
        assert "send_page_snapshot" in agent.tool_manager.list_tools()
        # Offered on every turn: it is the last step of making a page.
        assert "send_page_snapshot" in agent.semantic_tool_router.always_visible
        agent.mcp_manager = NotionMCP(
            {"ok": True, "result": {"content": [{"type": "text", "text": FETCHED}]}}
        )
        result, _ = agent.tool_manager.execute_tool("send_page_snapshot", {"url": PAGE})
        assert result_display(result)["title"] == "Launch plan"
    finally:
        agent.close()

    plain = MemAgent(memory_provider=False, auto_register=False)
    try:
        assert "send_page_snapshot" not in plain.tool_manager.list_tools()
    finally:
        plain.close()
