"""Signing in to a connected app from the conversation, not another page."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

NOTION = {
    "name": "notion",
    "transport": "streamable_http",
    "url": "https://mcp.notion.com/mcp",
    "auth": {
        "type": "oauth",
        "redirect_uri": "http://testserver/api/mcp/oauth/callback",
    },
}


@pytest.fixture()
def console(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")
    from fastapi.testclient import TestClient

    from memorizz.memagent.models import MemAgentModel
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.ui import state
    from memorizz.ui.app import create_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_ENV_FILE", str(tmp_path / "home" / ".env"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    provider.store_memagent(
        MemAgentModel(agent_id="pm", name="Project manager", mcp_servers=[NOTION])
    )
    provider.store_memagent(MemAgentModel(agent_id="plain", name="Plain"))
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "read_only": False,
    }
    try:
        with patch.dict(state._state, values):
            yield TestClient(create_app(), follow_redirects=False)
    finally:
        provider.close()


def _needed(page: str):
    return json.loads(re.search(r"pgDrawSignIn\((\[.*?\])\);", page, re.S).group(1))


def test_the_playground_carries_a_sign_in_bar_for_a_signed_out_app(console):
    page = console.get("/agents/pm/playground").text

    assert 'id="pg-signin"' in page
    # What needs signing in comes with the page: the bar needs no second read.
    (row,) = _needed(page)
    assert (row["status"], row["via"], row["service"]) == (
        "needs_sign_in",
        "notion",
        "Notion",
    )
    assert row["offers"] == [
        {
            "kind": "mcp_sign_in",
            "server": "notion",
            "title": "Sign in to Notion",
            "needs_client": False,
        }
    ]
    # The sign-in opens in its own window and reports back to this page.
    assert "window.open('', 'memorizz-sign-in'" in page
    assert "return_to: popup ? '/mcp/oauth/done' : window.location.pathname" in page
    assert "new BroadcastChannel('memorizz-oauth')" in page

    # An agent with nothing to sign in to has an empty bar.
    assert _needed(console.get("/agents/plain/playground").text) == []


def test_a_sign_in_window_ends_on_a_page_that_tells_the_playground(console):
    done = console.get("/mcp/oauth/done?oauth=success&server=notion")

    assert done.status_code == 200
    assert done.headers["cache-control"] == "no-store"
    assert "BroadcastChannel('memorizz-oauth')" in done.text
    assert "window.opener.postMessage(message, location.origin)" in done.text
    assert "window.close()" in done.text
    # Nothing from the address is written into the page as markup.
    crafted = console.get("/mcp/oauth/done?oauth=success&server=<script>x</script>")
    assert "<script>x</script>" not in crafted.text


def test_the_model_is_told_the_button_is_in_the_conversation():
    from memorizz.memagent.templates import PRODUCTIVITY_ASSISTANT, PROJECT_MANAGER

    for template in (PROJECT_MANAGER, PRODUCTIVITY_ASSISTANT):
        text = " ".join(template.instruction.split())
        assert "Sign in button" in text and "another page" in text
        assert "sign in on MCP connections" not in text
