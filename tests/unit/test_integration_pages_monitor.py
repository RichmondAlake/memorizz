"""Monitor views for the MCP connections, Vercel skills and Settings pages."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from memorizz.ui import integrations_view as view

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
OWNER = "agent-1"


def _entry(minutes, server, operation, ok, *, tool=None, code=None, owner=OWNER):
    return {
        "timestamp": (NOW - timedelta(minutes=minutes)).isoformat(),
        "owner_id": owner,
        "server_name": server,
        "operation": operation,
        "tool_name": tool,
        "ok": ok,
        "error_code": code,
    }


SERVERS = [
    {
        "name": "files",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "fs-server"],
        "auth": {"type": "none"},
        "allowed_tools": ["read_file"],
        "env_keys": ["LOG_LEVEL"],
    },
    {
        "name": "notion",
        "transport": "streamable_http",
        "url": "https://mcp.notion.com/mcp",
        "auth": {"type": "oauth", "scopes": ["read"]},
    },
    {
        "name": "github",
        "transport": "streamable_http",
        "url": "https://api.example.com/mcp",
        "auth": {"type": "bearer"},
    },
    {"name": "fresh", "transport": "sse", "url": "https://sse.example.com"},
    {"name": "off", "transport": "stdio", "command": "tool", "enabled": False},
]
STATUSES = [
    {"name": "files", "authenticated": True, "state": "configured"},
    {"name": "notion", "authenticated": False, "state": "reauth_required"},
    {"name": "github", "authenticated": True, "state": "configured"},
    {"name": "fresh", "authenticated": True, "state": "configured"},
    {"name": "off", "ok": False, "error_code": "configuration_error"},
]
AUDIT = [
    _entry(60 * 24 * 10, "files", "tools/call", False, tool="read_file", code="x"),
    _entry(120, "files", "tools/list", True),
    _entry(90, "files", "tools/call", True, tool="read_file"),
    _entry(80, "files", "tools/call", True, tool="read_file"),
    _entry(70, "files", "tools/call", False, tool="list", code="protocol_error"),
    _entry(30, "github", "tools/list", True),
    _entry(20, "github", "tools/list", False, code="connection_error"),
    _entry(10, "files", "tools/list", True, owner="someone-else"),
    {"owner_id": OWNER, "server_name": "files", "timestamp": "not a time"},
]


def _build(**kwargs):
    params = dict(
        owner_id=OWNER,
        tool_cache={
            OWNER: {
                "files": {
                    "endpoint": "npx -y fs-server",
                    "checked_at": (NOW - timedelta(minutes=5)).isoformat(),
                    "tool_count": 2,
                    "tools": [{"name": "read_file", "description": "Read"}],
                },
                "github": {"endpoint": "https://moved.example.com", "tool_count": 9},
            }
        },
        agents=[
            {"agent_id": OWNER, "name": "Me", "servers": SERVERS},
            {
                "agent_id": "agent-2",
                "name": "Other",
                "servers": [{"name": "n", "url": "https://mcp.notion.com/mcp"}],
            },
            {"agent_id": "agent-3", "name": "Idle", "servers": []},
        ],
        now=NOW,
    )
    params.update(kwargs)
    return view.build_mcp_view(SERVERS, STATUSES, AUDIT, **params)


@pytest.mark.unit
def test_mcp_view_derives_state_worst_first():
    result = _build()
    rows = {row["name"]: row for row in result["rows"]}
    assert rows["files"]["state"] == "ready"
    assert rows["notion"]["state"] == "needs_auth"
    assert rows["github"]["state"] == "failing"
    assert rows["fresh"]["state"] == "unchecked"
    assert rows["off"]["state"] == "disabled"
    assert [row["name"] for row in result["rows"]] == [
        "github",
        "notion",
        "fresh",
        "files",
        "off",
    ]
    assert rows["github"]["health"] == "failing"
    assert rows["notion"]["state_label"] == "Needs sign-in"


@pytest.mark.unit
def test_mcp_view_counts_window_activity_and_hints():
    rows = {row["name"]: row for row in _build()["rows"]}
    files = rows["files"]
    # The 10-day-old failure is outside the 7-day window; other owners never count.
    assert files["calls"] == 3
    assert files["failed_calls"] == 1
    assert files["error_count"] == 1
    assert files["errors"][0]["code"] == "protocol_error"
    assert "MCP" in files["errors"][0]["hint"]
    assert files["tools_used"][0] == {"name": "read_file", "calls": 2, "failed": 0}
    assert files["last_check"] == NOW - timedelta(minutes=5)  # cached list is newer
    github = rows["github"]
    assert github["errors"][0]["hint"] == view.ERROR_HINTS["connection_error"]
    assert github["last_check_ok"] is False


@pytest.mark.unit
def test_mcp_view_uses_cached_tools_only_for_the_same_endpoint():
    rows = {row["name"]: row for row in _build()["rows"]}
    assert rows["files"]["tool_count"] == 2
    assert rows["files"]["tools"][0]["name"] == "read_file"
    assert rows["github"]["tool_count"] is None  # server moved since it was listed
    totals = _build()["totals"]
    assert totals["tools"] == 2 and totals["tools_known"] == 1
    assert _build(tool_cache={})["totals"]["tools"] is None


@pytest.mark.unit
def test_mcp_view_totals_adoption_and_shared_servers():
    result = _build()
    totals = result["totals"]
    assert totals["servers"] == 5
    assert (totals["ready"], totals["needs_auth"], totals["failing"]) == (1, 1, 1)
    assert totals["unchecked"] == 1 and totals["disabled"] == 1
    assert totals["agents_with_mcp"] == 2 and totals["agents"] == 3
    assert totals["errors"] == 2
    rows = {row["name"]: row for row in result["rows"]}
    assert rows["notion"]["shared_with"] == [{"agent_id": "agent-2", "name": "Other"}]
    assert rows["files"]["shared_with"] == []
    config = {item["label"]: item["value"] for item in rows["files"]["config"]}
    assert config["Endpoint"] == "npx -y fs-server"
    assert config["Allowed tools"] == "read_file"
    assert config["Environment keys"] == "LOG_LEVEL"
    assert "oauth" in rows["notion"]["tags"].split()
    assert "fs-server" in rows["files"]["search"]


@pytest.mark.unit
def test_mcp_view_handles_no_servers():
    result = view.build_mcp_view([], [], [], owner_id=OWNER, now=NOW)
    assert result["rows"] == []
    assert result["totals"]["servers"] == 0
    assert result["totals"]["last_check"] is None


@pytest.mark.unit
def test_error_hint_falls_back_for_unknown_codes():
    assert view.error_hint("authorization_required").startswith("Sign-in")
    assert view.error_hint("something_new") == view.DEFAULT_HINT
    assert view.error_hint(None) == view.DEFAULT_HINT


@pytest.mark.unit
def test_read_jsonl_tail_is_bounded_and_tolerant(tmp_path):
    path = tmp_path / "audit.jsonl"
    assert view.read_jsonl_tail(path) == []
    lines = [json.dumps({"n": index}) for index in range(50)]
    path.write_text("\n".join(lines) + "\nnot json\n[1, 2]\n")
    assert [item["n"] for item in view.read_jsonl_tail(path)] == list(range(50))
    tail = view.read_jsonl_tail(path, max_bytes=60)
    assert tail and all(isinstance(item["n"], int) for item in tail)
    assert tail[-1]["n"] == 49 and len(tail) < 50


@pytest.mark.unit
def test_tool_cache_round_trip_is_bounded(tmp_path):
    path = tmp_path / "home" / "cache.json"
    server = {"name": "files", "command": "npx", "args": ["fs"]}
    tools = [{"name": f"tool_{i}", "description": "x" * 900} for i in range(250)]
    tools.append({"description": "nameless"})
    view.store_tool_list(path, OWNER, server, {"ok": False, "tools": tools}, NOW)
    assert not path.exists()  # failures never overwrite a good list
    view.store_tool_list(
        path,
        OWNER,
        server,
        {"ok": True, "tools": tools, "server_info": {"name": "fs", "version": "1"}},
        NOW,
    )
    entry = view.load_tool_cache(path)[OWNER]["files"]
    assert entry["endpoint"] == "npx fs"
    assert entry["tool_count"] == 250
    assert len(entry["tools"]) == view.CACHED_TOOLS_LIMIT
    assert len(entry["tools"][0]["description"]) == view.TOOL_DESCRIPTION_CHARS
    assert entry["server_name"] == "fs" and entry["checked_at"] == NOW.isoformat()
    path.write_text("{broken")
    assert view.load_tool_cache(path) == {}


@pytest.mark.unit
def test_skills_marketplace_agents_matches_the_agent_form_normalization():
    agents = [
        SimpleNamespace(
            agent_id="a1", name="Alpha", skills_marketplace_provider="Vercel"
        ),
        {
            "agent_id": "a2",
            "persona": {"name": "Beta"},
            "skills_marketplace_provider": {"provider": "vercel"},
        },
        SimpleNamespace(
            agent_id="a3", name="Gamma", skills_marketplace_provider="skillsmp"
        ),
        {"agent_id": "a4", "name": "Delta"},
    ]
    assert view.skills_marketplace_agents(agents) == [
        {"agent_id": "a1", "name": "Alpha"},
        {"agent_id": "a2", "name": "Beta"},
    ]


@pytest.mark.unit
def test_settings_summary_counts_keys_and_labels_defaults():
    sections = [
        {
            "title": "LLM & embeddings",
            "fields": [
                {"env": "OPENAI_API_KEY", "is_set": True, "current_value": ""},
                {"env": "ANTHROPIC_API_KEY", "is_set": False, "current_value": ""},
                {
                    "env": "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER",
                    "field_type": "select",
                    "is_set": True,
                    "current_value": "voyageai",
                    "options": [
                        {"value": "", "label": "Not set"},
                        {"value": "voyageai", "label": "Voyage AI"},
                    ],
                },
                {
                    "env": "MEMORIZZ_DEFAULT_EMBEDDING_MODEL",
                    "field_type": "text",
                    "is_set": False,
                    "current_value": "",
                },
            ],
        },
        {
            "title": "Default agent model",
            "fields": [
                {
                    "env": "MEMORIZZ_DEFAULT_LLM_PROVIDER",
                    "field_type": "select",
                    "is_set": False,
                    "current_value": "openai",
                    "options": [{"value": "openai", "label": "OpenAI"}],
                },
                {
                    "env": "MEMORIZZ_DEFAULT_LLM_MODEL",
                    "field_type": "select",
                    "is_set": False,
                    "current_value": "gpt-5.2",
                },
            ],
        },
        {"title": "LLM & Embeddings", "fields": []},
    ]
    summary = view.settings_summary(sections)
    assert (summary["keys_set"], summary["keys_total"]) == (1, 2)
    assert summary["embedding_provider"] == "Voyage AI"
    assert summary["embedding_model"] == ""
    assert summary["llm_provider"] == "OpenAI" and summary["llm_model"] == "gpt-5.2"
    assert summary["saved_backend"] == "" and summary["sandbox"] == ""
    ids = [item["id"] for item in summary["sections"]]
    assert ids == [
        "settings-llm-embeddings",
        "settings-default-agent-model",
        "settings-llm-embeddings-2",
    ]
    assert summary["sections"][0]["set"] == 2 and summary["sections"][0]["total"] == 4


# ------------------------------------------------------------------ routes

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402


class _Provider:
    def __init__(self, servers):
        self.agents = [
            SimpleNamespace(agent_id="agent-1", name="MCP agent", mcp_servers=servers),
            SimpleNamespace(
                agent_id="agent-2",
                name="Skills agent",
                mcp_servers=[],
                skills_marketplace_provider="vercel",
            ),
        ]

    def list_memagents(self):
        return list(self.agents)

    def retrieve_memagent(self, agent_id):
        return next((a for a in self.agents if a.agent_id == agent_id), None)

    def store_memagent(self, agent):
        return agent.agent_id


@pytest.fixture
def connected(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _Provider(
        [
            {"name": "files", "transport": "stdio", "command": "npx", "args": ["fs"]},
            {
                "name": "notion",
                "transport": "streamable_http",
                "url": "https://mcp.notion.com/mcp",
                "auth": {"type": "oauth"},
            },
        ]
    )
    (tmp_path / "mcp_audit.jsonl").write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "owner_id": "agent-1",
                "server_name": "files",
                "operation": "tools/list",
                "ok": False,
                "error_code": "connection_error",
            }
        )
        + "\n"
    )
    with patch.dict(
        state._state,
        {"provider": provider, "provider_type": "filesystem", "connection_info": {}},
    ):
        yield TestClient(create_app(), follow_redirects=False), tmp_path


@pytest.mark.unit
def test_mcp_page_renders_the_server_grid(connected):
    client, _ = connected
    html = client.get("/mcp?agent_id=agent-1").text
    assert 'id="mcp-table"' in html and 'class="fleet-tape"' in html
    assert 'data-key="files"' in html and 'data-key="notion"' in html
    assert "fleet-health--failing" in html  # files failed its last check
    assert view.ERROR_HINTS["connection_error"] in html
    assert "Needs sign-in" in html  # notion has no OAuth tokens
    assert 'id="mcp-form"' in html and 'id="mcp-allowed-tools"' in html
    assert "Expose MemoRizz as an MCP server" in html
    assert "Test all" in html


@pytest.mark.unit
def test_mcp_tools_endpoint_remembers_the_tool_list(connected, monkeypatch):
    client, home = connected
    from memorizz.mcp import MCPClientManager

    monkeypatch.setattr(
        MCPClientManager,
        "list_tools",
        lambda self, name: {
            "ok": True,
            "tools": [{"name": "read_file", "description": "Read one file"}],
        },
    )
    response = client.get("/api/mcp/agents/agent-1/servers/files/tools")
    assert response.status_code == 200 and response.json()["ok"] is True
    cache = view.load_tool_cache(home / "mcp_tools_cache.json")
    assert cache["agent-1"]["files"]["tool_count"] == 1
    html = client.get("/mcp?agent_id=agent-1").text
    assert "Read one file" in html
    assert 'data-mcp-call="read_file"' in html


@pytest.mark.unit
def test_vercel_skills_page_shows_marketplace_agents(connected):
    client, _ = connected
    html = client.get("/vercel-skills").text
    assert (
        'id="skills-table"' in html and 'data-monitor-filter-for="skills-table"' in html
    )
    assert "Skills agent" in html
    for element_id in (
        "search-input",
        "repo-input",
        "skill-name-input",
        "skill-detail-panel",
        "detail-skill-instructions",
    ):
        assert f'id="{element_id}"' in html


@pytest.mark.unit
def test_settings_page_is_a_console_with_unchanged_fields(connected):
    client, _ = connected
    with patch.dict(os.environ, {"OPENAI_API_KEY": "private-settings-key"}):
        html = client.get("/settings").text
    assert "private-settings-key" not in html
    assert 'class="fleet-tape settings-tape"' in html
    assert 'data-section-link="settings-llm-embeddings"' in html
    assert 'id="settings-llm-embeddings"' in html
    assert 'name="OPENAI_API_KEY"' in html and 'id="OPENAI_API_KEY"' in html
    assert 'form method="post" action="/settings"' in html
    assert "GraalPy (Local)" in html  # sandbox readiness checks render again
