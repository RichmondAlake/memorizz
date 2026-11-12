# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The inbox bell, the store switcher, first steps on the Overview and the
actions a table offers for several selected rows."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.approval import SQLiteApprovalStore
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.ui import inbox, state
from memorizz.ui.app import create_app
from memorizz.ui.dashboard import getting_started

UI = Path(__file__).resolve().parents[2] / "src" / "memorizz" / "ui"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A scratch MEMORIZZ_HOME with one FileSystem store connected."""
    home = tmp_path / "home"
    monkeypatch.setenv("MEMORIZZ_HOME", str(home))
    monkeypatch.delenv("MEMORIZZ_MEMORY_ROOT", raising=False)
    root = home / "memory"
    provider = FileSystemProvider(
        FileSystemConfig(root_path=root, lazy_vector_indexes=True)
    )
    provider.store_memagent(MemAgentModel(agent_id="agent-1", name="Support"))
    provider.store_memagent(MemAgentModel(agent_id="agent-2", name="Research"))
    inbox.clear_cache()
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"path": str(root)},
            "read_only": False,
        },
    ):
        yield provider, home
    inbox.clear_cache()
    provider.close()


@pytest.fixture()
def client(store):
    return TestClient(create_app(), follow_redirects=False)


@pytest.mark.unit
def test_inbox_lists_approvals_for_this_stores_agents_only(store, client):
    _, home = store
    approvals = SQLiteApprovalStore(home / "approvals.sqlite3")
    for owner, tool in (
        ("agent-1", "mcp:linear:create_issue"),
        ("agent-1", "mcp:linear:close_issue"),
        ("agent-2", "send_email"),
        ("someone-elses-agent", "mcp:linear:create_issue"),
        ("operator", "metaharness.run"),
    ):
        approvals.propose(
            owner_id=owner, tool_name=tool, arguments={}, policy_reason="mutating"
        )
    with patch("memorizz.approval.default_approval_store", lambda: approvals):
        data = client.get("/api/inbox").json()
    titles = {item["title"]: item for item in data["items"]}
    assert data["count"] == 3
    assert titles["1 harness run waiting for approval"]["href"] == "/harnesses"
    mcp = titles["2 MCP tool calls waiting for approval"]
    assert mcp["href"] == "/mcp?agent_id=agent-1"
    assert mcp["detail"] == "Asked for by Support."
    assert (
        titles["1 tool call waiting for approval"]["href"]
        == "/agents/agent-2/playground"
    )


@pytest.mark.unit
def test_inbox_lists_plans_left_for_review(store, client):
    provider, _ = store
    from memorizz.memagent.core import _shared_upkeep

    _shared_upkeep(provider, "agent-1").plans["plan-1"] = {"plan_id": "plan-1"}
    saved = provider.retrieve_memagent("agent-2")
    provider.store_memagent(
        saved.model_copy(
            update={"persona_learning_state": {"pending": {"field": "goals"}}}
        )
    )
    items = client.get("/api/inbox").json()["items"]
    by_href = {item["href"]: item["title"] for item in items}
    assert by_href == {
        "/learning-control-plane?agent_id=agent-1#memory-consolidation": (
            "1 consolidation plan to review"
        ),
        "/learning-control-plane?agent_id=agent-2#persona-learning": (
            "A persona change to review"
        ),
    }


@pytest.mark.unit
def test_inbox_survives_a_source_that_fails(store, client):
    def boom():
        raise RuntimeError("approval store is locked")

    with patch("memorizz.approval.default_approval_store", boom):
        assert client.get("/api/inbox").json() == {"count": 0, "items": []}


@pytest.mark.unit
def test_every_page_carries_the_bell_and_the_store_switcher(client):
    html = client.get("/memory/knowledge-base").text
    # The bell sits in the top bar and, on a phone, in the top strip.
    assert html.count("data-inbox-open") == 2
    # The chip (top bar and phone strip) and the sidebar foot open the switcher.
    assert html.count("data-store-switch") == 3
    assert html.count("/static/js/shell-menus.js") == 1


@pytest.mark.unit
def test_local_stores_marks_the_connected_one(store, client):
    _, home = store
    other = home / "team-store" / "agents"
    other.mkdir(parents=True)
    (other / "a.json").write_text("{}")
    stores = client.get("/api/local-stores").json()["stores"]
    current = [item for item in stores if item["current"]]
    assert len(current) == 1 and current[0]["path"].endswith("/memory")
    assert current[0]["agents"] == 2
    assert [item["agents"] for item in stores if not item["current"]] == [1]


@pytest.mark.unit
def test_first_steps_follow_what_the_store_holds():
    steps = getting_started({"agents": 1, "conversation_memory": 4})
    assert [step["key"] for step in steps if step["done"]] == ["agent", "chat"]
    # Opening a trace leaves nothing in the store: the page ticks that step.
    assert [step["key"] for step in steps if step.get("visit")] == ["trace"]
    # A store in regular use is not asked to get started.
    assert getting_started({"agents": 3, "conversation_memory": 50}) == []


@pytest.mark.unit
def test_overview_shows_first_steps_for_a_new_store_only(client):
    stats = {"agents": 1}
    with patch("memorizz.ui.app._get_memory_stats", lambda: stats):
        html = client.get("/dashboard").text
    assert 'id="ov-start"' in html
    assert html.count('class="ov-start-step is-done"') == 1
    done = {
        "agents": 1,
        "conversation_memory": 2,
        "knowledge_base": 1,
        "automations": 1,
    }
    with patch("memorizz.ui.app._get_memory_stats", lambda: done):
        assert 'id="ov-start"' not in client.get("/dashboard").text


@pytest.mark.unit
def test_several_agents_are_deleted_and_their_memory_kept(store, client):
    provider, _ = store
    response = client.post(
        "/api/agents/bulk-delete", json={"agent_ids": ["agent-1", "missing"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] == ["agent-1"] and body["ok"] is False
    assert body["failed"] == [{"agent_id": "missing", "reason": "Not found"}]
    assert provider.retrieve_memagent("agent-1") is None
    assert provider.retrieve_memagent("agent-2") is not None
    assert client.post("/api/agents/bulk-delete", json={}).status_code == 400


@pytest.mark.unit
def test_several_agents_are_marked_as_favorites(store, client):
    provider, _ = store
    body = client.post(
        "/api/agents/bulk-favorite",
        json={"agent_ids": ["agent-1", "agent-2"], "is_favorite": True},
    ).json()
    assert body == {"ok": True, "updated": ["agent-1", "agent-2"], "failed": []}
    assert provider.retrieve_memagent("agent-2").is_favorite is True


@pytest.mark.unit
def test_a_read_only_ui_refuses_bulk_changes(store, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    provider, _ = store
    client = TestClient(create_app(), follow_redirects=False)
    response = client.post("/api/agents/bulk-delete", json={"agent_ids": ["agent-1"]})
    assert response.status_code in (403, 405)
    assert provider.retrieve_memagent("agent-1") is not None


@pytest.mark.unit
def test_tables_offer_bulk_actions_chips_and_views():
    monitor = (UI / "static/js/monitor.js").read_text()
    for name in ("function bulk(", "function initViews(", "function renderChips("):
        assert name in monitor
    assert (
        "MemorizzMonitor.bulk('fleet-table'"
        in (UI / "templates/agents.html").read_text()
    )
    assert (
        "MemorizzMonitor.bulk('mem-table'"
        in (UI / "templates/memory_list.html").read_text()
    )
    # A destructive bulk action asks first, in the page.
    agents = (UI / "templates/agents.html").read_text()
    assert agents.index("MemorizzDialogs.confirm('Delete '") < agents.index(
        "post('/api/agents/bulk-delete'"
    )


@pytest.mark.unit
def test_playground_home_opens_on_a_composer_and_hands_the_message_on(client):
    html = client.get("/playground").text
    # One composer; it posts the message to the chosen agent's playground.
    assert html.count('id="chat-home-form"') == 1
    assert 'name="prompt"' in html
    assert '<option value="agent-1" data-name="Support"' in html
    assert html.count('class="chat-home-starter"') == 3
    # The agent table is still there for browsing.
    assert 'id="pg-table"' in html and 'data-key="agent-2"' in html
    playground = (UI / "templates/playground.html").read_text()
    # The playground takes the message once, then drops it from the address.
    assert "params.get('prompt')" in playground
    assert playground.index("params.delete('prompt')") < playground.index(
        "if (document.body.dataset.readOnly !== 'true') void sendMessage();"
    )


@pytest.mark.unit
def test_a_new_conversation_is_empty_not_an_error():
    """The server knows a conversation from its first message on; before
    that the thread request answers 404, which the page reads as empty."""
    playground = (UI / "templates/playground.html").read_text()
    assert "const unsaved = response.status === 404;" in playground
    assert "if (!response.ok && !unsaved)" in playground


# -- Activity log, operator identity, budget meter -----------------------------


@pytest.mark.unit
def test_activity_names_the_change_and_skips_chat_and_reads():
    from memorizz.ui import activity

    assert activity.describe("GET", "/agents") is None
    assert activity.describe("POST", "/agents/a1/playground/stream") is None
    assert activity.describe("POST", "/traces/reveal") is None
    assert activity.describe("POST", "/agents/a1/edit") == {
        "area": "Agents",
        "action": "Edited an agent",
        "target": "a1",
    }
    approve = activity.describe("POST", "/api/mcp/agents/a1/approvals/p9/approve")
    assert approve == {
        "area": "Approvals",
        "action": "Approved a request",
        "target": "p9",
    }
    # A route nobody listed is still recorded, by its path.
    assert activity.describe("POST", "/something/new")["action"] == (
        "Changed /something/new"
    )


@pytest.mark.unit
def test_a_change_is_logged_with_the_operators_name_and_no_content(store, client):
    from memorizz.ui import activity

    client.cookies.set(activity.OPERATOR_COOKIE, "Ada%20Lovelace")
    client.post(
        "/api/agents/bulk-favorite",
        json={"agent_ids": ["agent-1"], "is_favorite": True},
    )
    # A refused request changes nothing, so it is not a line.
    client.post("/api/agents/bulk-delete", json={})
    rows = activity.recent()
    assert len(rows) == 1
    row = rows[0]
    assert (row["actor"], row["area"], row["action"]) == (
        "Ada Lovelace",
        "Agents",
        "Changed favorites",
    )
    assert set(row) == {
        "schema_version",
        "timestamp",
        "actor",
        "method",
        "path",
        "area",
        "action",
        "target",
    }
    html = client.get("/activity").text
    assert "Ada Lovelace" in html and "Changed favorites" in html
    assert 'href="/activity"' in client.get("/dashboard").text


@pytest.mark.unit
def test_without_a_name_the_log_says_local_operator(store, client):
    from memorizz.ui import activity

    client.post(
        "/api/agents/bulk-favorite",
        json={"agent_ids": ["agent-1"], "is_favorite": False},
    )
    assert activity.recent()[0]["actor"] == "Local operator"


@pytest.mark.unit
def test_budget_meter_reports_nothing_until_a_budget_is_set(store, client):
    assert client.get("/api/budget").json()["set"] is False
    html = client.get("/dashboard").text
    assert 'id="sidebar-budget"' in html and "data-profile-open" in html


# -- Overview: comparison, trends, customising ---------------------------------


@pytest.mark.unit
def test_comparisons_that_would_overlap_the_window_are_not_offered():
    from datetime import datetime, timedelta, timezone

    from memorizz.ui.dashboard import compare_choices, format_period

    end = datetime(2026, 11, 11, tzinfo=timezone.utc)
    day = {"start": end - timedelta(hours=24), "end": end}
    month = {"start": end - timedelta(days=30), "end": end}
    assert [c["key"] for c in compare_choices(day)] == ["previous", "week"]
    assert [c["key"] for c in compare_choices(month)] == ["previous"]
    assert format_period(3600) == "1 hour" and format_period(21600) == "6 hours"


@pytest.mark.unit
def test_row_trends_count_calls_per_chart_period():
    from datetime import datetime, timedelta, timezone

    from memorizz.ui.dashboard import row_trends

    start = datetime(2026, 11, 11, tzinfo=timezone.utc)
    overview = {"window_start": start, "bucket_seconds": 3600, "series": [{}] * 3}

    def event(kind, minutes, **fields):
        stamp = (start + timedelta(minutes=minutes)).isoformat()
        return {"trace_kind": kind, "timestamp": stamp, **fields}

    trends = row_trends(
        [
            event("tool_result", 5, tool_name="lookup"),
            event("tool_result", 130, tool_name="lookup"),
            event("model_result", 70, model="gpt-4o-mini"),
            event("model_result", 500, model="gpt-4o-mini"),  # after the window
            event("turn_result", 10),
        ],
        overview,
    )
    assert trends == {
        "tools": {"lookup": [1, 0, 1]},
        "models": {"gpt-4o-mini": [0, 1, 0]},
    }


@pytest.mark.unit
def test_overview_sections_can_be_hidden_and_reordered(client):
    html = client.get("/dashboard").text
    assert 'id="ov-customize-btn"' in html
    assert 'data-widget="platform"' in html
    assert "memorizz_ui_overview_layout" in html


@pytest.mark.unit
def test_tables_get_export_row_menus_a_drawer_and_a_preview_pane():
    monitor = (UI / "static/js/monitor.js").read_text()
    for name in ("function initExport(", "function rowMenu(", "function openDrawer("):
        assert name in monitor
    # A cell that starts like a formula is neutralised in the CSV.
    assert "would run as a formula in a spreadsheet" in monitor
    preview = (UI / "static/js/chat-preview.js").read_text()
    # Model-written HTML renders with scripts off unless the reader allows them,
    # and never with access to this page.
    assert "frame.setAttribute('sandbox', scripts ? 'allow-scripts' : '');" in preview
    assert "allow-same-origin" not in preview
    assert (
        "/static/js/chat-preview.js" in (UI / "templates/playground.html").read_text()
    )
