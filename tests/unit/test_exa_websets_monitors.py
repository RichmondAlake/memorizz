"""Exa's Websets and Monitors: the calls, the tables, the agent tools."""

import io

import pytest
from rich.console import Console

from memorizz.internet_access import exa_tools
from memorizz.internet_access.exa_tools import (
    ExaRecords,
    build_exa_tools,
    monitor_view,
    row_note,
    save_table_to_memory,
    table_text,
    webset_table,
)
from memorizz.internet_access.providers.exa import ExaError, ExaProvider
from memorizz.tool_outcomes import ToolResult, normalize_tool_result, result_display

WEBSET = {
    "id": "webset_1",
    "status": "idle",
    "title": "Memory startups",
    "dashboardUrl": "https://websets.exa.ai/webset_1",
    "searches": [
        {
            "query": "AI agent memory startups",
            "count": 2,
            "entity": {"type": "company"},
            "criteria": [{"description": "builds agent memory", "successRate": 50}],
            "progress": {"found": 2, "analyzed": 40, "completion": 100},
        }
    ],
    "enrichments": [{"id": "enr_1", "description": "Year founded", "format": "number"}],
}
ITEMS = [
    {
        "id": "item_1",
        "properties": {
            "type": "company",
            "url": "https://acme.example",
            "description": "Acme builds memory for agents.",
            "company": {"name": "Acme", "location": "Paris", "employees": 12},
        },
        "evaluations": [
            {"criterion": "builds agent memory", "satisfied": "yes", "reasoning": "Says so."}
        ],
        "enrichments": [
            {"enrichmentId": "enr_1", "status": "completed", "result": ["2024"], "reasoning": "About page"}
        ],
    },
    {
        "id": "item_2",
        "properties": {
            "type": "company",
            "url": "https://bolt.example",
            "description": "Bolt.",
            "company": {"name": "Bolt", "location": None},
        },
        "evaluations": [{"criterion": "builds agent memory", "satisfied": "unclear"}],
        "enrichments": [{"enrichmentId": "enr_1", "status": "pending", "result": None}],
    },
]  # fmt: skip
MONITOR = {
    "id": "mon_1",
    "name": "Acme news",
    "status": "active",
    "search": {"query": "Acme launches"},
    "trigger": {"type": "interval", "period": "1d"},
    "nextRunAt": "2026-11-20T10:00:00Z",
}
RUNS = [
    {"id": "run_1", "status": "completed", "createdAt": "2026-11-18T10:00:00Z",
     "output": {"content": "Nothing new.", "results": []}},
    {"id": "run_2", "status": "completed", "createdAt": "2026-11-19T10:00:00Z",
     "completedAt": "2026-11-19T10:00:05Z",
     "output": {"content": "Acme shipped v2.", "results": [
         {"title": "Acme v2", "url": "https://acme.example/v2", "highlights": ["Acme v2 is out."]}]}},
]  # fmt: skip


class Reply:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status
        self.content = b"x" if body is not None else b""
        self.text = str(body)

    def json(self):
        return self._body


class Session:
    """Stands in for Exa: answers by method and path, remembers the calls."""

    def __init__(self):
        self.calls = []
        self.headers = {}
        self.refuse = None

    def request(self, method, url, json=None, params=None, timeout=None):
        path = url.replace("https://api.exa.ai", "")
        self.calls.append((method, path, json, params))
        if self.refuse:
            return Reply({"error": self.refuse}, 402)
        if (method, path) == ("POST", "/websets/v0/websets"):
            return Reply({**WEBSET, "status": "running"})
        if (method, path) == ("GET", "/websets/v0/websets/webset_1"):
            return Reply(WEBSET)
        if (method, path) == ("GET", "/websets/v0/websets/webset_1/items"):
            return Reply({"data": ITEMS, "hasMore": False, "nextCursor": None})
        if (method, path) == ("POST", "/monitors"):
            return Reply({**MONITOR, "webhookSecret": "whsec"})
        if (method, path) == ("GET", "/monitors/mon_1"):
            return Reply(MONITOR)
        if (method, path) == ("GET", "/monitors/mon_1/runs"):
            return Reply({"data": RUNS})
        if (method, path) == ("POST", "/monitors/mon_1/trigger"):
            return Reply({"triggered": True})
        if (method, path) == ("PATCH", "/monitors/mon_1"):
            return Reply({**MONITOR, "status": json["status"]})
        if (method, path) == ("DELETE", "/monitors/mon_1"):
            return Reply(None)
        return Reply({"error": "not found"}, 404)

    def close(self):
        pass


class Agent:
    agent_id = "agent-1"
    _current_user_id = None

    def __init__(self):
        self.memory_provider = object()
        self.knowledge_base_ids = []


class Knowledge:
    """Stands in for the knowledge base: keeps what was ingested."""

    def __init__(self, provider):
        self.provider = provider
        if not hasattr(Knowledge, "entries"):
            Knowledge.entries = []

    def ingest_knowledge(self, corpus, namespace, *, chunking_strategy, **options):
        Knowledge.entries.append(
            {"namespace": namespace, "chunks": chunking_strategy(corpus), **options}
        )
        return f"kb-{len(Knowledge.entries)}"

    def attach_to_agent(self, agent, knowledge_base_id, **_):
        agent.knowledge_base_ids.append(knowledge_base_id)
        return True


@pytest.fixture
def exa(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.delenv("EXA_MONITOR_WEBHOOK_URL", raising=False)
    import memorizz.long_term.semantic.knowledge_base as knowledge_base

    Knowledge.entries = []
    monkeypatch.setattr(knowledge_base, "KnowledgeBase", Knowledge)
    provider = ExaProvider(api_key="test-key")
    provider._session = Session()
    return provider


def tools_of(agent, provider):
    return {tool.__name__: tool for tool in build_exa_tools(agent, provider)}


@pytest.mark.unit
def test_a_webset_request_is_shaped_the_way_exa_expects(exa):
    exa.create_webset(
        "AI agent memory startups",
        count=5,
        entity="company",
        criteria=["builds agent memory"],
        enrichments=["Year founded", {"description": "CEO email", "format": "email"}],
        title="Memory startups",
    )

    method, path, body, _ = exa._session.calls[-1]
    assert (method, path) == ("POST", "/websets/v0/websets")
    assert body["search"] == {
        "query": "AI agent memory startups",
        "count": 5,
        "entity": {"type": "company"},
        "criteria": [{"description": "builds agent memory"}],
    }
    assert body["enrichments"] == [
        {"description": "Year founded"},
        {"description": "CEO email", "format": "email"},
    ]
    # Something Exa has no entity for is described instead.
    exa.create_webset("hackathons in Lisbon", entity="hackathon events")
    assert exa._session.calls[-1][2]["search"]["entity"] == {
        "type": "custom",
        "description": "hackathon events",
    }


@pytest.mark.unit
def test_a_webset_becomes_one_table(exa):
    table = webset_table(WEBSET, ITEMS)

    assert (table["status"], table["found"], table["wanted"]) == ("done", 2, 2)
    assert [(c["label"], c["kind"]) for c in table["columns"]][-2:] == [
        ("builds agent memory", "criterion"),
        ("Year founded", "enrichment"),
    ]
    acme, bolt = table["rows"]
    assert (acme["name"], acme["url"]) == ("Acme", "https://acme.example")
    assert acme["cells"]["location"] == "Paris"
    assert acme["cells"]["criterion_0"] == "yes"
    assert acme["cells"]["enrichment_enr_1"] == "2024"
    assert acme["why"]["criterion_0"] == "Says so."
    # A column still being looked up is marked, and left out of flat rows.
    assert bolt["cells"]["criterion_0"] == "unclear"
    assert bolt["cells"]["enrichment_enr_1"] == "…"
    assert table_text(table) == [
        {"name": "Acme", "url": "https://acme.example", "Location": "Paris",
         "Employees": "12", "builds agent memory": "yes", "Year founded": "2024"},
        {"name": "Bolt", "url": "https://bolt.example", "builds agent memory": "unclear"},
    ]  # fmt: skip
    note = row_note(table, acme)
    assert "Acme (https://acme.example)" in note and "Year founded: 2024" in note
    assert "Meets: builds agent memory" in note and "Memory startups" in note


@pytest.mark.unit
def test_an_agent_builds_reads_and_keeps_a_webset(exa):
    agent = Agent()
    tools = tools_of(agent, exa)

    made = tools["exa_create_webset"](
        "AI agent memory startups", count=500, entity="company", columns="Year founded"
    )
    assert isinstance(made, ToolResult)
    assert made.value["webset_id"] == "webset_1"
    # Never more than the provider allows; every row is billed.
    assert exa._session.calls[-1][2]["search"]["count"] == exa.webset_max_count
    assert made.display == {
        "kind": "exa_webset",
        "id": "webset_1",
        "title": "AI agent memory startups",
        "query": "AI agent memory startups",
        "status": "running",
    }
    # The display travels beside the result, not in what the model reads.
    value, outcome = normalize_tool_result(made)
    assert "display" not in value and outcome.ok
    assert result_display(made)["id"] == "webset_1"

    read = tools["exa_get_webset"]("webset_1", wait_seconds=0)
    assert read.value["status"] == "done" and read.value["found"] == 2
    assert read.value["rows"][0]["Year founded"] == "2024"

    kept = tools["exa_save_webset_to_memory"]("webset_1")
    assert kept.value["saved"] == 2 and kept.value["in_memory"] == 2
    # One knowledge-base entry for the dataset, a chunk per row, attached to
    # the agent so any conversation can recall it.
    (entry,) = Knowledge.entries
    assert entry["namespace"] == "webset: Memory startups"
    assert (
        len(entry["chunks"]) == 2
        and "Acme (https://acme.example)" in entry["chunks"][0]
    )
    assert entry["metadata"]["webset_id"] == "webset_1"
    assert agent.knowledge_base_ids == kept.value["knowledge_base_ids"] == ["kb-1"]
    # Saving again writes nothing twice.
    again = tools["exa_save_webset_to_memory"]("webset_1")
    assert again.value["saved"] == 0 and len(Knowledge.entries) == 1
    assert tools["exa_get_webset"]("webset_1", wait_seconds=0).value["in_memory"] == 2


@pytest.mark.unit
def test_an_agent_reaches_only_what_it_made_and_reports_refusals(exa):
    tools = tools_of(Agent(), exa)

    other = tools["exa_get_webset"]("webset_1", wait_seconds=0)
    assert other.value["ok"] is False and other.value["error_code"] == "not_found"
    assert not exa._session.calls

    exa._session.refuse = "Insufficient credits"
    refused = tools["exa_create_webset"]("anything")
    assert (
        refused.value["ok"] is False
        and "Insufficient credits" in refused.value["error"]
    )
    assert not normalize_tool_result(refused)[1].ok
    assert ExaRecords("agent-1").all("websets") == []
    with pytest.raises(ExaError):
        exa.get_webset("webset_1")


@pytest.mark.unit
def test_a_monitor_needs_approval_and_reports_to_your_address_or_a_temporary_one(
    exa, monkeypatch
):
    tools = tools_of(Agent(), exa)
    create = tools["exa_create_monitor"]
    assert create.__memorizz_tool_policy__.requires_approval is True
    assert (
        tools["exa_create_webset"].__memorizz_tool_policy__.requires_approval is False
    )

    assert (
        create("Acme launches", every="soon").value["error_code"] == "invalid_arguments"
    )
    assert not exa._session.calls

    # With no address of yours, a temporary one is made, as Exa's dashboard
    # does. Turned off, nothing is sent and the tool says what is missing.
    exa.temporary_webhooks = False
    missing = create("Acme launches", every="7d")
    assert missing.value["ok"] is False
    assert "EXA_MONITOR_WEBHOOK_URL" in missing.value["error"]
    assert not exa._session.calls
    exa.temporary_webhooks = True
    asked = []
    monkeypatch.setattr(
        exa, "temporary_webhook", lambda: asked.append(1) or "https://webhook.site/abc"
    )
    create("Acme launches", every="7d")
    create("Bolt launches", every="7d")
    assert exa._session.calls[-1][2]["webhook"]["url"] == "https://webhook.site/abc"
    # One address for the machine's monitors, not one each.
    assert len(asked) == 1
    view = monitor_view({**MONITOR, "webhook": {"url": "https://webhook.site/abc"}}, [])
    assert view["reports_to"] == "temporary"
    assert monitor_view(MONITOR, [])["reports_to"] == "yours"

    monkeypatch.setenv("EXA_MONITOR_WEBHOOK_URL", "https://hooks.example/exa")
    made = create("Acme launches", every="1d", name="Acme news")
    body = exa._session.calls[-1][2]
    assert body["trigger"] == {"type": "interval", "period": "1d"}
    assert body["webhook"] == {
        "url": "https://hooks.example/exa",
        "events": ["monitor.run.completed"],
    }
    assert made.value["monitor_id"] == "mon_1" and made.display["kind"] == "exa_monitor"
    # The address can carry a token, so it is never in the saved settings.
    assert "hooks.example" not in str(exa.get_config())

    assert tools["exa_list_monitors"]().value["monitors"][0]["monitor_id"] == "mon_1"
    found = tools["exa_monitor_results"]("mon_1", run_now=True)
    assert found.value["run_started"] is True
    assert found.value["latest"]["summary"] == "Acme shipped v2."
    assert found.value["latest"]["results"][0]["url"] == "https://acme.example/v2"

    assert tools["exa_stop_monitor"]("mon_1").value["status"] == "paused"
    assert tools["exa_resume_monitor"]("mon_1").value["status"] == "active"
    assert tools["exa_stop_monitor"]("mon_1", delete=True).value["status"] == "deleted"
    assert ExaRecords("agent-1").all("monitors") == []


@pytest.mark.unit
def test_a_monitor_view_puts_the_newest_run_first():
    view = monitor_view(MONITOR, RUNS)

    assert (view["name"], view["every"], view["status"]) == (
        "Acme news",
        "1d",
        "active",
    )
    assert [run["id"] for run in view["runs"]] == ["run_2", "run_1"]
    assert view["runs"][0]["results"][0]["snippet"] == "Acme v2 is out."


@pytest.mark.unit
def test_only_new_rows_are_saved_when_a_webset_has_grown(tmp_path):
    records = ExaRecords("agent-1", folder=tmp_path)
    kept = []

    def keep(title, notes, metadata):
        kept.append(notes)
        return f"kb-{len(kept)}"

    first = save_table_to_memory(webset_table(WEBSET, ITEMS[:1]), records, keep)
    assert first == {"saved": 1, "in_memory": 1, "knowledge_base_ids": ["kb-1"]}
    second = save_table_to_memory(webset_table(WEBSET, ITEMS), records, keep)
    assert second == {
        "saved": 1,
        "in_memory": 2,
        "knowledge_base_ids": ["kb-1", "kb-2"],
    }
    assert len(kept[1]) == 1 and kept[1][0].startswith("Bolt")
    assert (
        save_table_to_memory(webset_table(WEBSET, ITEMS), records, keep)["saved"] == 0
    )


@pytest.mark.unit
def test_a_display_is_small_or_it_is_trimmed():
    from memorizz.tool_outcomes import ToolOutcome, ToolOutcomeStatus

    ok = ToolOutcome(status=ToolOutcomeStatus.SUCCESS)
    assert result_display({"display": {"kind": "x"}}) is None  # only on a ToolResult
    assert result_display(ToolResult(value={}, outcome=ok)) is None
    assert result_display(ToolResult(value={}, outcome=ok, display={"id": "1"})) is None
    big = ToolResult(
        value={},
        outcome=ok,
        display={"kind": "exa_webset", "id": "w", "rows": ["x" * 9000]},
    )
    assert result_display(big) == {"kind": "exa_webset", "id": "w", "trimmed": True}


@pytest.mark.unit
def test_the_cli_prints_a_webset_and_a_search_as_tables(exa):
    from memorizz.cli.displays import print_displays

    out = io.StringIO()
    console = Console(file=out, width=140, force_terminal=False)
    print_displays(
        console,
        [
            {"kind": "web_search", "provider": "exa", "query": "acme",
             "results": [{"title": "Acme home", "url": "https://acme.example"}]},
            {"kind": "exa_webset", "id": "webset_1"},
            {"kind": "exa_webset", "id": "webset_1"},
            {"kind": "exa_monitor", "id": "mon_1"},
            {"kind": "something_else"},
        ],
        exa,
    )  # fmt: skip

    text = out.getvalue()
    assert "Searched with exa: acme" in text and "Acme home" in text
    assert text.count("Webset: Memory startups") == 1
    assert "Acme" in text and "2024" in text and "2 of 2 found, done" in text
    assert "Monitor: Acme news" in text and "Acme shipped v2." in text


@pytest.mark.unit
def test_a_temporary_address_is_asked_for_without_the_exa_key(exa, monkeypatch):
    import memorizz.internet_access.providers.exa as module

    seen = {}

    class Made:
        def raise_for_status(self):
            pass

        def json(self):
            return {"uuid": "abc"}

    class Made:
        status_code = 201

        def raise_for_status(self):
            pass

        def json(self):
            return {"uuid": "abc"}

    class Busy(Made):
        status_code = 429

    replies = [Busy(), Made()]
    monkeypatch.setattr(exa, "TEMPORARY_WEBHOOK_RETRY_SECONDS", 0)

    def post(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return replies.pop(0)

    monkeypatch.setattr(module.requests, "post", post)
    assert exa.temporary_webhook() == "https://webhook.site/abc"
    assert seen["url"] == "https://webhook.site/token"
    assert "test-key" not in str(seen["headers"])
    assert exa.get_config()["temporary_webhooks"] is True

    def down(*_, **__):
        raise module.requests.ConnectionError("down")

    monkeypatch.setattr(module.requests, "post", down)
    with pytest.raises(ExaError, match="EXA_MONITOR_WEBHOOK_URL"):
        exa.temporary_webhook()
