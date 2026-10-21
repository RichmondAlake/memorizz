"""Agent templates on the create page: built-in templates fill the form in,
the form saves as a template of the user's own, and a template's MCP
connections are attached when the agent is created."""

from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memagent import templates as agent_templates  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.agent_templates import values_from_form  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.helpers import (  # noqa: E402
    DEFAULT_LLM_MODEL_BY_PROVIDER,
    LLM_MODEL_CATALOG,
)

pytestmark = pytest.mark.unit


class _Provider:
    def __init__(self):
        self.agents = {
            "agent-1": SimpleNamespace(
                agent_id="agent-1", name="Assistant", persona=None, mcp_servers=[]
            )
        }

    def list_memagents(self):
        return list(self.agents.values())

    def retrieve_memagent(self, agent_id):
        return self.agents.get(agent_id)

    def store_memagent(self, agent):
        agent_id = getattr(agent, "agent_id", None) or f"agent-{len(self.agents)}"
        agent.agent_id = agent_id
        self.agents[agent_id] = agent
        return agent_id

    def list_personas(self, *args, **kwargs):
        return []

    def store_persona(self, persona, *args, **kwargs):
        return "persona-1"


@pytest.fixture
def ui(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _Provider()
    client = TestClient(create_app(), follow_redirects=False)
    # After the app loads any .env: no search provider has a key.
    for name in (
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
        "MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    with monkeypatch.context() as patcher:
        patcher.setitem(state._state, "provider", provider)
        patcher.setitem(state._state, "provider_type", "filesystem")
        patcher.setitem(state._state, "connection_info", {"path": "/data/memory"})
        yield client, provider, tmp_path


def _page_templates(html):
    marker = "window.AGENT_FORM_TEMPLATES = "
    start = html.index(marker) + len(marker)
    return json.loads(html[start : html.index(";\n", start)])


def _form(**fields):
    data = {
        "agent_name": "Ops helper",
        "instruction": "Help the ops team.",
        "application_mode": "assistant",
        "max_steps": "20",
        "tool_access": "private",
        "llm_provider": "openai",
        "llm_model": "gpt-6.1-sol",
        "enable_entity_memory": "on",
        "automations_enabled": "1",
        "skill_injection_role": "user",
    }
    data.update(fields)
    return data


def test_the_create_page_offers_every_template_and_the_overview(ui):
    client, _, _ = ui
    html = client.get("/agents/new").text
    cards = {card["key"]: card for card in _page_templates(html)}
    assert list(cards) == [
        "productivity",
        "finance",
        "research",
        "support",
        "writing",
        "tutor",
    ]
    for key in cards:
        assert f'data-use-template="{key}"' in html
    support = cards["support"]["values"]
    assert support["sealed_conversations"] is True
    assert support["automations_enabled"] is False
    assert "Sealed conversations" in cards["support"]["chips"]
    assert cards["productivity"]["needs_google_client"] is True
    # No key for a search provider: the template leaves web search off.
    assert cards["research"]["values"]["internet_provider"] == ""
    assert any("Tavily or Firecrawl key" in chip for chip in cards["research"]["chips"])
    assert (
        "not a licensed financial adviser" in cards["finance"]["values"]["instruction"]
    )

    for section in ("identity", "model", "memory", "tools", "runtime", "channels"):
        assert f'id="section-{section}"' in html
    assert 'id="builder-code"' in html and 'id="save-template-toggle"' in html
    builder = html.split("window.AGENT_BUILDER = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(builder)["store"] == {
        "type": "filesystem",
        "path": "/data/memory",
    }

    applied = client.get("/agents/new?template=finance").text
    assert 'window.AGENT_FORM_APPLY_TEMPLATE = "finance"' in applied


def test_a_template_with_connections_attaches_them_when_the_agent_is_saved(ui):
    client, provider, _ = ui
    response = client.post(
        "/agents/new",
        data=_form(
            template_key="productivity",
            google_client_id="client.apps.googleusercontent.com",
            google_client_secret="s3cret",
        ),
    )
    assert response.status_code == 303, response.text
    location = response.headers["location"]
    assert location.startswith("/mcp?agent_id=") and "created=productivity" in location
    agent = provider.agents[location.split("agent_id=")[1].split("&")[0]]
    assert {server["name"] for server in agent.mcp_servers} == {
        "gmail",
        "calendar",
        "notion",
    }
    assert "s3cret" not in json.dumps(agent.mcp_servers)

    plain = client.post("/agents/new", data=_form(template_key="finance"))
    assert plain.status_code == 302
    assert plain.headers["location"].endswith("/playground")


def test_saving_the_form_as_a_template_keeps_settings_but_no_secrets(ui):
    client, _, home = ui
    response = client.post(
        "/agents/templates/save",
        data=_form(
            template_title="Ops helper",
            template_summary="Runbooks and incidents",
            template_key="productivity",
            llm_config_json=json.dumps(
                {"temperature": 0.2, "api_key": "sk-live", "nested": {"token": "t"}}
            ),
            memory_ids="mem-1",
            delegate_ids="agent-1",
            google_client_secret="s3cret",
            sealed_conversations="on",
            memory_consolidation="on",
            consolidation_every_n_turns="12",
        ),
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["key"] == "my-ops-helper"
    assert body["url"].startswith("/agents/new?template=my-ops-helper")

    path = home / "agent-templates" / "my-ops-helper.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    raw = path.read_text()
    for secret in ("sk-live", "s3cret", '"t"', "mem-1", "agent-1"):
        assert secret not in raw
    saved = json.loads(raw)
    values = saved["values"]
    assert json.loads(values["llm_config_json"]) == {"temperature": 0.2, "nested": {}}
    assert values["sealed_conversations"] is True and values["semantic_cache"] is False
    assert values["consolidation_every_n_turns"] == "12"
    assert values["llm_model"] == "gpt-6.1-sol"
    assert saved["mcp_presets"] == ["gmail", "google-calendar", "notion"]

    cards = {
        card["key"]: card for card in _page_templates(client.get(body["url"]).text)
    }
    mine = cards["my-ops-helper"]
    assert mine["builtin"] is False and mine["title"] == "Ops helper"
    assert "openai · gpt-6.1-sol" in mine["chips"]

    # An agent made from it gets the template's connections too.
    created = client.post("/agents/new", data=_form(template_key="my-ops-helper"))
    assert "created=my-ops-helper" in created.headers["location"]

    deleted = client.post("/agents/templates/my-ops-helper/delete")
    assert deleted.status_code == 303 and not path.exists()
    assert client.post("/agents/templates/finance/delete").status_code == 404
    assert client.post("/agents/templates/save", data={}).status_code == 400


def test_the_edit_page_saves_templates_but_has_no_gallery(ui):
    client, provider, _ = ui
    created = client.post("/agents/new", data=_form())
    agent_id = created.headers["location"].split("/")[2]
    html = client.get(f"/agents/{agent_id}/edit").text
    assert 'id="save-template-toggle"' in html
    assert "window.AGENT_FORM_TEMPLATES" not in html
    assert 'data-is-edit="1"' in html
    assert f'name="source_agent_id" value="{agent_id}"' in html


def test_a_template_saved_from_an_agent_keeps_its_preset_connections(ui):
    client, provider, home = ui
    created = client.post(
        "/agents/new", data=_form(template_key="productivity", agent_name="Exec")
    )
    agent_id = created.headers["location"].split("agent_id=")[1].split("&")[0]
    provider.agents[agent_id].mcp_servers.append(
        {"name": "internal", "url": "https://mcp.internal.example/mcp"}
    )
    html = client.get(f"/agents/{agent_id}/edit").text
    builder = json.loads(html.split("window.AGENT_BUILDER = ", 1)[1].split(";\n", 1)[0])
    assert sorted(builder["mcp_servers"]) == ["calendar", "gmail", "internal", "notion"]

    saved = client.post(
        "/agents/templates/save",
        data=_form(template_title="Exec copy", source_agent_id=agent_id),
        headers={"Accept": "application/json"},
    )
    assert saved.status_code == 200, saved.text
    raw = json.loads((home / "agent-templates" / "my-exec-copy.json").read_text())
    assert sorted(raw["mcp_presets"]) == ["gmail", "google-calendar", "notion"]


def test_saving_under_the_same_name_replaces_the_template(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    first = agent_templates.save_user_template("Weekly planner", {"agent_name": "A"})
    agent_templates.save_user_template("Weekly  Planner", {"agent_name": "B"})
    saved = agent_templates.load_user_templates()
    assert [template.key for template in saved] == [first.key] == ["my-weekly-planner"]
    assert agent_templates.get_template(first.key).form_values()["agent_name"] == "B"
    with pytest.raises(ValueError):
        agent_templates.save_user_template("  ", {})
    assert agent_templates.delete_user_template("finance") is False


def test_values_from_form_reads_unchecked_boxes_as_off():
    values = values_from_form({"agent_name": " A ", "tool_cache": "on"})
    assert values["agent_name"] == "A"
    assert values["tool_cache"] is True and values["semantic_cache"] is False
    assert "memory_ids" not in values
    assert values_from_form({"llm_config_json": "not json"})["llm_config_json"] == ""


def test_the_model_picker_lists_current_openai_and_anthropic_models():
    openai = [model["value"] for model in LLM_MODEL_CATALOG["openai"]]
    anthropic = [model["value"] for model in LLM_MODEL_CATALOG["anthropic"]]
    assert openai[:3] == ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"]
    assert anthropic[:4] == [
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-fable-5-1",
    ]
    assert "gpt-4o" not in openai and "claude-3-haiku-20240307" not in anthropic
    assert DEFAULT_LLM_MODEL_BY_PROVIDER["openai"] in openai
    assert DEFAULT_LLM_MODEL_BY_PROVIDER["anthropic"] == "claude-opus-5-5"
