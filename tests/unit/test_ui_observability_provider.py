"""Observability in the UI: the agent form's section, the Settings keys and
the link that opens an agent's traces at its provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui import state  # noqa: E402
from memorizz.ui.agent_templates import values_from_form  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.helpers import observability_link  # noqa: E402

from .test_ui_agent_templates import _form, _Provider  # noqa: E402

pytestmark = pytest.mark.unit

SENTRY = {
    "observability_form": "1",
    "observability": "on",
    "observability_provider": "sentry",
    "observability_org": "acme",
}


@pytest.fixture
def ui(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _Provider()
    client = TestClient(create_app(), follow_redirects=False)
    # After the app loads any .env: Sentry is not set up.
    for name in ("MEMORIZZ_SENTRY_DSN", "SENTRY_DSN", "SENTRY_ORG", "SENTRY_SPOTLIGHT"):
        monkeypatch.delenv(name, raising=False)
    with monkeypatch.context() as patcher:
        patcher.setitem(state._state, "provider", provider)
        patcher.setitem(state._state, "provider_type", "filesystem")
        patcher.setitem(state._state, "connection_info", {"path": "/data/memory"})
        yield client, provider


def _create(client, **fields):
    created = client.post("/agents/new", data=_form(**fields))
    assert created.status_code == 302, created.text
    return created.headers["location"].split("/")[2]


def test_the_form_has_an_observability_section_traced_in_memorizz_by_default(ui):
    client, _ = ui
    html = client.get("/agents/new").text
    assert 'id="section-observability"' in html
    assert 'name="observability_form" value="1"' in html
    assert 'id="observability" name="observability" value="on" checked' in html
    assert '<option value="memorizz" selected>MemoRizz (default)</option>' in html
    assert '<option value="sentry" >Sentry</option>' in html
    assert 'id="observability-sentry" hidden>' in html
    assert 'href="#section-observability"' in html  # in the section list


def test_an_agent_saved_without_touching_the_section_is_traced_in_memorizz(ui):
    client, provider = ui
    agent_id = _create(client, observability_form="1", observability="on")
    agent = provider.agents[agent_id]
    assert agent.observability is True and agent.observability_config is None


def test_sentry_is_saved_shown_again_and_linked(ui, monkeypatch):
    client, provider = ui
    agent_id = _create(client, **SENTRY, observability_include_content="on")
    agent = provider.agents[agent_id]
    assert agent.observability is True
    assert agent.observability_config == {
        "provider": "sentry",
        "org": "acme",
        "include_content": True,
    }

    html = client.get(f"/agents/{agent_id}/edit").text
    assert '<option value="sentry" selected>Sentry</option>' in html
    assert 'name="observability_org" value="acme"' in html
    assert 'name="observability_include_content" value="on" checked' in html
    assert 'id="observability-sentry" >' in html  # shown
    # No DSN in this UI's environment: it says so, and still links out.
    assert "Needs setup" in html and "SENTRY_DSN" in html
    assert 'href="https://acme.sentry.io/explore/traces/"' in html

    monkeypatch.setenv("SENTRY_DSN", "https://key@o1.ingest.us.sentry.io/4242")
    html = client.get(f"/agents/{agent_id}/edit").text
    assert 'href="https://acme.sentry.io/explore/traces/?project=4242"' in html
    assert "key@o1" not in html  # the DSN is never put in a page


def test_turning_observability_off_keeps_the_provider(ui):
    client, provider = ui
    agent_id = _create(client, **SENTRY)
    off = {**SENTRY}
    off.pop("observability")
    client.post(f"/agents/{agent_id}/edit", data=_form(**off))
    agent = provider.agents[agent_id]
    assert agent.observability is False
    assert agent.observability_config == {"provider": "sentry", "org": "acme"}

    # Back to MemoRizz: nothing is left to configure.
    client.post(
        f"/agents/{agent_id}/edit",
        data=_form(
            observability_form="1",
            observability="on",
            observability_provider="memorizz",
        ),
    )
    agent = provider.agents[agent_id]
    assert agent.observability is True and agent.observability_config is None


def test_a_form_without_the_section_changes_nothing(ui):
    client, provider = ui
    agent_id = _create(client, **SENTRY)
    client.post(f"/agents/{agent_id}/edit", data=_form())
    agent = provider.agents[agent_id]
    assert agent.observability is True
    assert agent.observability_config == {"provider": "sentry", "org": "acme"}

    plain = _create(client)  # no section posted at all
    assert provider.agents[plain].observability is True
    assert provider.agents[plain].observability_config is None


def test_settings_the_form_does_not_show_survive_an_edit(ui):
    client, provider = ui
    agent_id = _create(client, **SENTRY)
    provider.agents[agent_id].observability_config = {
        "provider": "sentry",
        "org": "acme",
        "environment": "staging",
        "sample_rate": 0.25,
    }
    client.post(f"/agents/{agent_id}/edit", data=_form(**SENTRY))
    assert provider.agents[agent_id].observability_config == {
        "provider": "sentry",
        "org": "acme",
        "environment": "staging",
        "sample_rate": 0.25,
    }


def test_the_agents_page_links_to_the_provider(ui):
    client, provider = ui
    sentry_id = _create(client, agent_name="Sentry agent", **SENTRY)
    provider.agents["quiet"] = SimpleNamespace(
        agent_id="quiet",
        name="Quiet agent",
        persona=None,
        mcp_servers=[],
        observability=False,
        observability_config=None,
    )

    page = client.get("/agents").text
    link = (
        'href="https://acme.sentry.io/explore/traces/" target="_blank" '
        'rel="noopener noreferrer"'
    )
    # By the name, in the detail's actions and in its Configuration list.
    assert page.count(link) == 3
    assert page.count("Not traced") == 1
    assert provider.agents[sentry_id].observability is True


def test_the_link_describes_where_traces_go(monkeypatch):
    plain = observability_link({"agent_id": "a 1", "name": "Ada"})
    assert (plain["provider"], plain["external"], plain["enabled"]) == (
        "memorizz",
        False,
        True,
    )
    assert plain["url"] == "/traces?agent_id=a%201"

    sentry = observability_link(
        {
            "agent_id": "a1",
            "observability": False,
            "observability_config": {"provider": "sentry", "org": "acme"},
        }
    )
    assert sentry["enabled"] is False and sentry["external"] is True
    assert sentry["label"] == "Sentry" and sentry["ready"] is False
    assert sentry["url"] == "https://acme.sentry.io/explore/traces/"

    # A provider this install does not know falls back to MemoRizz.
    gone = observability_link({"observability_config": {"provider": "gone"}})
    assert gone["provider"] == "memorizz"


def test_the_local_server_shows_as_local_and_links_to_its_own_page(ui, monkeypatch):
    client, _ = ui
    agent_id = _create(client, **SENTRY)
    monkeypatch.setenv("SENTRY_SPOTLIGHT", "http://localhost:8969/stream")

    html = client.get(f"/agents/{agent_id}/edit").text
    assert "local development server only" in html
    assert "Needs setup" not in html

    link = observability_link(
        {"agent_id": agent_id, "observability_config": {"provider": "sentry"}}
    )
    assert (link["label"], link["ready"], link["local_only"]) == (
        "Sentry (local)",
        True,
        True,
    )
    assert link["url"].startswith("https://local.sentry.dev/?url=")
    # A turn that went only to the local server has no page on sentry.io.
    from memorizz.ui.helpers import observability_trace_link

    assert observability_trace_link({}, "sentry-local", "root-1") is None


def test_settings_has_the_sentry_keys_and_never_shows_the_dsn(ui, monkeypatch):
    client, _ = ui
    monkeypatch.setenv("SENTRY_DSN", "https://verysecretkey@o1.ingest.sentry.io/1")
    page = client.get("/settings").text
    assert 'id="settings-observability"' in page
    assert "SENTRY_DSN" in page and "SENTRY_ORG" in page
    assert "verysecretkey" not in page


def test_templates_keep_the_observability_fields():
    values = values_from_form(
        {
            "observability": "on",
            "observability_provider": "sentry",
            "observability_org": "acme",
            "observability_include_content": "on",
        }
    )
    assert values["observability"] is True
    assert values["observability_provider"] == "sentry"
    assert values["observability_org"] == "acme"
    assert values["observability_include_content"] is True
