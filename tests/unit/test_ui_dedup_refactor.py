# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The helpers that replaced repeated blocks in the agent form and playground routes.

``_render_agent_form`` builds the one ``agent_form.html`` context, the parse
helpers hold the checks create and edit both ran, ``_config_error_redirect``
sends the playground's config panel back with its error, and the page scripts
share ``base.html``'s ``escapeHtml``.
"""

import re
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.routers import agents_crud, playground  # noqa: E402

UI_DIR = Path(agents_crud.__file__).resolve().parents[1]
PAGE_SCRIPTS = ("harness-judge.js", "harness-trajectory.js", "memory-evolution.js")
AGENT_FORM = {
    "agent_name": "Dedup Agent",
    "instruction": "You are a test agent.",
    "application_mode": "assistant",
    "max_steps": "10",
    "tool_access": "private",
    "llm_provider": "openai",
    "llm_model": "gpt-4.1-mini",
}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def connected(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "ui-dedup", lazy_vector_indexes=True)
    )
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    ):
        yield provider


def _saved_agent_id(client):
    resp = client.post("/agents/new", data=AGENT_FORM)
    assert resp.status_code in (302, 303), resp.text[:300]
    return resp.headers["location"].split("/agents/")[1].split("/")[0]


# --- _render_agent_form -------------------------------------------------------


class _Templates:
    """Stands in for the Jinja templates and records the context it was given."""

    def __init__(self):
        self.calls = []

    def TemplateResponse(self, name, context):
        self.calls.append((name, context))
        return "rendered"


@pytest.fixture()
def rendered(monkeypatch):
    stub = _Templates()
    monkeypatch.setattr(agents_crud, "templates", stub)
    monkeypatch.setattr(
        agents_crud,
        "_build_agent_nav_items",
        lambda active_agent_id: ["nav", active_agent_id],
    )
    with patch.dict(
        state._state,
        {"provider_type": "filesystem", "connection_info": {"root_path": "/r"}},
    ):
        yield stub


def test_create_form_context_has_no_agent_nav(rendered):
    result = agents_crud._render_agent_form(
        "req", error=None, form_data={"agent_id": "", "agent_name": "N"}
    )
    assert result == "rendered"
    name, context = rendered.calls[0]
    assert name == "agent_form.html"
    assert context == {
        "request": "req",
        "provider_type": "filesystem",
        "connection_info": {"root_path": "/r"},
        "active_page": "agents",
        "form_title": "Create Agent",
        "form_action": "/agents/new",
        "is_edit": False,
        "error": None,
        "agent_id": "",
        "agent_name": "N",
    }


def test_edit_form_context_carries_the_agent_nav_and_error(rendered):
    agents_crud._render_agent_form(
        "req", error="boom", form_data={"agent_id": "a1"}, agent_id="a1"
    )
    _, context = rendered.calls[0]
    assert context["agents_nav"] == ["nav", "a1"]
    assert context["active_agent_id"] == "a1"
    assert context["form_title"] == "Edit Agent"
    assert context["form_action"] == "/agents/a1/edit"
    assert context["is_edit"] is True
    assert context["error"] == "boom"
    assert context["agent_id"] == "a1"


# --- the parse helpers create and edit share ----------------------------------


def test_harness_fields_normalise_and_resolve_the_workspace(tmp_path):
    parsed = agents_crud._parse_harness_form_fields(
        " Delegate ", "Claude_Code", str(tmp_path), None
    )
    assert parsed == ("delegate", "claude-code", str(tmp_path.resolve()), None)
    assert agents_crud._parse_harness_form_fields("", "", "", None) == (
        "",
        "auto",
        "",
        None,
    )


@pytest.mark.parametrize(
    ("mode", "harness", "workspace", "message"),
    [
        (
            "weird",
            "auto",
            "",
            "Meta-harness mode must be disabled, delegate, or runtime",
        ),
        ("", "nope", "", "Default harness must be auto, codex, claude-code"),
        ("", "auto", "{missing}", "Harness workspace must be an existing directory"),
        ("", "auto", "{file}", "Harness workspace must be an existing directory"),
    ],
)
def test_harness_fields_report_the_first_failing_check(
    tmp_path, mode, harness, workspace, message
):
    (tmp_path / "f.txt").write_text("x")
    workspace = workspace.format(missing=tmp_path / "missing", file=tmp_path / "f.txt")
    *_, error = agents_crud._parse_harness_form_fields(mode, harness, workspace, None)
    assert error.startswith(message)


def test_harness_fields_keep_an_earlier_error_and_the_workspace_as_typed(tmp_path):
    parsed = agents_crud._parse_harness_form_fields(
        "weird", "nope", str(tmp_path / "missing"), "earlier"
    )
    assert parsed == ("weird", "nope", str(tmp_path / "missing"), "earlier")
    # Within the helper the mode check comes first.
    *_, error = agents_crud._parse_harness_form_fields("weird", "nope", "", None)
    assert error.startswith("Meta-harness mode")


def test_default_timezone_blank_is_unset_and_unknown_is_the_error():
    assert agents_crud._parse_default_timezone("  ", None) == (None, None)
    assert agents_crud._parse_default_timezone("Europe/London", None) == (
        "Europe/London",
        None,
    )
    value, error = agents_crud._parse_default_timezone("Not/AZone", None)
    assert value == "Not/AZone" and error.startswith("Invalid timezone")
    assert agents_crud._parse_default_timezone("Not/AZone", "earlier") == (
        "Not/AZone",
        "earlier",
    )


def test_whatsapp_values_only_build_a_config_when_enabled():
    assert agents_crud._whatsapp_form_values(None, "hello") == (False, None)
    assert agents_crud._whatsapp_form_values("on", "  hello  ") == (
        True,
        {"welcome_message": "hello", "auto_reply_enabled": True, "timeout_seconds": 60},
    )
    assert agents_crud._whatsapp_form_values("on", "")[1]["welcome_message"] is None


# --- the routes, end to end ---------------------------------------------------


def test_create_form_shows_the_harness_error_with_the_submitted_values(
    client, connected
):
    resp = client.post(
        "/agents/new",
        data={**AGENT_FORM, "default_harness": "nope", "agent_name": "Kept Name"},
    )
    assert resp.status_code == 200
    assert "Default harness must be" in resp.text
    assert "Kept Name" in resp.text
    assert not connected.list_memagents()


def test_create_form_reports_an_unknown_timezone(client, connected):
    resp = client.post(
        "/agents/new", data={**AGENT_FORM, "default_timezone": "Not/AZone"}
    )
    assert resp.status_code == 200
    assert "Invalid timezone" in resp.text


def test_edit_form_shows_the_meta_harness_error_and_keeps_the_agent(client, connected):
    agent_id = _saved_agent_id(client)
    resp = client.post(
        f"/agents/{agent_id}/edit",
        data={**AGENT_FORM, "instruction": "changed", "meta_harness_mode": "weird"},
    )
    assert resp.status_code == 200
    assert "Meta-harness mode must be" in resp.text
    assert f"/agents/{agent_id}/edit" in resp.text
    existing = connected.retrieve_memagent(agent_id)
    assert getattr(existing, "instruction") == AGENT_FORM["instruction"]


def test_config_error_redirect_quotes_the_message():
    response = playground._config_error_redirect("a1", "bad <value> & more")
    assert response.status_code == 302
    assert response.headers["location"] == (
        "/agents/a1/playground?config_error=" + quote("bad <value> & more")
    )


def test_playground_config_redirects_with_the_timezone_error(client, connected):
    agent_id = _saved_agent_id(client)
    resp = client.post(
        f"/agents/{agent_id}/playground/config",
        data={
            "llm_provider": "openai",
            "llm_model": "gpt-4.1-mini",
            "max_steps": "10",
            "default_timezone": "Not/AZone",
        },
    )
    assert resp.status_code == 302
    location = resp.headers["location"]
    assert location.startswith(f"/agents/{agent_id}/playground?config_error=")
    assert "Invalid%20timezone" in location


def test_playground_imports_quote_once():
    source = Path(playground.__file__).read_text(encoding="utf-8")
    assert source.count("from urllib.parse import quote") == 1


# --- one escapeHtml for the page scripts --------------------------------------

ALIAS = re.compile(
    r"const esc = typeof escapeHtml === [\"']function[\"'] \? escapeHtml :"
)


@pytest.mark.parametrize("name", PAGE_SCRIPTS)
def test_page_script_aliases_the_shared_escape_helper(name):
    source = (UI_DIR / "static" / "js" / name).read_text(encoding="utf-8")
    assert len(ALIAS.findall(source)) == 1
    assert "function escapeHtml" not in source


def test_base_defines_escape_html_before_the_page_scripts_load():
    base = (UI_DIR / "templates" / "base.html").read_text(encoding="utf-8")
    assert base.index("function escapeHtml(") < base.index("{% block head %}")
    pages = (
        "harnesses.html",
        "harness_chat.html",
        "memory_evolution.html",
        "observability.html",
        "playground.html",
    )
    for template in pages:
        source = (UI_DIR / "templates" / template).read_text(encoding="utf-8")
        assert source.lstrip().startswith('{% extends "base.html" %}'), template
        assert any(script in source for script in PAGE_SCRIPTS), template


def test_connect_page_is_standalone_and_keeps_its_own_escape_helper():
    # connect.html is its own document (no base.html), so the global is not
    # in scope there and its escapeHtml stays.
    connect = (UI_DIR / "templates" / "connect.html").read_text(encoding="utf-8")
    assert "{% extends" not in connect
    assert connect.count("function escapeHtml(") == 1
