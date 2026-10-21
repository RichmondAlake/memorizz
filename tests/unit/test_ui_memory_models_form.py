"""The agent form's Memory models group: reranker and memory routing."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui import state  # noqa: E402
from memorizz.ui.agent_templates import values_from_form  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

from .test_ui_agent_templates import _form, _Provider  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def ui(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _Provider()
    client = TestClient(create_app(), follow_redirects=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "set")
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with monkeypatch.context() as patcher:
        patcher.setitem(state._state, "provider", provider)
        patcher.setitem(state._state, "provider_type", "filesystem")
        patcher.setitem(state._state, "connection_info", {"path": "/data/memory"})
        yield client, provider


def test_the_form_offers_hosted_rerankers_and_routers(ui):
    client, _ = ui
    html = client.get("/agents/new").text
    assert 'id="memory-models"' in html
    assert '<option value="voyage:rerank-2.5" >Voyage · rerank-2.5</option>' in html
    assert "Cohere · rerank-v3.5 (needs COHERE_API_KEY)" in html
    assert 'value="openai_decisions:gpt-6-luna"' in html
    assert '<option value="" selected>Follow the store (none set)</option>' in html


def test_reranker_and_routing_are_saved_and_shown_again(ui):
    client, provider = ui
    created = client.post(
        "/agents/new",
        data=_form(
            reranker="voyage:rerank-2.5-lite",
            memory_routing="openai_decisions:gpt-6-luna",
            memory_routing_capture="on",
        ),
    )
    assert created.status_code == 302, created.text
    agent_id = created.headers["location"].split("/")[2]
    agent = provider.agents[agent_id]
    assert agent.reranker == {"provider": "voyage", "model": "rerank-2.5-lite"}
    assert agent.memory_routing == {
        "enabled": True,
        "router": "openai_decisions",
        "capture_turns": True,
        "min_confidence": 0.5,
        "model": "gpt-6-luna",
    }

    html = client.get(f"/agents/{agent_id}/edit").text
    assert '<option value="voyage:rerank-2.5-lite" selected>' in html
    assert '<option value="openai_decisions:gpt-6-luna" selected>' in html

    # Routing only on request, then both off.
    data = _form(reranker="off", memory_routing="llm", memory_models_form="1")
    client.post(f"/agents/{agent_id}/edit", data=data)
    agent = provider.agents[agent_id]
    assert agent.reranker == {"provider": "off"}
    assert agent.memory_routing["router"] == "llm"
    assert agent.memory_routing["capture_turns"] is False
    client.post(
        f"/agents/{agent_id}/edit",
        data=_form(reranker="", memory_routing="", memory_models_form="1"),
    )
    agent = provider.agents[agent_id]
    assert agent.reranker is None and agent.memory_routing is None


def test_templates_keep_the_memory_models():
    values = values_from_form(
        {
            "reranker": "voyage:rerank-2.5",
            "memory_routing": "auto",
            "memory_routing_capture": "on",
        }
    )
    assert values["reranker"] == "voyage:rerank-2.5"
    assert values["memory_routing"] == "auto"
    assert values["memory_routing_capture"] is True
