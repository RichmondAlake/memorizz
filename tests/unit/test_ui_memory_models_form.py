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


def test_the_form_offers_decision_models_and_their_switches(ui, monkeypatch):
    client, _ = ui
    monkeypatch.setenv("TYPESAFE_API_KEY", "set")
    monkeypatch.delenv("LIQUID_API_KEY", raising=False)
    html = client.get("/agents/new").text
    assert 'id="decision_model" name="decision_model"' in html
    assert '<option value="" selected>None</option>' in html
    assert '<option value="jev:jev-1.14.0" >TypeSafe Jev · jev-1.14.0</option>' in html
    assert "Liquid d1 (hosted) · d1.1 (needs LIQUID_API_KEY)" in html
    for key in ("extraction_gate", "read_routing", "tool_selection", "cache_verifier"):
        assert f'name="decision_{key}"' in html


def test_the_decision_model_and_switches_are_saved_and_shown_again(ui):
    client, provider = ui
    created = client.post(
        "/agents/new",
        data=_form(
            decision_model="d1",
            decision_read_routing="on",
            decision_tool_selection="on",
            decision_extraction_gate="on",
            memory_models_form="1",
        ),
    )
    assert created.status_code == 302, created.text
    agent_id = created.headers["location"].split("/")[2]
    agent = provider.agents[agent_id]
    assert agent.decision_model == {"backend": "d1"}
    assert agent.retrieval_policy["read_routing"] is True
    assert agent.context_policy == {"tool_selection": True}
    assert agent.memory_routing["extraction_gate"] is True
    assert agent.memory_routing["enabled"] is False  # the gate alone

    html = client.get(f"/agents/{agent_id}/edit").text
    assert '<option value="d1" selected>' in html
    assert 'name="decision_read_routing" value="on" checked' in html
    assert 'name="decision_dedupe_writes" value="on" >' in html

    # A model named in code for one feature stays when its switch stays on.
    provider.agents[agent_id] = agent.model_copy(
        update={"context_policy": {"tool_selection": "jev", "tool_top_k": 3}}
    )
    client.post(
        f"/agents/{agent_id}/edit",
        data=_form(
            decision_model="openai_decisions:gpt-6-luna",
            decision_tool_selection="on",
            decision_cache_verifier="on",
            decision_relations="on",
            memory_models_form="1",
        ),
    )
    agent = provider.agents[agent_id]
    assert agent.decision_model == {
        "backend": "openai_decisions",
        "model": "gpt-6-luna",
    }
    assert agent.context_policy == {"tool_selection": "jev", "tool_top_k": 3}
    assert "read_routing" not in (agent.retrieval_policy or {})
    assert agent.memory_routing is None
    assert agent.semantic_cache_config == {"verifier": True}
    assert agent.memory_consolidation == {"enabled": False, "relations": True}

    # Everything off.
    client.post(
        f"/agents/{agent_id}/edit",
        data=_form(decision_model="", memory_models_form="1"),
    )
    agent = provider.agents[agent_id]
    assert agent.decision_model is None
    assert agent.context_policy == {"tool_top_k": 3}
    assert agent.semantic_cache_config is None and agent.memory_consolidation is None


def test_a_form_without_the_group_keeps_the_switches(ui):
    client, provider = ui
    created = client.post(
        "/agents/new",
        data=_form(
            decision_model="jev", decision_read_routing="on", memory_models_form="1"
        ),
    )
    agent_id = created.headers["location"].split("/")[2]
    client.post(f"/agents/{agent_id}/edit", data=_form())
    agent = provider.agents[agent_id]
    assert agent.decision_model == {"backend": "jev"}
    assert agent.retrieval_policy["read_routing"] is True


def test_templates_keep_the_decision_model():
    values = values_from_form({"decision_model": "d1", "decision_read_routing": "on"})
    assert values["decision_model"] == "d1"
    assert values["decision_read_routing"] is True
    assert values["decision_tool_selection"] is False
