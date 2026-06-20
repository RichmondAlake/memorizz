# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Route tests for the read-only agents JSON API.

First stateful router test: the connected provider is injected by patching the
shared ui.state._state dict, so no real database is touched.
"""

import types
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


class _Provider:
    def __init__(self, agents=None, byid=None):
        self._agents = agents or []
        self._byid = byid or {}

    def list_memagents(self):
        return self._agents

    def retrieve_memagent(self, agent_id):
        return self._byid.get(agent_id)


def _agent(**kw):
    return types.SimpleNamespace(**kw)


@pytest.mark.unit
def test_status_disconnected(client):
    with patch.dict(
        state._state,
        {"provider": None, "provider_type": None, "connection_info": {}},
    ):
        resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.json()["connected"] is False


@pytest.mark.unit
def test_status_connected(client):
    with patch.dict(
        state._state,
        {
            "provider": object(),
            "provider_type": "filesystem",
            "connection_info": {"a": 1},
        },
    ):
        resp = client.get("/api/status")
    body = resp.json()
    assert body["connected"] is True
    assert body["provider_type"] == "filesystem"


@pytest.mark.unit
def test_agents_not_connected(client):
    with patch.dict(state._state, {"provider": None}):
        resp = client.get("/api/agents")
    assert resp.status_code == 400


@pytest.mark.unit
def test_agents_list(client):
    prov = _Provider(
        agents=[_agent(agent_id="a1", instruction="hi", memory_provider="X", model="Y")]
    )
    with patch.dict(state._state, {"provider": prov}):
        resp = client.get("/api/agents")
    body = resp.json()
    assert body["count"] == 1
    assert body["agents"][0]["agent_id"] == "a1"
    # _serialize_agent strips non-serializable fields
    assert "memory_provider" not in body["agents"][0]
    assert "model" not in body["agents"][0]


@pytest.mark.unit
def test_get_agent_not_found(client):
    with patch.dict(state._state, {"provider": _Provider(byid={})}):
        resp = client.get("/api/agents/nope")
    assert resp.status_code == 404


@pytest.mark.unit
def test_get_agent_found(client):
    prov = _Provider(byid={"a1": _agent(agent_id="a1", instruction="hi")})
    with patch.dict(state._state, {"provider": prov}):
        resp = client.get("/api/agents/a1")
    assert resp.status_code == 200
    assert resp.json()["agent_id"] == "a1"


@pytest.mark.unit
def test_system_prompt(client):
    fake_agent = MagicMock()
    fake_agent._build_system_prompt.return_value = "SYSTEM PROMPT HERE"
    fake_agent.persona_manager = None
    fake_agent.tool_manager = None
    with patch.dict(state._state, {"provider": object()}), patch(
        "memorizz.memagent.MemAgent.load", return_value=fake_agent
    ):
        resp = client.get("/api/agents/a1/system-prompt")
    assert resp.status_code == 200
    body = resp.json()
    assert body["system_prompt"] == "SYSTEM PROMPT HERE"
    assert body["length"] == len("SYSTEM PROMPT HERE")
    assert body["persona"] is None
