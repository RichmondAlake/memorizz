"""The playground's code sandbox choice: the console's default, or none."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.helpers import (
    _resolve_sandbox_provider_config,
    _validate_sandbox_provider_choice,
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", "graalpy")
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    provider.store_memagent(MemAgentModel(agent_id="agent-1", name="Vera"))
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {},
            "read_only": False,
        },
    ):
        yield TestClient(create_app(), follow_redirects=False)
    provider.close()


@pytest.mark.unit
def test_none_is_kept_instead_of_falling_back_to_the_default(client):
    url = "/agents/agent-1/playground"

    before = client.get(url).text
    assert "Default (GraalPy, from Settings)" in before
    assert '<option value="none" >None (disabled)</option>' in before

    saved = client.post(url + "/config", data={"sandbox_provider": "none"})
    assert saved.status_code in (302, 303)
    assert "config_error" not in saved.headers["location"]

    stored = state._state["provider"].retrieve_memagent("agent-1")
    assert stored.sandbox_provider == "none"
    after = client.get(url).text
    assert '<option value="none" selected>None (disabled)</option>' in after
    # Nothing is wrong with a sandbox that is off.
    assert _validate_sandbox_provider_choice("none") is None
    assert _resolve_sandbox_provider_config("none") is None

    # Going back to the default clears the choice; other saves leave it.
    client.post(url + "/config", data={"sandbox_provider": ""})
    assert (
        state._state["provider"].retrieve_memagent("agent-1").sandbox_provider is None
    )
    client.post(url + "/config", data={"sandbox_provider": "none"})
    client.post(url + "/config", data={"max_steps": "12"})
    assert (
        state._state["provider"].retrieve_memagent("agent-1").sandbox_provider == "none"
    )


@pytest.mark.unit
def test_an_agent_saved_with_none_loads_without_a_sandbox(client):
    from memorizz.memagent import MemAgent

    client.post("/agents/agent-1/playground/config", data={"sandbox_provider": "none"})

    agent = MemAgent.load("agent-1", memory_provider=state._state["provider"])

    assert agent.has_sandbox() is False
