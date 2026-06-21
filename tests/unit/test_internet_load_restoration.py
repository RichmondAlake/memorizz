# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Internet access survives save/load — the capability automations depend on.

An automation runs ``MemAgent.load(agent_id, provider)`` then ``agent.run(...)``,
so the loaded agent must regain its internet provider + the ``internet_search``
tool. No network: the provider is a fake and create_internet_access_provider is
patched for the load path.
"""

from unittest.mock import patch

import pytest

from memorizz.internet_access import InternetAccessProvider
from memorizz.llms.llm_factory import create_llm_provider
from memorizz.memagent import MemAgent
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider


class _FakeInternet(InternetAccessProvider):
    provider_name = "tavily"

    def search(self, query, **kwargs):
        return {"query": query, "results": []}

    def fetch_url(self, url, **kwargs):
        return {"url": url, "content": ""}

    def get_provider_name(self):
        return "tavily"

    def is_enabled(self):
        return True


def _agent(tmp_path, internet=True):
    prov = FileSystemProvider(FileSystemConfig(root_path=str(tmp_path)))
    model = create_llm_provider({"provider": "ollama", "model": "qwen2.5:7b"})
    builder = (
        MemAgentBuilder()
        .with_model(model)
        .with_memory_provider(prov)
        .with_automations_enabled(False)
    )
    if internet:
        builder = builder.with_internet_access_provider(_FakeInternet())
    return prov, builder.build()


@pytest.mark.unit
def test_internet_tools_register_and_unregister(tmp_path):
    _, agent = _agent(tmp_path, internet=True)
    assert agent.has_internet_access()
    assert "internet_search" in agent.tool_manager.tools
    agent.with_internet_access_provider(None)
    assert "internet_search" not in agent.tool_manager.tools


@pytest.mark.unit
def test_load_restores_internet_access(tmp_path):
    prov, agent = _agent(tmp_path, internet=True)
    agent.save()
    with patch(
        "memorizz.internet_access.create_internet_access_provider",
        return_value=_FakeInternet(),
    ):
        loaded = MemAgent.load(agent.agent_id, memory_provider=prov)
    assert loaded.has_internet_access()
    assert loaded.get_internet_access_provider_name() == "tavily"
    assert "internet_search" in loaded.tool_manager.tools
