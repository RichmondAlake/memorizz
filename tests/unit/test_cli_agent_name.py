"""The CLI shows the agent's name and /name changes it in the store."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console

from memorizz import MemAgent
from memorizz.cli import commands
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit


@pytest.fixture
def session(tmp_path, monkeypatch):
    from memorizz.llms.openai import OpenAI

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", Mock())
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    agent = MemAgent(
        model=OpenAI(model="gpt-test"), memory_provider=provider, auto_register=False
    )
    agent.instruction = "stored instruction"
    agent.save()
    # What the CLI changes on its running agent is never saved by /name.
    agent.instruction = "runtime only"
    output = StringIO()
    yield SimpleNamespace(
        agent=agent,
        provider=provider,
        console=Console(file=output, color_system=None, width=160),
        output=output,
    )
    agent.close()


def test_name_shows_then_renames_only_the_name(session):
    commands.dispatch("/name", session)
    assert "This agent has no name" in session.output.getvalue()
    assert "Usage: /name <new name>" in session.output.getvalue()

    commands.dispatch("/name   Productivity   engine ", session)
    assert "Agent renamed → Productivity engine" in session.output.getvalue()
    assert session.agent.name == "Productivity engine"
    stored = session.provider.retrieve_memagent(session.agent.agent_id)
    assert stored.name == "Productivity engine"
    assert stored.instruction == "stored instruction"

    commands.dispatch("/rename Coach", session)
    assert session.provider.retrieve_memagent(session.agent.agent_id).name == "Coach"
    commands.dispatch("/name", session)
    assert "Agent name: Coach" in session.output.getvalue()


def test_a_rename_made_elsewhere_reaches_the_running_agent(session):
    record = session.provider.retrieve_memagent(session.agent.agent_id)
    session.provider.store_memagent(record.model_copy(update={"name": "From the UI"}))
    commands.refresh_agent_name(session)
    assert session.agent.name == "From the UI"


def test_short_n_still_starts_a_new_conversation():
    assert commands.ALIASES["n"] == "new"
    assert commands.ALIASES["rename"] == "name"
