"""Tests for actionable configuration reporting in the CLI."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from memorizz.cli import commands
from memorizz.cli.app import app


class _Console:
    def __init__(self):
        self.output = []

    def print(self, *values, **kwargs):
        self.output.append(" ".join(str(value) for value in values))


@pytest.mark.unit
def test_config_reports_oracle_embedding_and_learning_mode(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_BACKEND", "oracle")
    monkeypatch.setenv("MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING", "false")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "256")
    monkeypatch.setenv("MEMORIZZ_CONTINUAL_LEARNING", "1")

    with patch(
        "memorizz.cli.agent_factory.detect_llm_config",
        return_value={"provider": "openai", "model": "gpt-4.1-mini"},
    ):
        result = CliRunner().invoke(app, ["config"])

    assert result.exit_code == 0
    assert "memory backend: oracle" in result.stdout
    assert "openai / text-embedding-3-small (256 dimensions)" in result.stdout
    assert "continual learning: enabled" in result.stdout


@pytest.mark.unit
def test_repl_config_reports_live_provider_and_learning_state():
    console = _Console()
    provider = SimpleNamespace(
        config=SimpleNamespace(
            in_database_embedding=False,
            embedding_provider="openai",
            embedding_config={"dimensions": 256},
        )
    )
    session = SimpleNamespace(
        console=console,
        provider=provider,
        provider_name="openai",
        model_name="gpt-4.1-mini",
        code_mode=False,
        agent=SimpleNamespace(
            continual_learning_manager=object(),
            get_internet_access_provider_name=lambda: None,
        ),
    )

    commands.cmd_config(session, "")

    output = "\n".join(console.output)
    assert "embedding:      openai (256 dimensions)" in output
    assert "continual learning: enabled" in output
