"""Oracle embedding compatibility tests for CLI provider construction."""

import os
from unittest.mock import patch

import pytest

from memorizz.cli import agent_factory, legacy


def _set_oracle_env(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_BACKEND", "oracle")
    monkeypatch.setenv("ORACLE_USER", "memorizz_user")
    monkeypatch.setenv("ORACLE_PASSWORD", "secret")
    monkeypatch.setenv("ORACLE_DSN", "localhost:1521/FREEPDB1")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "256")
    monkeypatch.delenv("MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING", raising=False)


@pytest.mark.unit
def test_interactive_cli_preserves_external_oracle_embedding_mode(monkeypatch):
    _set_oracle_env(monkeypatch)

    with patch(
        "memorizz.memory_provider.oracle.OracleProvider",
        side_effect=lambda config: config,
    ):
        config = agent_factory.detect_memory_provider(
            {"provider": "openai", "model": "gpt-4.1-mini"}, []
        )

    assert config.in_database_embedding is False
    assert config.lazy_vector_indexes is True


@pytest.mark.unit
def test_cli_automations_use_shared_provider_factory(monkeypatch):
    _set_oracle_env(monkeypatch)
    monkeypatch.delenv("MEMORIZZ_BACKEND")
    provider = object()

    with (
        patch(
            "memorizz.cli.agent_factory.detect_memory_provider",
            return_value=provider,
        ) as detect_provider,
        patch(
            "memorizz.automation.store.factory.get_automation_store",
            return_value=object(),
        ),
        patch("memorizz.automation.worker.run_worker"),
    ):
        assert legacy.run_automations() is True

    detect_provider.assert_called_once_with({}, [])
    assert os.environ["MEMORIZZ_BACKEND"] == "oracle"
