"""Tests for Oracle embedding-mode selection in the local UI."""

import pytest

from memorizz._env_io import resolve_oracle_in_database_embedding_from_env


@pytest.mark.unit
def test_oracle_ui_defaults_to_in_database_embeddings(monkeypatch):
    monkeypatch.delenv("MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING", raising=False)
    monkeypatch.delenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", raising=False)

    assert resolve_oracle_in_database_embedding_from_env() is True


@pytest.mark.unit
def test_oracle_ui_preserves_external_embedding_defaults(monkeypatch):
    monkeypatch.delenv("MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING", raising=False)
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "openai")

    assert resolve_oracle_in_database_embedding_from_env() is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("configured_value", "expected"),
    [("true", True), ("false", False)],
)
def test_oracle_ui_explicit_embedding_mode_wins(
    monkeypatch, configured_value, expected
):
    monkeypatch.setenv("MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING", configured_value)
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "openai")

    assert resolve_oracle_in_database_embedding_from_env() is expected


@pytest.mark.unit
def test_oracle_ui_rejects_invalid_embedding_mode(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING", "sometimes")

    with pytest.raises(ValueError, match="must be true or false"):
        resolve_oracle_in_database_embedding_from_env()
