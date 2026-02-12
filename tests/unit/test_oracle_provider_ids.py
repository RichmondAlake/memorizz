"""Unit tests for Oracle RAW UUID normalization helpers."""

import uuid

import pytest

from memorizz.memory_provider.oracle.provider import OracleConfig, OracleProvider


@pytest.mark.unit
def test_normalize_raw_uuid_accepts_raw_bytes():
    """Existing RAW(16) values should pass through unchanged."""
    raw_value = uuid.uuid4().bytes

    normalized = OracleProvider._normalize_raw_uuid(raw_value)

    assert normalized == raw_value


@pytest.mark.unit
def test_normalize_raw_uuid_accepts_string_uuid():
    """String UUID values should convert to RAW(16)."""
    value = uuid.uuid4()

    normalized = OracleProvider._normalize_raw_uuid(str(value))

    assert normalized == value.bytes


@pytest.mark.unit
def test_normalize_raw_uuid_rejects_invalid_payload():
    """Invalid byte payloads should raise ValueError."""
    with pytest.raises(ValueError):
        OracleProvider._normalize_raw_uuid(b"invalid")


@pytest.mark.unit
def test_resolve_embedding_defaults_from_env_openai(monkeypatch):
    """OpenAI defaults should map dimensions/api key into embedding config."""
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    resolved = OracleProvider._resolve_embedding_defaults_from_env()

    assert resolved is not None
    provider, config = resolved
    assert provider == "openai"
    assert config["model"] == "text-embedding-3-small"
    assert config["dimensions"] == 1536
    assert config["api_key"] == "test-openai-key"


@pytest.mark.unit
def test_resolve_embedding_defaults_from_env_voyageai_uses_output_dimension(
    monkeypatch,
):
    """VoyageAI defaults should map dimensions to output_dimension."""
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "voyageai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "voyage-3-lite")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")

    resolved = OracleProvider._resolve_embedding_defaults_from_env()

    assert resolved is not None
    provider, config = resolved
    assert provider == "voyageai"
    assert config["model"] == "voyage-3-lite"
    assert config["output_dimension"] == 1024
    assert config["api_key"] == "test-voyage-key"
    assert "dimensions" not in config


@pytest.mark.unit
def test_resolve_embedding_defaults_from_env_unsupported_provider_returns_none(
    monkeypatch,
):
    """Unsupported providers should be ignored safely."""
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "unsupported-provider")

    assert OracleProvider._resolve_embedding_defaults_from_env() is None


@pytest.mark.unit
def test_apply_embedding_defaults_from_env_merges_with_explicit_embedding_config(
    monkeypatch,
):
    """Explicit embedding_config values should override env defaults."""
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    config = OracleConfig(
        user="user",
        password="pass",
        dsn="localhost:1521/FREEPDB1",
        embedding_provider=None,
        embedding_config={"model": "text-embedding-3-large", "dimensions": 1024},
    )

    OracleProvider._apply_embedding_defaults_from_env_if_needed(config)

    assert config.embedding_provider == "openai"
    assert config.embedding_config["model"] == "text-embedding-3-large"
    assert config.embedding_config["dimensions"] == 1024
    assert config.embedding_config["api_key"] == "env-key"
