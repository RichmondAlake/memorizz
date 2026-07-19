"""Unit tests for Oracle AI Database ONNX embeddings."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.memory_provider.oracle.embedding import (
    DEFAULT_MODEL_DIMENSIONS,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_URL,
    OracleInDatabaseEmbeddingProvider,
    in_database_embedding_options,
)
from memorizz.memory_provider.oracle.provider import OracleConfig, OracleProvider


class _FakeLob:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, chunk, offset=1):
        index = offset - 1
        end = index + len(chunk)
        if len(self.data) < end:
            self.data.extend(b"\x00" * (end - len(self.data)))
        self.data[index:end] = chunk

    def close(self):
        self.closed = True


class _FakeCursor:
    def __init__(self, pool):
        self.pool = pool
        self.row = None
        self.closed = False

    def execute(self, sql, params=None):
        params = params or {}
        normalized = " ".join(sql.split()).upper()
        self.pool.executions.append((normalized, params))
        if "FROM USER_MINING_MODELS" in normalized:
            self.row = (1 if self.pool.model_loaded else 0,)
        elif "DBMS_VECTOR.LOAD_ONNX_MODEL" in normalized:
            self.pool.model_loaded = True
            self.pool.loaded_bytes = bytes(params["model_data"].data)
            self.row = None
        elif "VECTOR_EMBEDDING" in normalized:
            self.pool.embedding_calls += 1
            self.row = ([0.25] * self.pool.dimensions,)

    def fetchone(self):
        return self.row

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self, pool):
        self.pool = pool
        self.commits = 0
        self.last_lob = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return _FakeCursor(self.pool)

    def createlob(self, _lob_type):
        self.last_lob = _FakeLob()
        return self.last_lob

    def commit(self):
        self.commits += 1


class _FakePool:
    def __init__(self, *, model_loaded=False, dimensions=3):
        self.model_loaded = model_loaded
        self.dimensions = dimensions
        self.executions = []
        self.embedding_calls = 0
        self.loaded_bytes = b""
        self.connection = _FakeConnection(self)

    def acquire(self):
        return self.connection


@pytest.mark.unit
def test_in_database_options_use_oracle_defaults(monkeypatch):
    for name in (
        "ORACLE_EMBEDDING_MODEL",
        "ORACLE_EMBEDDING_DIM",
        "ORACLE_EMBEDDING_ONNX_PATH",
        "ORACLE_EMBEDDING_ONNX_URL",
        "ORACLE_EMBEDDING_INSTALL_IF_MISSING",
    ):
        monkeypatch.delenv(name, raising=False)

    options = in_database_embedding_options({})

    assert options["model_name"] == DEFAULT_MODEL_NAME
    assert options["dimensions"] == DEFAULT_MODEL_DIMENSIONS
    assert options["install_if_missing"] is True
    assert options["onnx_url"].endswith("/all_MiniLM_L12_v2.onnx")


@pytest.mark.unit
def test_existing_model_is_not_reinstalled():
    pool = _FakePool(model_loaded=True, dimensions=3)
    provider = OracleInDatabaseEmbeddingProvider(
        pool,
        dimensions=3,
        install_if_missing=True,
    )

    assert provider.ensure_model() is False
    assert not any("LOAD_ONNX_MODEL" in sql for sql, _ in pool.executions)


@pytest.mark.unit
def test_model_install_and_embedding_are_database_backed_and_cached(tmp_path: Path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model-bytes")
    pool = _FakePool(dimensions=3)
    provider = OracleInDatabaseEmbeddingProvider(
        pool,
        dimensions=3,
        onnx_path=str(model_path),
        onnx_url=None,
    )

    assert provider.ensure_model() is True
    assert pool.loaded_bytes == b"model-bytes"

    first = provider.get_embedding("refund completed order")
    second = provider.get_embedding("refund completed order")

    assert first == [0.25, 0.25, 0.25]
    assert second == first
    assert pool.embedding_calls == 1
    assert provider.get_provider_info()["provider"] == "oracle_in_database"


@pytest.mark.unit
def test_missing_model_can_be_required_without_auto_install():
    provider = OracleInDatabaseEmbeddingProvider(
        _FakePool(model_loaded=False),
        dimensions=3,
        install_if_missing=False,
    )

    with pytest.raises(RuntimeError, match="install_if_missing is disabled"):
        provider.ensure_model()


@pytest.mark.unit
def test_model_name_must_be_safe_oracle_identifier():
    with pytest.raises(ValueError, match="unquoted Oracle identifiers"):
        OracleInDatabaseEmbeddingProvider(
            _FakePool(),
            model_name="MODEL); DROP TABLE TOOLBOX; --",
            dimensions=3,
        )


@pytest.mark.unit
def test_oracle_vector_dimension_parser_handles_quoted_ddl():
    ddl = 'CREATE TABLE "MEMORIZZ"."TOOLBOX" ' '("EMBEDDING" VECTOR(384, FLOAT32))'

    assert OracleProvider._vector_dimension_from_ddl(ddl, "EMBEDDING") == 384


@pytest.mark.unit
def test_oracle_dimension_validation_fails_before_first_write():
    provider = OracleProvider.__new__(OracleProvider)
    provider.config = SimpleNamespace(
        schema="MEMORIZZ_USER",
        user="MEMORIZZ_USER",
    )
    provider._embedding_provider = Mock()
    provider._embedding_provider.get_dimensions.return_value = 384
    provider.get_vector_schema_dimensions = Mock(
        return_value={
            "SKILLBOX.EMBEDDING": 384,
            "TOOLBOX.EMBEDDING": 256,
            "WORKFLOW_MEMORY.EMBEDDING": 256,
        }
    )

    with pytest.raises(RuntimeError, match="TOOLBOX.EMBEDDING=256"):
        provider.validate_vector_schema_dimensions()


@pytest.mark.unit
def test_oracle_dimension_validation_returns_matching_schema():
    provider = OracleProvider.__new__(OracleProvider)
    provider.config = SimpleNamespace(schema="GUIDE", user="GUIDE")
    provider._embedding_provider = Mock()
    provider._embedding_provider.get_dimensions.return_value = 384
    declared = {
        "SKILLBOX.EMBEDDING": 384,
        "TOOLBOX.EMBEDDING": 384,
    }
    provider.get_vector_schema_dimensions = Mock(return_value=declared)

    assert provider.validate_vector_schema_dimensions() == declared


@pytest.mark.unit
def test_vector_index_skips_tables_without_embedding_columns():
    provider = OracleProvider.__new__(OracleProvider)
    provider._vector_indexes_created = set()
    provider._table_has_column = Mock(return_value=False)
    provider._get_connection = Mock(
        side_effect=AssertionError("database should not be queried")
    )

    provider._ensure_vector_index(MemoryType.TOOL_LOG)

    provider._table_has_column.assert_called_once_with("tool_log", "embedding")
    provider._get_connection.assert_not_called()
    assert "idx_tool_log_vec" in provider._vector_indexes_created


@pytest.mark.unit
def test_oracle_provider_uses_in_database_adapter_by_default(monkeypatch):
    for name in (
        "ORACLE_EMBEDDING_MODEL",
        "ORACLE_EMBEDDING_DIM",
        "ORACLE_EMBEDDING_ONNX_PATH",
        "ORACLE_EMBEDDING_ONNX_URL",
        "ORACLE_EMBEDDING_INSTALL_IF_MISSING",
        "ORACLE_EMBEDDING_DOWNLOAD_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)

    pool = object()
    adapter = Mock()
    adapter.get_default_model.return_value = DEFAULT_MODEL_NAME
    adapter.get_dimensions.return_value = DEFAULT_MODEL_DIMENSIONS
    adapter_factory = Mock(return_value=adapter)
    set_global = Mock()

    monkeypatch.setattr(
        "memorizz.memory_provider.oracle.embedding."
        "OracleInDatabaseEmbeddingProvider",
        adapter_factory,
    )
    monkeypatch.setattr(
        "memorizz.embeddings.set_global_embedding_manager",
        set_global,
    )

    provider = OracleProvider.__new__(OracleProvider)
    provider.pool = pool
    config = OracleConfig(
        user="user",
        password="pass",
        dsn="localhost:1521/FREEPDB1",
    )

    result = provider._setup_embedding_provider(config)

    assert result is adapter
    adapter_factory.assert_called_once_with(
        pool,
        model_name=DEFAULT_MODEL_NAME,
        dimensions=DEFAULT_MODEL_DIMENSIONS,
        onnx_path=None,
        onnx_url=DEFAULT_MODEL_URL,
        install_if_missing=True,
        request_timeout=600.0,
    )
    adapter.ensure_model.assert_called_once_with()
    set_global.assert_called_once_with(adapter)
