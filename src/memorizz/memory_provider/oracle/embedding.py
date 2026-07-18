# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Oracle AI Database-backed text embeddings."""

import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import oracledb
import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "ALL_MINILM_L12_V2"
DEFAULT_MODEL_DIMENSIONS = 384
DEFAULT_MODEL_URL = (
    "https://objectstorage.us-ashburn-1.oraclecloud.com/n/adwc4pm/"
    "b/OML-Resources/o/all_MiniLM_L12_v2.onnx"
)

_MODEL_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]{0,122}$")
_CACHE_MAX_ENTRIES = 512
_CHUNK_SIZE = 1024 * 1024


class OracleInDatabaseEmbeddingProvider:
    """Generate embeddings with an ONNX model loaded into Oracle AI Database.

    The provider deliberately implements the same small interface as MemoRizz's
    ``EmbeddingManager`` so it can be installed as the global embedding manager.
    This keeps workflow, skill, toolbox, and query embeddings in one vector
    space even when those components call the module-level embedding helpers.
    """

    def __init__(
        self,
        pool: Any,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        dimensions: int = DEFAULT_MODEL_DIMENSIONS,
        onnx_path: Optional[str] = None,
        onnx_url: Optional[str] = DEFAULT_MODEL_URL,
        install_if_missing: bool = True,
        request_timeout: float = 600.0,
    ) -> None:
        normalized_model = str(model_name or "").strip().upper()
        if not _MODEL_IDENTIFIER.fullmatch(normalized_model):
            raise ValueError(
                "Oracle in-database embedding model names must be unquoted Oracle "
                "identifiers (letters, digits, _, $, or #; starting with a letter)."
            )
        if int(dimensions) <= 0:
            raise ValueError("Embedding dimensions must be a positive integer")

        self.pool = pool
        self.model_name = normalized_model
        self.dimensions = int(dimensions)
        self.onnx_path = str(onnx_path) if onnx_path else None
        self.onnx_url = str(onnx_url) if onnx_url else None
        self.install_if_missing = bool(install_if_missing)
        self.request_timeout = float(request_timeout)
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_lock = threading.Lock()

    def ensure_model(self) -> bool:
        """Install the configured ONNX model when absent.

        Returns ``True`` when this call installed the model and ``False`` when
        the model was already present.
        """
        if self.is_model_loaded():
            return False
        if not self.install_if_missing:
            raise RuntimeError(
                f"Oracle ONNX model {self.model_name!r} is not installed and "
                "install_if_missing is disabled."
            )

        with self.pool.acquire() as conn:
            cursor = conn.cursor()
            model_lob = None
            try:
                model_lob = conn.createlob(oracledb.DB_TYPE_BLOB)
                self._write_model_blob(model_lob)
                cursor.execute(
                    """
                    BEGIN
                        DBMS_VECTOR.LOAD_ONNX_MODEL(
                            :model_name,
                            :model_data,
                            JSON(:metadata)
                        );
                    END;
                    """,
                    {
                        "model_name": self.model_name,
                        "model_data": model_lob,
                        "metadata": (
                            '{"function":"embedding",'
                            '"embeddingOutput":"embedding",'
                            '"input":{"input":["DATA"]}}'
                        ),
                    },
                )
                conn.commit()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not install Oracle ONNX embedding model "
                    f"{self.model_name!r}. The database user needs CREATE MINING "
                    "MODEL and EXECUTE ON DBMS_VECTOR. Configure a readable "
                    "embedding_config['onnx_path'] or reachable "
                    "embedding_config['onnx_url']. "
                    f"Oracle error: {exc}"
                ) from exc
            finally:
                cursor.close()
                if model_lob is not None:
                    try:
                        model_lob.close()
                    except Exception:
                        pass

        logger.info("Installed Oracle ONNX embedding model %s", self.model_name)
        return True

    def is_model_loaded(self) -> bool:
        """Return whether the configured model exists in the current schema."""
        with self.pool.acquire() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM user_mining_models
                    WHERE model_name = :model_name
                    """,
                    {"model_name": self.model_name},
                )
                row = cursor.fetchone()
                return bool(row and int(row[0]) > 0)
            finally:
                cursor.close()

    def get_embedding(self, text: str, **kwargs: Any) -> List[float]:
        """Embed one text value with ``VECTOR_EMBEDDING`` inside Oracle."""
        if kwargs:
            raise TypeError(
                "Oracle in-database embeddings use the model configured on "
                "OracleConfig; per-call overrides are not supported."
            )
        value = str(text)
        if not value:
            raise ValueError("Cannot embed empty text")

        with self._cache_lock:
            cached = self._cache.get(value)
            if cached is not None:
                self._cache.move_to_end(value)
                return list(cached)

        with self.pool.acquire() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    f"""
                    SELECT VECTOR_EMBEDDING(
                        {self.model_name} USING :text AS DATA
                    )
                    FROM dual
                    """,
                    {"text": value},
                )
                row = cursor.fetchone()
            finally:
                cursor.close()

        if not row or row[0] is None:
            raise RuntimeError(
                f"Oracle model {self.model_name!r} returned no embedding"
            )
        result = [float(component) for component in row[0]]
        if len(result) != self.dimensions:
            raise RuntimeError(
                f"Oracle model {self.model_name!r} returned {len(result)} "
                f"dimensions; configured dimensions are {self.dimensions}."
            )

        with self._cache_lock:
            self._cache[value] = list(result)
            self._cache.move_to_end(value)
            while len(self._cache) > _CACHE_MAX_ENTRIES:
                self._cache.popitem(last=False)
        return result

    def get_dimensions(self) -> int:
        return self.dimensions

    def get_default_model(self) -> str:
        return self.model_name

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": "oracle_in_database",
            "model": self.model_name,
            "dimensions": self.dimensions,
            "config": {"install_if_missing": self.install_if_missing},
        }

    def _write_model_blob(self, model_lob: Any) -> None:
        path = Path(self.onnx_path).expanduser() if self.onnx_path else None
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"ONNX model file not found: {path}")
            with path.open("rb") as source:
                chunks = iter(lambda: source.read(_CHUNK_SIZE), b"")
                self._copy_chunks_to_lob(chunks, model_lob)
            return

        if not self.onnx_url:
            raise RuntimeError(
                "No ONNX source configured. Set embedding_config['onnx_path'] "
                "or embedding_config['onnx_url']."
            )
        with requests.get(
            self.onnx_url,
            stream=True,
            timeout=self.request_timeout,
        ) as response:
            response.raise_for_status()
            self._copy_chunks_to_lob(
                response.iter_content(chunk_size=_CHUNK_SIZE),
                model_lob,
            )

    @staticmethod
    def _copy_chunks_to_lob(chunks: Any, model_lob: Any) -> None:
        offset = 1
        bytes_written = 0
        for chunk in chunks:
            if not chunk:
                continue
            model_lob.write(chunk, offset=offset)
            offset += len(chunk)
            bytes_written += len(chunk)
        if bytes_written == 0:
            raise RuntimeError("The configured ONNX model source was empty")


def in_database_embedding_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve Oracle in-database embedding options from config and env."""
    values = dict(config or {})
    model_name = (
        values.get("model") or os.getenv("ORACLE_EMBEDDING_MODEL") or DEFAULT_MODEL_NAME
    )
    dimensions = (
        values.get("dimensions")
        or os.getenv("ORACLE_EMBEDDING_DIM")
        or DEFAULT_MODEL_DIMENSIONS
    )
    onnx_path = values.get("onnx_path") or os.getenv("ORACLE_EMBEDDING_ONNX_PATH")
    onnx_url = (
        values.get("onnx_url")
        or os.getenv("ORACLE_EMBEDDING_ONNX_URL")
        or DEFAULT_MODEL_URL
    )
    install_if_missing = values.get(
        "install_if_missing",
        os.getenv("ORACLE_EMBEDDING_INSTALL_IF_MISSING", "1").strip().lower()
        not in {"0", "false", "no", "off"},
    )
    request_timeout = values.get(
        "request_timeout",
        os.getenv("ORACLE_EMBEDDING_DOWNLOAD_TIMEOUT", "600"),
    )
    return {
        "model_name": str(model_name),
        "dimensions": int(dimensions),
        "onnx_path": onnx_path,
        "onnx_url": onnx_url,
        "install_if_missing": bool(install_if_missing),
        "request_timeout": float(request_timeout),
    }
