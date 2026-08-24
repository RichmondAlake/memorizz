"""Persistent, content-addressed corpus embedding snapshots."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .models import MemoryDocument

_CACHE_SCHEMA = "memorizz.corpus-embeddings.v1"


def _document_digest(documents: Sequence[MemoryDocument]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(str(document.source_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(document.parent_source_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class CorpusEmbeddingCache:
    """Read and atomically write local embedding snapshots.

    Snapshot files contain vectors and content fingerprints, not source text.
    The configured model digest is part of the key; callers should provide an
    immutable digest when a mutable local model alias can change over time.
    """

    def __init__(self, root: str | Path, *, enabled: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        self.enabled = bool(enabled)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cache_key(
        documents: Sequence[MemoryDocument], identity: Mapping[str, Any]
    ) -> Tuple[str, str]:
        corpus_digest = _document_digest(documents)
        payload = {
            "schema": _CACHE_SCHEMA,
            "corpus_digest": corpus_digest,
            "embedding_identity": dict(identity),
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), corpus_digest

    def load(
        self,
        documents: Sequence[MemoryDocument],
        identity: Mapping[str, Any],
    ) -> Tuple[list[list[float]] | None, Dict[str, Any]]:
        started = time.perf_counter()
        key, corpus_digest = self.cache_key(documents, identity)
        path = self.root / f"{key}.json.gz"
        metadata = {
            "enabled": self.enabled,
            "hit": False,
            "key": f"sha256:{key}",
            "corpus_digest": f"sha256:{corpus_digest}",
            "path": str(path),
            "embedding_identity": dict(identity),
            "read_seconds": 0.0,
            "bytes": 0,
        }
        if not self.enabled or not path.exists():
            metadata["read_seconds"] = time.perf_counter() - started
            return None, metadata
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            vectors = payload.get("embeddings") if isinstance(payload, dict) else None
            if (
                payload.get("schema") != _CACHE_SCHEMA
                or payload.get("corpus_digest") != corpus_digest
                or payload.get("embedding_identity") != dict(identity)
                or not isinstance(vectors, list)
                or len(vectors) != len(documents)
            ):
                metadata["invalid"] = True
                return None, metadata
            normalized = [[float(value) for value in vector] for vector in vectors]
            metadata.update(
                {
                    "hit": True,
                    "bytes": path.stat().st_size,
                    "read_seconds": time.perf_counter() - started,
                }
            )
            return normalized, metadata
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            metadata.update(
                {
                    "invalid": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "read_seconds": time.perf_counter() - started,
                }
            )
            return None, metadata

    def store(
        self,
        documents: Sequence[MemoryDocument],
        identity: Mapping[str, Any],
        embeddings: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        key, corpus_digest = self.cache_key(documents, identity)
        path = self.root / f"{key}.json.gz"
        metadata = {
            "enabled": self.enabled,
            "key": f"sha256:{key}",
            "corpus_digest": f"sha256:{corpus_digest}",
            "path": str(path),
            "write_seconds": 0.0,
            "bytes": 0,
        }
        if not self.enabled:
            return metadata
        if len(embeddings) != len(documents):
            raise ValueError("Embedding count does not match the corpus")
        payload = {
            "schema": _CACHE_SCHEMA,
            "corpus_digest": corpus_digest,
            "embedding_identity": dict(identity),
            "source_ids": [document.source_id for document in documents],
            "embeddings": [list(vector) for vector in embeddings],
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        metadata.update(
            {
                "write_seconds": time.perf_counter() - started,
                "bytes": path.stat().st_size,
            }
        )
        return metadata


__all__ = ["CorpusEmbeddingCache"]
