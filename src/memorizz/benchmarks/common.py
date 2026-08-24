"""Shared, intentionally small benchmark-provider helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def normalize_memory_backend(value: Optional[str]) -> str:
    backend = str(value or "filesystem").strip().lower()
    if backend not in {"filesystem", "oracle"}:
        raise ValueError("memory_backend must be 'filesystem' or 'oracle'")
    return backend


def create_benchmark_memory_provider(
    memory_backend: str,
    *,
    filesystem_root: Path,
    embedding_provider: Optional[Any] = None,
):
    """Create a benchmark provider without coupling evaluators to one backend."""
    backend = normalize_memory_backend(memory_backend)
    if backend == "oracle":
        from memorizz.memory_provider.oracle import OracleProvider

        overrides = {}
        if embedding_provider is not None:
            overrides.update(
                embedding_provider=embedding_provider,
                in_database_embedding=False,
            )
        provider = OracleProvider.from_env(
            provision_if_missing=False,
            index_policy="lazy",
            **overrides,
        )
        preflight = provider.preflight()
        if preflight.get("embedding_dimension_compatible") is False:
            configured = (preflight.get("embedding") or {}).get("dimensions")
            declared = preflight.get("vector_dimensions") or {}
            provider.close()
            raise ValueError(
                "Oracle benchmark embedding dimensions do not match the existing "
                f"schema (configured={configured}, declared={declared}). Set the "
                "benchmark embedding dimension to the schema dimension before any "
                "paid model call."
            )
        return provider

    from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider

    return FileSystemProvider(
        FileSystemConfig(
            root_path=Path(filesystem_root),
            lazy_vector_indexes=True,
            # Exact cosine search avoids loading FAISS alongside the official
            # LongMemEval PyTorch tokenizer runtime on macOS.
            use_faiss=False,
            embedding_provider=embedding_provider,
        )
    )


__all__ = ["create_benchmark_memory_provider", "normalize_memory_backend"]
