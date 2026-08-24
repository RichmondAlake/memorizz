# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import threading
from pathlib import Path

from ..enums.memory_type import MemoryType
from .base import MemoryProvider, MemoryProviderCapabilities

_DEFAULT_PROVIDER_LOCK = threading.Lock()
_DEFAULT_PROVIDERS = {}


def create_default_memory_provider(root_path=None):
    """Create the zero-configuration filesystem memory provider.

    The root resolves from ``MEMORIZZ_MEMORY_ROOT`` when set, otherwise from
    ``MEMORIZZ_HOME/memory`` (default ``~/.memorizz/memory``). Vector indexes
    are lazy and FAISS remains optional.
    """

    from .._env_io import memory_root
    from .filesystem import FileSystemConfig, FileSystemProvider

    resolved_root = (
        Path(memory_root() if root_path is None else root_path).expanduser().resolve()
    )
    cache_key = str(resolved_root)
    with _DEFAULT_PROVIDER_LOCK:
        provider = _DEFAULT_PROVIDERS.get(cache_key)
        if provider is None:
            provider = FileSystemProvider(
                FileSystemConfig(
                    root_path=resolved_root,
                    lazy_vector_indexes=True,
                )
            )
            _DEFAULT_PROVIDERS[cache_key] = provider
        return provider


# Lazy imports for optional dependencies
def _lazy_import_mongodb():
    """Lazy import MongoDB provider (requires pymongo)."""
    try:
        from .mongodb import MongoDBProvider

        return MongoDBProvider
    except ImportError as e:
        raise ImportError(
            'MongoDB provider requires pymongo. Install with: pip install "memorizz[mongodb]"'
        ) from e


def _lazy_import_oracle():
    """Lazy import Oracle provider (requires oracledb)."""
    try:
        from .oracle import OracleProvider

        return OracleProvider
    except ImportError as e:
        raise ImportError(
            'Oracle provider requires oracledb. Install with: pip install "memorizz[oracle]"'
        ) from e


# Make providers available via module-level getattr
def __getattr__(name):
    if name == "MongoDBProvider":
        return _lazy_import_mongodb()
    elif name == "OracleProvider":
        return _lazy_import_oracle()
    elif name in ("FileSystemProvider", "FileSystemConfig"):
        try:
            from .filesystem import FileSystemConfig, FileSystemProvider

            return (
                FileSystemProvider if name == "FileSystemProvider" else FileSystemConfig
            )
        except ImportError as e:
            raise ImportError(
                "Filesystem provider requires faiss-cpu for vector search. "
                'Install with: pip install "memorizz[filesystem]"'
            ) from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "MemoryProvider",
    "MemoryProviderCapabilities",
    "MongoDBProvider",
    "OracleProvider",
    "FileSystemProvider",
    "FileSystemConfig",
    "create_default_memory_provider",
    "MemoryType",
]
