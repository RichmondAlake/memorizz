# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from ..enums.memory_type import MemoryType
from .base import MemoryProvider


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
    "MongoDBProvider",
    "OracleProvider",
    "FileSystemProvider",
    "FileSystemConfig",
    "MemoryType",
]
