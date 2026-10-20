# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""FileSystem memory stores on this computer, for the connect page.

The CLI's store is ``memory_root()``: ``~/.memorizz/memory`` unless
``MEMORIZZ_MEMORY_ROOT`` names another folder. Any other folder directly
under ``MEMORIZZ_HOME`` that holds an ``agents`` folder is a store too, such
as one an earlier connection created. Each is listed with its agent count so
the one with your agents is obvious.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .._env_io import memorizz_home, memory_root


def _agent_count(path: Path) -> int:
    try:
        return sum(
            1 for item in (path / "agents").glob("*.json") if item.name != "index.json"
        )
    except OSError:
        return 0


def display_path(path: Path) -> str:
    """``~/…`` for a folder under the home directory, else the full path."""
    try:
        return "~/" + str(path.relative_to(Path.home().resolve()))
    except ValueError:
        return str(path)


def detect_local_stores(limit: int = 8) -> List[Dict[str, Any]]:
    """The CLI's store first, then other stores under ``MEMORIZZ_HOME``."""
    default = Path(memory_root()).expanduser().resolve()
    candidates = [default]
    try:
        candidates += sorted(
            child.resolve()
            for child in memorizz_home().iterdir()
            if child.is_dir() and (child / "agents").is_dir()
        )
    except OSError:
        pass
    stores: List[Dict[str, Any]] = []
    for path in dict.fromkeys(candidates):
        exists = (path / "agents").is_dir()
        if not exists and path != default:
            continue
        stores.append(
            {
                "path": str(path),
                "display": display_path(path),
                "agents": _agent_count(path) if exists else 0,
                "exists": exists,
                "default": path == default,
            }
        )
    return stores[:limit]


__all__ = ["detect_local_stores", "display_path"]
