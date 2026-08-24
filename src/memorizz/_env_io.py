# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Shared environment-file resolution and persistence for memorizz.

This module is the single source of truth for *where* memorizz reads and writes
its ``.env`` so that the CLI (``memorizz init`` / ``/login``), the local web UI
Settings page, and ``memorizz ui`` all agree on one file.

Historically the UI computed ``ENV_FILE_PATH`` as ``<package>/../../../.env``
(``ui/app.py``). Under a pip/uv install that path lands inside ``site-packages``
where nothing ever reads it, and a globally-installed CLI cannot rely on
``$CWD/.env`` either. We centralize on ``~/.memorizz/.env``.

Canonical file resolution (for *writes* and the authoritative read target)::

    1. $MEMORIZZ_ENV_FILE          (explicit file override)
    2. $MEMORIZZ_HOME/.env
    3. ~/.memorizz/.env            (default)

``load_layered_env()`` additionally honors a project-local ``./.env`` for
backwards-compatibility with the old CLI behavior (``override=False`` so the
real process environment always wins).
"""

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

__all__ = [
    "memorizz_home",
    "ensure_home",
    "resolve_env_file",
    "memory_root",
    "history_file",
    "load_layered_env",
    "resolve_oracle_in_database_embedding_from_env",
    "format_env_value",
    "update_env_file",
    "apply_env_updates",
]

DEFAULT_HOME = "~/.memorizz"


def memorizz_home() -> Path:
    """Return the memorizz home directory (``$MEMORIZZ_HOME`` or ``~/.memorizz``)."""
    raw = os.environ.get("MEMORIZZ_HOME")
    base = raw if raw else DEFAULT_HOME
    return Path(base).expanduser()


def ensure_home() -> Path:
    """Create and return the memorizz home directory."""
    home = memorizz_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


def resolve_env_file() -> Path:
    """Return the canonical ``.env`` path used for reads and writes."""
    raw = os.environ.get("MEMORIZZ_ENV_FILE")
    if raw:
        return Path(raw).expanduser()
    return memorizz_home() / ".env"


def memory_root() -> Path:
    """Default on-disk root for the FileSystem memory provider."""
    explicit = os.environ.get("MEMORIZZ_MEMORY_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    return memorizz_home() / "memory"


def history_file() -> Path:
    """Default path for the interactive REPL history file."""
    return memorizz_home() / "history"


def load_layered_env(extra_paths: Optional[Iterable[Path]] = None) -> List[Path]:
    """Load ``.env`` files into ``os.environ`` without clobbering existing values.

    Resolution order (earliest wins, because ``override=False``):

        process env  >  ``$CWD/.env``  >  ``~/.memorizz/.env``

    Returns the list of files that were actually loaded. A no-op (returns ``[]``)
    when ``python-dotenv`` is unavailable.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a base dependency
        return []

    candidates: List[Path] = [Path.cwd() / ".env", resolve_env_file()]
    if extra_paths:
        candidates.extend(Path(p) for p in extra_paths)

    loaded: List[Path] = []
    seen = set()
    for candidate in candidates:
        try:
            path = candidate.expanduser()
        except Exception:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


def resolve_oracle_in_database_embedding_from_env() -> bool:
    """Resolve the embedding mode used by first-party Oracle clients.

    The SDK's ``OracleConfig`` default remains in-database embeddings. The UI
    and CLI call this helper so an explicit mode wins, while legacy external
    embedding defaults continue to select the provider that created an
    existing schema.
    """
    explicit_mode = os.environ.get("MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING")
    if explicit_mode is not None and explicit_mode.strip():
        normalized = explicit_mode.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError("MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING must be true or false")

    external_provider = os.environ.get(
        "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", ""
    ).strip()
    return not bool(external_provider)


def format_env_value(value: str) -> str:
    """Format an env var value for safe storage in a ``.env`` file.

    Verbatim port of ``ui/app.py:_format_env_value`` so the CLI and UI write
    byte-identical files.
    """
    if value == "":
        return ""
    if any(ch.isspace() for ch in value) or "#" in value or "=" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def update_env_file(env_path: Path, updates: Dict[str, str]) -> None:
    """Update or append environment variables in a ``.env`` file.

    Creates the file (and parent directories) if missing. Preserves comments,
    blank lines, and unrelated entries. Verbatim port of
    ``ui/app.py:_update_env_file`` with directory auto-creation added.
    """
    env_path = Path(env_path).expanduser()
    env_path.parent.mkdir(parents=True, exist_ok=True)

    if env_path.exists():
        lines = env_path.read_text().splitlines()
    else:
        lines = []

    updated_lines: List[str] = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue
        key, _value = line.split("=", 1)
        key = key.strip()
        if key in updates:
            updated_lines.append(f"{key}={format_env_value(updates[key])}")
            seen.add(key)
        else:
            updated_lines.append(line)

    for key, value in updates.items():
        if key in seen:
            continue
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append(f"{key}={format_env_value(value)}")

    env_path.write_text("\n".join(updated_lines) + "\n")


def apply_env_updates(updates: Dict[str, str]) -> Optional[str]:
    """Apply updates to ``os.environ`` and persist to the canonical ``.env``.

    Returns ``None`` on success, or a human-readable error string if the file
    could not be written (the in-process env is still updated either way).
    """
    for key, value in updates.items():
        os.environ[key] = value
    try:
        update_env_file(resolve_env_file(), updates)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        return str(exc)
    return None
