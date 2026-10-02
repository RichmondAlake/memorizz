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

import io
import os
import re
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
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
    "validate_env_updates",
    "environment_source",
    "env_override_warnings",
]

DEFAULT_HOME = "~/.memorizz"
_ENV_SOURCES = {}
_WRITE_LOCK = threading.RLock()
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


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
            before = dict(os.environ)
            load_dotenv(path, override=False)
            for name, value in os.environ.items():
                if name not in before:
                    _ENV_SOURCES[name] = (value, "file", str(path.resolve()))
            loaded.append(path)
    return loaded


def env_text(name: str) -> Optional[str]:
    """An environment variable without surrounding whitespace; None when unset
    or blank."""
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def env_bool(name: str, default: bool = False) -> bool:
    """A boolean environment flag: 1, true, yes or on (any case) mean true."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def validate_env_updates(updates: Dict[str, str]) -> None:
    """Reject invalid keys and values without putting credentials in errors."""
    for key, value in updates.items():
        if not isinstance(key, str) or not _ENV_KEY.fullmatch(key):
            raise ValueError(
                "Use a variable name containing letters, digits and underscores, not an assignment."
            )
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("Setting values must be strings without NUL characters.")
        # python-dotenv interpolates even single-quoted ${...}. Reject these
        # new values rather than silently changing a password on the next load.
        # Existing interpolated bindings are preserved unchanged.
        if "${" in value:
            raise ValueError(
                "Enter a concrete value; ${...} expansion cannot be safely saved by this editor."
            )


def format_env_value(value: str) -> str:
    """Round-trip quotes, slashes and multiline values through python-dotenv."""
    validate_env_updates({"VALUE": value})
    if not value or re.fullmatch(r"[A-Za-z0-9_./:@,+%-]+", value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    for actual, encoded in (
        ("\n", "\\n"),
        ("\r", "\\r"),
        ("\t", "\\t"),
        ("\b", "\\b"),
        ("\f", "\\f"),
        ("\v", "\\v"),
        ("\a", "\\a"),
    ):
        escaped = escaped.replace(actual, encoded)
    return f'"{escaped}"'


@contextmanager
def _env_file_lock(path: Path):
    """Serialize cooperating CLI/UI writers, including separate processes."""
    lock_path = path.with_name(path.name + ".lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    with _WRITE_LOCK:
        descriptor = os.open(lock_path, flags, 0o600)
        locked = False
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Configuration lock must be a regular file.")
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            deadline = time.monotonic() + 10
            if os.name == "nt":  # pragma: no cover - Windows CI
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
            else:
                import fcntl
            while not locked:
                try:
                    if os.name == "nt":  # pragma: no cover - Windows CI
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Another process is updating configuration."
                        ) from None
                    time.sleep(0.05)
            yield
        finally:
            try:
                if locked:
                    if os.name == "nt":  # pragma: no cover - Windows CI
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def update_env_file(env_path: Path, updates: Dict[str, str]) -> None:
    """Update or append environment variables in a ``.env`` file.

    Atomic, owner-only writes preserve unrelated bindings (including multiline
    values), comments and blank lines. Symlink targets are rejected. The small
    owner-only lock file is retained so concurrent writers use one lock inode.
    """
    from dotenv.parser import parse_stream

    validate_env_updates(updates)
    if not updates:
        return
    env_path = Path(env_path).expanduser().absolute()
    env_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _env_file_lock(env_path):
        if env_path.is_symlink():
            raise ValueError(
                "Refusing to replace a symlink; choose its intended target explicitly."
            )
        if env_path.exists() and not env_path.is_file():
            raise ValueError("Configuration must be a regular file.")
        original = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        parts, seen = [], set()
        for binding in parse_stream(io.StringIO(original)):
            if binding.error:
                raise ValueError(
                    "The existing .env has invalid syntax; correct it before saving."
                )
            if binding.key in updates:
                if binding.key not in seen:
                    prefix = re.match(r"\s*", binding.original.string).group()
                    parts.append(
                        prefix
                        + binding.key
                        + "="
                        + format_env_value(updates[binding.key])
                        + "\n"
                    )
                    seen.add(binding.key)
            else:
                parts.append(binding.original.string)
        content = "".join(parts)
        if content and not content.endswith("\n"):
            content += "\n"
        for key, value in updates.items():
            if key not in seen:
                content += key + "=" + format_env_value(value) + "\n"
        descriptor, temporary = tempfile.mkstemp(
            prefix="." + env_path.name + ".", suffix=".tmp", dir=env_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, env_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def environment_source(key: str) -> Dict[str, str]:
    """Describe effective provenance without returning the setting's value."""
    if key not in os.environ:
        return {"kind": "unset"}
    recorded = _ENV_SOURCES.get(key)
    if recorded and recorded[0] == os.environ[key]:
        return {"kind": recorded[1], "path": recorded[2]}
    return {"kind": "process environment"}


def env_override_warnings(
    updates: Dict[str, str], env_path: Optional[Path] = None
) -> List[str]:
    """Warn before saving defaults that higher-priority configuration can mask."""
    from dotenv import dotenv_values

    validate_env_updates(updates)
    target = Path(env_path or resolve_env_file()).expanduser().resolve()
    project = (Path.cwd() / ".env").resolve()
    project_values = (
        dotenv_values(project, interpolate=False) if project.exists() else {}
    )
    warnings = []
    for key, value in updates.items():
        if (
            environment_source(key)["kind"] == "process environment"
            and os.environ.get(key) != value
        ):
            warnings.append(
                f"{key}: a process environment value may override this saved default on restart; update or unset its export."
            )
        if target != project and key in project_values and project_values[key] != value:
            warnings.append(
                f"{key}: the project .env takes precedence over this file; use --project or update that project setting."
            )
    return warnings


def apply_env_updates(updates: Dict[str, str]) -> Optional[str]:
    """Apply updates to ``os.environ`` and persist to the canonical ``.env``.

    Returns ``None`` on success, or a human-readable error string if the file
    could not be written (the in-process env is still updated either way).
    """
    try:
        validate_env_updates(updates)
    except ValueError as exc:
        return str(exc)
    for key, value in updates.items():
        os.environ[key] = value
        _ENV_SOURCES[key] = (value, "session", str(resolve_env_file().resolve()))
    try:
        update_env_file(resolve_env_file(), updates)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        return f"Configuration could not be saved ({type(exc).__name__}); check the target path and permissions."
    return None
