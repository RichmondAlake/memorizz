# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Best-effort release notices for interactive sessions only."""

import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread

import requests
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from rich.text import Text

from .. import __version__
from . import config as cfg

PYPI_URL = "https://pypi.org/pypi/memorizz/json"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT = (2, 2)


def _disabled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("MEMORIZZ_NO_UPDATE_CHECK", "CI")
    )


def _stable_version(value) -> Version:
    if not isinstance(value, str):
        raise ValueError("Missing release version")
    version = Version(value)
    if version.is_prerelease or version.is_devrelease or version.local:
        raise ValueError("Not a public stable release")
    return version


def _read_cache() -> dict:
    try:
        cache = json.loads(
            (cfg.memorizz_home() / "update-check.json").read_text(encoding="utf-8")
        )
        if cache["python_version"] != platform.python_version():
            return {}
        if not math.isfinite(float(cache["checked_at"])):
            return {}
        if cache["latest_version"] is not None:
            _stable_version(cache["latest_version"])
        return cache
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _write_cache(latest_version: str | None) -> None:
    temporary = None
    try:
        root = cfg.ensure_home()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=".update-check-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                {
                    "checked_at": time.time(),
                    "python_version": platform.python_version(),
                    "latest_version": latest_version,
                },
                handle,
            )
        temporary.replace(root / "update-check.json")
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _fetch_latest_version() -> str | None:
    with requests.get(
        PYPI_URL,
        headers={"Accept": "application/json", "User-Agent": f"memorizz/{__version__}"},
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()
        info = response.json()["info"]
    version = _stable_version(info["version"])
    if info.get("yanked", False):
        return None
    requirements = SpecifierSet(info.get("requires_python") or "")
    if not requirements.contains(platform.python_version()):
        return None
    return str(version)


def upgrade_command() -> str:
    """Use the owning installer, including the npm wrapper's pinned runtime."""
    if os.environ.get("MEMORIZZ_INSTALL_METHOD") == "npm":
        return "npm install -g memorizz@latest"
    prefix = Path(sys.prefix).resolve()
    if "Cellar" in prefix.parts and "memorizz" in prefix.parts:
        return "brew upgrade memorizz"
    if (prefix / "uv-receipt.toml").is_file():
        return "uv tool upgrade memorizz"
    if (prefix / "pipx_metadata.json").is_file():
        return "pipx upgrade memorizz"
    args = [sys.executable, "-m", "pip", "install", "--upgrade", "memorizz"]
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


def check_for_update(current_version: str = __version__) -> str | None:
    """Return an actionable notice, using a daily cache and quiet failure."""
    if _disabled():
        return None
    try:
        installed = Version(current_version)
    except InvalidVersion:
        return None

    cache = _read_cache()
    latest = cache.get("latest_version")
    age = time.time() - float(cache.get("checked_at", 0))
    if not cache or not 0 <= age < CHECK_INTERVAL_SECONDS:
        try:
            latest = _fetch_latest_version()
        except (requests.RequestException, OSError, ValueError, KeyError, TypeError):
            # A previously discovered update remains useful while offline.
            pass
        else:
            _write_cache(latest)
    if latest and _stable_version(latest) > installed:
        return (
            f"Update available: memorizz {current_version} → {latest}\n"
            f"Upgrade: {upgrade_command()}"
        )
    return None


@contextmanager
def update_notifier(console):
    """Notify above the prompt without waiting for the network or on exit.

    The caller holds prompt_toolkit's ``patch_stdout`` for this context so a
    worker's output cannot overwrite input. No checks run for pipes or CI.
    """
    stopped = Event()

    def notify():
        try:
            notice = check_for_update()
            if notice and not stopped.is_set():
                console.print(Text(notice, style="yellow"))
        except Exception:
            # An optional notification must never interrupt an agent session.
            pass

    if console.is_terminal and not console.is_dumb_terminal and not _disabled():
        try:
            Thread(target=notify, name="memorizz-update-check", daemon=True).start()
        except RuntimeError:
            pass
    try:
        yield
    finally:
        stopped.set()
