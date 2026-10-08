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
from .._env_io import env_bool
from . import config as cfg

PYPI_URL = "https://pypi.org/pypi/memorizz/json"
# Re-ask PyPI at most this often. A daily cache hid same-day releases for up
# to a day; ten minutes keeps rapid relaunches cheap and notices current.
CHECK_INTERVAL_SECONDS = 10 * 60
REQUEST_TIMEOUT = (2, 2)


def _disabled() -> bool:
    return any(env_bool(name) for name in ("MEMORIZZ_NO_UPDATE_CHECK", "CI"))


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


def _newer(latest, installed: Version) -> str | None:
    try:
        return str(latest) if latest and _stable_version(latest) > installed else None
    except (InvalidVersion, ValueError):
        return None


def known_update(current_version: str = __version__) -> dict | None:
    """An update already discovered by an earlier check, from the cache only.

    No network: this is what the banner can show at once. ``None`` when the
    cache is empty, stale-formatted, opted out, or not newer than installed.
    """
    if _disabled():
        return None
    try:
        installed = Version(current_version)
    except InvalidVersion:
        return None
    latest = _newer(_read_cache().get("latest_version"), installed)
    return {"latest": latest, "command": upgrade_command()} if latest else None


def latest_update(
    current_version: str = __version__, *, force: bool = False
) -> dict | None:
    """The newest compatible stable release when it is newer than installed.

    PyPI is asked when the cache is older than ``CHECK_INTERVAL_SECONDS`` (or
    ``force``); failures stay quiet and a previously discovered update remains
    useful offline. Returns ``{"latest", "command"}`` or ``None``.
    """
    if _disabled():
        return None
    try:
        installed = Version(current_version)
    except InvalidVersion:
        return None
    cache = _read_cache()
    latest = cache.get("latest_version")
    age = time.time() - float(cache.get("checked_at", 0))
    if force or not cache or not 0 <= age < CHECK_INTERVAL_SECONDS:
        try:
            latest = _fetch_latest_version()
        except (requests.RequestException, OSError, ValueError, KeyError, TypeError):
            # A previously discovered update remains useful while offline.
            pass
        else:
            _write_cache(latest)
    newer = _newer(latest, installed)
    return {"latest": newer, "command": upgrade_command()} if newer else None


def check_for_update(
    current_version: str = __version__, *, force: bool = False
) -> str | None:
    """Return an actionable notice, or None (quiet on failure)."""
    update = latest_update(current_version, force=force)
    if not update:
        return None
    return (
        f"Update available: memorizz {current_version} → {update['latest']}\n"
        f"Upgrade: {update['command']}"
    )


@contextmanager
def update_notifier(console, session=None):
    """Notify above the prompt without waiting for the network or on exit.

    The caller holds prompt_toolkit's ``patch_stdout`` for this context so a
    worker's output cannot overwrite input. No checks run for pipes or CI.
    With a ``session`` the update is also pinned on it (``update_notice``) so
    the status bar keeps showing it until the person upgrades.
    """
    stopped = Event()

    def notify():
        try:
            update = latest_update(__version__)
            if update and not stopped.is_set():
                if session is not None:
                    session.update_notice = update
                    _redraw_prompt()
                console.print(
                    Text(
                        f"Update available: memorizz {__version__} → {update['latest']}\n"
                        f"Upgrade: {update['command']}   (/update for details)",
                        style="yellow",
                    )
                )
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


def _redraw_prompt() -> None:
    """Ask a running prompt_toolkit app to repaint (its status bar), thread-safe."""
    try:
        from prompt_toolkit.application.current import get_app_or_none

        app = get_app_or_none()
        if app is not None:
            app.invalidate()
    except Exception:
        pass
