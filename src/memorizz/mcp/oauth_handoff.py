# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Pass an OAuth callback from the local UI to the terminal that signed in.

A preset's redirect address is the local UI's (127.0.0.1:8765). When the CLI
signs in while that UI is running, the browser lands on the UI, which knows
nothing of the CLI's flow. The CLI leaves a marker named after the flow's
``state``; the UI's callback writes the code beside it and the CLI collects
it. The files are private to the user and removed once read; a code is
useless without the PKCE verifier, which never leaves the CLI process.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from .._env_io import memorizz_home

# Matches the sign-in timeout (manager.SIGN_IN_SECONDS).
WAIT_SECONDS = 600.0

_SAFE_STATE = re.compile(r"[A-Za-z0-9_-]{8,256}\Z")


def _directory() -> Path:
    return memorizz_home() / "oauth-handoff"


def _paths(state: str) -> Optional[tuple]:
    if not _SAFE_STATE.match(str(state or "")):
        return None
    directory = _directory()
    return directory / f"{state}.wait", directory / f"{state}.json"


def _prune(directory: Path) -> None:
    cutoff = time.time() - WAIT_SECONDS
    for path in directory.iterdir():
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def expect(state: str) -> bool:
    """Mark ``state`` as a sign-in this process is waiting for."""
    paths = _paths(state)
    if paths is None:
        return False
    marker, _ = paths
    try:
        marker.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _prune(marker.parent)
        os.close(os.open(marker, os.O_WRONLY | os.O_CREAT, 0o600))
    except OSError:
        return False
    return True


def deliver(state: str, params: Dict[str, Optional[str]]) -> bool:
    """Hand a callback to the process waiting for ``state``; False if none is."""
    paths = _paths(state)
    if paths is None:
        return False
    marker, result = paths
    try:
        if time.time() - marker.stat().st_mtime > WAIT_SECONDS:
            return False
        payload = {key: value for key, value in params.items() if value}
        temporary = result.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(temporary, result)
    except OSError:
        return False
    return True


def page(error: Optional[str] = None) -> str:
    """The browser tab's last page: go back to the terminal."""
    if error:
        heading = "Sign-in did not finish"
        body = f"The provider said: {html.escape(error)}. Return to your terminal."
    else:
        heading = "Signed in"
        body = "Return to your terminal; MemoRizz is finishing the connection."
    return (
        "<!doctype html><meta charset='utf-8'><title>MemoRizz sign-in</title>"
        "<body style='font-family:system-ui,sans-serif;margin:3rem;"
        "color:#111;background:#fff'>"
        f"<h1>{heading}</h1><p>{body}</p><p>You can close this tab.</p>"
    )


def discard(state: str) -> None:
    for path in _paths(state) or ():
        try:
            path.unlink()
        except OSError:
            pass


def collect(
    state: str,
    *,
    timeout: float = WAIT_SECONDS,
    cancelled: Optional[threading.Event] = None,
) -> Optional[Dict[str, str]]:
    """Wait for the callback for ``state``; None on timeout or cancel."""
    paths = _paths(state)
    if paths is None:
        return None
    _, result = paths
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if cancelled is not None and cancelled.is_set():
                return None
            try:
                value = json.loads(result.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                time.sleep(0.2)
                continue
            if not isinstance(value, dict):
                return None
            return {str(key): str(item) for key, item in value.items()}
        return None
    finally:
        discard(state)


__all__ = ["WAIT_SECONDS", "collect", "deliver", "discard", "expect", "page"]
