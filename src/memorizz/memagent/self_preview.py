# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Run one version of an agent's own source as a second console, to look at.

The console a person is using runs the installed MemoRizz, so a change an
agent makes to the console's pages cannot be seen there. A preview starts
another console from the folder of one kept version, on a free local port,
against the same store but read-only: the agent's code is shown, and cannot
write anything.
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

from .self_source import SelfSource, SelfSourceError

logger = logging.getLogger(__name__)

#: A preview nobody stops is stopped after this long.
PREVIEW_SECONDS = 45 * 60
#: How long a preview may take to answer its first request.
START_SECONDS = 60

_lock = threading.Lock()
# agent_id -> the one preview that agent has running.
_running: Dict[str, Dict[str, Any]] = {}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _answers(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/login", timeout=2
        ) as reply:
            return reply.status < 500
    except urllib.error.HTTPError as exc:
        return exc.code < 500
    except Exception:
        return False


def _public(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: entry[key]
        for key in (
            "version",
            "number",
            "port",
            "url",
            "started_at",
            "ends_at",
            "read_only",
        )
    }


def status(agent_id: str) -> Optional[Dict[str, Any]]:
    """The agent's running preview, or None. A preview that has exited is forgotten."""
    with _lock:
        entry = _running.get(str(agent_id))
        if entry is None:
            return None
        if entry["process"].poll() is not None:
            _running.pop(str(agent_id), None)
            return None
        return _public(entry)


def stop(agent_id: str) -> bool:
    """Stop the agent's preview. True when one was running."""
    with _lock:
        entry = _running.pop(str(agent_id), None)
    if entry is None:
        return False
    process = entry["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
    try:
        entry["log"].close()
    except Exception:
        pass
    return True


def start(
    source: SelfSource,
    version: str,
    *,
    store: Optional[List[str]] = None,
    label: str = "",
    read_only: bool = True,
) -> Dict[str, Any]:
    """Start a console from one kept version and wait until it answers.

    ``store`` is the console's own arguments for the store to open
    (``["--backend", "filesystem", "--path", ...]``); without them the
    preview connects as the environment says, or opens on the connect page.
    An agent has one preview at a time: starting another stops the first.
    """
    kept = {row["id"]: row for row in source.versions()}
    row = next(
        (
            item
            for key, item in kept.items()
            if key == version or item["commit"] == version
        ),
        None,
    )
    if row is None:
        raise SelfSourceError("A version is named by its ID")
    folder = source.checkout(row["id"])
    stop(source.agent_id)
    port = _free_port()
    log_path = source.folder / "previews" / "preview.log"
    log = log_path.open("w", encoding="utf-8")
    command = [sys.executable, "-m", "memorizz", "ui", "--port", str(port), "--no-open"]
    if read_only:
        command.append("--read-only")
    command += list(store or [])
    environment = {
        **os.environ,
        # The version's folder comes before the installed package.
        "PYTHONPATH": str(folder),
        "PYTHONDONTWRITEBYTECODE": "1",
        # One console runs the scheduled work; a preview only shows pages.
        "MEMORIZZ_AUTOMATIONS_UI_WORKER": "0",
        "MEMORIZZ_UI_OPEN_BROWSER": "false",
        "MEMORIZZ_UI_PREVIEW_LABEL": label or f"v{row['number']}",
    }
    process = subprocess.Popen(
        command,
        cwd=str(folder),
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
            raise SelfSourceError(
                "This version's console did not start:\n" + tail.strip()
            )
        if _answers(port):
            break
        time.sleep(0.4)
    else:
        process.terminate()
        log.close()
        raise SelfSourceError("This version's console did not answer in time")
    now = time.time()
    entry = {
        "process": process,
        "log": log,
        "version": row["id"],
        "number": row["number"],
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "started_at": now,
        "ends_at": now + PREVIEW_SECONDS,
        "read_only": bool(read_only),
        "folder": str(folder),
    }
    with _lock:
        _running[str(source.agent_id)] = entry
    timer = threading.Timer(
        PREVIEW_SECONDS, _expire, args=(str(source.agent_id), process)
    )
    timer.daemon = True
    timer.start()
    return _public(entry)


def _expire(agent_id: str, process: Any) -> None:
    with _lock:
        entry = _running.get(agent_id)
        current = entry is not None and entry["process"] is process
    if current:
        stop(agent_id)


def stop_all() -> None:
    for agent_id in list(_running):
        try:
            stop(agent_id)
        except Exception:
            logger.debug("Stopping a preview failed", exc_info=True)


atexit.register(stop_all)

__all__ = ["PREVIEW_SECONDS", "start", "status", "stop", "stop_all"]
