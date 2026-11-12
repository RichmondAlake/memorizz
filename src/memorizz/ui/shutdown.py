# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Starting and stopping the local UI server.

Stopping: uvicorn waits for every open connection to finish before it shuts
the app down, and a page left open keeps an event stream going for minutes
(the playground's live turns, Oracle's container logs). So Ctrl+C used to
print "Waiting for connections to close" and hang, and a second Ctrl+C
force-quit with two tracebacks of ``CancelledError``. Now the first Ctrl+C
sets :func:`requested`, which the streams check, so they end at once; a
request still running after ``GRACE_SECONDS`` is cancelled, and a request
cancelled because the server is stopping is not reported as an error.

Starting: :func:`serve` opens the UI in the browser once the server is
listening (``memorizz ui --no-open``, or ``MEMORIZZ_UI_OPEN_BROWSER=false``,
turns that off; it is off over SSH, in CI and on a Linux machine without a
display).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import webbrowser
from typing import Any, Optional

#: Seconds a request may keep running after Ctrl+C before it is cancelled.
GRACE_SECONDS = 3

_stopping = threading.Event()
_OFF = {"0", "false", "no", "off"}
_ON = {"1", "true", "yes", "on"}


def requested() -> bool:
    """The server was asked to stop: a stream should end now."""
    return _stopping.is_set()


def request() -> None:
    _stopping.set()


def reset() -> None:
    _stopping.clear()


class QuietShutdown:
    """The ASGI app, with a request cancelled because the server is stopping
    ending quietly instead of as "Exception in ASGI application"."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "lifespan":
            send = self._lifespan_send(send)
        try:
            await self.app(scope, receive, send)
        except asyncio.CancelledError:
            if requested():
                return
            raise

    @staticmethod
    def _lifespan_send(send: Any) -> Any:
        """A second Ctrl+C cancels the app's lifespan task, and Starlette
        reports that as a failed shutdown, with its traceback; it is just
        the server stopping."""

        async def quiet(message: Any) -> None:
            if (
                requested()
                and isinstance(message, dict)
                and message.get("type") == "lifespan.shutdown.failed"
                and "CancelledError" in str(message.get("message") or "")
            ):
                message = {"type": "lifespan.shutdown.complete"}
            await send(message)

        return quiet


def browser_wanted(open_browser: Optional[bool] = None) -> bool:
    """Whether to open the UI in a browser: as asked, else as
    ``MEMORIZZ_UI_OPEN_BROWSER`` says, else wherever a browser can open."""
    if open_browser is not None:
        return bool(open_browser)
    setting = os.environ.get("MEMORIZZ_UI_OPEN_BROWSER", "").strip().lower()
    if setting in _OFF:
        return False
    if setting in _ON:
        return True
    if any(os.environ.get(name) for name in ("CI", "SSH_CONNECTION", "SSH_TTY")):
        return False
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    return True


def browser_url(host: str, port: int) -> str:
    """The address to open: a wildcard host is reached on this machine."""
    host = (host or "").strip()
    if host in {"", "0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    elif ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _open_when_listening(server: Any, url: str, timeout: float = 60.0) -> None:
    def wait() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(server, "started", False):
                try:
                    webbrowser.open(url, new=2)
                except Exception:
                    pass
                return
            if getattr(server, "should_exit", False):
                return
            time.sleep(0.1)

    threading.Thread(target=wait, name="memorizz-ui-browser", daemon=True).start()


def serve(app: Any, host: str, port: int, *, open_browser: bool = False) -> None:
    """Run ``app`` on uvicorn until Ctrl+C, stopping promptly and quietly."""
    import uvicorn

    class Server(uvicorn.Server):
        def handle_exit(self, sig: int, frame: Any) -> None:
            request()
            super().handle_exit(sig, frame)

    reset()
    server = Server(
        uvicorn.Config(
            QuietShutdown(app),
            host=host,
            port=port,
            log_level="info",
            timeout_graceful_shutdown=GRACE_SECONDS,
        )
    )
    if open_browser:
        _open_when_listening(server, browser_url(host, port))
    server.run()


__all__ = [
    "GRACE_SECONDS",
    "QuietShutdown",
    "browser_url",
    "browser_wanted",
    "request",
    "requested",
    "reset",
    "serve",
]
