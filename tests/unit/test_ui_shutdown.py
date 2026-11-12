"""`memorizz ui` stops on the first Ctrl+C without tracebacks, and opens the
UI in the browser once it is listening."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from memorizz.ui import shutdown

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    shutdown.reset()
    yield
    shutdown.reset()


def _run(coro):
    return asyncio.run(coro)


def test_a_request_cancelled_while_stopping_ends_quietly():
    async def app(scope, receive, send):
        raise asyncio.CancelledError()

    wrapped = shutdown.QuietShutdown(app)
    # Not stopping: a cancellation is a real one and propagates.
    with pytest.raises(asyncio.CancelledError):
        _run(wrapped({"type": "http"}, None, None))
    shutdown.request()
    _run(wrapped({"type": "http"}, None, None))  # no error


def test_a_cancelled_lifespan_is_not_reported_as_a_failed_shutdown():
    sent = []

    async def send(message):
        sent.append(message)

    async def app(scope, receive, send):
        await send(
            {
                "type": "lifespan.shutdown.failed",
                "message": "Traceback ...\nasyncio.exceptions.CancelledError\n",
            }
        )
        raise asyncio.CancelledError()

    wrapped = shutdown.QuietShutdown(app)
    shutdown.request()
    _run(wrapped({"type": "lifespan"}, None, send))
    assert sent == [{"type": "lifespan.shutdown.complete"}]

    # Any other failure is still reported, stopping or not.
    async def failing(scope, receive, send):
        await send({"type": "lifespan.shutdown.failed", "message": "ValueError: x"})

    sent.clear()
    _run(shutdown.QuietShutdown(failing)({"type": "lifespan"}, None, send))
    assert sent[0]["type"] == "lifespan.shutdown.failed"


def test_the_live_stream_ends_when_the_server_stops(monkeypatch):
    from memorizz.ui.routers import playground

    class Tail:
        def __init__(self, *args):
            pass

        def poll(self):
            return []

    class Request:
        async def is_disconnected(self):
            return False

    monkeypatch.setattr("memorizz.live_turns.LiveTail", Tail)
    monkeypatch.setattr("memorizz.ui.helpers._load_agent", lambda agent_id: None)
    response = _run(playground.agent_live_turns(Request(), "agent-1", wait=300))

    async def read():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
            shutdown.request()  # Ctrl+C after the first event
        return chunks

    assert _run(asyncio.wait_for(read(), timeout=5)) == ["retry: 2000\n\n"]


@pytest.mark.parametrize(
    "env, platform, wanted",
    [
        ({}, "darwin", True),
        ({"MEMORIZZ_UI_OPEN_BROWSER": "false"}, "darwin", False),
        ({"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}, "darwin", False),
        ({"CI": "true"}, "darwin", False),
        ({"CI": "true", "MEMORIZZ_UI_OPEN_BROWSER": "1"}, "darwin", True),
        ({}, "linux", False),
        ({"DISPLAY": ":0"}, "linux", True),
    ],
)
def test_when_the_browser_opens(monkeypatch, env, platform, wanted):
    for name in (
        "MEMORIZZ_UI_OPEN_BROWSER",
        "SSH_CONNECTION",
        "SSH_TTY",
        "CI",
        "DISPLAY",
        "WAYLAND_DISPLAY",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "platform", platform)
    assert shutdown.browser_wanted() is wanted
    # Asked for in so many words, it is what was asked.
    assert shutdown.browser_wanted(False) is False
    assert shutdown.browser_wanted(True) is True


def test_the_address_opened():
    assert shutdown.browser_url("127.0.0.1", 8765) == "http://127.0.0.1:8765"
    assert shutdown.browser_url("0.0.0.0", 9000) == "http://127.0.0.1:9000"
    assert shutdown.browser_url("::1", 8765) == "http://[::1]:8765"


def test_the_browser_opens_once_the_server_is_listening():
    server = SimpleNamespace(started=False, should_exit=False)
    opened = []
    with patch.object(shutdown.webbrowser, "open", lambda url, new: opened.append(url)):
        shutdown._open_when_listening(server, "http://127.0.0.1:8765", timeout=5)
        asyncio.run(asyncio.sleep(0.3))
        assert opened == []
        server.started = True
        for _ in range(30):
            if opened:
                break
            asyncio.run(asyncio.sleep(0.1))
    assert opened == ["http://127.0.0.1:8765"]


def test_ui_takes_open_and_no_open(monkeypatch):
    from typer.testing import CliRunner

    from memorizz.cli import operations
    from memorizz.cli.app import app as cli_app

    calls = []
    monkeypatch.setattr(
        operations, "run_local", lambda **kwargs: calls.append(kwargs) or True
    )
    runner = CliRunner()
    runner.invoke(cli_app, ["ui", "--no-connect"])
    runner.invoke(cli_app, ["ui", "--no-connect", "--no-open"])
    runner.invoke(cli_app, ["ui", "--no-connect", "--open"])
    assert [call["open_browser"] for call in calls] == [None, False, True]
