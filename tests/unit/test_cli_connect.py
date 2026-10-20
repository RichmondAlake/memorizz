"""Connecting what a request needs from the CLI: the offer after a turn,
/connect, and a browser sign-in that finishes without a pasted address."""

import http.server
import json
import socket
import threading
import urllib.error
import urllib.request
from io import StringIO
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from rich.console import Console

from memorizz.cli import connect
from memorizz.mcp import catalog, oauth_handoff

pytestmark = pytest.mark.unit

STATE = "st4te-for-this-sign-in"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    return tmp_path


def _console():
    out = StringIO()
    return Console(file=out, width=200, color_system=None), out


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, ""


class FakeManager:
    """authorize_interactive's contract: show the URL, read the callback."""

    def __init__(self, redirect_uri):
        self.server = SimpleNamespace(
            name="notion", auth=SimpleNamespace(redirect_uri=redirect_uri)
        )

    def get_server(self, name):
        return self.server

    def authorize_interactive(self, name, *, on_authorization_url, callback_reader):
        on_authorization_url(f"https://auth.example/authorize?state={STATE}")
        query = parse_qs(urlparse(callback_reader()).query)
        return {"ok": query.get("code") == ["c0de"], "state": query.get("state")}


def _browser(monkeypatch, redirect_uri):
    """The browser: after sign-in the provider redirects to the app."""
    pages = []

    def open_page(url):
        target = f"{redirect_uri}?code=c0de&state={STATE}"
        threading.Thread(target=lambda: pages.append(_get(target))).start()
        return True

    monkeypatch.setattr(connect.webbrowser, "open", open_page)
    return pages


class _App(http.server.BaseHTTPRequestHandler):
    """Something already on the redirect port: a MemoRizz UI or not."""

    hands_off = True

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path)
        if path.path == "/api/mcp/oauth/handoff" and self.hands_off:
            body = json.dumps({"ok": True, "handoff": True}).encode()
        elif path.path == "/api/mcp/oauth/callback" and self.hands_off:
            query = {k: v[0] for k, v in parse_qs(path.query).items()}
            assert oauth_handoff.deliver(query["state"], query)
            body = oauth_handoff.page().encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def app_on_port():
    servers = []

    def start(hands_off):
        handler = type("App", (_App,), {"hands_off": hands_off})
        server = http.server.HTTPServer(("127.0.0.1", _free_port()), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}/api/mcp/oauth/callback"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_the_handoff_is_private_single_use_and_only_for_a_waiting_sign_in(home):
    assert oauth_handoff.deliver(STATE, {"code": "x"}) is False  # nobody waiting
    assert oauth_handoff.expect("../escape") is False
    assert oauth_handoff.expect(STATE)
    assert oauth_handoff.deliver(STATE, {"code": "c0de", "iss": None})
    directory = home / "oauth-handoff"
    assert directory.stat().st_mode & 0o777 == 0o700
    assert (directory / f"{STATE}.json").stat().st_mode & 0o777 == 0o600
    assert oauth_handoff.collect(STATE, timeout=1) == {"code": "c0de"}
    assert not any(directory.iterdir())
    assert oauth_handoff.collect(STATE, timeout=0.3) is None


def test_the_ui_passes_a_cli_sign_in_back_to_the_terminal():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memorizz.ui.app import create_app

    client = TestClient(create_app())
    assert client.get("/api/mcp/oauth/handoff").json()["handoff"] is True
    assert (
        client.get(f"/api/mcp/oauth/callback?state={STATE}&code=c").status_code == 400
    )

    oauth_handoff.expect(STATE)
    page = client.get(f"/api/mcp/oauth/callback?state={STATE}&code=c0de")
    assert page.status_code == 200 and "Return to your terminal" in page.text
    assert oauth_handoff.collect(STATE, timeout=1) == {"code": "c0de", "state": STATE}


def test_sign_in_catches_the_redirect_itself_when_the_port_is_free(monkeypatch):
    redirect = f"http://127.0.0.1:{_free_port()}/api/mcp/oauth/callback"
    pages = _browser(monkeypatch, redirect)
    console, out = _console()
    result = connect.sign_in(FakeManager(redirect), "notion", console, title="Notion")
    assert result == {"ok": True, "state": [STATE]}
    assert pages[0][0] == 200 and "Return to your terminal" in pages[0][1]
    assert "Sign in to Notion in your browser" in out.getvalue()
    assert not connect._in_use("127.0.0.1", urlparse(redirect).port)  # closed


def test_sign_in_goes_through_the_local_ui_on_the_redirect_port(
    monkeypatch, app_on_port
):
    redirect = app_on_port(hands_off=True)
    pages = _browser(monkeypatch, redirect)
    console, _ = _console()
    result = connect.sign_in(FakeManager(redirect), "notion", console)
    assert result["ok"] is True
    assert pages[0][0] == 200


def test_sign_in_asks_for_the_address_when_the_app_cannot_pass_it_on(
    monkeypatch, app_on_port
):
    redirect = app_on_port(hands_off=False)
    monkeypatch.setattr(connect.webbrowser, "open", lambda url: True)
    prompts = []

    def paste(prompt):
        prompts.append(prompt)
        return f"{redirect}?code=c0de&state={STATE}"

    monkeypatch.setattr("builtins.input", paste)
    console, out = _console()
    assert connect.sign_in(FakeManager(redirect), "notion", console)["ok"] is True
    assert len(prompts) == 1 and "OAuth state is invalid" in out.getvalue()

    monkeypatch.setattr("builtins.input", lambda prompt: "")
    result = connect.sign_in(FakeManager(redirect), "notion", console)
    assert result["ok"] is False and result["error"] == "sign-in cancelled"


@pytest.fixture
def session():
    from memorizz import MemAgent

    agent = MemAgent(memory_provider=False, auto_register=False)
    console, out = _console()
    yield SimpleNamespace(
        agent=agent,
        provider=None,
        console=console,
        out=out,
        offered_capabilities=set(),
    )
    agent.close()


def test_a_turn_that_needs_notion_offers_to_connect_it(session, monkeypatch):
    from memorizz.cli import mcp_config

    agent = session.agent
    prompts = []
    monkeypatch.setattr("builtins.input", lambda q: prompts.append(q) or "")
    signed_in = []
    monkeypatch.setattr(
        connect,
        "sign_in",
        lambda manager, name, console, **k: signed_in.append(name) or {"ok": True},
    )
    monkeypatch.setattr(
        agent.mcp_manager,
        "cached_tools",
        lambda name: [{"name": "notion-search", "inputSchema": {"type": "object"}}],
    )

    connect.offer(session, ["notes", "notes"])
    assert prompts == [
        "Notion isn't connected to this agent. Connect it now? You'll sign in "
        "to Notion in your browser. [Y/n] "
    ]
    assert signed_in == ["notion"]
    assert [s["url"] for s in agent.mcp_servers] == [catalog.NOTION_URL]
    saved = mcp_config.load_servers(agent.agent_id)
    assert [s["name"] for s in saved] == ["notion"]
    assert any("notion" in name for name in agent.tool_manager.list_tools())
    text = session.out.getvalue()
    assert "Added Notion to this agent." in text
    assert "Notion connected · 1 tools" in text

    connect.offer(session, ["notes"])  # asked once per session
    assert len(prompts) == 1
    assert "/connect notion" in session.out.getvalue()


def test_declining_leaves_the_command_and_settings_only_get_a_hint(
    session, monkeypatch
):
    monkeypatch.setattr("builtins.input", lambda q: "n")
    connect.offer(session, ["notes", "email", "browser"])
    text = session.out.getvalue()
    assert "Later: /connect notion" in text
    assert "Email isn't enabled for this agent: /connect gmail shows how." in text
    assert "Browse websites isn't enabled for this agent: /browser on" in text
    assert session.agent.mcp_servers == []


def test_gmail_needs_its_own_client_so_connect_explains_the_step(session):
    assert connect.connect(session, "gmail") is False
    text = session.out.getvalue()
    assert "--client-id" in text and catalog.DEFAULT_REDIRECT_URI in text
    assert session.agent.mcp_servers == []


def test_slash_connect_lists_what_can_be_connected(session):
    from memorizz.cli import commands

    assert commands.dispatch("/connect", session)
    assert commands.dispatch("/mcp", session)
    text = session.out.getvalue()
    assert text.count("Connections") == 2
    assert "Notes and docs" in text and "/connect notion" in text
    assert "/connect google-calendar" in text
    commands.dispatch("/connect teleport", session)
    assert "Unknown connection: teleport" in session.out.getvalue()


def test_the_repl_offers_after_the_answer(monkeypatch):
    from memorizz.cli import repl, streaming

    class Stream:
        def __init__(self, events):
            self.events = list(events)

        def poll(self, timeout=None):
            if not self.events:
                raise StopIteration
            return self.events.pop(0)

        def close(self):
            pass

    events = [
        {"type": "run.started", "memory_id": "m1", "thread_id": "t1"},
        {"type": "capability.suggested", "capabilities": ["notes"]},
        {"type": "answer.delta", "delta": "Notion isn't set up yet."},
        {"type": "capability.requested", "capability": "notes"},
        {"type": "run.done", "status": "completed"},
    ]
    agent = SimpleNamespace(
        agent_id="agent-1", run_stream_events=lambda query, **k: Stream(events)
    )
    session = SimpleNamespace(
        agent=agent,
        memory_id=None,
        thread_id=None,
        user_id=None,
        console=Console(file=StringIO(), force_terminal=True, color_system=None),
    )
    offered = []
    monkeypatch.setattr(streaming, "save_stream_session", lambda s: "written")
    monkeypatch.setattr(connect, "offer", lambda s, ids: offered.append(list(ids)))
    repl._stream_turn(session, "can you connect to notion?")
    assert offered == [["notes", "notes"]]
