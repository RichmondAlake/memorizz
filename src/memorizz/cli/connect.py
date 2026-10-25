# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Connect what a request needs (Notion, Gmail, Google Calendar) from the CLI.

A turn that needs a capability the agent lacks ends with an offer to connect
it (the agent emits ``capability.suggested`` or ``capability.requested``);
``/connect`` does the same on demand. Connecting attaches the preset to the
agent, opens the provider's sign-in in the browser and finishes it without a
paste wherever the redirect lands: on a listener started here when the
redirect port is free, or on the local UI already running there. Only a UI
too old to pass the callback on needs the address pasted.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time
import urllib.request
import webbrowser
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from rich.markup import escape

from .. import capability_gaps
from ..mcp import catalog, oauth_handoff

# Capabilities turned on by a setting rather than a connection.
SETTING_HINTS = {
    "web_search": "/web tavily or /web firecrawl turns it on",
    "browser": "/browser on turns it on",
    "desktop": "/desktop on turns it on",
    "code_execution": "choose a sandbox in the agent's settings (/ui)",
}


def _rows(agent) -> List[Dict[str, Any]]:
    try:
        return capability_gaps.state_for_agent(agent)
    except Exception:
        return []


def _preset_key(row: Dict[str, Any]) -> Optional[str]:
    capability = capability_gaps.BY_ID.get(row["id"])
    if capability is None or not capability.preset:
        return None
    return catalog.preset_key(capability.preset)


def _attached(manager, key: str) -> Optional[str]:
    """The attached server that is this preset, whatever it was named."""
    url = catalog.PRESETS[key]["url"].rstrip("/")
    for server in manager.server_dicts():
        if str(server.get("url") or "").rstrip("/") == url:
            return str(server["name"])
    return None


def _resolve(target: str, manager) -> tuple:
    """``(preset key or None, attached server name or None)`` for ``target``."""
    names = {str(server.get("name")) for server in manager.server_dicts()}
    if target in names:
        return None, target
    try:
        key = catalog.preset_key(target)
    except ValueError:
        capability = capability_gaps.BY_ID.get(capability_gaps.resolve(target) or "")
        if capability is None or not capability.preset:
            return None, None
        key = catalog.preset_key(capability.preset)
    return key, _attached(manager, key)


def _sync_shared_config(agent) -> None:
    """Pick up servers ``memorizz mcp add`` saved for this agent meanwhile."""
    from . import mcp_config

    agent_id = str(getattr(agent, "agent_id", "") or "")
    if not agent_id or not mcp_config.has_server_config(agent_id):
        return
    servers = mcp_config.load_servers(agent_id)
    if servers != list(agent.mcp_servers or []):
        agent.with_mcp_servers(servers)


def _apply(session, servers: List[Dict[str, Any]]) -> None:
    """Give the running agent ``servers`` and save them with the agent."""
    from . import mcp_config
    from .commands import save_agent_fields

    agent = session.agent
    agent.with_mcp_servers(servers)
    agent_id = getattr(agent, "agent_id", None)
    if agent_id:
        mcp_config.save_servers(list(agent.mcp_servers or []), str(agent_id))
    save_agent_fields(session, mcp_servers=list(agent.mcp_servers or []))


def _explain_client(console, key: str) -> None:
    preset = catalog.PRESETS[key]
    name = preset.get("name") or key
    console.print(
        f"{preset['title']} signs in through your own Google OAuth client, so it "
        "takes one setup step first."
    )
    console.print(f"[dim]{escape(preset['setup'])}[/dim]")
    console.print(f"Redirect URI: [cyan]{catalog.DEFAULT_REDIRECT_URI}[/cyan]")
    console.print(
        f"Then run [cyan]memorizz mcp add {name} --preset {key} --client-id <id> "
        f"--client-secret <secret>[/cyan] and [cyan]/connect {key}[/cyan], or "
        "connect it from the UI's MCP page (/ui)."
    )
    console.print(f"[dim]Guide: {preset['docs']}[/dim]")


def connect(session, target: str) -> bool:
    """Attach ``target`` (a preset or an attached server) and sign in to it."""
    console = session.console
    agent = session.agent
    manager = getattr(agent, "mcp_manager", None)
    if manager is None:
        console.print("[red]This agent cannot use MCP servers.[/red]")
        return False
    try:
        _sync_shared_config(agent)
    except Exception as exc:
        console.print(
            f"[yellow]Saved MCP servers not loaded:[/yellow] {escape(str(exc))}"
        )
    key, name = _resolve(target.strip(), manager)
    if key is None and name is None:
        attached = sorted(str(s.get("name")) for s in manager.server_dicts())
        console.print(
            f"[red]Unknown connection:[/red] {escape(target)}. Use "
            + ", ".join([*catalog.PRESETS, *attached])
            + "."
        )
        return False
    title = catalog.PRESETS[key]["title"] if key else name
    if name is None:
        if catalog.PRESETS[key]["needs_client"]:
            _explain_client(console, key)
            return False
        config = catalog.preset_config(key)
        try:
            manager.upsert_server(config)
            _apply(session, manager.server_dicts())
        except Exception as exc:
            console.print(f"[red]Could not add {title}:[/red] {escape(str(exc))}")
            return False
        name = str(config["name"])
        console.print(f"Added {title} to this agent.")
    status = (manager.connection_status(name).get("servers") or [{}])[0]
    if not status.get("authenticated"):
        result = sign_in(manager, name, console, title=title)
        if not result.get("ok"):
            error = result.get("error") or result.get("message") or "unknown error"
            console.print(
                f"[red]{escape(title)} sign-in did not finish:[/red] "
                f"{escape(str(error))}"
            )
            console.print(f"[dim]Try again with /connect {key or name}.[/dim]")
            return False
    tools = manager.cached_tools(name)
    if not tools and manager.list_tools(name).get("ok"):
        tools = manager.cached_tools(name)
    registered = agent._register_mcp_server_tools(name, tools)
    console.print(
        f"[green]{escape(title)} connected[/green] · {len(registered)} tools. "
        "Ask away."
    )
    return True


def show(session) -> None:
    """What the agent can reach and how to connect the rest."""
    from rich.table import Table

    console = session.console
    rows = _rows(session.agent)
    table = Table(title="Connections", title_justify="left")
    for heading in ("Capability", "Status", "How"):
        table.add_column(heading)
    for row in rows:
        key = _preset_key(row)
        if row["status"] == "enabled":
            status, how = f"[green]on[/green] · {escape(str(row['via']))}", ""
        elif row["status"] == "needs_sign_in":
            status = "[yellow]connected, not signed in[/yellow]"
            how = f"/connect {row['via']}"
        else:
            status = "[dim]off[/dim]"
            how = f"/connect {key}" if key else SETTING_HINTS.get(row["id"], "")
        table.add_row(row["title"], status, escape(how))
    console.print(table)
    console.print("[dim]Usage: /connect <notion|gmail|google-calendar|server>[/dim]")


def command(session, args: str) -> None:
    target = args.strip()
    if target:
        connect(session, target)
    else:
        show(session)


def _confirm(question: str) -> bool:
    try:
        answer = input(f"{question} [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"", "y", "yes"}


def offer(session, capability_ids: Iterable[Optional[str]]) -> None:
    """After a turn: offer to connect what the request needed.

    Each capability is asked about once per session; later turns that need
    it get a one-line reminder of the command instead.
    """
    wanted = list(dict.fromkeys(item for item in capability_ids if item))
    agent = session.agent
    if not wanted or getattr(agent, "mcp_manager", None) is None:
        return
    offered = getattr(session, "offered_capabilities", None)
    if offered is None:
        offered = set()
        try:
            session.offered_capabilities = offered
        except Exception:
            pass
    console = session.console
    rows = {row["id"]: row for row in _rows(agent)}
    for capability_id in wanted:
        row = rows.get(capability_id)
        if row is None or row["status"] == "enabled":
            continue
        key = _preset_key(row)
        if row["status"] == "needs_sign_in":
            target = str(row["via"])
            question = (
                f"{target} is connected to this agent but not signed in. "
                "Sign in now?"
            )
            reminder = f"{target} isn't signed in: /connect {target}"
        elif key and not catalog.PRESETS[key]["needs_client"]:
            title = catalog.PRESETS[key]["title"]
            target = key
            question = (
                f"{title} isn't connected to this agent. Connect it now? "
                f"You'll sign in to {title} in your browser."
            )
            reminder = f"{title} isn't connected: /connect {key}"
        else:
            how = f"/connect {key} shows how" if key else SETTING_HINTS.get(row["id"])
            if how and capability_id not in offered:
                console.print(
                    f"[dim]{row['title']} isn't enabled for this agent: {how}.[/dim]"
                )
            offered.add(capability_id)
            continue
        if capability_id in offered:
            console.print(f"[dim]{escape(reminder)}[/dim]")
            continue
        offered.add(capability_id)
        if _confirm(question):
            connect(session, target)
        else:
            console.print(f"[dim]Later: /connect {escape(target)}[/dim]")


# ------------------------------------------------------------------ sign-in


def _in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _passes_callbacks(origin: str) -> bool:
    """Whether the app at ``origin`` hands a CLI sign-in back (MemoRizz UI)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{origin}/api/mcp/oauth/handoff", timeout=2) as response:
            return bool(json.loads(response.read().decode("utf-8")).get("handoff"))
    except Exception:
        return False


class _Catcher:
    """Wait for the browser's redirect: on a listener here, through the local
    UI already on the redirect port, or, failing both, a pasted address."""

    def __init__(self, redirect_uri: str):
        parsed = urlparse(redirect_uri)
        self.path = parsed.path or "/"
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.cancelled = threading.Event()
        self.received = threading.Event()
        self.paste_needed = threading.Event()
        self.callback_url = ""
        self.state: Optional[str] = None
        self.failure: Optional[str] = None
        self.server: Optional[http.server.HTTPServer] = None
        self.mode = "paste"
        host = {"localhost": "127.0.0.1"}.get(parsed.hostname or "", parsed.hostname)
        if parsed.scheme != "http" or host != "127.0.0.1":
            return
        port = parsed.port or 80
        if not _in_use(host, port):
            self.server = self._listen(host, port)
            if self.server is not None:
                self.mode = "listener"
                return
        if _passes_callbacks(self.origin):
            self.mode = "handoff"

    def _listen(self, host: str, port: int) -> Optional[http.server.HTTPServer]:
        catcher = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server's name
                if urlparse(self.path).path != catcher.path:
                    self.send_error(404)
                    return
                query = parse_qs(urlparse(self.path).query)
                error = (
                    query.get("error_description") or query.get("error") or [None]
                )[0]
                body = oauth_handoff.page(error).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                catcher.callback_url = catcher.origin + self.path
                catcher.received.set()

            def log_message(self, *args):
                pass

        class Server(http.server.HTTPServer):
            # Never share the port with an app already listening on it.
            allow_reuse_address = False

        try:
            server = Server((host, port), Handler)
        except OSError:
            return None
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
        ).start()
        return server

    def started(self, url: str) -> None:
        # Before the browser opens, so only this sign-in's redirect counts.
        self.received.clear()
        self.callback_url = ""
        self.state = (parse_qs(urlparse(url).query).get("state") or [None])[0]
        if self.mode == "handoff" and not (
            self.state and oauth_handoff.expect(self.state)
        ):
            self.mode = "paste"

    def _wait_for(self, event: threading.Event) -> bool:
        deadline = time.monotonic() + oauth_handoff.WAIT_SECONDS
        while not event.wait(0.2):
            if self.cancelled.is_set():
                return False
            if time.monotonic() > deadline:
                self.failure = "no sign-in arrived within 10 minutes"
                return False
        return True

    def wait(self) -> str:
        """The callback address; called by the sign-in on its own thread."""
        if self.mode == "listener":
            return self.callback_url if self._wait_for(self.received) else ""
        if self.mode == "handoff":
            params = oauth_handoff.collect(self.state or "", cancelled=self.cancelled)
            if params is None:
                if not self.cancelled.is_set():
                    self.failure = "no sign-in arrived within 10 minutes"
                return ""
            return "?" + urlencode(params)
        self.paste_needed.set()
        return self.callback_url if self._wait_for(self.received) else ""

    def pasted(self, value: str) -> None:
        if not value:
            self.failure = "sign-in cancelled"
        self.callback_url = value
        self.received.set()

    def close(self) -> None:
        self.cancelled.set()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.state:
            oauth_handoff.discard(self.state)


def _read_paste(console, origin: str) -> str:
    console.print(
        f"The browser returns to {origin}, which can't pass the sign-in back "
        "here (an older MemoRizz UI shows 'OAuth state is invalid or "
        "expired'). Copy the page's full address from the address bar."
    )
    try:
        return input("Paste it here (Enter on its own cancels): ").strip()
    except EOFError:
        return ""


def sign_in(
    manager,
    server_name: str,
    console,
    *,
    title: Optional[str] = None,
    open_browser: bool = True,
) -> Dict[str, Any]:
    """Run the provider's sign-in for ``server_name`` and wait for it."""
    server = manager.get_server(server_name)
    title = title or server.name
    catcher = _Catcher(server.auth.redirect_uri or catalog.DEFAULT_REDIRECT_URI)
    outcome: Dict[str, Any] = {}

    def show_url(url: str) -> None:
        catcher.started(url)
        console.print(f"Sign in to {escape(title)} in your browser.")
        console.print(f"[dim]If no browser opened, visit:\n{escape(url)}[/dim]")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        if catcher.mode != "paste":
            console.print("[dim]Waiting for the sign-in… (Ctrl-C cancels)[/dim]")

    def work() -> None:
        try:
            outcome["result"] = manager.authorize_interactive(
                server_name,
                on_authorization_url=show_url,
                callback_reader=catcher.wait,
            )
        except BaseException as exc:
            outcome["result"] = {"ok": False, "error": str(exc)}

    # The sign-in runs on its own thread so Ctrl-C and a pasted address are
    # read here, on the terminal's thread.
    thread = threading.Thread(target=work, name="memorizz-sign-in", daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            if catcher.paste_needed.wait(0.2):
                catcher.paste_needed.clear()
                catcher.pasted(_read_paste(console, catcher.origin))
    except KeyboardInterrupt:
        catcher.cancelled.set()
        thread.join(5)
        return {"ok": False, "error": "sign-in cancelled"}
    finally:
        catcher.close()
    result = outcome.get("result") or {"ok": False, "error": "sign-in did not finish"}
    if not result.get("ok") and catcher.failure:
        result = {**result, "error": catcher.failure}
    return result


__all__ = ["command", "connect", "offer", "show", "sign_in"]
