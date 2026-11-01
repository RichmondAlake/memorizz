# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""``memorizz mcp apps``: connect cloud apps (Grok Bot, ChatGPT dots) to memory.

``serve`` puts the MCP server online with OAuth sign-in, ``pair`` makes the
one-time code that approves an app, and ``list``/``revoke`` manage the apps.
"""

from __future__ import annotations

import atexit
import getpass
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import typer

apps_app = typer.Typer(
    help="Connect cloud apps such as Grok Bot and ChatGPT dots to your memory.",
    no_args_is_help=True,
)

_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_PRINCIPAL = re.compile(r"[^A-Za-z0-9_.:@-]")


def default_principal() -> str:
    """Whose memory approved apps use unless told otherwise: this Mac's user."""
    try:
        name = _PRINCIPAL.sub("-", getpass.getuser()).strip("-.")
    except Exception:
        name = ""
    return name[:64] or "owner"


def memory_owner(principal: Optional[str], shared: bool) -> str:
    """The principal a pairing code connects an app to."""
    from ..mcp_server.config import _PRINCIPAL_PATTERN, LOCAL_AGENTS_PRINCIPAL

    if shared:
        if principal:
            raise typer.BadParameter(
                "--shared uses your local agents' memory; leave out --principal."
            )
        return LOCAL_AGENTS_PRINCIPAL
    who = principal or default_principal()
    if not _PRINCIPAL_PATTERN.fullmatch(who):
        raise typer.BadParameter(
            "A principal starts with a letter or number and uses only letters, "
            "numbers, '.', '_', ':', '@' or '-'."
        )
    return who


def owner_label(principal: str) -> str:
    """How the terminal names whose memory an app uses."""
    from ..mcp_server.config import LOCAL_AGENTS_PRINCIPAL

    return "your local agents" if principal == LOCAL_AGENTS_PRINCIPAL else principal


_SHARED_WARNING = (
    "The app uses the same memory as your local agents (Claude Code, Codex): every "
    "project, including past sessions. It can change those memories too, unless "
    "it's read only."
)


def _scopes(read_only: bool) -> list:
    from ..mcp_server.config import READ_SCOPE, WRITE_SCOPE

    return [READ_SCOPE] if read_only else [READ_SCOPE, WRITE_SCOPE]


def _console():
    from rich.console import Console

    return Console()


def _when(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "never"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def start_tunnel(port: int, *, timeout: float = 45.0) -> Tuple[subprocess.Popen, str]:
    """A Cloudflare quick tunnel to the local port, and its public URL."""
    binary = os.environ.get("MEMORIZZ_CLOUDFLARED") or shutil.which("cloudflared")
    if not binary:
        raise typer.BadParameter(
            "--tunnel uses Cloudflare's free quick tunnel, and cloudflared isn't "
            "installed. Get it with `brew install cloudflared` or from "
            "https://github.com/cloudflare/cloudflared/releases (set "
            "MEMORIZZ_CLOUDFLARED to its path if it isn't on PATH). Or run "
            "`tailscale funnel --bg 8766` and pass --public-url with the "
            "https://….ts.net address it prints; that address never changes."
        )
    process = subprocess.Popen(
        [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    atexit.register(process.terminate)
    lines: "queue.Queue[str]" = queue.Queue()

    def drain() -> None:
        # Read cloudflared's log for the URL, then keep reading so it never
        # blocks on a full pipe.
        for line in process.stderr:
            lines.put(line)

    threading.Thread(target=drain, daemon=True).start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            if process.poll() is not None:
                break
            continue
        match = _TUNNEL_URL.search(line)
        if match:
            return process, match.group(0)
    process.terminate()
    raise typer.BadParameter(
        "Cloudflare did not give a tunnel address in time; try again."
    )


@apps_app.command("serve")
def serve_apps(
    public_url: Optional[str] = typer.Option(
        None,
        "--public-url",
        help="Your HTTPS address for this server, e.g. a Tailscale Funnel or named tunnel.",
    ),
    tunnel: bool = typer.Option(
        False,
        "--tunnel",
        help="Get a temporary public address from Cloudflare (needs cloudflared).",
    ),
    port: int = typer.Option(8766, "--port", min=1, max=65535),
    principal: Optional[str] = typer.Option(
        None,
        "--principal",
        help="Whose memory the first pairing code connects to (default: your Mac user).",
    ),
    read_only: bool = typer.Option(
        False, "--read-only", help="Let apps read but not save memories."
    ),
    shared: bool = typer.Option(
        False,
        "--shared",
        help="The first pairing code shares your local agents' memory, in every project.",
    ),
):
    """Put your memory online for Grok Bot and ChatGPT dots, with sign-in."""
    from .config import load_layered_env

    load_layered_env()
    try:
        from ..mcp_server import MemorizzMCPServerConfig, run_memorizz_mcp_server
        from ..mcp_server.oauth_store import PAIRING_TTL_SECONDS, OAuthStore
    except ImportError as exc:
        raise typer.BadParameter(
            "Connecting apps needs the MCP server; install `memorizz[mcp]`."
        ) from exc
    who = memory_owner(principal, shared)
    if tunnel and public_url:
        raise typer.BadParameter("Use --tunnel or --public-url, not both.")
    if not tunnel and not public_url:
        raise typer.BadParameter(
            "Grok and ChatGPT reach your memory only over public HTTPS. Pass "
            "--public-url https://… (a Tailscale Funnel or named Cloudflare tunnel "
            "keeps one address), or --tunnel for a temporary one."
        )
    if tunnel:
        _, public_url = start_tunnel(port)
    try:
        config = MemorizzMCPServerConfig.from_env(
            transport="streamable-http",
            host="127.0.0.1",
            port=port,
            public_url=public_url,
            oauth=True,
            allow_writes=not read_only,
            allow_agent_execution=False,
            allow_harness_execution=False,
            allow_training=False,
            tool_set="memory",
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    code, _ = OAuthStore().create_pairing_code(who, _scopes(read_only))
    url = f"{config.public_url}{config.path}"
    minutes = PAIRING_TTL_SECONDS // 60
    console = _console()
    console.print(
        f"\n[green]Your MemoRizz memory is online for your apps[/green] as "
        f"[bold]{owner_label(who)}[/bold]"
    )
    console.print(f"  Server URL    [bold]{url}[/bold]")
    console.print(
        f"  Pairing code  [bold]{code}[/bold]  [dim](one app, {minutes} minutes; "
        "`memorizz mcp apps pair` makes another)[/dim]\n"
    )
    if shared:
        console.print(f"  [yellow]{_SHARED_WARNING}[/yellow]\n")
    console.print(
        "  [bold]Grok Bot[/bold]   Connect apps → Add a custom connector → name it memorizz, "
        "type Remote HTTPS, paste the URL (leave the OAuth fields empty), and sign in "
        "with a pairing code."
    )
    console.print(
        "  [bold]Grok[/bold]  grok.com/connectors → New Connector → Custom → paste the "
        "URL, and sign in with a pairing code."
    )
    console.print(
        "  [bold]ChatGPT dots[/bold]  Plugins → + → Add custom MCP server (Developer mode) → "
        "paste the URL, choose OAuth, and sign in with a pairing code. Your dot uses "
        "your plugins."
    )
    console.print(
        "  [bold]Claude[/bold]  claude.ai: Customize → Connectors → Add custom connector → "
        f"paste the URL. Claude Code: `claude mcp add --transport http memorizz {url}`, "
        "then `claude mcp login memorizz`."
    )
    if tunnel:
        console.print(
            "\n  [yellow]This tunnel address changes every time it starts,[/yellow] so the "
            "apps must be connected again. Use --public-url with a fixed address to keep them."
        )
    console.print(
        "\n  [dim]Keep this running; Ctrl-C takes the memory offline.[/dim]\n"
    )
    run_memorizz_mcp_server(config)


@apps_app.command("pair")
def pair_app(
    principal: Optional[str] = typer.Option(
        None,
        "--principal",
        help="Whose memory the app connects to (default: your Mac user).",
    ),
    read_only: bool = typer.Option(
        False, "--read-only", help="The app may read but not save memories."
    ),
    shared: bool = typer.Option(
        False,
        "--shared",
        help="Share your local agents' memory (Claude Code, Codex), in every project.",
    ),
    minutes: int = typer.Option(10, "--minutes", min=1, max=60),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Make a one-time code that approves one app on its sign-in page."""
    from ..mcp_server.oauth_store import OAuthStore

    who = memory_owner(principal, shared)
    code, expires_at = OAuthStore().create_pairing_code(
        who, _scopes(read_only), ttl_seconds=minutes * 60
    )
    if raw_json:
        typer.echo(
            json.dumps({"code": code, "principal": who, "expires_at": expires_at})
        )
        return
    console = _console()
    console.print(
        f"Pairing code [bold]{code}[/bold] connects one app to the memory of "
        f"[bold]{owner_label(who)}[/bold]{' (read only)' if read_only else ''}. "
        f"It works once, until {_when(expires_at)}."
    )
    if shared:
        console.print(f"[yellow]{_SHARED_WARNING}[/yellow]")


@apps_app.command("list")
def list_apps(
    include_revoked: bool = typer.Option(
        False, "--all", help="Include disconnected apps."
    ),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Show the apps connected to your memory."""
    from ..mcp_server.oauth_store import OAuthStore

    grants = OAuthStore().list_grants(include_revoked=include_revoked)
    if raw_json:
        typer.echo(json.dumps(grants))
        return
    if not grants:
        _console().print(
            "No apps are connected. Start with `memorizz mcp apps serve --tunnel`."
        )
        return
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    # Revoking needs the whole ID, so it wraps rather than being cut short.
    table.add_column("ID", overflow="fold", min_width=20)
    for column in ("App", "Memory of", "Can", "Connected", "Last used"):
        table.add_column(column)
    for grant in grants:
        can = "read, save" if "memorizz:write" in grant["scopes"] else "read"
        if grant["revoked_at"]:
            can = "disconnected"
        table.add_row(
            grant["grant_id"],
            grant["app_name"],
            owner_label(grant["principal"]),
            can,
            _when(grant["created_at"]),
            _when(grant["last_used_at"]),
        )
    _console().print(table)


@apps_app.command("revoke")
def revoke_app(grant_id: str = typer.Argument(..., help="The app's ID from `list`.")):
    """Disconnect an app now; it has to be approved again to come back."""
    from ..mcp_server.oauth_store import OAuthStore

    store = OAuthStore()
    grant = store.grant(grant_id)
    if grant is None or not store.revoke_grant(grant_id):
        _console().print(f"No connected app has the ID {grant_id}.")
        raise typer.Exit(1)
    _console().print(f"Disconnected {grant['app_name']}.")
