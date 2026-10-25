# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Typer commands that let agents open, read and operate the apps on this Mac."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional, Tuple

import typer

desktop_app = typer.Typer(
    help="Let agents open, read and operate the apps on this Mac (MCP).",
    no_args_is_help=True,
)

_TERMINALS = {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "vscode": "VS Code"}


def _console():
    from rich.console import Console

    return Console()


def _state_path() -> str:
    from . import config as cfg

    return str(cfg.memorizz_home() / "desktop" / "last-read.json")


def server_config(
    name: str = "desktop", *, approval: bool = True, screenshots: bool = True
) -> dict:
    """The MCP connection that runs ``memorizz desktop serve`` for an agent."""
    return {
        "name": name,
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "memorizz", "desktop", "serve", "--state", _state_path()]
        + ([] if screenshots else ["--no-screenshots"]),
        "timeout": 60.0,
        "require_approval": approval,
    }


def _terminal() -> str:
    """The app macOS asks about: the ``.app`` this process was started from."""
    pid = os.getpid()
    for _ in range(25):
        try:
            line = subprocess.run(
                ["ps", "-o", "ppid=,comm=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            parent, command = line.split(None, 1)
            pid = int(parent)
        except (OSError, ValueError, subprocess.SubprocessError):
            break
        if ".app/" in command:
            return command.split(".app/")[0].rsplit("/", 1)[-1]
        if pid <= 1:
            break
    program = os.environ.get("TERM_PROGRAM", "")
    return _TERMINALS.get(program, program) or "your terminal app"


@desktop_app.command("serve")
def serve(
    state: str = typer.Option(
        None, "--state", help="File that keeps the last read's numbers between calls."
    ),
    screenshots: bool = typer.Option(
        True,
        "--screenshots/--no-screenshots",
        help="Save a screenshot of the app's window after each step (next to --state).",
    ),
):
    """Run the desktop MCP server on stdio (memorizz starts it for the agent)."""
    from pathlib import Path

    from ..desktop_control.server import run_desktop_mcp_server

    state = state or _state_path()
    run_desktop_mcp_server(
        state, str(Path(state).parent / "screenshots") if screenshots else None
    )


@desktop_app.command("doctor")
def doctor(
    ask: bool = typer.Option(
        False, "--ask", help="Show macOS's permission prompts for whatever is missing."
    ),
):
    """Check that this Mac lets MemoRizz read and operate apps."""
    from ..desktop_control import DesktopError, MacDesktop

    console = _console()
    try:
        status = MacDesktop().permissions(prompt=ask)
    except DesktopError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if not _report(console, status):
        raise typer.Exit(1)


@desktop_app.command("connect")
def connect(
    name: str = typer.Option("desktop", "--name", help="Connection name."),
    approval: bool = typer.Option(
        True,
        "--approval/--no-approval",
        help="Ask before each app launch, click, keystroke or scroll (reading never asks).",
    ),
    screenshots: bool = typer.Option(
        True,
        "--screenshots/--no-screenshots",
        help="Keep a screenshot of each step with the run's workflow memory.",
    ),
):
    """Attach the desktop tools to the chat agent; restart memorizz to use them.

    Runs are kept as workflow (procedural) memory: each step, the window it
    read and, with Screen Recording allowed, a screenshot.
    """
    from . import mcp_config

    console = _console()
    manager = mcp_config.build_manager()
    manager.upsert_server(
        server_config(name, approval=approval, screenshots=screenshots)
    )
    path = mcp_config.save_servers(manager.server_dicts())
    # Listing caches the tools, so the chat agent gets one tool per action.
    listing = manager.list_tools(name)
    if not listing.get("ok"):
        console.print(
            f"[red]Saved {name}, but the server did not start:[/red] {listing.get('error')}"
        )
        raise typer.Exit(1)
    tools = [tool.get("name") for tool in listing.get("tools") or []]
    console.print(f"[green]Connected[/green] {name} ({len(tools)} tools) in {path}")
    console.print(
        "Reading windows never asks; "
        + (
            "opening apps and links, clicks, typing and keys ask you first."
            if approval
            else "[yellow]opening apps and links, clicks, typing and keys run without asking.[/yellow]"
        )
    )
    console.print(
        "Each run is kept as workflow memory (its steps"
        + (" and screenshots" if screenshots else "")
        + ")."
    )
    console.print(
        "Next: [cyan]memorizz desktop doctor[/cyan], then restart memorizz "
        "(or use [cyan]/desktop on[/cyan] in a chat, which needs no restart)."
    )


# -- /desktop in a chat ---------------------------------------------------------


def _attached(manager) -> Optional[str]:
    """The attached desktop connection, whatever it was named."""
    for server in manager.server_dicts():
        args = [str(arg) for arg in server.get("args") or []]
        if "desktop" in args and "serve" in args:
            return str(server.get("name"))
    return None


def _permissions(prompt: bool = False) -> Tuple[Optional[dict], Optional[str]]:
    from ..desktop_control import DesktopError, MacDesktop

    try:
        return MacDesktop().permissions(prompt=prompt), None
    except DesktopError as exc:
        return None, str(exc)


def _report(console, status: dict) -> bool:
    """Print what macOS allows; True when agents can read and operate apps."""
    app = _terminal()
    if status["screen_recording"]:
        console.print(
            "[green]Screen Recording: allowed.[/green] Each step keeps a screenshot."
        )
    else:
        console.print(
            f"[yellow]Screen Recording: not allowed for {app}.[/yellow] Steps are "
            "remembered without screenshots; turn it on under Privacy & Security "
            "> Screen Recording to keep them."
        )
    if status["accessibility"]:
        console.print(
            "[green]Accessibility: allowed.[/green] Agents can read and operate apps."
        )
        return True
    console.print(
        f"[yellow]Accessibility: not allowed for {app}.[/yellow] Agents can "
        "open apps and links but not read or operate them.\n"
        f"Turn on {app} in System Settings > Privacy & Security > "
        f"Accessibility, then run this again (restart {app} if it still "
        "says no)."
    )
    return False


def chat_command(session, args: str) -> None:
    """``/desktop [on|off|status|doctor]``: computer use for the chat agent.

    ``on`` attaches the desktop tools to the running agent and saves them
    with it, so no restart is needed. Each action asks in the chat first.
    """
    from rich.markup import escape

    from . import connect

    console = session.console
    agent = session.agent
    words = args.split()
    sub = words[0].lower() if words else "status"
    manager = getattr(agent, "mcp_manager", None)
    if manager is None:
        console.print("[red]This agent cannot use MCP servers.[/red]")
        return
    try:
        connect._sync_shared_config(agent)
    except Exception:
        pass
    name = _attached(manager)

    if sub == "doctor":
        status, error = _permissions(prompt="--ask" in words)
        if status is None:
            console.print(f"[red]{escape(error or '')}[/red]")
        else:
            _report(console, status)
        return

    if sub in {"on", "connect"}:
        status, error = _permissions()
        if status is None:
            console.print(
                f"[red]Desktop control is not available:[/red] {escape(error or '')}"
            )
            return
        config = server_config(
            name or "desktop", screenshots="--no-screenshots" not in words
        )
        try:
            manager.upsert_server(config)
            connect._apply(session, manager.server_dicts())
            listing = manager.list_tools(config["name"])
        except Exception as exc:
            console.print(
                f"[red]Could not turn on desktop control:[/red] {escape(str(exc))}"
            )
            return
        if not listing.get("ok"):
            console.print(
                "[red]The desktop server did not start:[/red] "
                f"{escape(str(listing.get('error')))}"
            )
            return
        tools = agent._register_mcp_server_tools(
            config["name"], manager.cached_tools(config["name"])
        )
        console.print(
            f"[green]Desktop control on[/green] · {len(tools)} tools. Reading "
            "windows never asks; opening apps and links, clicks, typing and keys "
            "ask you here first."
        )
        if not status["accessibility"]:
            console.print(
                f"[yellow]{escape(_terminal())} doesn't have Accessibility yet[/yellow], "
                "so the agent can open apps and links but not read or operate "
                "them: [cyan]/desktop doctor --ask[/cyan]"
            )
        return

    if sub in {"off", "disconnect"}:
        if name is None:
            console.print("Desktop control is already off.")
            return
        agent._register_mcp_server_tools(name, [])
        manager.remove_server(name, delete_credentials=True)
        connect._apply(session, manager.server_dicts())
        console.print("[yellow]Desktop control OFF[/yellow]")
        return

    console.print(
        "Desktop control: "
        + (f"[green]on[/green] · {escape(name)}" if name else "[cyan]off[/cyan]")
    )
    status, error = _permissions()
    if status is None:
        console.print(f"[yellow]{escape(error or '')}[/yellow]")
    else:
        _report(console, status)
    console.print("Usage: /desktop on | off | status | doctor [--ask]")
