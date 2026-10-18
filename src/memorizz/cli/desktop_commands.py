# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Typer commands that let agents open, read and operate the apps on this Mac."""

from __future__ import annotations

import os
import subprocess
import sys

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
        return
    console.print(
        f"[yellow]Accessibility: not allowed for {app}.[/yellow] Agents can "
        "open apps but not read or operate them.\n"
        f"Turn on {app} in System Settings > Privacy & Security > "
        f"Accessibility, then run this again (restart {app} if it still "
        "says no)."
    )
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
        {
            "name": name,
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "memorizz", "desktop", "serve", "--state", _state_path()]
            + ([] if screenshots else ["--no-screenshots"]),
            "timeout": 60.0,
            "require_approval": approval,
        }
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
            "opening apps, clicks, typing and keys wait for /approvals."
            if approval
            else "[yellow]opening apps, clicks, typing and keys run without asking.[/yellow]"
        )
    )
    console.print(
        "Each run is kept as workflow memory (its steps"
        + (" and screenshots" if screenshots else "")
        + ")."
    )
    console.print("Next: [cyan]memorizz desktop doctor[/cyan], then restart memorizz.")
