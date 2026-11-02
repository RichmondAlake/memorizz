# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""``/timer``, and telling the person when a countdown ends."""

from __future__ import annotations

import re
from contextlib import contextmanager
from threading import Event, Thread
from typing import Iterator, Optional

from rich.markup import escape
from rich.text import Text

from .. import timers
from .commands import _con
from .updates import _redraw_prompt

_LENGTH_RE = re.compile(r"^(\d+(?:\.\d+)?)(s|sec|secs|m|min|mins|h|hr|hrs)?$", re.I)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600}


def parse_length(text: str) -> Optional[float]:
    """Seconds in ``25``, ``90s``, ``25m`` or ``1h``; a bare number is minutes."""
    match = _LENGTH_RE.match(text.strip())
    if match is None:
        return None
    return float(match.group(1)) * _UNIT_SECONDS[(match.group(2) or "m")[0].lower()]


def _usage(console) -> None:
    usage = "Usage: /timer 25 focus · /timer 90s tea · /timer 1h · /timer cancel [id]"
    console.print(f"[dim]{escape(usage)}[/dim]")


def cmd_timer(session, args: str) -> None:
    console = _con(session)
    parts = args.split()
    if not parts:
        active = timers.running()
        if not active:
            console.print("[dim]No timers running.[/dim]")
            _usage(console)
            return
        for item in active:
            console.print(
                f"  [cyan]{timers.format_time(item['remaining'])}[/cyan] "
                f"{escape(item['label'])} [dim]{item['timer_id']}[/dim]"
            )
        return
    if parts[0].lower() in {"cancel", "stop"}:
        stopped = timers.cancel(parts[1] if len(parts) > 1 else "")
        if stopped is None:
            console.print(
                "[yellow]Say which timer.[/yellow] /timer lists them with their ids."
            )
            return
        console.print(f"[dim]Cancelled {escape(stopped['label'])}.[/dim]")
        return
    seconds = parse_length(parts[0])
    if seconds is None:
        _usage(console)
        return
    try:
        started = timers.start(
            seconds,
            " ".join(parts[1:]),
            agent_id=getattr(session.agent, "agent_id", None),
        )
    except (ValueError, OSError) as exc:
        console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        return
    console.print(
        f"[green]{escape(started['label'])}: "
        f"{timers.format_time(started['seconds'])}[/green] "
        "[dim]counting down in the status bar[/dim]"
    )
    _redraw_prompt()


def announce_finished(console) -> None:
    """Say which timers have ended, including any that ended with the CLI closed."""
    try:
        finished = timers.take_finished()
    except Exception:
        # A timer must never stop the CLI starting.
        return
    for item in finished:
        console.print(
            Text(
                f"⏰ {item['label']} finished ({timers.format_time(item['seconds'])})",
                style="bold yellow",
            )
        )
        console.bell()


@contextmanager
def timer_watch(console, *, every: float = 1.0) -> Iterator[None]:
    """Keep the status bar's countdown moving and ring when a timer ends.

    The caller holds prompt_toolkit's ``patch_stdout``, so a line printed here
    lands above the prompt. A running timer redraws the bar; none costs nothing
    but one small file check a second.
    """
    stopped = Event()

    def watch() -> None:
        while not stopped.wait(every):
            try:
                announce_finished(console)
                if timers.running():
                    _redraw_prompt()
            except Exception:
                # A timer must never interrupt an agent session.
                pass

    if console.is_terminal and not console.is_dumb_terminal:
        Thread(target=watch, name="memorizz-timer-watch", daemon=True).start()
    try:
        yield
    finally:
        stopped.set()


__all__ = ["announce_finished", "cmd_timer", "parse_length", "timer_watch"]
