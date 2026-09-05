# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Interactive REPL: stdin → canonical answer events → live Rich render.

Design notes:
- ``patch_stdout()`` keeps stray library logging from corrupting the prompt.
- A ``console.status`` spinner runs until the first token (local models can
  take 30-60s to first byte), then a ``rich.live.Live`` renders incremental
  Markdown.
- Ctrl-C mid-reply aborts only that turn (``gen.close()``); Ctrl-D / ``/exit``
  ends the session. We never hold a ``Live`` open across a prompt.
"""

import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .. import __version__
from . import commands
from . import config as cfg

_TOOL_OUTCOME_LABELS = {
    "success": "Completed",
    "empty": "Completed · no results",
    "degraded": "Completed with limitations",
    "fallback": "Completed via fallback",
    "provider_error": "Provider error",
    "error": "Failed",
}


class SlashCompleter(Completer):
    """Complete ``/command`` names, only when the line starts with ``/``."""

    def __init__(self, names):
        self.names = names

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for name in self.names:
            if name.startswith(text):
                yield Completion(name, start_position=-len(text))


def _quiet_logging() -> None:
    logging.getLogger("memorizz").setLevel(logging.WARNING)


def _banner(console: Console, session) -> None:
    mode = "coding" if session.code_mode else "memory assistant"
    store = type(session.provider).__name__ if session.provider else "(none)"
    root = getattr(session.provider, "root_path", None)
    store_line = f"{store}" + (f"  [dim]{root}[/dim]" if root else "")
    continual_learning = bool(
        getattr(session.agent, "continual_learning_manager", None)
    )
    learning_status = (
        "[green]enabled[/green]" if continual_learning else "[dim]disabled[/dim]"
    )
    try:
        browser_provider = session.agent.get_browser_control_provider_name()
    except Exception:
        browser_provider = None
    browser_status = (
        f"[green]{browser_provider}[/green]"
        if browser_provider
        else "[dim]disabled[/dim]"
    )
    body = (
        f"[bold]memorizz {__version__}[/bold] — {mode}\n"
        f"provider: [cyan]{session.provider_name}[/cyan]   "
        f"model: [cyan]{session.model_name}[/cyan]\n"
        f"memory:   {store_line}\n"
        f"learning: continual {learning_status}\n"
        f"browser:  {browser_status}\n"
        f"[dim]Type /help for commands · /browser for browser control · "
        f"/exit to quit[/dim]"
    )
    console.print(Panel(body, expand=False, border_style="green"))
    for warning in session.warnings or []:
        console.print(f"[yellow]![/yellow] {warning}")


def _stream_turn(session, query: str) -> None:
    """Render public deltas and tool status with bounded refresh frequency."""
    import time

    from .streaming import consume_stream, save_stream_session

    console = session.console or Console()
    if not console.is_terminal or console.is_dumb_terminal:
        # Rich intentionally defers Live output on dumb terminals/pipes.
        # Preserve incremental delivery there using the plain-text projection.
        consume_stream(session, query, stdout=console.file)
        return

    answer, status, tools = "", "preparing", []
    stream = None
    last_refresh = 0.0
    with Live(console=console, refresh_per_second=12, auto_refresh=False) as live:

        def refresh(force=False):
            nonlocal last_refresh
            now = time.monotonic()
            if not force and now - last_refresh < 1 / 12:
                return
            parts = [Text(status, style="dim")]
            if tools:
                parts.append(
                    Text("Tools: " + ", ".join(dict.fromkeys(tools)), style="dim cyan")
                )
            if answer:
                parts.append(Markdown(answer))
            live.update(Group(*parts), refresh=True)
            last_refresh = now

        try:
            stream = session.agent.run_stream_events(
                query,
                memory_id=session.memory_id,
                thread_id=session.thread_id,
                user_id=session.user_id,
            )
            while True:
                try:
                    event = stream.poll(timeout=1 / 12)
                except StopIteration:
                    break
                if event is None:
                    refresh()
                    continue
                kind = event["type"]
                if kind in {"run.started", "run.done"}:
                    session.memory_id = event.get("memory_id") or session.memory_id
                    session.thread_id = event.get("thread_id") or session.thread_id
                if kind == "answer.delta":
                    answer += event["delta"]
                elif kind == "status":
                    status = event.get("stage", "working").replace("_", " ")
                elif kind == "tool.started":
                    tools.append(event.get("tool_name", "tool"))
                elif kind == "approval.required":
                    status = "paused for approval"
                elif kind == "answer.done":
                    status = "answer complete; saving"
                elif kind == "run.done":
                    status = event["status"]
                refresh(force=kind in {"answer.done", "run.done"})
        except KeyboardInterrupt:
            status = "interrupted; partial answer retained"
        except Exception:
            status = "stream failed; partial answer retained"
        finally:
            if stream is not None:
                stream.close()
            refresh(force=True)
    if save_stream_session(session) == "failed":
        session.console.print("[yellow]Session state could not be saved.[/yellow]")


def run_repl(session) -> None:
    """Run the interactive loop until ``/exit`` / Ctrl-D."""
    _quiet_logging()
    console = session.console or Console()
    session.console = console

    cfg.ensure_home()
    ptk = PromptSession(
        history=FileHistory(str(cfg.history_file())),
        completer=SlashCompleter(commands.command_completions()),
        complete_while_typing=True,
    )

    _banner(console, session)

    while True:
        prompt_text = "code> " if session.code_mode else "memorizz> "
        try:
            with patch_stdout():
                line = ptk.prompt(prompt_text)
        except (KeyboardInterrupt, EOFError):
            # Ctrl-C or Ctrl-D at the prompt: exit. (Ctrl-C *during* a streaming
            # reply only aborts that reply — handled in _stream_turn.)
            commands.cmd_exit(session, "")
            break

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            if not commands.dispatch(line, session):
                break
            continue

        _stream_turn(session, line)
