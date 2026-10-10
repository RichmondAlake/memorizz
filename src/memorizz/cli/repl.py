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
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .. import __version__
from . import commands
from . import config as cfg
from . import harness_session, ui
from .updates import known_update, update_notifier


class SlashCompleter(Completer):
    """Complete ``/command`` names, only when the line starts with ``/``."""

    def __init__(self, names):
        self.names = names

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for name in self.names:
            if name.startswith(text):
                yield Completion(name, start_position=-len(text))


class SafeFileHistory(FileHistory):
    """Never retain credential/configuration input, including abbreviations."""

    @staticmethod
    def safe(value):
        for line in value.splitlines():
            stripped = line.lstrip()
            if not stripped.startswith("/"):
                continue
            name = (
                stripped[1:].split(maxsplit=1)[0].lower()
                if stripped[1:].strip()
                else ""
            )
            if name and any(
                command.startswith(name)
                for command in ("login", "config", "memory-provider")
            ):
                return False
        return True

    def append_string(self, string):
        if self.safe(string):
            super().append_string(string)

    def store_string(self, string):
        if self.safe(string):
            super().store_string(string)

    def load_history_strings(self):
        for string in super().load_history_strings():
            if self.safe(string):
                yield string


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
        f"[dim]Type /help for commands · /config for settings · "
        f"/memory-provider for memory setup · "
        f"/exit to quit[/dim]"
    )
    update = known_update()
    if update:
        session.update_notice = update
        body += (
            f"\n[bold yellow]⬆ Update available:[/bold yellow] {__version__} → "
            f"{update['latest']}   [yellow]{update['command']}[/yellow]"
        )
    panel = Panel(body, expand=False, border_style="green")
    if ui.should_animate(console):
        console.print(Columns([ui.crest_text(), panel], padding=(0, 2)))
    else:
        console.print(panel)
    for warning in session.warnings or []:
        console.print(f"[yellow]![/yellow] {warning}")


def _stream_turn(session, query: str) -> None:
    """Render public deltas and tool status with bounded refresh frequency."""
    import time

    from ..llms.streaming import provider_error_message
    from ..streaming import harness_event_listener
    from .harness_feed import HarnessFeed
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
    started, finished = time.monotonic(), False
    # A harness's own output (messages, commands, tool calls) prints above
    # the live status as it happens.
    feed = HarnessFeed()
    listening = harness_event_listener.set(feed.listener)
    with ui.quiet_keys(), Live(
        console=console, refresh_per_second=12, auto_refresh=False
    ) as live:

        def show_feed(lines):
            for line in lines:
                live.console.print(line, overflow="fold")

        def refresh(force=False):
            nonlocal last_refresh
            now = time.monotonic()
            if not force and now - last_refresh < 1 / 12:
                return
            line = status
            if not answer and not finished:
                line += ui.elapsed_suffix(now - started)
            parts = [Text(line, style="dim")]
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
                show_feed(feed.drain())
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
                    status = event.get("message") or event.get(
                        "stage", "working"
                    ).replace("_", " ")
                elif kind == "tool.started":
                    tools.append(event.get("tool_name", "tool"))
                elif kind == "approval.required":
                    status = "paused for approval"
                elif kind == "answer.done":
                    status, finished = "answer complete; saving", True
                    show_feed(feed.finish(answer))
                elif kind == "run.done":
                    finished = True
                    show_feed(feed.finish(answer))
                    status = (
                        provider_error_message(event.get("error_code"))
                        or event["status"]
                    )
                refresh(force=kind in {"answer.done", "run.done"})
        except KeyboardInterrupt:
            status, finished = "interrupted; partial answer retained", True
        except Exception:
            status, finished = "stream failed; partial answer retained", True
        finally:
            if stream is not None:
                stream.close()
            harness_event_listener.reset(listening)
            refresh(force=True)
    if save_stream_session(session) == "failed":
        session.console.print("[yellow]Session state could not be saved.[/yellow]")


def run_repl(session) -> None:
    """Run the interactive loop until ``/exit`` / Ctrl-D."""
    _quiet_logging()
    console = session.console or Console()
    session.console = console

    cfg.ensure_home()
    ui.ignore_focus_reports()
    ptk = PromptSession(
        history=SafeFileHistory(str(cfg.history_file())),
        completer=SlashCompleter(commands.command_completions()),
        complete_while_typing=True,
        key_bindings=ui.key_bindings(session),
        bottom_toolbar=lambda: ui.toolbar(session),
        style=ui.STYLE,
        color_depth=ui.color_depth(),
    )

    ui.animate_crest(console)
    _banner(console, session)

    with patch_stdout(raw=True), update_notifier(console, session):
        while True:
            try:
                line = ptk.prompt(ui.prompt_fragments(session))
            except (KeyboardInterrupt, EOFError):
                # Ctrl-C or Ctrl-D at the prompt exits; _stream_turn handles
                # interrupts during a reply without ending the session.
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
            harness_session.report_pending_approvals(session, console)
