# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Interactive REPL: stdin → ``agent.run_stream`` → live rich render.

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
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from . import commands
from . import config as cfg


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
    body = (
        f"[bold]memorizz[/bold] — {mode}\n"
        f"provider: [cyan]{session.provider_name}[/cyan]   "
        f"model: [cyan]{session.model_name}[/cyan]\n"
        f"memory:   {store_line}\n"
        f"[dim]Type /help for commands · /code for coding tools · /exit to quit[/dim]"
    )
    console.print(Panel(body, expand=False, border_style="green"))
    for warning in session.warnings or []:
        console.print(f"[yellow]![/yellow] {warning}")


def _stream_turn(session, query: str) -> None:
    console = session.console
    agent = session.agent
    acc = ""
    gen = None
    started = False
    live = None
    interrupted = False
    error = None

    status = console.status("[dim]thinking…[/dim]", spinner="dots")
    status.start()
    try:
        gen = agent.run_stream(
            query,
            memory_id=session.memory_id,
            thread_id=session.thread_id,
            user_id=session.user_id,
        )
        for chunk in gen:
            if not chunk:
                continue
            if not started:
                status.stop()
                started = True
                live = Live(console=console, refresh_per_second=12, auto_refresh=False)
                live.start()
            acc += chunk
            live.update(Markdown(acc))
            live.refresh()
    except KeyboardInterrupt:
        interrupted = True
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
    except Exception as exc:  # surface, don't crash the REPL
        error = exc
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
    finally:
        if not started:
            try:
                status.stop()
            except Exception:
                pass
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass

    if error is not None:
        console.print(f"[red]Error:[/red] {error}")
    elif interrupted:
        console.print("[yellow]⏹ interrupted[/yellow]")
    elif not acc.strip():
        console.print("[dim](no output)[/dim]")

    session.sync_ids()


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
        except KeyboardInterrupt:
            # Ctrl-C at the prompt: clear the line, keep going.
            continue
        except EOFError:
            # Ctrl-D: graceful exit.
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
