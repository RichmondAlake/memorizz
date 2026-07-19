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
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .. import __version__
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
    continual_learning = bool(
        getattr(session.agent, "continual_learning_manager", None)
    )
    learning_status = (
        "[green]enabled[/green]" if continual_learning else "[dim]disabled[/dim]"
    )
    body = (
        f"[bold]memorizz {__version__}[/bold] — {mode}\n"
        f"provider: [cyan]{session.provider_name}[/cyan]   "
        f"model: [cyan]{session.model_name}[/cyan]\n"
        f"memory:   {store_line}\n"
        f"learning: continual {learning_status}\n"
        f"[dim]Type /help for commands · /code for coding tools · /exit to quit[/dim]"
    )
    console.print(Panel(body, expand=False, border_style="green"))
    for warning in session.warnings or []:
        console.print(f"[yellow]![/yellow] {warning}")


def _stream_turn(session, query: str) -> None:
    """Stream one turn: dim reasoning + tool activity above the live answer."""
    console = session.console
    agent = session.agent

    turn = {"reasoning": "", "answer": "", "tools": [], "error": None}
    live = None

    def renderable():
        parts = []
        if turn["reasoning"].strip():
            parts.append(
                Panel(
                    Text(turn["reasoning"].strip(), style="dim italic"),
                    title="reasoning",
                    border_style="grey42",
                    expand=False,
                )
            )
        if turn["tools"]:
            uniq = list(dict.fromkeys(turn["tools"]))
            parts.append(Text("⚙ " + ", ".join(uniq), style="dim cyan"))
        if turn["error"]:
            parts.append(Text("⚠ " + str(turn["error"]), style="red"))
        elif turn["answer"].strip():
            parts.append(Markdown(turn["answer"]))
        return Group(*parts) if parts else Text("thinking…", style="dim")

    def refresh():
        if live is not None:
            live.update(renderable())
            live.refresh()

    def on_event(ev):
        etype = ev.get("type")
        if etype == "trace":
            kind = ev.get("trace_kind")
            if kind == "reasoning":
                turn["reasoning"] += ev.get("content", "") or ""
            elif kind == "tool_call":
                turn["tools"].append(
                    str(ev.get("tool_name") or ev.get("title") or "tool")
                )
        elif etype == "error":
            turn["error"] = ev.get("message") or turn["error"]
        refresh()

    gen = None
    interrupted = False
    agent.set_stream_event_callback(on_event)
    live = Live(console=console, refresh_per_second=12, auto_refresh=False)
    live.start()
    refresh()
    try:
        gen = agent.run_stream(
            query,
            memory_id=session.memory_id,
            thread_id=session.thread_id,
            user_id=session.user_id,
        )
        for chunk in gen:
            if chunk:
                turn["answer"] += chunk
                refresh()
    except KeyboardInterrupt:
        interrupted = True
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
    except Exception as exc:  # surface, don't crash the REPL
        turn["error"] = turn["error"] or str(exc)
        refresh()
    finally:
        try:
            live.stop()
        except Exception:
            pass
        try:
            agent.set_stream_event_callback(None)
        except Exception:
            pass

    if turn["error"]:
        if "does not support tools" in str(turn["error"]).lower():
            console.print(
                "[yellow]This model can't tool-call.[/yellow] Try a tool-capable "
                "model, e.g. [cyan]ollama pull llama3.1:8b[/cyan] then "
                "[cyan]/model llama3.1:8b[/cyan]."
            )
    elif interrupted:
        console.print("[yellow]⏹ interrupted[/yellow]")
    elif not turn["answer"].strip() and not turn["reasoning"].strip():
        console.print("[dim](no output)[/dim]")

    session.sync_ids()
    try:
        cfg.save_state({"memory_id": session.memory_id})
    except Exception:
        pass


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
