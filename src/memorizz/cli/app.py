# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The ``memorizz`` Typer application and ``main()`` entry point.

Lazy-import discipline: only ``typer`` and the light ``cli.config`` module are
imported at module load. Everything that pulls in ``memorizz`` core (agent
factory, REPL, providers) is imported inside the command handlers so that
``memorizz --help`` / ``--version`` / ``init`` stay sub-second.
"""

import os
import sys
from typing import List, Optional

import typer

from . import config as cfg

app = typer.Typer(
    name="memorizz",
    help="memorizz — a local agent CLI with persistent memory.",
    no_args_is_help=False,
    add_completion=False,
)


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _deprecate(msg: str) -> None:
    _eprint(f"note: {msg}")


def _make_console():
    from rich.console import Console

    return Console()


def _print_version() -> None:
    try:
        from importlib.metadata import version

        v = version("memorizz")
    except Exception:
        v = "unknown"
    print(f"memorizz {v}")


# --------------------------------------------------------------------------- #
# Root callback: --version and the no-subcommand REPL launch
# --------------------------------------------------------------------------- #


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Show the installed version and exit."
    ),
):
    if version:
        _print_version()
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _launch_repl(code_mode=False)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@app.command(help="Start the interactive REPL (this is also the default).")
def chat(
    code: bool = typer.Option(
        False, "--code", help="Enable coding tools (file edits + commands)."
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Force an LLM provider."
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Force a model."),
):
    _launch_repl(code_mode=code, provider=provider, model=model)


@app.command(
    help='Run a single prompt and print the reply (e.g. memorizz run "hello").'
)
def run(
    prompt: Optional[List[str]] = typer.Argument(None),
    code: bool = typer.Option(False, "--code", help="Enable coding tools."),
):
    if not prompt:
        _eprint('Usage: memorizz run "<prompt>"')
        raise typer.Exit(2)
    _run_oneshot(" ".join(prompt), code_mode=code)


@app.command(help="Launch the local web UI (requires memorizz[ui]).")
def ui(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
):
    from .legacy import run_local

    ok = run_local(host=host, port=port)
    raise typer.Exit(0 if ok else 1)


@app.command(help="Set up ~/.memorizz (API keys or the local Ollama stack).")
def init(
    local: bool = typer.Option(
        False, "--local", help="Configure the fully-local Ollama stack."
    ),
    force: bool = typer.Option(False, "--force"),
):
    _init(local=local, force=force)


@app.command(name="config", help="Show resolved configuration and paths.")
def config_cmd():
    _show_config()


# --- Oracle sub-app (legacy commands preserved) ---
oracle_app = typer.Typer(help="Oracle database helpers.")
app.add_typer(oracle_app, name="oracle")


@oracle_app.command("install", help="Install the Oracle container.")
def oracle_install(image: Optional[str] = typer.Option(None, "--image", "-i")):
    from .legacy import install_oracle

    raise typer.Exit(0 if install_oracle(image=image) else 1)


@oracle_app.command("setup", help="Set up the Oracle user/schema.")
def oracle_setup():
    from .legacy import setup_oracle

    raise typer.Exit(0 if setup_oracle() else 1)


@oracle_app.command("setup-schema", help="Apply Oracle schema updates (no user drop).")
def oracle_setup_schema():
    from .legacy import setup_oracle_schema

    raise typer.Exit(0 if setup_oracle_schema() else 1)


@oracle_app.command("teardown", help="Teardown the Oracle installation.")
def oracle_teardown(
    mode: Optional[str] = typer.Option(
        None, "--mode", help="drop-user|remove-container|full"
    ),
    force: bool = typer.Option(False, "--force"),
):
    from .legacy import teardown_oracle

    raise typer.Exit(0 if teardown_oracle(mode=mode, force=force) else 1)


# --- Automations sub-app (legacy command preserved) ---
automations_app = typer.Typer(help="Automations worker.")
app.add_typer(automations_app, name="automations")


@automations_app.command("run", help="Run the always-on automations worker (Oracle).")
def automations_run(
    poll_interval: int = typer.Option(5, "--poll-interval"),
    lease_seconds: int = typer.Option(120, "--lease-seconds"),
    concurrency: int = typer.Option(2, "--concurrency"),
):
    from .legacy import run_automations

    ok = run_automations(
        poll_interval=poll_interval,
        lease_seconds=lease_seconds,
        concurrency=concurrency,
    )
    raise typer.Exit(0 if ok else 1)


# --------------------------------------------------------------------------- #
# Build / wizard helpers
# --------------------------------------------------------------------------- #


def _load_env() -> None:
    cfg.load_layered_env()


def _resolve_llm_config(provider, model):
    from . import agent_factory

    if provider:
        c = agent_factory.config_for_provider(provider)
        if model:
            c["model"] = model
        return c
    if model:
        c = agent_factory.detect_llm_config()  # may raise; caller handles
        c["model"] = model
        return c
    return None


def _build_or_wizard(code_mode, provider=None, model=None, console=None):
    from . import agent_factory

    console = console or _make_console()
    try:
        llm_config = _resolve_llm_config(provider, model)
        session = agent_factory.build_session_agent(
            code_mode=code_mode, llm_config=llm_config
        )
        session.console = console
        return session
    except agent_factory.NeedsOllamaPull as exc:
        console.print(
            f"[yellow]Ollama is running at {exc.host} but has no models.[/yellow]"
        )
        console.print(
            f"Run:  [cyan]ollama pull {cfg.DEFAULT_OLLAMA_LLM}[/cyan]  then try again."
        )
        return _wizard(console, code_mode)
    except agent_factory.NoProviderConfigured:
        return _wizard(console, code_mode)
    except Exception as exc:
        console.print(f"[red]Could not start agent:[/red] {exc}")
        return None


def _wizard(console, code_mode, depth: int = 0):
    from rich.panel import Panel

    from . import agent_factory, ollama_probe

    if depth > 3:
        return None

    console.print(
        Panel(
            "No LLM provider is configured yet. Choose one:\n\n"
            "  [1] Paste an OpenAI API key\n"
            "  [2] Paste an Anthropic API key\n"
            "  [3] Use local Ollama (free, no key)\n"
            "  [q] Quit",
            title="Welcome to memorizz",
            border_style="green",
            expand=False,
        )
    )
    try:
        choice = input("Choose [1/2/3/q]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    import getpass

    if choice == "1":
        key = getpass.getpass("OPENAI_API_KEY (hidden): ").strip()
        if key:
            cfg.apply_env_updates({"OPENAI_API_KEY": key})
    elif choice == "2":
        key = getpass.getpass("ANTHROPIC_API_KEY (hidden): ").strip()
        if key:
            cfg.apply_env_updates({"ANTHROPIC_API_KEY": key})
    elif choice == "3":
        host = ollama_probe.resolve_host()
        if not ollama_probe.reachable(host):
            console.print(
                f"[yellow]No Ollama daemon at {host}.[/yellow] Install from "
                f"https://ollama.com, run `ollama serve`, then "
                f"`ollama pull {cfg.DEFAULT_OLLAMA_LLM}`."
            )
            return None
        cfg.apply_env_updates({"MEMORIZZ_DEFAULT_LLM_PROVIDER": "ollama"})
    else:
        return None

    try:
        session = agent_factory.build_session_agent(code_mode=code_mode)
        session.console = console
        return session
    except agent_factory.NeedsOllamaPull:
        console.print(
            f"[yellow]Pull a model first:[/yellow] ollama pull {cfg.DEFAULT_OLLAMA_LLM}"
        )
        return None
    except agent_factory.NoProviderConfigured:
        return _wizard(console, code_mode, depth + 1)
    except Exception as exc:
        console.print(f"[red]Could not start agent:[/red] {exc}")
        return None


def _launch_repl(code_mode=False, provider=None, model=None):
    _load_env()
    console = _make_console()
    session = _build_or_wizard(code_mode, provider, model, console)
    if session is None:
        raise typer.Exit(1)
    from .repl import run_repl

    run_repl(session)


def _run_oneshot(text, code_mode=False):
    _load_env()
    console = _make_console()
    session = _build_or_wizard(code_mode, console=console)
    if session is None:
        raise typer.Exit(1)
    result = session.agent.run(
        text,
        memory_id=session.memory_id,
        thread_id=session.thread_id,
        user_id=session.user_id,
    )
    print(result)
    session.sync_ids()
    try:
        cfg.save_state({"memory_id": session.memory_id})
        if getattr(session.agent, "memory_provider", None) is not None:
            session.agent.save()
    except Exception:
        pass


def _init(local, force):
    _load_env()
    console = _make_console()
    home = cfg.ensure_home()
    console.print(f"[green]memorizz home:[/green] {home}")
    if local:
        cfg.apply_env_updates({"MEMORIZZ_DEFAULT_LLM_PROVIDER": "ollama"})
        console.print("Configured the fully-local Ollama stack. Next steps:")
        console.print("  • install Ollama: https://ollama.com")
        console.print("  • ollama serve")
        console.print(f"  • ollama pull {cfg.DEFAULT_OLLAMA_LLM}")
        console.print(f"  • ollama pull {cfg.DEFAULT_OLLAMA_EMBED_MODEL}")
    else:
        _wizard(console, code_mode=False)
    console.print(f"[green]Done.[/green] Config file: {cfg.resolve_env_file()}")


def _show_config():
    _load_env()
    console = _make_console()
    console.print("[bold]memorizz config[/bold]")
    console.print(f"  home:        {cfg.memorizz_home()}")
    console.print(f"  env file:    {cfg.resolve_env_file()}")
    console.print(f"  memory root: {cfg.memory_root()}")
    try:
        from . import agent_factory

        llm = agent_factory.detect_llm_config()
        model = llm.get("model") or llm.get("deployment_name")
        console.print(f"  detected llm: [green]{llm.get('provider')}[/green] / {model}")
    except Exception as exc:
        console.print(f"  detected llm: [yellow]none[/yellow] ({type(exc).__name__})")


# --------------------------------------------------------------------------- #
# Back-compat argv shim + entry point
# --------------------------------------------------------------------------- #

_LEGACY_ORACLE = {
    "install-oracle": ["oracle", "install"],
    "setup-oracle": ["oracle", "setup"],
    "setup-oracle-schema": ["oracle", "setup-schema"],
    "teardown-oracle": ["oracle", "teardown"],
}


def _rewrite_argv(argv: List[str]) -> List[str]:
    """Map the old space-separated commands onto the new Typer tree."""
    if not argv:
        return argv
    first = argv[0]
    if first == "run" and len(argv) >= 2 and argv[1] in ("local", "automations"):
        if argv[1] == "local":
            _deprecate("`memorizz run local` is now `memorizz ui`")
            return ["ui", *argv[2:]]
        _deprecate("`memorizz run automations` is now `memorizz automations run`")
        return ["automations", "run", *argv[2:]]
    if first in _LEGACY_ORACLE:
        mapped = _LEGACY_ORACLE[first]
        _deprecate(f"`memorizz {first}` is now `memorizz {' '.join(mapped)}`")
        return [*mapped, *argv[1:]]
    return argv


def main():
    """Console-script entry point (``memorizz`` / ``python -m memorizz``)."""
    os.environ.setdefault("MEMORIZZ_LOG_LEVEL", "WARNING")
    argv = _rewrite_argv(sys.argv[1:])
    app(args=argv, prog_name="memorizz")


if __name__ == "__main__":
    main()
