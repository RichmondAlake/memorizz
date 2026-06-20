# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Slash-command registry for the interactive REPL.

A single ``COMMANDS`` table feeds both the REPL dispatcher and the
prompt_toolkit completer. Each handler has the signature
``handler(session, args: str) -> Optional[bool]`` and returns ``False`` to end
the REPL (any other value continues).
"""

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import agent_factory
from . import config as cfg
from . import ollama_probe


@dataclass
class Command:
    handler: Callable
    help: str
    usage: str = ""


def _con(session):
    """Return the session console, creating a fallback if needed."""
    if getattr(session, "console", None) is not None:
        return session.console
    from rich.console import Console

    session.console = Console()
    return session.console


# --------------------------------------------------------------------------- #
# Model / provider switching
# --------------------------------------------------------------------------- #


def _carry_api_key(existing, new_config: Dict[str, object]) -> None:
    """Copy the API key off the current provider into ``new_config`` in place.

    Saved/derived configs deliberately omit secrets; when changing only the
    model within the same provider we reuse the live key (mirrors ui/app.py).
    """
    if existing is None or "api_key" in new_config:
        return
    for attr in ("api_key", "_api_key"):
        key = getattr(existing, attr, None)
        if key:
            new_config["api_key"] = key
            return
    client = getattr(existing, "client", None)
    if client is not None:
        key = getattr(client, "api_key", None)
        if key:
            new_config["api_key"] = key


def _swap_model(session, new_config: Dict[str, object], carry_key: bool) -> None:
    from ..llms.llm_factory import create_llm_provider

    agent = session.agent
    if carry_key and new_config.get("provider") != "ollama":
        _carry_api_key(getattr(agent, "model", None), new_config)
    agent.model = create_llm_provider(new_config)
    agent.llm_config = dict(new_config)
    agent._llm_init_error = None
    session.llm_config = dict(new_config)


def cmd_model(session, args: str):
    console = _con(session)
    name = args.strip()
    if not name:
        console.print(
            f"Model: [cyan]{session.model_name}[/cyan]  (provider: {session.provider_name})"
        )
        console.print("Usage: /model <model-name>")
        return
    new_config = dict(session.llm_config)
    new_config["model"] = name
    _swap_model(session, new_config, carry_key=True)
    console.print(f"[green]Model →[/green] {name}")


def cmd_provider(session, args: str):
    console = _con(session)
    name = args.strip().lower()
    if not name:
        console.print(f"Provider: [cyan]{session.provider_name}[/cyan]")
        console.print(
            "Usage: /provider <openai|anthropic|ollama|azure|huggingface|mlx>"
        )
        return
    new_config = agent_factory.config_for_provider(name)
    _swap_model(session, new_config, carry_key=False)
    console.print(
        f"[green]Provider →[/green] {new_config.get('provider')} "
        f"({new_config.get('model') or new_config.get('deployment_name')})"
    )


def cmd_ollama(session, args: str):
    console = _con(session)
    parts = args.split(maxsplit=1)
    sub = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub == "host":
        if not rest:
            console.print(f"OLLAMA_HOST: [cyan]{ollama_probe.resolve_host()}[/cyan]")
            return
        os.environ["OLLAMA_HOST"] = rest
        cfg.apply_env_updates({"OLLAMA_HOST": rest})
        console.print(f"[green]OLLAMA_HOST →[/green] {rest}")
        return

    if sub == "pull":
        if not rest:
            console.print("Usage: /ollama pull <model-tag>")
            return
        console.print(f"Pulling [cyan]{rest}[/cyan] … (this can take a while)")
        ok, msg = ollama_probe.pull(rest)
        console.print(
            ("[green]" + msg + "[/green]") if ok else ("[red]" + msg + "[/red]")
        )
        return

    # default: list
    result = ollama_probe.probe()
    if not result["reachable"]:
        console.print(
            f"[red]Ollama not reachable[/red] at {result['host']}: {result['error']}"
        )
        return
    models = result["models"] or []
    if not models:
        console.print(
            f"Ollama is up at {result['host']} but has no models. Try /ollama pull llama3.1"
        )
        return
    console.print(f"[bold]Ollama models[/bold] ({result['host']}):")
    for m in models:
        console.print(f"  • {m}")


# --------------------------------------------------------------------------- #
# Internet access
# --------------------------------------------------------------------------- #


def cmd_web(session, args: str):
    console = _con(session)
    agent = session.agent
    arg = args.strip().lower()

    if not arg:
        try:
            name = agent.get_internet_access_provider_name()
        except Exception:
            name = None
        console.print(f"Internet access: [cyan]{name or 'off'}[/cyan]")
        console.print("Usage: /web on|off|tavily|firecrawl")
        return

    if arg in ("off", "false", "no", "0"):
        try:
            agent.with_internet_access_provider(None)
            console.print("[yellow]Internet access OFF[/yellow]")
        except Exception as exc:
            console.print(f"[red]Failed:[/red] {exc}")
        return

    try:
        if arg in ("on", "true", "yes"):
            from ..internet_access import get_default_internet_access_provider

            provider = get_default_internet_access_provider()
        else:
            from ..internet_access import create_internet_access_provider

            key = os.environ.get(f"{arg.upper()}_API_KEY")
            provider = create_internet_access_provider(
                arg, {"api_key": key} if key else {}
            )
        if provider is None or getattr(provider, "provider_name", None) == "offline":
            console.print(
                "[yellow]No internet key configured.[/yellow] Add one with "
                "/login tavily (or /login firecrawl)."
            )
            return
        agent.with_internet_access_provider(provider)
        name = agent.get_internet_access_provider_name() or arg
        console.print(f"[green]Internet access →[/green] {name}")
    except Exception as exc:
        console.print(f"[red]Could not enable internet access:[/red] {exc}")
        console.print("Add a key first with /login tavily (or /login firecrawl).")


# --------------------------------------------------------------------------- #
# Coding mode
# --------------------------------------------------------------------------- #


def cmd_code(session, args: str):
    console = _con(session)
    arg = args.strip().lower()
    turn_on = arg not in ("off", "false", "no", "0")
    session.agent.with_self_aware(turn_on, {"allow_writes": True} if turn_on else None)
    session.code_mode = turn_on
    if turn_on:
        try:
            scoped = session.agent.get_self_aware_config().get("root_paths") or [
                os.getcwd()
            ]
        except Exception:
            scoped = [os.getcwd()]
        console.print(
            "[green]Coding mode ON[/green] — file read/write + bounded commands enabled."
        )
        console.print(f"  scoped to: {', '.join(scoped)}  (deletes stay off)")
    else:
        console.print("[yellow]Coding mode OFF[/yellow]")


# --------------------------------------------------------------------------- #
# Memory / threads / agents
# --------------------------------------------------------------------------- #


def cmd_memory(session, args: str):
    console = _con(session)
    mid = args.strip()
    if not mid:
        console.print(
            f"memory_id: [cyan]{session.memory_id or '(new on first turn)'}[/cyan]"
        )
        console.print(
            f"thread_id: [cyan]{session.thread_id or '(new on first turn)'}[/cyan]"
        )
        return
    session.memory_id = mid
    try:
        history = session.agent.load_conversation_history(mid) or []
        console.print(
            f"[green]Switched to memory[/green] {mid}  ({len(history)} entries)"
        )
    except Exception as exc:
        console.print(
            f"[green]Switched to memory[/green] {mid}  (history unavailable: {exc})"
        )


def _render_history_entry(entry) -> Optional[str]:
    if isinstance(entry, dict):
        role = entry.get("role") or ("user" if entry.get("query") else "assistant")
        content = (
            entry.get("content") or entry.get("response") or entry.get("query") or ""
        )
        content = str(content).strip().replace("\n", " ")
        if not content:
            return None
        return f"[dim]{role}:[/dim] {content[:200]}"
    text = str(entry).strip()
    return text[:200] if text else None


def cmd_history(session, args: str):
    console = _con(session)
    try:
        history = session.agent.load_conversation_history(session.memory_id) or []
    except Exception as exc:
        console.print(f"[red]Could not load history:[/red] {exc}")
        return
    if not history:
        console.print("[dim]No conversation history yet.[/dim]")
        return
    console.print(f"[bold]Conversation history[/bold] ({len(history)} entries):")
    for entry in history[-20:]:
        line = _render_history_entry(entry)
        if line:
            console.print("  " + line)


def cmd_forget(session, args: str):
    console = _con(session)
    target = args.strip()
    if not target:
        console.print("Usage: /forget <id>   (a stored memory or knowledge-base id)")
        return
    provider = getattr(session, "provider", None)
    if provider is None:
        console.print("[yellow]No memory provider.[/yellow]")
        return

    from ..enums.memory_type import MemoryType

    skip = {MemoryType.MEMAGENT, MemoryType.PERSONAS, MemoryType.TOOLBOX}
    deleted_from = None
    for memory_type in MemoryType:
        if memory_type in skip:
            continue
        try:
            if provider.delete_by_id(target, memory_type):
                deleted_from = memory_type.value
                break
        except Exception:
            pass

    if deleted_from:
        console.print(f"[green]Forgot[/green] {target}  [dim]({deleted_from})[/dim]")
    else:
        console.print(f"[yellow]No memory found with id[/yellow] {target}")


def cmd_agents(session, args: str):
    console = _con(session)
    try:
        agents = session.provider.list_memagents() or []
    except Exception as exc:
        console.print(f"[red]Could not list agents:[/red] {exc}")
        return
    if not agents:
        console.print("[dim]No saved agents.[/dim]")
        return
    console.print(f"[bold]Saved agents[/bold] ({len(agents)}):")
    for a in agents:
        agent_id = getattr(a, "agent_id", None) or getattr(a, "id", "?")
        name = getattr(a, "name", None) or ""
        mode = getattr(a, "application_mode", None) or ""
        fav = "★ " if getattr(a, "is_favorite", False) else ""
        console.print(f"  {fav}{agent_id}  [cyan]{name}[/cyan] [dim]{mode}[/dim]")


def cmd_agent(session, args: str):
    console = _con(session)
    agent_id = args.strip()
    if not agent_id:
        console.print("Usage: /agent <agent-id>   (see /agents)")
        return
    from ..memagent import MemAgent

    loaded = MemAgent.load(agent_id, memory_provider=session.provider)
    session.agent = loaded
    session.memory_id = None
    session.thread_id = None
    try:
        session.llm_config = dict(getattr(loaded, "llm_config", {}) or {})
    except Exception:
        pass
    console.print(f"[green]Loaded agent[/green] {agent_id}")


def _persona_usage(console):
    console.print("[dim]Usage:[/dim]   /persona <name> [| goals | background]")
    console.print(
        "[dim]Example:[/dim] /persona Ada | tutor me in mathematics | "
        "a patient senior engineer"
    )
    console.print(
        "[dim](separate name, goals, and background with the | character)[/dim]"
    )


def cmd_persona(session, args: str):
    console = _con(session)
    pm = getattr(session.agent, "persona_manager", None)
    text = args.strip()

    if not text:
        cur = getattr(pm, "current_persona", None) if pm else None
        if not cur:
            console.print("[dim]No persona set.[/dim]")
            _persona_usage(console)
            return
        console.print("[bold]Persona[/bold]")
        console.print(f"  name: [cyan]{getattr(cur, 'name', '?')}[/cyan]")
        console.print(f"  role: {getattr(cur, 'role', '?')}")
        goals = str(getattr(cur, "goals", "") or "").strip()
        background = str(getattr(cur, "background", "") or "").strip()
        if goals:
            console.print(f"  goals: {goals[:300]}")
        if background:
            console.print(f"  background: {background[:300]}")
        console.print()
        _persona_usage(console)
        return

    parts = [p.strip() for p in text.split("|")]
    name = parts[0]
    goals = parts[1] if len(parts) > 1 else ""
    background = parts[2] if len(parts) > 2 else ""

    from ..long_term.semantic.persona import Persona

    persona = Persona(name=name, goals=goals, background=background)
    if session.agent.set_persona(persona):
        console.print(f"[green]Persona set →[/green] {name}")
    else:
        console.print("[red]Failed to set persona.[/red]")


def cmd_persona_reset(session, args: str):
    console = _con(session)
    agent = session.agent
    pm = getattr(agent, "persona_manager", None)
    cur = getattr(pm, "current_persona", None) if pm else None
    storage_id = None
    if cur is not None:
        storage_id = getattr(cur, "_storage_id", None) or getattr(
            cur, "persona_id", None
        )

    try:
        agent.set_persona(None)
    except Exception as exc:
        console.print(f"[red]Failed to clear persona:[/red] {exc}")
        return

    # Best-effort: drop the saved persona doc and persist the cleared state.
    provider = getattr(session, "provider", None)
    if storage_id and provider is not None:
        try:
            from ..enums.memory_type import MemoryType

            provider.delete_by_id(storage_id, MemoryType.PERSONAS)
        except Exception:
            pass
    try:
        if provider is not None:
            agent.save()
    except Exception:
        pass

    console.print(
        "[green]Persona reset.[/green] The agent reverts to its default persona."
    )


def cmd_new(session, args: str):
    console = _con(session)
    session.agent.reset_thread_state()
    session.memory_id = None
    session.thread_id = None
    console.print("[green]Started a fresh conversation thread.[/green]")


def cmd_tools(session, args: str):
    console = _con(session)
    manager = getattr(session.agent, "tool_manager", None)
    if manager is None or not hasattr(manager, "list_tools"):
        console.print("[dim]No tool manager available.[/dim]")
        return
    try:
        tools = manager.list_tools() or []
    except Exception as exc:
        console.print(f"[red]Could not list tools:[/red] {exc}")
        return
    if not tools:
        console.print("[dim]No tools registered.[/dim]")
        return
    console.print(f"[bold]Tools[/bold] ({len(tools)}):")
    for t in tools:
        if isinstance(t, dict):
            name = t.get("name") or t.get("function", {}).get("name") or "?"
            desc = (
                t.get("description") or t.get("function", {}).get("description") or ""
            )
        else:
            name = getattr(t, "name", str(t))
            desc = getattr(t, "description", "")
        desc = str(desc).strip().replace("\n", " ")
        console.print(f"  • [cyan]{name}[/cyan] [dim]{desc[:80]}[/dim]")


def cmd_ingest(session, args: str):
    console = _con(session)
    target = args.strip().strip('"').strip("'")
    if not target:
        console.print("Usage: /ingest <path-to-file>")
        return
    path = Path(target).expanduser()
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        return
    from ..long_term.semantic import KnowledgeBase

    kb = KnowledgeBase(memory_provider=session.provider)
    kb_id = kb.ingest_file(str(path))
    console.print(f"[green]Ingested[/green] {path.name} → knowledge_base_id {kb_id}")


# --------------------------------------------------------------------------- #
# UI / config / session
# --------------------------------------------------------------------------- #


def cmd_ui(session, args: str):
    console = _con(session)
    host, port = "127.0.0.1", 8765
    tokens = args.split()
    i = 0
    while i < len(tokens):
        if tokens[i] == "--port" and i + 1 < len(tokens):
            try:
                port = int(tokens[i + 1])
            except ValueError:
                console.print(f"[red]Invalid port:[/red] {tokens[i + 1]}")
                return
            i += 2
        elif tokens[i] == "--host" and i + 1 < len(tokens):
            host = tokens[i + 1]
            i += 2
        else:
            i += 1

    existing = getattr(session, "ui_proc", None)
    if existing is not None and existing.poll() is None:
        console.print("[yellow]UI already running.[/yellow]")
        return

    import subprocess

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "memorizz",
                "ui",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        console.print(f"[red]Failed to launch UI:[/red] {exc}")
        return
    session.ui_proc = proc
    console.print(
        f"[green]Local UI →[/green] http://{host}:{port}  (stops when you /exit)"
    )


_LOGIN_PROVIDERS = [
    ("openai", "OPENAI_API_KEY", "OpenAI — LLM + embeddings"),
    ("anthropic", "ANTHROPIC_API_KEY", "Anthropic — LLM"),
    ("azure", "AZURE_OPENAI_API_KEY", "Azure OpenAI"),
    ("tavily", "TAVILY_API_KEY", "Tavily — internet search"),
    ("firecrawl", "FIRECRAWL_API_KEY", "Firecrawl — internet search"),
    ("voyage", "VOYAGE_API_KEY", "Voyage AI — embeddings"),
]


def cmd_login(session, args: str):
    console = _con(session)
    import getpass

    by_name = {p[0]: p for p in _LOGIN_PROVIDERS}
    choice = args.strip().lower()

    # No provider given -> list the platforms and let the user pick.
    if not choice:
        console.print("[bold]Log in to a platform[/bold]")
        for i, (name, env_var, desc) in enumerate(_LOGIN_PROVIDERS, 1):
            status = " [green]✓ set[/green]" if os.environ.get(env_var) else ""
            console.print(f"  [cyan]{i}[/cyan]. {name:<10} [dim]{desc}[/dim]{status}")
        try:
            choice = input("Select a platform [number or name]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Cancelled.[/yellow]")
            return
        if not choice:
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # Resolve selection: number, known name, or an arbitrary KEY name.
    if choice.isdigit():
        idx = int(choice) - 1
        if not (0 <= idx < len(_LOGIN_PROVIDERS)):
            console.print(f"[yellow]Invalid selection:[/yellow] {choice}")
            return
        key_name = _LOGIN_PROVIDERS[idx][1]
    elif choice in by_name:
        key_name = by_name[choice][1]
    else:
        key_name = choice.upper()
    try:
        value = getpass.getpass(f"Paste {key_name} (hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Cancelled.[/yellow]")
        return
    if not value:
        console.print("[yellow]No value entered; nothing saved.[/yellow]")
        return
    err = cfg.apply_env_updates({key_name: value})
    if err:
        console.print(
            f"[yellow]Set for this session, but failed to persist:[/yellow] {err}"
        )
    else:
        console.print(f"[green]Saved[/green] {key_name} → {cfg.resolve_env_file()}")

    # Internet keys take effect immediately: attach the provider to the live
    # agent so the user doesn't also have to run /web.
    if key_name in ("TAVILY_API_KEY", "FIRECRAWL_API_KEY"):
        provider_name = "tavily" if key_name == "TAVILY_API_KEY" else "firecrawl"
        try:
            from ..internet_access import create_internet_access_provider

            provider = create_internet_access_provider(
                provider_name, {"api_key": value}
            )
            if provider is not None:
                session.agent.with_internet_access_provider(provider)
                console.print(
                    f"[green]Internet access enabled →[/green] {provider_name}"
                )
        except Exception as exc:
            console.print(f"[yellow]Saved, but enabling web failed:[/yellow] {exc}")


def cmd_config(session, args: str):
    console = _con(session)
    provider_type = type(session.provider).__name__ if session.provider else "(none)"
    console.print("[bold]memorizz config[/bold]")
    console.print(f"  home:          {cfg.memorizz_home()}")
    console.print(f"  env file:      {cfg.resolve_env_file()}")
    console.print(f"  memory root:   {cfg.memory_root()}")
    console.print(f"  llm provider:  {session.provider_name}")
    console.print(f"  llm model:     {session.model_name}")
    console.print(f"  memory store:  {provider_type}")
    console.print(f"  coding mode:   {'on' if session.code_mode else 'off'}")
    try:
        net = session.agent.get_internet_access_provider_name()
    except Exception:
        net = None
    console.print(f"  internet:      {net or 'off'}")


_DOCS_BASE = "https://richmondalake.github.io/memorizz"
_DOCS_PAGES = {
    "": "/",
    "cli": "/getting-started/cli/",
    "ui": "/getting-started/local-ui/",
    "quickstart": "/getting-started/python-sdk-quickstart/",
    "concepts": "/getting-started/concepts/",
}


def cmd_docs(session, args: str):
    console = _con(session)
    key = args.strip().lower()
    url = _DOCS_BASE + _DOCS_PAGES.get(key, "/")

    import webbrowser

    try:
        opened = webbrowser.open(url)
    except Exception:
        opened = False

    if opened:
        console.print(f"[green]Opening docs →[/green] {url}")
    else:
        console.print(f"[bold]Docs:[/bold] {url}")
    if not key:
        console.print(
            "[dim]Jump to a page: /docs cli · /docs ui · /docs quickstart[/dim]"
        )


def cmd_cls(session, args: str):
    _con(session).clear()


def cmd_clear(session, args: str):
    """Erase the agent's stored memory after a typed confirmation."""
    console = _con(session)
    provider = getattr(session, "provider", None)
    if provider is None:
        console.print("[yellow]No memory provider; nothing to clear.[/yellow]")
        return

    console.print(
        "[red bold]This permanently erases this agent's stored memory[/red bold] "
        "— conversations, knowledge base, entities, summaries, workflows, caches "
        "and tool logs. The agent itself (persona + tools) is kept."
    )
    try:
        answer = (
            input("Type 'wipe' to confirm (anything else cancels): ").strip().lower()
        )
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Cancelled.[/yellow]")
        return
    if answer != "wipe":
        console.print("[yellow]Cancelled.[/yellow]")
        return

    from ..enums.memory_type import MemoryType

    keep = {MemoryType.PERSONAS, MemoryType.TOOLBOX, MemoryType.MEMAGENT}
    cleared = 0
    for memory_type in MemoryType:
        if memory_type in keep:
            continue
        try:
            if provider.delete_all(memory_type):
                cleared += 1
        except Exception:
            pass

    try:
        session.agent.reset_thread_state()
    except Exception:
        pass
    session.memory_id = None
    session.thread_id = None
    try:
        cfg.clear_state(["memory_id"])
    except Exception:
        pass
    console.print(
        f"[green]Memory wiped.[/green] Cleared {cleared} memory store(s); "
        "the agent starts fresh."
    )


def cmd_exit(session, args: str):
    console = _con(session)
    agent = getattr(session, "agent", None)
    if agent is not None and getattr(agent, "memory_provider", None) is not None:
        try:
            agent.save()
        except Exception:
            pass
    proc = getattr(session, "ui_proc", None)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    console.print("[dim]Goodbye.[/dim]")
    return False


def cmd_help(session, args: str):
    console = _con(session)
    console.print("[bold]Slash commands[/bold]")
    for name in sorted(COMMANDS):
        cmd = COMMANDS[name]
        usage = cmd.usage or f"/{name}"
        console.print(f"  [cyan]{usage:<26}[/cyan] {cmd.help}")
    console.print(
        "\nType plain text to chat. Ctrl-C / Ctrl-D / /exit quits; "
        "Ctrl-C during a reply aborts just that reply."
    )


# --------------------------------------------------------------------------- #
# Registry + dispatch
# --------------------------------------------------------------------------- #

COMMANDS: Dict[str, Command] = {
    "help": Command(cmd_help, "Show this help.", "/help"),
    "model": Command(cmd_model, "Show or switch the chat model.", "/model [name]"),
    "provider": Command(cmd_provider, "Switch the LLM provider.", "/provider [name]"),
    "ollama": Command(
        cmd_ollama,
        "List/pull Ollama models or set host.",
        "/ollama [list|pull <t>|host <u>]",
    ),
    "web": Command(
        cmd_web,
        "Enable/disable internet access (Tavily/Firecrawl).",
        "/web [on|off|tavily|firecrawl]",
    ),
    "code": Command(
        cmd_code, "Toggle coding tools (file edits + commands).", "/code [on|off]"
    ),
    "memory": Command(
        cmd_memory, "Show or switch the active memory id.", "/memory [id]"
    ),
    "history": Command(
        cmd_history, "Print the current conversation history.", "/history"
    ),
    "forget": Command(
        cmd_forget, "Delete a single stored memory by id.", "/forget <id>"
    ),
    "agents": Command(cmd_agents, "List saved agents.", "/agents"),
    "agent": Command(cmd_agent, "Load a saved agent by id.", "/agent <id>"),
    "new": Command(cmd_new, "Start a fresh conversation thread.", "/new"),
    "tools": Command(cmd_tools, "List the agent's tools.", "/tools"),
    "ingest": Command(
        cmd_ingest, "Ingest a file into the knowledge base.", "/ingest <file>"
    ),
    "ui": Command(cmd_ui, "Launch the local web UI.", "/ui [--port N]"),
    "login": Command(
        cmd_login,
        "Log in / save an API key (lists platforms if none given).",
        "/login [provider]",
    ),
    "config": Command(cmd_config, "Show resolved config + paths.", "/config"),
    "docs": Command(
        cmd_docs, "Open the documentation in your browser.", "/docs [cli|ui]"
    ),
    "persona": Command(
        cmd_persona,
        "Show or set the agent's persona.",
        "/persona [name | goals | background]",
    ),
    "persona-reset": Command(
        cmd_persona_reset,
        "Clear the agent's persona (revert to default).",
        "/persona-reset",
    ),
    "clear": Command(
        cmd_clear,
        "Erase the agent's stored memory (asks to confirm).",
        "/clear",
    ),
    "cls": Command(cmd_cls, "Clear the terminal screen.", "/cls"),
    "exit": Command(cmd_exit, "Save and quit.", "/exit"),
}

ALIASES: Dict[str, str] = {
    "quit": "exit",
    "q": "exit",
    "h": "help",
    "?": "help",
    "models": "model",
}


def command_completions() -> List[str]:
    """Return ``/name`` strings for the prompt_toolkit completer."""
    return [f"/{name}" for name in sorted(COMMANDS)]


def dispatch(line: str, session) -> bool:
    """Dispatch a ``/command`` line. Returns False to end the REPL."""
    console = _con(session)
    body = line[1:] if line.startswith("/") else line
    parts = body.split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    name = ALIASES.get(name, name)

    cmd = COMMANDS.get(name)
    if cmd is None:
        # Resolve unambiguous abbreviations: /lo -> /login, /mo -> /model.
        matches = sorted(n for n in COMMANDS if n.startswith(name))
        if len(matches) == 1:
            name = matches[0]
            cmd = COMMANDS[name]
        elif len(matches) > 1:
            console.print(
                f"[yellow]Ambiguous command:[/yellow] /{name} → "
                + ", ".join("/" + m for m in matches)
            )
            return True
        else:
            console.print(f"[yellow]Unknown command:[/yellow] /{name}  (try /help)")
            return True
    try:
        result = cmd.handler(session, args)
    except Exception as exc:
        console.print(f"[red]/{name} failed:[/red] {exc}")
        return True
    return result is not False
