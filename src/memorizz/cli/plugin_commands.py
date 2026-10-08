"""MemoRizz as a Codex or Claude Code plugin: install it, and answer its hooks.

The plugins (``plugins/codex/memorizz`` and ``plugins/claude-code/memorizz``
in the repository, packaged with MemoRizz) give the coding agent MemoRizz's
MCP tools, skills for using them, and a SessionStart hook that tells it the
project's memory ID and recent memories. Both agents use the same memory ID
for a project, so what one learns the other can recall.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

plugin_app = typer.Typer(
    help=(
        "Use MemoRizz memory from Codex and Claude Code: install the plugin "
        "and run its hooks."
    ),
    no_args_is_help=True,
)
hook_app = typer.Typer(
    help="Hooks the plugin runs (they read the agent's hook JSON on stdin).",
    no_args_is_help=True,
)
plugin_app.add_typer(hook_app, name="hook")

PLUGIN_NAME = "memorizz"
MARKETPLACE_NAME = "memorizz"
AGENTS = {"codex": "Codex", "claude-code": "Claude Code"}
# Hook output is extra context for the model; keep it small.
CONTEXT_CHARS = 2_000
RECENT_MEMORIES = 8
PROMPT_RECALL_ENV = "MEMORIZZ_PROMPT_RECALL"
CAPTURE_ENV = "MEMORIZZ_SESSION_CAPTURE"  # turns (default) or off
SUMMARY_ENV = "MEMORIZZ_SESSION_SUMMARY"  # true (default) or false
# Record each session as a run on the UI's Harnesses page: true (default) or false.
RUNS_ENV = "MEMORIZZ_PLUGIN_RUNS"
PRINCIPAL_ENV = "MEMORIZZ_MCP_SERVER_LOCAL_PRINCIPAL"
# A hosted MemoRizz MCP server the plugin uses instead of the local store.
REMOTE_URL_ENV = "MEMORIZZ_PLUGIN_REMOTE_URL"
REMOTE_TOKEN_ENV_ENV = "MEMORIZZ_PLUGIN_REMOTE_TOKEN_ENV"
DEFAULT_TOKEN_ENV = "MEMORIZZ_MCP_TOKEN"


def _setting(name: str, default: str = "") -> str:
    from .config import load_layered_env

    load_layered_env()
    return str(os.environ.get(name, default) or default).strip()


def _user_id() -> Optional[str]:
    """The user memories are saved for: the MCP server's local principal, so
    hooks and tools agree (unset for a single-person store)."""
    return _setting(PRINCIPAL_ENV) or None


def _agent_name() -> str:
    """Which agent runs the hook. Codex sets PLUGIN_ROOT; Claude Code only
    CLAUDE_PLUGIN_ROOT."""
    return "codex" if os.environ.get("PLUGIN_ROOT") else "claude-code"


def _sessions_dir() -> Path:
    from .._env_io import memorizz_home

    folder = memorizz_home() / "plugin-sessions"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    return folder


def _safe_session(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._:-]", "-", str(value or "session"))[:120]


def project_memory_id(path: Any) -> str:
    """The memory ID for a project: ``project-<folder>-<hash>``, from its Git
    root (or the folder itself), so each checkout keeps its own memories and
    every agent working in it shares them."""
    folder = Path(str(path or os.getcwd())).expanduser()
    try:
        folder = folder.resolve()
    except OSError:
        pass
    root = folder
    try:
        found = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if found.returncode == 0 and found.stdout.strip():
            root = Path(found.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    slug = re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-")[:40] or "project"
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:6]
    return f"project-{slug}-{digest}"


def _hook_payload() -> Dict[str, Any]:
    try:
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        payload = json.loads(raw) if raw.strip() else {}
    except (OSError, ValueError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _store():
    from .agent_commands import _provider

    provider, _warnings = _provider()
    return provider


def _remote_url() -> Optional[str]:
    return _setting(REMOTE_URL_ENV) or None


def remote_tool(name: str, arguments: Dict[str, Any], timeout: float = 15.0) -> Any:
    """Call a tool on the hosted MemoRizz MCP server, authenticated with the
    bearer token in MEMORIZZ_PLUGIN_REMOTE_TOKEN_ENV (default MEMORIZZ_MCP_TOKEN)."""
    import asyncio

    url = _remote_url()
    token_env = _setting(REMOTE_TOKEN_ENV_ENV, DEFAULT_TOKEN_ENV)
    token = os.environ.get(token_env) or _setting(token_env)
    if not url or not token:
        raise RuntimeError("The remote MemoRizz server or its token isn't configured")

    async def call() -> Any:
        import httpx2
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client

        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            trust_env=False,
        ) as http:
            async with Client(
                streamable_http_client(url, http_client=http),
                read_timeout_seconds=timeout,
                raise_exceptions=True,
                mode="auto",
            ) as client:
                result = await client.call_tool(name, arguments)
        structured = getattr(result, "structured_content", None) or getattr(
            result, "structuredContent", None
        )
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        for part in getattr(result, "content", None) or []:
            text = getattr(part, "text", None)
            if text:
                return json.loads(text)
        return structured

    return asyncio.run(asyncio.wait_for(call(), timeout + 5))


def _remote_rows(memory_type: str, memory_id: str, limit: int) -> List[Dict[str, Any]]:
    value = remote_tool(
        "memorizz_list_memories",
        {"memory_type": memory_type, "memory_id": memory_id, "limit": limit},
    )
    rows = value.get("memories") if isinstance(value, dict) else None
    return [row for row in rows or [] if isinstance(row, dict)]


def _close(provider: Any) -> None:
    close = getattr(provider, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _text(record: Dict[str, Any]) -> str:
    return " ".join(str(record.get("content") or record.get("text") or "").split())


def _recent_memories(provider: Any, memory_id: str, limit: int) -> List[Dict[str, Any]]:
    from ..enums import MemoryType

    user_id = _user_id()
    rows = [
        row
        for row in provider.list_all(MemoryType.KNOWLEDGE_BASE) or []
        if isinstance(row, dict)
        and row.get("memory_id") == memory_id
        and _text(row)
        and row.get("status") != "superseded"
        and (user_id is None or row.get("user_id") == user_id)
    ]
    rows.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
    return rows[:limit]


def _last_summary(provider: Any, memory_id: str) -> Optional[Dict[str, Any]]:
    from ..enums import MemoryType

    user_id = _user_id()
    rows = [
        row
        for row in provider.list_all(MemoryType.SUMMARIES) or []
        if isinstance(row, dict)
        and row.get("memory_id") == memory_id
        and _text(row)
        and (user_id is None or row.get("user_id") == user_id)
    ]
    rows.sort(
        key=lambda row: _day(row.get("created_at") or row.get("period_end"))
        + str(row.get("created_at") or ""),
        reverse=True,
    )
    return rows[0] if rows else None


def _day(value: Any) -> str:
    """A record's date as YYYY-MM-DD, from an ISO string or epoch seconds."""
    if (
        isinstance(value, (int, float))
        or str(value or "").replace(".", "", 1).isdigit()
    ):
        from datetime import datetime, timezone

        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        except (OverflowError, OSError, ValueError):
            return ""
    return str(value or "")[:10]


def _bullets(rows: List[Dict[str, Any]], budget: int) -> List[str]:
    lines: List[str] = []
    used = 0
    for row in rows:
        day = _day(row.get("timestamp") or row.get("created_at"))
        line = f"- {day + ': ' if day else ''}{_text(row)}"
        if len(line) > 400:
            line = line[:399] + "…"
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line)
    return lines


def session_context(cwd: Any, provider: Any = None) -> str:
    """What the SessionStart hook tells the agent: the project's memory ID,
    how to use it, and its most recent memories."""
    memory_id = project_memory_id(cwd)
    lines = [
        "MemoRizz memory is connected (the `memorizz` MCP tools).",
        f"This project's MemoRizz memory ID is `{memory_id}`. Before re-deriving "
        "project facts, search it with `memorizz_search_memories` "
        f'(memory_id "{memory_id}"); save durable facts, decisions and preferences '
        f'with `memorizz_store_memory` (memory_type "knowledge_base", memory_id '
        f'"{memory_id}"). Never store secrets.',
    ]
    if _setting(CAPTURE_ENV, "turns") != "off":
        lines.append(
            "Each turn of this session (your request and final answer, secrets "
            "removed) is saved to MemoRizz conversation memory; search past sessions "
            'with memory_type "conversation_memory" or "summaries".'
        )
    owned = provider is None
    try:
        if provider is None and _remote_url():

            def newest(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
                return sorted(
                    rows, key=lambda row: str(row.get(key) or ""), reverse=True
                )

            recent = newest(
                [
                    row
                    for row in _remote_rows("knowledge_base", memory_id, 50)
                    if _text(row)
                ],
                "timestamp",
            )[:RECENT_MEMORIES]
            summaries = newest(_remote_rows("summaries", memory_id, 20), "created_at")
            summary = summaries[0] if summaries else None
            owned = False
        else:
            provider = provider or _store()
            recent = _recent_memories(provider, memory_id, RECENT_MEMORIES)
            summary = _last_summary(provider, memory_id)
    except Exception:
        recent, summary = [], None
    finally:
        if owned and provider is not None:
            _close(provider)
    if summary:
        when = _day(summary.get("created_at") or summary.get("period_end"))
        text = _text(summary)
        lines.append(
            f"Last session summary ({when}, {summary.get('agent_id') or 'agent'}): "
            + (text[:700] + "…" if len(text) > 700 else text)
        )
    if recent:
        lines.append("Recent project memories (newest first):")
        lines.extend(_bullets(recent, CONTEXT_CHARS - sum(len(line) for line in lines)))
    return "\n".join(lines)


def _keyword_matches(
    rows: List[Dict[str, Any]], query: str, limit: int
) -> List[Dict[str, Any]]:
    """Rank rows by how many of the query's words they contain (used when
    semantic search is unavailable)."""
    words = {word for word in re.findall(r"[a-z0-9]{3,}", query.lower())}
    scored = []
    for row in rows:
        text = _text(row).lower()
        hits = sum(1 for word in words if word in text)
        if hits:
            scored.append((hits, str(row.get("timestamp") or ""), row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _hits, _when, row in scored[:limit]]


def prompt_context(cwd: Any, prompt: str, provider: Any = None) -> str:
    """What the UserPromptSubmit hook tells the agent: project memories
    related to the prompt (semantic search), or nothing."""
    text = " ".join(str(prompt or "").split())
    if len(text) < 12:
        return ""
    from ..enums import MemoryType

    memory_id = project_memory_id(cwd)
    user_id = _user_id()
    if provider is None and _remote_url():
        try:
            found = remote_tool(
                "memorizz_search_memories",
                {"query": text, "memory_id": memory_id, "limit": 4},
            )
        except Exception:
            return ""
        rows = [
            row
            for row in (found or {}).get("memories") or []
            if isinstance(row, dict) and _text(row)
        ]
        if not rows:
            return ""
        return "\n".join(
            [
                "MemoRizz project memories that may be relevant:",
                *_bullets(rows, CONTEXT_CHARS),
            ]
        )
    owned = provider is None
    try:
        provider = provider or _store()
        try:
            rows = provider.retrieve_by_query(
                text,
                memory_store_type=MemoryType.KNOWLEDGE_BASE,
                memory_id=memory_id,
                limit=4,
            )
        except Exception:
            # No embedding model: match on words instead.
            rows = _keyword_matches(_recent_memories(provider, memory_id, 500), text, 4)
    except Exception:
        return ""
    finally:
        if owned and provider is not None:
            _close(provider)
    rows = [
        row
        for row in rows or []
        if isinstance(row, dict)
        and _text(row)
        and row.get("status") != "superseded"
        and (user_id is None or row.get("user_id") in (None, user_id))
    ]
    if not rows:
        return ""
    return "\n".join(
        [
            "MemoRizz project memories that may be relevant:",
            *_bullets(rows, CONTEXT_CHARS),
        ]
    )


def _wants_prompt_recall() -> bool:
    from .._env_io import env_bool
    from .config import load_layered_env

    load_layered_env()
    return env_bool(PROMPT_RECALL_ENV, False)


@hook_app.command("session-start")
def hook_session_start() -> None:
    """SessionStart: print the project's memory ID and recent memories."""
    payload = _hook_payload()
    try:
        typer.echo(session_context(payload.get("cwd")))
    except Exception:
        pass  # A memory hook must never stop the agent.
    _prune_sessions()


@hook_app.command("prompt")
def hook_prompt() -> None:
    """UserPromptSubmit: print project memories related to the prompt.

    Off unless MEMORIZZ_PROMPT_RECALL is true (it searches on every prompt):
    `memorizz config set MEMORIZZ_PROMPT_RECALL true`.
    """
    payload = _hook_payload()
    try:
        if not _wants_prompt_recall():
            return
        context = prompt_context(payload.get("cwd"), str(payload.get("prompt") or ""))
        if context:
            typer.echo(context)
    except Exception:
        pass


def capture_turn(
    payload: Dict[str, Any], provider: Any = None, prompt_file: Optional[Path] = None
) -> Optional[str]:
    """Save one turn (the pending prompt and the final answer) to the project's
    conversation memory. Returns the thread ID, or None when nothing was saved.

    The Stop hook script moves the pending prompt to ``prompt_file`` first, so
    the next prompt can't be paired with this answer."""
    if _setting(CAPTURE_ENV, "turns") == "off":
        return None
    answer = str(payload.get("last_assistant_message") or "").strip()
    session = _safe_session(payload.get("session_id"))
    pending = (
        Path(prompt_file) if prompt_file else _sessions_dir() / f"{session}.prompt.json"
    )
    try:
        prompt = str(
            json.loads(pending.read_text(encoding="utf-8")).get("prompt") or ""
        ).strip()
    except (OSError, ValueError, AttributeError):
        prompt = ""
    if not prompt or not answer:
        return None
    from ..episodic_capture import record_turn

    agent = _agent_name()
    thread_id = f"{agent}-{session}"
    if provider is None and _remote_url():
        remote_tool(
            "memorizz_record_turn",
            {
                "memory_id": project_memory_id(payload.get("cwd")),
                "thread_id": thread_id,
                "user_message": prompt,
                "assistant_message": answer,
                "agent_id": agent,
            },
            timeout=60,
        )
        try:
            pending.unlink()
        except OSError:
            pass
        return thread_id
    owned = provider is None
    provider = provider or _store()
    try:
        record_turn(
            provider,
            memory_id=project_memory_id(payload.get("cwd")),
            thread_id=thread_id,
            user_message=prompt,
            assistant_message=answer,
            agent_id=agent,
            user_id=_user_id(),
        )
    finally:
        if owned:
            _close(provider)
    try:
        pending.unlink()
    except OSError:
        pass
    return thread_id


def summarize_session(payload: Dict[str, Any], provider: Any = None) -> List[str]:
    """Summarize a session's saved turns with MemoRizz's default model (if one
    is configured). Returns the new summary IDs."""
    if _setting(SUMMARY_ENV, "true").lower() in {"false", "0", "off", "no"}:
        return []
    from ..episodic_capture import default_model, summarize_thread

    if provider is None and _remote_url():
        agent = _agent_name()
        value = remote_tool(
            "memorizz_summarize_session",
            {
                "memory_id": project_memory_id(payload.get("cwd")),
                "thread_id": f"{agent}-{_safe_session(payload.get('session_id'))}",
                "agent_id": agent,
            },
            timeout=180,
        )
        return list((value or {}).get("summary_ids") or [])
    model = default_model()
    if model is None:
        return []
    agent = _agent_name()
    owned = provider is None
    provider = provider or _store()
    try:
        return summarize_thread(
            provider,
            memory_id=project_memory_id(payload.get("cwd")),
            thread_id=f"{agent}-{_safe_session(payload.get('session_id'))}",
            agent_id=agent,
            user_id=_user_id(),
            model=model,
        )
    finally:
        if owned:
            _close(provider)


def record_session_run(payload: Dict[str, Any], store: Any = None) -> Optional[str]:
    """Record the session as a run on the Harnesses page (its steps read from
    the agent's own session log), or bring it up to date. Returns the run ID.

    Local stores only: a hosted server can't read this machine's session log."""
    if _setting(RUNS_ENV, "true").lower() in {"false", "0", "off", "no"}:
        return None
    if store is None and _remote_url():
        return None
    from ..metaharness.agent_sessions import find_session_log, record_session

    agent = _agent_name()
    path = payload.get("transcript_path") or payload.get("agent_transcript_path")
    log = Path(str(path)).expanduser() if path else None
    if log is None or not log.is_file():
        log = find_session_log(agent, str(payload.get("session_id") or ""))
    if log is None:
        return None
    run = record_session(
        log,
        agent=agent,
        memory_id=project_memory_id(payload.get("cwd")),
        store=store,
    )
    return run.run_id if run else None


@hook_app.command("stop")
def hook_stop(
    prompt_file: Optional[Path] = typer.Option(
        None, "--prompt-file", help="The turn's prompt, set aside by the Stop hook."
    ),
) -> None:
    """Stop (run detached by the hook script): save the turn that just ended."""
    payload = _hook_payload()
    try:
        capture_turn(payload, prompt_file=prompt_file)
    except Exception:
        pass
    try:
        record_session_run(payload)
    except Exception:
        pass


# Turn saves still running when a session is summarized: at most this long.
SAVE_WAIT_SECONDS = 60


def _wait_for_saves(session: str, limit: float = SAVE_WAIT_SECONDS) -> None:
    """Wait for the session's detached turn saves (``<session>.saving.*``
    files, removed when each save ends) so the summary includes the last turn."""
    import time

    deadline = time.monotonic() + limit
    while time.monotonic() < deadline and any(
        _sessions_dir().glob(f"{session}.saving.*")
    ):
        time.sleep(0.5)


def _prune_sessions(days: int = 7) -> None:
    """Remove pending prompts and save files a session left behind."""
    import time

    cutoff = time.time() - days * 86_400
    for path in _sessions_dir().glob("*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


@hook_app.command("summarize")
def hook_summarize() -> None:
    """SessionEnd / PreCompact (run detached): summarize the session."""
    payload = _hook_payload()
    try:
        _wait_for_saves(_safe_session(payload.get("session_id")))
        summarize_session(payload)
    except Exception:
        pass
    try:
        record_session_run(payload)  # with the totals written at session end
    except Exception:
        pass


@plugin_app.command("import-session")
def import_session(
    logs: List[Path] = typer.Argument(
        ...,
        help=(
            "Session logs: a Codex rollout (~/.codex/sessions/…/rollout-*.jsonl) "
            "or a Claude Code transcript (~/.claude/projects/<project>/<session>.jsonl)."
        ),
        exists=True,
        dir_okay=False,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Show past Codex or Claude Code sessions on the UI's Harnesses page,
    with their steps. The plugin does this for new sessions as they run."""
    from ..metaharness.agent_sessions import detect_agent, read_session
    from ..metaharness.store import SQLiteHarnessRunStore

    recorded = []
    store = SQLiteHarnessRunStore()
    try:
        for log in logs:
            agent = detect_agent(log)
            built = read_session(log, agent=agent) if agent else None
            if built is None:
                recorded.append({"log": str(log), "run_id": None})
                continue
            run, events = built
            # The project's memory ID, as the plugin names it.
            run.task["memory_id"] = project_memory_id(run.task.get("workspace"))
            store.replace(run, events)
            recorded.append(
                {
                    "log": str(log),
                    "run_id": run.run_id,
                    "agent": agent,
                    "task": run.task.get("task"),
                }
            )
    finally:
        store.close()
    if as_json:
        typer.echo(json.dumps({"runs": recorded}, indent=2))
        return
    for item in recorded:
        if item["run_id"]:
            typer.echo(
                f"{AGENTS[item['agent']]} session → run {item['run_id'][:8]}: "
                f"{str(item['task'])[:70]}"
            )
        else:
            typer.echo(
                f"Skipped {item['log']}: not a Codex or Claude Code session log."
            )
    if any(item["run_id"] for item in recorded):
        typer.echo("Open `memorizz ui` → Harnesses to see their steps.")


@plugin_app.command("export-memory")
def export_project_memory(
    output: Path = typer.Argument(..., help="Destination .memorizz.json file."),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Export this plugin project's memories, including session history."""
    from ..memory_archive import MemoryArchive, write_archive

    provider = None
    try:
        namespace = project_memory_id(workspace or Path.cwd())
        if _remote_url():
            response = remote_tool(
                "memorizz_export_memories", {"memory_id": namespace}, timeout=60
            )
            if not response or not response.get("ok"):
                raise ValueError("The remote server did not export an archive")
            archive = response["archive"]
        else:
            provider = _store()
            archive = MemoryArchive(provider).export(
                memory_id=namespace, user_id=_user_id()
            )
        write_archive(archive, output, overwrite=overwrite)
        typer.echo(
            json.dumps({"path": str(output), "manifest": archive["manifest"]}, indent=2)
        )
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    finally:
        if provider:
            provider.close()


@plugin_app.command("import-memory")
def import_project_memory(
    archive: Path = typer.Argument(..., exists=True, dir_okay=False),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
    apply: bool = typer.Option(False, "--apply"),
    conflict: str = typer.Option("error", "--conflict"),
    new_ids: bool = typer.Option(False, "--new-ids"),
    preserve_namespace: bool = typer.Option(
        False,
        "--preserve-namespace",
        help="Keep the original project namespace instead of restoring to this project.",
    ),
):
    """Preview/restore a project archive into the current project's memory."""
    from ..memory_archive import MemoryArchive, read_archive

    provider = None
    try:
        value = read_archive(archive)
        options = {
            "dry_run": not apply,
            "conflict": conflict,
            "id_strategy": "new" if new_ids else "preserve",
            "target_memory_id": None
            if preserve_namespace
            else project_memory_id(workspace or Path.cwd()),
        }
        if _remote_url():
            report = remote_tool(
                "memorizz_import_memories", {"archive": value, **options}, timeout=60
            )
        else:
            provider = _store()
            report = MemoryArchive(provider).import_archive(
                value, user_id=_user_id(), **options
            )
        typer.echo(json.dumps(report, indent=2))
        if not report or not report.get("ok"):
            raise typer.Exit(1)
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    finally:
        if provider:
            provider.close()


@plugin_app.command("memory-id")
def memory_id(
    path: Optional[Path] = typer.Argument(
        None, help="Project folder (default: the current folder)."
    ),
) -> None:
    """Print the memory ID the plugins use for a project."""
    typer.echo(project_memory_id(path or Path.cwd()))


def plugin_source(agent: str) -> Path:
    """The packaged plugin folder for an agent (or the repository's, in a
    checkout)."""
    packaged = (
        Path(__file__).resolve().parents[1] / "agent_plugins" / agent / PLUGIN_NAME
    )
    checkout = Path(__file__).resolve().parents[3] / "plugins" / agent / PLUGIN_NAME
    for folder in (packaged, checkout):
        if any(
            (folder / marker / "plugin.json").is_file()
            for marker in (".codex-plugin", ".claude-plugin")
        ):
            return folder
    raise FileNotFoundError(
        f"The MemoRizz {AGENTS[agent]} plugin files are missing from this install"
    )


def _marketplace_root() -> Path:
    from .._env_io import memorizz_home

    return memorizz_home() / "plugin-marketplace"


def _agent(value: str) -> str:
    agent = str(value or "").strip().lower().replace("_", "-")
    agent = {"claude": "claude-code"}.get(agent, agent)
    if agent not in AGENTS:
        raise typer.BadParameter("Choose codex or claude-code")
    return agent


def _cli(agent: str) -> str:
    command = "codex" if agent == "codex" else "claude"
    found = shutil.which(command)
    if not found:
        raise typer.BadParameter(f"{AGENTS[agent]} isn't installed or isn't on PATH")
    return found


def _run(agent: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_cli(agent), *args], capture_output=True, text=True, timeout=180, check=False
    )


def _marketplace_entry() -> Dict[str, Any]:
    return {
        "name": PLUGIN_NAME,
        "source": {"source": "local", "path": f"./plugins/codex/{PLUGIN_NAME}"},
        "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
        "category": "Productivity",
    }


def remote_mcp_config(agent: str, url: str, token_env: str) -> Dict[str, Any]:
    """The plugin's MCP entry for a hosted server: Codex reads the bearer token
    from an environment variable; Claude Code expands it in a header."""
    if agent == "codex":
        local = json.loads(
            (plugin_source("codex") / ".mcp.json").read_text(encoding="utf-8")
        )
        server = local["mcpServers"][PLUGIN_NAME]
        return {
            "mcpServers": {
                PLUGIN_NAME: {
                    "type": "http",
                    "url": url,
                    "bearer_token_env_var": token_env,
                    "default_tools_approval_mode": server.get(
                        "default_tools_approval_mode", "writes"
                    ),
                    "tools": server.get("tools", {}),
                    "tool_timeout_sec": server.get("tool_timeout_sec", 300),
                }
            }
        }
    return {
        "mcpServers": {
            PLUGIN_NAME: {
                "type": "http",
                "url": url,
                "headers": {"Authorization": "Bearer ${" + token_env + "}"},
            }
        }
    }


def build_marketplace(
    root: Path, memorizz_bin: Optional[str], remote: Optional[Dict[str, str]] = None
) -> Dict[str, Path]:
    """Write a local marketplace holding both MemoRizz plugins, readable by
    Codex (.agents/plugins/marketplace.json) and Claude Code
    (.claude-plugin/marketplace.json). Each plugin remembers which `memorizz`
    to run, so it works when the agent's PATH differs."""
    plugins: Dict[str, Path] = {}
    for agent in AGENTS:
        target = root / "plugins" / agent / PLUGIN_NAME
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(plugin_source(agent), target)
        for script in (target / "scripts").glob("*.sh"):
            script.chmod(0o755)
        if memorizz_bin:
            (target / "scripts" / "memorizz-bin").write_text(
                memorizz_bin + "\n", encoding="utf-8"
            )
        if remote:
            (target / ".mcp.json").write_text(
                json.dumps(
                    remote_mcp_config(agent, remote["url"], remote["token_env"]),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        plugins[agent] = target
    codex = root / ".agents" / "plugins" / "marketplace.json"
    codex.parent.mkdir(parents=True, exist_ok=True)
    codex.write_text(
        json.dumps(
            {
                "name": MARKETPLACE_NAME,
                "interface": {"displayName": "MemoRizz"},
                "plugins": [_marketplace_entry()],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    claude = root / ".claude-plugin" / "marketplace.json"
    claude.parent.mkdir(parents=True, exist_ok=True)
    claude.write_text(
        json.dumps(
            {
                "name": MARKETPLACE_NAME,
                "owner": {"name": "MemoRizz"},
                "plugins": [
                    {
                        "name": PLUGIN_NAME,
                        "source": f"./plugins/claude-code/{PLUGIN_NAME}",
                        "description": "Durable, searchable project memory through MemoRizz.",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return plugins


def _current_memorizz() -> Optional[str]:
    """This `memorizz` executable, if it is one (not `python -m`)."""
    candidate = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if candidate and candidate.name == "memorizz" and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which("memorizz")
    return str(Path(found).resolve()) if found else None


def _step(
    steps: List[Dict[str, Any]], name: str, done: subprocess.CompletedProcess
) -> bool:
    steps.append(
        {
            "step": name,
            "code": done.returncode,
            "output": (done.stdout + done.stderr).strip(),
        }
    )
    return done.returncode == 0


def _save_choices(choices: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Save the install options to MemoRizz's .env, as `memorizz config set`
    would; options left out keep their current values."""
    from .._env_io import resolve_env_file
    from .settings_commands import save_settings

    updates = {key: value for key, value in choices.items() if value is not None}
    if updates:
        save_settings(updates, resolve_env_file())
    return updates


def _flag(value: Optional[bool]) -> Optional[str]:
    return None if value is None else ("true" if value else "false")


@plugin_app.command("install")
def install(
    agent: str = typer.Argument(..., help="codex or claude-code."),
    user: Optional[str] = typer.Option(
        None,
        "--user",
        help="Save memories for this user (for a store several people share).",
    ),
    capture: Optional[bool] = typer.Option(
        None,
        "--capture/--no-capture",
        help="Save each turn to conversation memory (default on).",
    ),
    summaries: Optional[bool] = typer.Option(
        None,
        "--summaries/--no-summaries",
        help="Summarize each session with MemoRizz's default model (default on).",
    ),
    prompt_recall: Optional[bool] = typer.Option(
        None,
        "--prompt-recall/--no-prompt-recall",
        help="Add related memories to every prompt (default off).",
    ),
    allow_agents: Optional[bool] = typer.Option(
        None,
        "--allow-agents/--no-agents",
        help="Let the agent run your saved MemoRizz agents.",
    ),
    allow_harness: Optional[bool] = typer.Option(
        None,
        "--allow-harness/--no-harness",
        help="Let the agent hand tasks to other harnesses through MemoRizz.",
    ),
    harness_roots: Optional[List[Path]] = typer.Option(
        None, "--harness-root", help="A folder harness runs may use; repeat for more."
    ),
    allow_traces: Optional[bool] = typer.Option(
        None,
        "--allow-traces/--no-traces",
        help="Let the agent query MemoRizz observability traces.",
    ),
    remote: Optional[str] = typer.Option(
        None,
        "--remote",
        help="Use a hosted MemoRizz MCP server at this URL (https://…/mcp) instead of the local store.",
    ),
    token_env: str = typer.Option(
        DEFAULT_TOKEN_ENV,
        "--token-env",
        help="Environment variable that holds the hosted server's API key.",
    ),
    local: bool = typer.Option(
        False, "--local", help="Stop using a hosted server; use the local store again."
    ),
    raw_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Install the MemoRizz plugin into Codex or Claude Code.

    Writes a local marketplace under ~/.memorizz/plugin-marketplace, adds it
    to the agent and installs the plugin. Options are saved to MemoRizz's
    settings (`memorizz config get` shows them). Start a new session to use it.
    """
    agent = _agent(agent)
    if remote and local:
        raise typer.BadParameter("Choose --remote or --local, not both")
    if remote:
        from urllib.parse import urlparse

        parsed = urlparse(remote)
        loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
            raise typer.BadParameter(
                "--remote must be an https:// URL (http:// only for this machine)"
            )
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", token_env):
            raise typer.BadParameter("--token-env must be an environment variable name")
    if (
        allow_harness
        and not harness_roots
        and not os.environ.get("MEMORIZZ_MCP_SERVER_HARNESS_WORKSPACE_ROOTS")
    ):
        raise typer.BadParameter(
            "--allow-harness needs at least one --harness-root folder"
        )
    saved = _save_choices(
        {
            PRINCIPAL_ENV: user.strip() if user and user.strip() else None,
            CAPTURE_ENV: None if capture is None else ("turns" if capture else "off"),
            SUMMARY_ENV: _flag(summaries),
            PROMPT_RECALL_ENV: _flag(prompt_recall),
            "MEMORIZZ_MCP_SERVER_ALLOW_AGENT_EXECUTION": _flag(allow_agents),
            "MEMORIZZ_MCP_SERVER_ALLOW_HARNESS_EXECUTION": _flag(allow_harness),
            "MEMORIZZ_MCP_SERVER_HARNESS_WORKSPACE_ROOTS": (
                ",".join(
                    str(Path(root).expanduser().resolve()) for root in harness_roots
                )
                if harness_roots
                else None
            ),
            "MEMORIZZ_MCP_SERVER_ALLOW_TRACE_QUERIES": _flag(allow_traces),
            REMOTE_URL_ENV: remote or ("" if local else None),
            REMOTE_TOKEN_ENV_ENV: token_env if remote else None,
        }
    )
    remote_url = remote or (None if local else _remote_url())
    root = _marketplace_root()
    plugins = build_marketplace(
        root,
        _current_memorizz(),
        {
            "url": remote_url,
            "token_env": token_env
            if remote
            else _setting(REMOTE_TOKEN_ENV_ENV, DEFAULT_TOKEN_ENV),
        }
        if remote_url
        else None,
    )
    steps: List[Dict[str, Any]] = []
    if agent == "codex":
        listed = _run(agent, "plugin", "marketplace", "list")
        ok = str(root) in listed.stdout or _step(
            steps,
            "marketplace add",
            _run(agent, "plugin", "marketplace", "add", str(root)),
        )
        ok = ok and _step(
            steps,
            "plugin add",
            _run(agent, "plugin", "add", f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"),
        )
        after = "Start a new Codex thread; the first time, review the MemoRizz hooks with /hooks."
    else:
        listed = _run(agent, "plugin", "marketplace", "list")
        ok = MARKETPLACE_NAME in listed.stdout or _step(
            steps,
            "marketplace add",
            _run(agent, "plugin", "marketplace", "add", str(root)),
        )
        ok = ok and _step(
            steps,
            "plugin install",
            _run(agent, "plugin", "install", f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"),
        )
        after = "Start a new Claude Code session; allow the memorizz tools when it first asks."
    payload = {
        "ok": ok,
        "agent": agent,
        "plugin": str(plugins[agent]),
        "marketplace": str(root),
        "steps": steps,
        "settings_saved": saved,
        "next": after,
    }
    typer.echo(json.dumps(payload, indent=None if raw_json else 2))
    if not ok:
        raise typer.Exit(1)


@plugin_app.command("uninstall")
def uninstall(
    agent: str = typer.Argument(..., help="codex or claude-code."),
    raw_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Remove the MemoRizz plugin and its marketplace from an agent. Memories
    stay."""
    agent = _agent(agent)
    if agent == "codex":
        removed = _run(agent, "plugin", "remove", f"{PLUGIN_NAME}@{MARKETPLACE_NAME}")
    else:
        removed = _run(
            agent, "plugin", "uninstall", f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"
        )
    dropped = _run(agent, "plugin", "marketplace", "remove", MARKETPLACE_NAME)
    payload = {
        "ok": removed.returncode == 0,
        "agent": agent,
        "plugin_removed": removed.returncode == 0,
        "marketplace_removed": dropped.returncode == 0,
        "output": (
            removed.stdout + removed.stderr + dropped.stdout + dropped.stderr
        ).strip(),
    }
    typer.echo(json.dumps(payload, indent=None if raw_json else 2))


__all__ = ["plugin_app", "project_memory_id", "prompt_context", "session_context"]
