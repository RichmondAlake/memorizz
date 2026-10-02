"""Small, secret-safe configuration editor shared by the CLI and REPL.

Saving defaults deliberately does not reconnect a running agent. This module
does not import providers or make network requests, including for ``--help``.
"""

import getpass
import os
import re
import shlex
import warnings
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from uuid import UUID

import typer

from .. import _env_io as env

config_app = typer.Typer(help="Inspect or save settings; credentials use hidden input.")

# An allowlist, not a blacklist: arbitrary integration keys stay hidden.
PUBLIC_SETTINGS = {
    "MEMORIZZ_BACKEND",
    "MEMORIZZ_MEMORY_ROOT",
    "MEMORIZZ_DEFAULT_LLM_PROVIDER",
    "MEMORIZZ_DEFAULT_LLM_MODEL",
    "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER",
    "MEMORIZZ_DEFAULT_EMBEDDING_MODEL",
    "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS",
    "MEMORIZZ_NOTION_DATA_SOURCE_ID",
    "MEMORIZZ_NOTION_SEMANTIC_BACKEND",
    "MEMORIZZ_NOTION_VECTOR_PATH",
    "MEMORIZZ_NOTION_STATE_PATH",
    "MEMORIZZ_NOTION_VECTOR_DB",
    "MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING",
    "MEMORIZZ_NO_UPDATE_CHECK",
    # The Codex and Claude Code plugins.
    "MEMORIZZ_PROMPT_RECALL",
    "MEMORIZZ_SESSION_CAPTURE",
    "MEMORIZZ_SESSION_SUMMARY",
    # The MCP server the plugins (and other MCP clients) run.
    "MEMORIZZ_MCP_SERVER_ALLOW_AGENT_EXECUTION",
    "MEMORIZZ_MCP_SERVER_ALLOW_HARNESS_EXECUTION",
    "MEMORIZZ_MCP_SERVER_HARNESS_WORKSPACE_ROOTS",
    "MEMORIZZ_MCP_SERVER_ALLOW_TRACE_QUERIES",
    "MEMORIZZ_MCP_SERVER_LOCAL_PRINCIPAL",
    "MEMORIZZ_MCP_SERVER_INGEST_ROOTS",
    "MEMORIZZ_PLUGIN_REMOTE_URL",
    "MEMORIZZ_PLUGIN_REMOTE_TOKEN_ENV",
}
SETTING_CHOICES = {
    "MEMORIZZ_BACKEND": ("filesystem", "mongodb", "oracle", "notion"),
    "MEMORIZZ_NOTION_SEMANTIC_BACKEND": ("filesystem", "mongodb", "oracle", "none"),
    "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": (
        "openai",
        "azure",
        "ollama",
        "voyageai",
        "huggingface",
    ),
    "MEMORIZZ_DEFAULT_LLM_PROVIDER": (
        "openai",
        "anthropic",
        "azure",
        "ollama",
        "huggingface",
        "mlx",
        "local-openai",
    ),
    "MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING": ("true", "false"),
    "MEMORIZZ_NO_UPDATE_CHECK": ("1", "0"),
    "MEMORIZZ_PROMPT_RECALL": ("true", "false"),
    "MEMORIZZ_SESSION_CAPTURE": ("turns", "off"),
    "MEMORIZZ_SESSION_SUMMARY": ("true", "false"),
    "MEMORIZZ_MCP_SERVER_ALLOW_AGENT_EXECUTION": ("true", "false"),
    "MEMORIZZ_MCP_SERVER_ALLOW_HARNESS_EXECUTION": ("true", "false"),
    "MEMORIZZ_MCP_SERVER_ALLOW_TRACE_QUERIES": ("true", "false"),
}


def secret_prompt(label: str) -> str:
    """Fail closed when a terminal cannot hide input (no echoed fallback)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass(label + " (hidden): ")
        except getpass.GetPassWarning:
            raise ValueError(
                "Hidden input is unavailable. Use an interactive terminal, "
                "or configure the variable through your secret manager."
            ) from None


def setting_key(key: str) -> str:
    key = key.strip().upper()
    env.validate_env_updates({key: ""})
    if key in {"MEMORIZZ_HOME", "MEMORIZZ_ENV_FILE"}:
        raise ValueError(
            "Set MEMORIZZ_HOME / MEMORIZZ_ENV_FILE in the launch environment; "
            "use --project or --env-file to choose this command's save target."
        )
    return key


def notion_identifier(value: str) -> str:
    """Accept a UUID or a Notion page/database URL, never an arbitrary API host."""
    value = value.strip()
    if "://" in value:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not any(
            host == domain or host.endswith("." + domain)
            for domain in ("notion.so", "notion.com")
        ):
            raise ValueError(
                "Use a Notion HTTPS URL or its page/database/data-source UUID."
            )
        value = parsed.path.rstrip("/").split("/")[-1]
        match = re.search(
            r"([0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})$",
            value,
        )
        value = match.group(1) if match else ""
    try:
        return str(UUID(value))
    except (ValueError, AttributeError):
        raise ValueError(
            "Use a Notion HTTPS URL or its page/database/data-source UUID."
        ) from None


def validate_setting(key: str, value: str) -> str:
    env.validate_env_updates({key: value})
    if key in SETTING_CHOICES:
        value = value.strip().lower()
        if value not in SETTING_CHOICES[key]:
            raise ValueError("Choose one of: " + ", ".join(SETTING_CHOICES[key]))
    elif key == "MEMORIZZ_NOTION_DATA_SOURCE_ID":
        # Here the ID must identify the data source, not its enclosing database.
        try:
            value = str(UUID(value.strip()))
        except ValueError:
            raise ValueError(
                "Enter a data-source UUID, or use 'memorizz notion connect' to resolve a database URL."
            ) from None
    elif key == "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS" and value:
        if not value.isascii() or not value.isdigit() or int(value) < 1:
            raise ValueError(
                "Embedding dimensions must be a positive integer, or blank for the model default."
            )
    return value


def save_target(project: bool = False, env_file: Optional[Path] = None) -> Path:
    if project and env_file is not None:
        raise ValueError("Choose --project or --env-file, not both.")
    target = Path.cwd() / ".env" if project else env_file or env.resolve_env_file()
    return Path(target).expanduser().absolute()


def describe_target(target: Path) -> None:
    typer.echo(f"Save target: {target}")
    typer.echo("Precedence: process exports > project .env > Memorizz .env.")
    if target.resolve() not in {
        (Path.cwd() / ".env").resolve(),
        env.resolve_env_file().resolve(),
    }:
        typer.echo(
            "This custom file is not automatically loaded. Set MEMORIZZ_ENV_FILE to this path when launching Memorizz."
        )


def save_settings(updates: dict, target: Path) -> None:
    """Persist as one transaction without mutating the current process/provider."""
    updates = {
        setting_key(key): validate_setting(setting_key(key), value)
        for key, value in updates.items()
    }
    for warning in env.env_override_warnings(updates, target):
        typer.echo("Warning: " + warning)
    try:
        env.update_env_file(target, updates)
    except (OSError, ValueError, TimeoutError) as exc:
        # Never include exception text; filesystem errors can contain input.
        raise ValueError(
            f"Nothing saved ({type(exc).__name__}). Check .env syntax, target permissions and symlinks."
        ) from None
    typer.echo(f"Saved {len(updates)} setting(s) to {target}.")
    typer.echo(
        "Restart required: saved defaults apply on the next launch. The current agent and memory provider are unchanged; existing memory is not migrated."
    )
    typer.echo(
        "Keep this file private and out of version control. On POSIX systems it is owner-only (0600)."
    )


@config_app.callback(invoke_without_command=True)
def show(ctx: typer.Context):
    """Show configuration, or use set/get/path/keys."""
    if ctx.invoked_subcommand is None:
        from .app import _show_config

        _show_config()


@config_app.command("path")
def path_command(
    project: bool = typer.Option(
        False, "--project", help="Use .env in the current directory."
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env-file", help="Use an explicit file."
    ),
):
    """Show the save target and precedence, without reading credentials."""
    env.load_layered_env()
    try:
        describe_target(save_target(project, env_file))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)


@config_app.command("keys")
def keys_command():
    """List ordinary settings. Other environment variables use hidden input."""
    for key in sorted(PUBLIC_SETTINGS):
        choices = SETTING_CHOICES.get(key)
        typer.echo(key + ("  (" + " | ".join(choices) + ")" if choices else ""))
    typer.echo(
        "Credentials: config set NOTION_TOKEN (or another ENV_VARIABLE), with no value argument."
    )


@config_app.command("set")
def set_command(
    key: str = typer.Argument(..., help="Environment variable name, not KEY=value."),
    value: Optional[str] = typer.Argument(
        None, help="Ordinary value; omit credentials for hidden input."
    ),
    project: bool = typer.Option(
        False, "--project", help="Save to this project's .env."
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env-file", help="Save to an explicit file."
    ),
):
    """Save a setting for the next launch. Omit VALUE for an interactive prompt."""
    env.load_layered_env()
    try:
        key = setting_key(key)
        target = save_target(project, env_file)
        describe_target(target)
        if key not in PUBLIC_SETTINGS:
            if value is not None:
                raise ValueError(
                    "Do not put credentials or unknown settings in command arguments. Run 'memorizz config set ENV_VARIABLE' without VALUE for hidden input."
                )
            value = secret_prompt(key)
            if not value:
                typer.echo("Cancelled; nothing saved.")
                return
        elif value is None:
            choices = SETTING_CHOICES.get(key)
            if choices:
                typer.echo("Choices: " + ", ".join(choices))
            value = typer.prompt(
                key, default=os.environ.get(key, ""), show_default=True
            )
        value = validate_setting(key, value)
        save_settings({key: value}, target)
    except (EOFError, KeyboardInterrupt, typer.Abort):
        typer.echo("Cancelled; nothing saved.")
        raise typer.Exit(1)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)


@config_app.command("get")
def get_command(
    key: str,
    project: bool = typer.Option(False, "--project"),
    env_file: Optional[Path] = typer.Option(None, "--env-file"),
):
    """Show effective and saved values plus their source; credentials stay redacted."""
    from dotenv import dotenv_values

    env.load_layered_env()
    try:
        key = setting_key(key)
        target = save_target(project, env_file)
        source = env.environment_source(key)
        saved = dotenv_values(target, interpolate=False) if target.is_file() else {}

        def display(value):
            if value is None:
                return "(unset)"
            return repr(value) if key in PUBLIC_SETTINGS else "(set; hidden)"

        typer.echo(f"{key}: {display(os.environ.get(key))}")
        typer.echo("Effective source: " + source.get("path", source["kind"]))
        typer.echo(f"Saved in {target}: {display(saved.get(key))}")
        typer.echo(
            "A running agent may still use its original connection; /config shows the active provider."
        )
    except (ValueError, OSError):
        typer.echo(
            "Cannot inspect that setting. Check the variable name and file permissions.",
            err=True,
        )
        raise typer.Exit(2)


def run_repl_command(application, args: str, name: str, console) -> None:
    """Reuse the same command contract; never reflect malformed secret input."""
    try:
        application(args=shlex.split(args), prog_name=name, standalone_mode=False)
    except (ValueError, typer.TyperException):
        console.print(
            f"Invalid arguments. Use {name} --help. Credentials must be entered at the hidden prompt."
        )
    except (typer.Abort, EOFError, KeyboardInterrupt):
        console.print("Cancelled; the current agent is unchanged.")
