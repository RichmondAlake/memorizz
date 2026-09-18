"""Guided memory setup, with explicit consent before remote provisioning."""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import typer

from .. import _env_io as env
from .settings_commands import (
    SETTING_CHOICES,
    describe_target,
    notion_identifier,
    save_settings,
    save_target,
    secret_prompt,
    validate_setting,
)

memory_app = typer.Typer(
    help="Configure the memory provider (separate from the LLM).", no_args_is_help=True
)


def _choice(label, choices, default):
    if default not in choices:
        default = choices[0]

    def parse(value):
        selected = value.strip().lower()
        if selected not in choices:
            raise typer.BadParameter("Choose one of: " + ", ".join(choices))
        return selected

    return typer.prompt(
        label + " (" + ", ".join(choices) + ")", value_proc=parse, default=default
    )


def _secret(key, *, fallback=None):
    current = os.getenv(key) or (os.getenv(fallback) if fallback else None)
    if current and typer.confirm(f"Keep the configured {key} (hidden)?", default=True):
        return current
    value = secret_prompt(key).strip()
    if not value:
        raise ValueError("A credential is required; nothing saved.")
    env.validate_env_updates({key: value})
    return value


def _ordinary(key, label, default=""):
    return validate_setting(key, typer.prompt(label, default=default).strip())


def _embedding_settings():
    choices = SETTING_CHOICES["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"]
    current_provider = os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "")
    provider = _choice(
        "Embedding provider",
        choices,
        current_provider if current_provider in choices else "ollama",
    )
    defaults = {"ollama": "nomic-embed-text", "openai": "text-embedding-3-small"}
    old_model = os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "")
    model = typer.prompt(
        "Embedding model / Azure deployment",
        default=(
            old_model if provider == current_provider else defaults.get(provider, "")
        ),
    ).strip()
    if not model:
        raise ValueError("An embedding model is required; nothing saved.")
    dimensions = _ordinary(
        "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS",
        "Embedding dimensions (blank = model default)",
        os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "")
        if provider == current_provider and model == old_model
        else "",
    )
    updates = {
        "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": provider,
        "MEMORIZZ_DEFAULT_EMBEDDING_MODEL": model,
        "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS": dimensions,
    }
    key = {
        "openai": "OPENAI_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "voyageai": "VOYAGE_API_KEY",
    }.get(provider)
    if key:
        updates[key] = _secret(key)
    if provider == "azure":
        updates["AZURE_OPENAI_ENDPOINT"] = _ordinary(
            "AZURE_OPENAI_ENDPOINT",
            "Azure OpenAI endpoint",
            os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        )
        if not updates["AZURE_OPENAI_ENDPOINT"]:
            raise ValueError("An Azure endpoint is required; nothing saved.")
    typer.echo(
        "Embedding settings are shared defaults. No model download or paid embedding call is made by setup; install/configure the chosen provider before sync."
    )
    return updates


def _storage_settings(backend, *, notion=False):
    updates = {}
    if backend == "filesystem":
        key = "MEMORIZZ_NOTION_VECTOR_PATH" if notion else "MEMORIZZ_MEMORY_ROOT"
        default = os.getenv(key, "") if notion else str(env.memory_root())
        updates[key] = _ordinary(
            key,
            "Vector directory (blank = separate directory per Notion data source)"
            if notion
            else "Memory directory",
            default,
        )
    elif backend == "mongodb":
        key = "MEMORIZZ_NOTION_MONGODB_URI" if notion else "MONGODB_URI"
        updates[key] = _secret(key, fallback="MONGODB_URI" if notion else None)
        db_key = "MEMORIZZ_NOTION_VECTOR_DB" if notion else "MONGODB_DB_NAME"
        updates[db_key] = _ordinary(
            db_key,
            "MongoDB database name",
            os.getenv(db_key, "memorizz_notion_vectors" if notion else "memorizz"),
        )
    elif backend == "oracle":
        for key, label in (
            ("ORACLE_USER", "Oracle user"),
            ("ORACLE_DSN", "Oracle DSN (host:port/service; no password)"),
        ):
            updates[key] = _ordinary(key, label, os.getenv(key, ""))
            if not updates[key]:
                raise ValueError("Oracle user and DSN are required; nothing saved.")
        updates["ORACLE_PASSWORD"] = _secret("ORACLE_PASSWORD")
        if not notion:
            updates["MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING"] = "false"
    if backend != "none":
        updates.update(_embedding_settings())
    typer.echo(
        "Vector connectivity, schema/index readiness and model availability are not tested by this configuration wizard."
    )
    return updates


def _resolve_data_source(client, identifier):
    from ..memory_provider.notion.client import NotionAPIError
    from ..memory_provider.notion.provider import PROPERTIES

    try:
        schema = client.request("GET", "/data_sources/" + identifier)
    except NotionAPIError as exc:
        if exc.status != 404:
            raise
        database = client.request("GET", "/databases/" + identifier)
        sources = database.get("data_sources") or []
        if not sources:
            raise ValueError(
                "This database has no accessible data source. Share it with your Notion connection."
            ) from None
        if len(sources) > 1:
            # IDs are unambiguous and do not render untrusted Notion names.
            for index, source in enumerate(sources, 1):
                typer.echo(f"  {index}. {notion_identifier(source['id'])}")
            selected = (
                int(
                    _choice(
                        "Data source number",
                        [str(index) for index in range(1, len(sources) + 1)],
                        "1",
                    )
                )
                - 1
            )
        else:
            selected = 0
        identifier = notion_identifier(sources[selected]["id"])
        schema = client.request("GET", "/data_sources/" + identifier)
    missing = [
        name
        for name, kind in PROPERTIES.values()
        if schema.get("properties", {}).get(name, {}).get("type") != kind
    ]
    if missing:
        raise ValueError(
            "Not a compatible Memorizz library. Required columns are missing or renamed: "
            + ", ".join(missing)
            + ". Use a library created by Memorizz, or restore its managed column names. No schema was changed."
        )
    typer.echo("Notion access and managed column types verified (read-only check).")
    return notion_identifier(schema["id"])


def _recovery_file(workspace, target):
    """Persist non-secret IDs so a later local save failure cannot lose them."""
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, path = tempfile.mkstemp(
        prefix="notion-setup-", suffix=".json", dir=target.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(workspace, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    typer.echo(
        f"Created-resource IDs saved to {path}. Keep this file; rerunning 'create' makes another area."
    )


def _notion_setup(target):
    from ..memory_provider.notion import provision_notion_workspace
    from ..memory_provider.notion.client import (
        NotionAPIError,
        NotionClient,
        NotionError,
    )

    typer.echo(
        "Notion stores the memory documents and human-readable views. A separate provider handles vectors. This suits modest-volume use; Notion is rate-limited and is not an interactive chat server."
    )
    typer.echo(
        "Create a Notion internal connection, then share the parent page/library with it using the page's Connections menu. Setup never edits an existing library's schema."
    )
    token = _secret("NOTION_TOKEN")
    action = _choice("Notion library", ("existing", "create"), "existing")
    identifier = notion_identifier(
        typer.prompt(
            "Shared parent page URL / ID"
            if action == "create"
            else "Memorizz database URL / data-source ID",
            default=os.getenv("MEMORIZZ_NOTION_DATA_SOURCE_ID", "")
            if action == "existing"
            else "",
        )
    )
    workspace = {}
    client = NotionClient(token)
    try:
        if action == "create":
            page = client.request("GET", "/pages/" + identifier)
            if page.get("in_trash") or page.get("archived"):
                raise ValueError(
                    "That parent page is archived. Restore it or choose an active page."
                )
            title = typer.prompt("New Memorizz area title", default="Memorizz").strip()
            if not title:
                raise ValueError("An area title is required; nothing created.")
        else:
            identifier = _resolve_data_source(client, identifier)
        semantic = _choice(
            "Semantic / vector backend",
            SETTING_CHOICES["MEMORIZZ_NOTION_SEMANTIC_BACKEND"],
            os.getenv("MEMORIZZ_NOTION_SEMANTIC_BACKEND", "filesystem"),
        )
        updates = {
            "MEMORIZZ_BACKEND": "notion",
            "NOTION_TOKEN": token,
            "MEMORIZZ_NOTION_SEMANTIC_BACKEND": semantic,
        }
        updates.update(_storage_settings(semantic, notion=True))
        if semantic == "none":
            typer.echo(
                "Semantic search is disabled. Memory CRUD and Notion tables still work; agent features requiring embeddings will not."
            )
        # Explicitly choose the journal rather than reusing a prior library's file.
        updates["MEMORIZZ_NOTION_STATE_PATH"] = _ordinary(
            "MEMORIZZ_NOTION_STATE_PATH",
            "Local sync journal path (blank = automatic per-library journal)",
            "",
        )
        for key, value in updates.items():
            validate_setting(key, value)
        for warning in env.env_override_warnings(updates, target):
            typer.echo("Warning: " + warning)
        typer.echo(
            f"Memory provider: notion; vector backend: {semantic}; save target: {target}"
        )
        if not typer.confirm(
            "Create a NEW Notion area with memory tables and agent views, then save these defaults?"
            if action == "create"
            else "Save these defaults for the next launch?",
            default=False,
        ):
            typer.echo("Cancelled; nothing saved or created.")
            return
        if action == "create":
            typer.echo(
                "Creating the requested Notion area and views. This can take a moment..."
            )
            try:
                workspace = provision_notion_workspace(
                    identifier, title=title, client=client
                )
            except BaseException as exc:
                workspace = getattr(exc, "notion_workspace", {})
                raise
            identifier = notion_identifier(workspace["data_source_id"])
        updates["MEMORIZZ_NOTION_DATA_SOURCE_ID"] = identifier
        if workspace:
            _recovery_file(workspace, target)
            typer.echo("Notion area: " + workspace["url"])
        save_settings(updates, target)
        typer.echo(
            "Next: restart Memorizz; run 'memorizz notion status', then 'memorizz notion sync' to populate/refresh vectors. Sync may call the configured embedding service and incur charges. Use /config to inspect the active provider."
        )
    except NotionAPIError as exc:
        typer.echo(
            f"Notion access failed (HTTP {exc.status}). Check the connection token, share the exact page/database with it, and check its read/insert/update permissions.",
            err=True,
        )
        raise typer.Exit(1) from None
    except NotionError:
        typer.echo(
            "Notion did not complete setup. Check the target area before retrying; a create may have succeeded remotely.",
            err=True,
        )
        raise typer.Exit(1) from None
    finally:
        client.close()
        if workspace:
            # Always print recoverable IDs, including partial provisioning and
            # failures creating the local manifest; never include the token.
            typer.echo("Recoverable Notion resource IDs (do not blindly rerun create):")
            typer.echo(json.dumps(workspace, indent=2))


@memory_app.command("configure")
def configure(
    provider: Optional[str] = typer.Argument(
        None, metavar="PROVIDER", help="filesystem, mongodb, oracle, or notion."
    ),
    project: bool = typer.Option(
        False, "--project", help="Save to the current project's .env."
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env-file", help="Save to an explicit .env file."
    ),
):
    """Guided setup. Does not migrate memory or switch the running agent."""
    env.load_layered_env()
    try:
        target = save_target(project, env_file)
        describe_target(target)
        choices = SETTING_CHOICES["MEMORIZZ_BACKEND"]
        provider = (
            provider.strip().lower()
            if provider
            else _choice(
                "Memory provider", choices, os.getenv("MEMORIZZ_BACKEND", "filesystem")
            )
        )
        validate_setting("MEMORIZZ_BACKEND", provider)
        if provider == "notion":
            _notion_setup(target)
            return
        updates = {"MEMORIZZ_BACKEND": provider, **_storage_settings(provider)}
        if typer.confirm("Save these defaults for the next launch?", default=False):
            save_settings(updates, target)
        else:
            typer.echo("Cancelled; nothing saved.")
    except (EOFError, KeyboardInterrupt, typer.Abort):
        typer.echo(
            "Cancelled; current agent unchanged. If creation started, inspect the printed resource IDs before retrying."
        )
        raise typer.Exit(1)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)
    except OSError:
        typer.echo(
            "Setup could not finish saving local files. Check file permissions and retain any printed Notion IDs before retrying.",
            err=True,
        )
        raise typer.Exit(1)
