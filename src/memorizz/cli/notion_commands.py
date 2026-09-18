"""Explicit Notion provisioning and semantic-index maintenance commands."""

import json
from pathlib import Path
from typing import Optional

import typer

from . import config as cfg

notion_app = typer.Typer(
    help="Set up Notion memory and maintain its separate semantic index.",
    no_args_is_help=True,
)


@notion_app.command("connect")
def connect(
    project: bool = typer.Option(
        False, "--project", help="Save to this project's .env."
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env-file", help="Save to an explicit file."
    ),
):
    """Guided Notion credentials, library and vector-store setup."""
    from .memory_commands import configure

    configure(provider="notion", project=project, env_file=env_file)


@notion_app.command("init")
def initialize(
    parent_page_id: str = typer.Option(..., "--parent-page-id"),
    title: str = typer.Option("Memorizz", "--title"),
    views: bool = typer.Option(True, "--views/--no-views"),
):
    """Create a NEW Memorizz area below the specified, shared Notion page."""
    from ..memory_provider.notion import provision_notion_workspace

    cfg.load_layered_env()
    result = provision_notion_workspace(parent_page_id, title=title, create_views=views)
    typer.echo(json.dumps(result, indent=2))
    typer.echo(
        "Set MEMORIZZ_BACKEND=notion and MEMORIZZ_NOTION_DATA_SOURCE_ID="
        + result["data_source_id"]
    )


def _provider(*, read_only=False):
    from ..memory_provider.notion.factory import create_notion_provider_from_env
    from .agent_factory import _choose_embedding

    cfg.load_layered_env()
    embed = _choose_embedding({})
    return create_notion_provider_from_env(
        read_only=read_only,
        embedding_provider=embed.get("embedding_provider"),
        embedding_config=embed.get("embedding_config"),
    )


@notion_app.command("status")
def status():
    """Show tracked index state, without scanning the remote memory library."""
    provider = _provider(read_only=True)
    try:
        typer.echo(json.dumps(provider.index_status(), indent=2))
    finally:
        provider.close()


@notion_app.command("sync")
def sync(
    memory_type: Optional[str] = typer.Option(None, "--memory-type"),
    force: bool = typer.Option(False, "--force"),
):
    """Synchronize Notion edits; --force rebuilds even unchanged vectors."""
    provider = _provider()
    try:
        typer.echo(json.dumps(provider.sync(memory_type, force=force), indent=2))
    finally:
        provider.close()


@notion_app.command("repair")
def repair():
    """Retry pending vector writes/deletes without recreating Notion records."""
    provider = _provider()
    try:
        typer.echo(json.dumps(provider.repair_index(), indent=2))
    finally:
        provider.close()
