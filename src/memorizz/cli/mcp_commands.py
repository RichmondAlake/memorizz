# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Typer commands for MCP connection configuration and diagnostics."""

from __future__ import annotations

import json
import webbrowser
from typing import List, Optional

import typer

from . import mcp_config

mcp_app = typer.Typer(
    help="Connect to MCP servers or expose Memorizz as an MCP server.",
    no_args_is_help=True,
)

NOTION_URL = "https://mcp.notion.com/mcp"
GOOGLE_CALENDAR_URL = "https://calendarmcp.googleapis.com/mcp/v1"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8765/api/mcp/oauth/callback"
GOOGLE_CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    "https://www.googleapis.com/auth/calendar.events.freebusy",
    "https://www.googleapis.com/auth/calendar.events.readonly",
]


def _console(*, stderr: bool = False):
    from rich.console import Console

    return Console(stderr=stderr)


def _print_result(value, *, raw_json: bool = False) -> None:
    console = _console()
    if raw_json:
        console.print_json(json.dumps(value, default=str))
    else:
        console.print(value)


def _exit_for_result(value) -> None:
    if isinstance(value, dict) and not value.get("ok", True):
        raise typer.Exit(1)


def _pairs(values: List[str], label: str) -> dict:
    result = {}
    for raw in values:
        if "=" not in raw:
            raise typer.BadParameter(f"{label} must use KEY=VALUE syntax: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"{label} key cannot be empty")
        result[key] = value
    return result


@mcp_app.command("serve")
def serve_memorizz(
    transport: Optional[str] = typer.Option(
        None, "--transport", "-t", help="stdio or streamable-http."
    ),
    host: Optional[str] = typer.Option(None, "--host", help="HTTP bind host."),
    port: Optional[int] = typer.Option(None, "--port", min=1, max=65535),
    path: Optional[str] = typer.Option(None, "--path", help="HTTP MCP endpoint path."),
    public_url: Optional[str] = typer.Option(
        None, "--public-url", help="External HTTPS origin used in auth metadata."
    ),
    agent_id: Optional[List[str]] = typer.Option(
        None,
        "--agent-id",
        help="Agent exposed to remote callers; repeat to expose more than one.",
    ),
    allow_anonymous: Optional[bool] = typer.Option(
        None,
        "--allow-anonymous/--require-auth",
        help="Explicitly permit unauthenticated HTTP access.",
    ),
    allow_writes: Optional[bool] = typer.Option(
        None,
        "--allow-writes/--read-only",
        help="Allow direct memory writes and conversation persistence.",
    ),
    allow_agent_execution: Optional[bool] = typer.Option(
        None,
        "--allow-agent-execution/--no-agent-execution",
        help="Allow exposed agents to run for MCP callers.",
    ),
    allow_harness_execution: Optional[bool] = typer.Option(
        None,
        "--allow-harness-execution/--no-harness-execution",
        help="Allow governed Codex, Claude Code, or OpenHands runs.",
    ),
    harness_workspace_root: Optional[List[str]] = typer.Option(
        None,
        "--harness-workspace-root",
        help="Allowed harness workspace root; repeat to add roots.",
    ),
    stateless_http: Optional[bool] = typer.Option(
        None,
        "--stateless-http/--stateful-http",
        help="Use stateless Streamable HTTP sessions (recommended).",
    ),
):
    """Expose Memorizz memory and agents to any standards-compliant MCP client."""
    # Load the same layered environment as the regular CLI. Bearer tokens are
    # intentionally accepted only through MEMORIZZ_MCP_SERVER_API_KEYS, never a
    # process-list-visible command-line option.
    from .config import load_layered_env

    load_layered_env()
    try:
        from ..mcp_server import MemorizzMCPServerConfig, run_memorizz_mcp_server
    except ImportError as exc:
        raise typer.BadParameter(
            "The MCP server requires optional dependencies; install " "`memorizz[mcp]`."
        ) from exc

    overrides = {
        "transport": transport,
        "host": host,
        "port": port,
        "path": path,
        "public_url": public_url,
        "exposed_agent_ids": set(agent_id) if agent_id is not None else None,
        "allow_anonymous_http": allow_anonymous,
        "allow_writes": allow_writes,
        "allow_agent_execution": allow_agent_execution,
        "allow_harness_execution": allow_harness_execution,
        "harness_workspace_roots": (
            set(harness_workspace_root) if harness_workspace_root is not None else None
        ),
        "stateless_http": stateless_http,
    }
    try:
        config = MemorizzMCPServerConfig.from_env(**overrides)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    console = _console(stderr=True)
    if config.transport == "stdio":
        console.print(
            "[dim]Serving Memorizz MCP over stdio; protocol output remains on stdout.[/dim]"
        )
    else:
        auth = "bearer auth" if config.auth_required else "anonymous access"
        endpoint = (
            f"{config.public_url}{config.path}"
            if config.public_url
            else f"http://{config.host}:{config.port}{config.path}"
        )
        console.print(f"[green]Serving Memorizz MCP[/green] at {endpoint} ({auth})")
    run_memorizz_mcp_server(config)


def _server_runtime():
    """Build the operator-side runtime sharing the durable approval store."""
    from ..mcp_server import MemorizzMCPServerConfig, MemorizzRuntime

    return MemorizzRuntime(
        MemorizzMCPServerConfig(
            transport="stdio",
            allow_writes=True,
            allow_agent_execution=False,
        )
    )


@mcp_app.command("server-approvals")
def list_server_approvals(
    status: Optional[str] = typer.Option(None, "--status"),
    principal: Optional[str] = typer.Option(None, "--principal"),
    limit: int = typer.Option(100, "--limit", min=1, max=500),
    raw_json: bool = typer.Option(False, "--json"),
):
    """List deletion proposals created by the Memorizz MCP server."""
    approvals = _server_runtime().list_approvals(
        status=status, principal=principal, limit=limit
    )
    _print_result(
        {"ok": True, "approvals": approvals, "count": len(approvals)},
        raw_json=raw_json,
    )


@mcp_app.command("server-approve")
def approve_server_proposal(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver"),
    reason: Optional[str] = typer.Option(None, "--reason"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Approve an exact MCP-server operation without executing it."""
    proposal = _server_runtime().approve_proposal(
        proposal_id, approver_id=approver_id, reason=reason
    )
    _print_result({"ok": True, "proposal": proposal}, raw_json=raw_json)


@mcp_app.command("server-reject")
def reject_server_proposal(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver"),
    reason: Optional[str] = typer.Option(None, "--reason"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Reject an MCP-server operation."""
    proposal = _server_runtime().reject_proposal(
        proposal_id, approver_id=approver_id, reason=reason
    )
    _print_result({"ok": True, "proposal": proposal}, raw_json=raw_json)


@mcp_app.command("server-resume")
def resume_server_proposal(
    proposal_id: str = typer.Argument(...),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Consume one approved proposal and resume its exact deletion checkpoint."""
    result = _server_runtime().resume_proposal(proposal_id)
    _print_result(result, raw_json=raw_json)
    _exit_for_result(result)


@mcp_app.command("server-cancel")
def cancel_server_proposal(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver"),
    reason: Optional[str] = typer.Option(None, "--reason"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Cancel a pending MCP-server operation."""
    proposal = _server_runtime().cancel_proposal(
        proposal_id, approver_id=approver_id, reason=reason
    )
    _print_result({"ok": True, "proposal": proposal}, raw_json=raw_json)


@mcp_app.command("list")
def list_servers(
    raw_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
):
    """List configured MCP servers and authentication state."""
    manager = mcp_config.build_manager()
    value = manager.connection_status()
    if raw_json:
        _print_result(value, raw_json=True)
        return
    from rich.table import Table

    table = Table(title="MCP connections")
    for heading in ("Name", "Transport", "Auth", "State", "Endpoint"):
        table.add_column(heading)
    configs = {server["name"]: server for server in manager.server_dicts()}
    for status in value.get("servers", []):
        server = configs.get(status.get("name"), {})
        endpoint = (
            server.get("url")
            or " ".join(
                [server.get("command") or "", *(server.get("args") or [])]
            ).strip()
        )
        table.add_row(
            str(status.get("name", "")),
            str(status.get("transport", "")),
            str(status.get("auth_type", "")),
            str(status.get("state", "")),
            endpoint,
        )
    _console().print(table)


@mcp_app.command("add")
def add_server(
    name: str = typer.Argument(..., help="Unique connection name."),
    preset: Optional[str] = typer.Option(
        None, "--preset", help="notion or google-calendar"
    ),
    transport: str = typer.Option("streamable_http", "--transport", "-t"),
    url: Optional[str] = typer.Option(None, "--url"),
    command: Optional[str] = typer.Option(None, "--command"),
    arg: Optional[List[str]] = typer.Option(
        None, "--arg", help="Repeat for each stdio argument."
    ),
    env: Optional[List[str]] = typer.Option(
        None, "--env", help="Encrypted KEY=VALUE; repeatable."
    ),
    header: Optional[List[str]] = typer.Option(
        None, "--header", help="Encrypted KEY=VALUE; repeatable."
    ),
    auth: str = typer.Option("none", "--auth", help="none, bearer, or oauth"),
    token: Optional[str] = typer.Option(
        None, "--token", help="Bearer token/PAT (encrypted at rest)."
    ),
    client_id: Optional[str] = typer.Option(None, "--client-id"),
    client_secret: Optional[str] = typer.Option(
        None, "--client-secret", help="OAuth client secret (encrypted at rest)."
    ),
    redirect_uri: Optional[str] = typer.Option(None, "--redirect-uri"),
    scope: Optional[List[str]] = typer.Option(
        None, "--scope", help="OAuth scope; repeatable."
    ),
    allow_tool: Optional[List[str]] = typer.Option(
        None, "--allow-tool", help="Tool allowlist; repeatable."
    ),
    block_tool: Optional[List[str]] = typer.Option(
        None, "--block-tool", help="Blocked tool; repeatable."
    ),
    read_only_tool: Optional[List[str]] = typer.Option(
        None, "--read-only-tool", help="Explicit read-only policy; repeatable."
    ),
    mutation_tool: Optional[List[str]] = typer.Option(
        None, "--mutation-tool", help="Explicit mutation policy; repeatable."
    ),
    require_approval: bool = typer.Option(
        True, "--require-approval/--no-require-approval"
    ),
    allow_private_network: bool = typer.Option(False, "--allow-private-network"),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0),
):
    """Add or replace a connection; secret options are encrypted immediately."""
    preset_value = (preset or "").strip().lower().replace("_", "-")
    scopes = list(scope or [])
    if preset_value == "notion":
        transport, url, auth = "streamable_http", NOTION_URL, "oauth"
        redirect_uri = redirect_uri or DEFAULT_REDIRECT_URI
    elif preset_value in {"calendar", "google-calendar", "google"}:
        transport, url, auth = "streamable_http", GOOGLE_CALENDAR_URL, "oauth"
        redirect_uri = redirect_uri or DEFAULT_REDIRECT_URI
        scopes = scopes or list(GOOGLE_CALENDAR_SCOPES)
        if not client_id:
            raise typer.BadParameter("Google Calendar requires --client-id")
    elif preset_value:
        raise typer.BadParameter("--preset must be notion or google-calendar")

    auth_value = auth.strip().lower()
    if auth_value not in {"none", "bearer", "oauth"}:
        raise typer.BadParameter("--auth must be none, bearer, or oauth")
    payload = {
        "name": name,
        "transport": transport,
        "timeout": timeout,
        "require_approval": require_approval,
        "allow_private_network": allow_private_network,
        "allowed_tools": list(allow_tool or []),
        "blocked_tools": list(block_tool or []),
        "read_only_tools": list(read_only_tool or []),
        "mutation_tools": list(mutation_tool or []),
        "auth": {
            "type": auth_value,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "scopes": scopes,
            "token": token,
        },
    }
    if transport.strip().lower().replace("-", "_") == "stdio":
        payload.update(
            {
                "command": command,
                "args": list(arg or []),
                "env": _pairs(list(env or []), "--env"),
            }
        )
    else:
        payload.update({"url": url, "headers": _pairs(list(header or []), "--header")})

    manager = mcp_config.build_manager()
    try:
        server = manager.upsert_server(payload)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = mcp_config.save_servers(manager.server_dicts())
    _console().print(f"[green]Saved[/green] {server['name']} in {path}")
    if auth_value == "oauth":
        _console().print(f"Next: [cyan]memorizz mcp login {server['name']}[/cyan]")


@mcp_app.command("remove")
def remove_server(name: str = typer.Argument(...)):
    """Remove a connection and its encrypted credentials."""
    manager = mcp_config.build_manager()
    if not manager.remove_server(name, delete_credentials=True):
        _console().print(f"[red]Unknown MCP server:[/red] {name}")
        raise typer.Exit(1)
    mcp_config.save_servers(manager.server_dicts())
    _console().print(f"[green]Removed[/green] {name}")


@mcp_app.command("status")
def status(
    name: Optional[str] = typer.Argument(None),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Show configuration, authentication, and local request metrics."""
    manager = mcp_config.build_manager()
    value = manager.connection_status(name)
    value["diagnostics"] = manager.diagnostics().get("metrics", {})
    _print_result(value, raw_json=raw_json)


def _operation(name: str, method_name: str, raw_json: bool = False):
    manager = mcp_config.build_manager()
    value = getattr(manager, method_name)(name)
    _print_result(value, raw_json=raw_json)
    _exit_for_result(value)


@mcp_app.command("test")
def test(
    name: str = typer.Argument(...), raw_json: bool = typer.Option(False, "--json")
):
    """Connect, initialize the protocol, and list tools."""
    _operation(name, "test_connection", raw_json)


@mcp_app.command("tools")
def tools(
    name: str = typer.Argument(...), raw_json: bool = typer.Option(False, "--json")
):
    """List the tools exposed by a server."""
    _operation(name, "list_tools", raw_json)


@mcp_app.command("resources")
def resources(
    name: str = typer.Argument(...), raw_json: bool = typer.Option(False, "--json")
):
    """List the resources exposed by a server."""
    _operation(name, "list_resources", raw_json)


@mcp_app.command("prompts")
def prompts(
    name: str = typer.Argument(...), raw_json: bool = typer.Option(False, "--json")
):
    """List the prompts exposed by a server."""
    _operation(name, "list_prompts", raw_json)


@mcp_app.command("call")
def call_tool(
    name: str = typer.Argument(..., help="Server name."),
    tool_name: str = typer.Argument(..., help="Tool name."),
    arguments: str = typer.Option("{}", "--arguments", "-a", help="JSON object."),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Call an MCP tool; mutations return a durable proposal ID."""
    try:
        values = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--arguments is invalid JSON: {exc}") from exc
    if not isinstance(values, dict):
        raise typer.BadParameter("--arguments must be a JSON object")
    result = mcp_config.build_manager().call_tool(name, tool_name, values)
    _print_result(result, raw_json=raw_json)
    _exit_for_result(result)


@mcp_app.command("approvals")
def list_approvals(
    status: Optional[str] = typer.Option(None, "--status"),
    limit: int = typer.Option(100, "--limit", min=1, max=500),
    raw_json: bool = typer.Option(False, "--json"),
):
    """List durable MCP tool-call approval proposals."""
    result = {
        "ok": True,
        "approvals": mcp_config.build_manager().list_tool_call_approvals(
            status=status, limit=limit
        ),
    }
    result["count"] = len(result["approvals"])
    _print_result(result, raw_json=raw_json)


@mcp_app.command("approve")
def approve_call(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver", help="Human approver identity."),
    reason: Optional[str] = typer.Option(None, "--reason"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Approve one exact MCP call without executing it."""
    result = mcp_config.build_manager().approve_tool_call(
        proposal_id, approver_id=approver_id, reason=reason
    )
    _print_result({"ok": True, "proposal": result}, raw_json=raw_json)


@mcp_app.command("reject")
def reject_call(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver", help="Human approver identity."),
    reason: Optional[str] = typer.Option(None, "--reason"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Reject one exact MCP call."""
    result = mcp_config.build_manager().reject_tool_call(
        proposal_id, approver_id=approver_id, reason=reason
    )
    _print_result({"ok": True, "proposal": result}, raw_json=raw_json)


@mcp_app.command("resume")
def resume_call(
    proposal_id: str = typer.Argument(...),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Consume an approved proposal and execute its stored MCP call once."""
    result = mcp_config.build_manager().resume_tool_call(proposal_id)
    _print_result(result, raw_json=raw_json)
    _exit_for_result(result)


@mcp_app.command("cancel")
def cancel_call(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver", help="Human operator identity."),
    reason: Optional[str] = typer.Option(None, "--reason"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Cancel one pending MCP call."""
    result = mcp_config.build_manager().cancel_tool_call(
        proposal_id, approver_id=approver_id, reason=reason
    )
    _print_result({"ok": True, "proposal": result}, raw_json=raw_json)


@mcp_app.command("login")
def login(
    name: str = typer.Argument(...),
    no_browser: bool = typer.Option(False, "--no-browser"),
):
    """Authorize an OAuth server, then paste the final callback URL."""
    console = _console()

    def show_url(url: str) -> None:
        console.print(f"Open this URL:\n[link={url}]{url}[/link]")
        if not no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass

    def read_callback() -> str:
        return typer.prompt("Paste the full callback URL")

    result = mcp_config.build_manager().authorize_interactive(
        name,
        on_authorization_url=show_url,
        callback_reader=read_callback,
    )
    _print_result(result)
    _exit_for_result(result)


@mcp_app.command("logout")
def logout(name: str = typer.Argument(...)):
    """Delete bearer/OAuth credentials without removing the connection."""
    manager = mcp_config.build_manager()
    deleted = manager.disconnect(name)
    _print_result({"ok": True, "server_name": name, "credentials_deleted": deleted})
