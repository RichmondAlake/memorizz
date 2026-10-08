# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Network, subprocess, policy, and redaction controls for MCP clients."""

from __future__ import annotations

import fnmatch
import ipaddress
import os
import shutil
import socket
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

from ..redaction import SENSITIVE_KEY_NAMES, KeyMatcher, RedactionPolicy
from ..redaction import redact as _redact
from .errors import MCPConfigurationError, MCPPolicyError
from .models import MCPServerConfig

# A connected tool is assumed to change data unless the server, the host or
# its name says it only reads. "export" is deliberately absent: exports often
# create files or share data.
_READ_ONLY_PREFIXES = frozenset(
    {
        "check",
        "count",
        "describe",
        "fetch",
        "find",
        "get",
        "info",
        "inspect",
        "list",
        "lookup",
        "preview",
        "query",
        "read",
        "search",
        "show",
        "status",
        "view",
    }
)

_CONNECTIVES = frozenset({"and", "then", "or", "with", "plus"})

# Verbs that always mean a mutation, wherever they appear in the name
# ("get_and_delete" is a delete). Unknown verbs are mutations too; this set
# only exists so a read verb earlier in the name cannot launder them.
_MUTATION_VERBS = frozenset(
    {
        "add",
        "append",
        "apply",
        "approve",
        "archive",
        "assign",
        "attach",
        "book",
        "buy",
        "call",
        "cancel",
        "clear",
        "close",
        "compact",
        "compile",
        "confirm",
        "continue",
        "create",
        "delete",
        "deploy",
        "destroy",
        "detach",
        "disable",
        "drop",
        "edit",
        "enable",
        "execute",
        "forget",
        "forward",
        "import",
        "ingest",
        "insert",
        "invite",
        "label",
        "launch",
        "merge",
        "move",
        "patch",
        "pause",
        "post",
        "publish",
        "purchase",
        "purge",
        "put",
        "record",
        "register",
        "reject",
        "remove",
        "rename",
        "reply",
        "rerun",
        "reschedule",
        "reset",
        "restore",
        "resume",
        "retry",
        "revoke",
        "run",
        "save",
        "schedule",
        "send",
        "set",
        "share",
        "start",
        "stop",
        "store",
        "submit",
        "summarize",
        "sync",
        "trash",
        "trigger",
        "unsubscribe",
        "update",
        "upload",
        "upsert",
        "write",
    }
)

# Whole-key matches only: MCP argument names are structured, and string values
# are tool traffic that must round-trip untouched.
_REDACTION = RedactionPolicy(
    keys=KeyMatcher(exact=SENSITIVE_KEY_NAMES), replacement="***", tuples="tuple"
)


def redact(value: Any) -> Any:
    """Recursively redact known credential-bearing fields."""
    return _redact(value, _REDACTION)


def validate_remote_url(url: str, allow_private_network: bool = False) -> None:
    """Reject unsafe MCP destinations and common SSRF targets."""
    parsed = urlparse(str(url or "").strip())
    if parsed.username or parsed.password:
        raise MCPConfigurationError("MCP URLs must not contain embedded credentials")
    if parsed.scheme not in ({"https", "http"} if allow_private_network else {"https"}):
        requirement = "http or https" if allow_private_network else "https"
        raise MCPConfigurationError(f"Remote MCP URL must use {requirement}")
    if not parsed.hostname:
        raise MCPConfigurationError("Remote MCP URL must include a hostname")

    configured_allowlist = [
        item.strip().lower()
        for item in os.environ.get("MEMORIZZ_MCP_HOST_ALLOWLIST", "").split(",")
        if item.strip()
    ]
    hostname = parsed.hostname.lower().rstrip(".")
    if configured_allowlist and not any(
        fnmatch.fnmatch(hostname, pattern) for pattern in configured_allowlist
    ):
        raise MCPConfigurationError(
            f"MCP hostname '{hostname}' is not in MEMORIZZ_MCP_HOST_ALLOWLIST"
        )

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as exc:
        raise MCPConfigurationError(
            f"MCP hostname '{hostname}' could not be resolved: {exc}"
        ) from exc

    if allow_private_network:
        return
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if any(
            (
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            )
        ):
            raise MCPConfigurationError(
                f"MCP hostname '{hostname}' resolves to a blocked address"
            )


def validate_stdio_command(command: str) -> str:
    """Resolve a stdio executable without invoking a shell."""
    configured_allowlist = [
        item.strip()
        for item in os.environ.get("MEMORIZZ_MCP_STDIO_ALLOWLIST", "").split(",")
        if item.strip()
    ]
    resolved = shutil.which(command)
    if not resolved:
        raise MCPConfigurationError(f"MCP stdio command '{command}' was not found")
    if configured_allowlist:
        names = {
            command,
            os.path.basename(command),
            resolved,
            os.path.basename(resolved),
        }
        if not any(
            fnmatch.fnmatch(candidate, pattern)
            for candidate in names
            for pattern in configured_allowlist
        ):
            raise MCPConfigurationError(
                f"MCP stdio command '{command}' is not in MEMORIZZ_MCP_STDIO_ALLOWLIST"
            )
    return resolved


def tool_is_mutating(
    tool_name: str,
    tool_metadata: Optional[Dict[str, Any]] = None,
    *,
    read_only_tools: Iterable[str] = (),
    mutation_tools: Iterable[str] = (),
) -> bool:
    """Whether a connected tool may change external data.

    A tool is read-only only when (a) the host's per-server ``read_only_tools``
    allowlist names it, (b) the server annotates it ``readOnlyHint=True``, or
    (c) the first word of its name is a read verb (``get``, ``list``,
    ``search``, ...). Everything else, including un-annotated names such as
    ``reply``, ``forward`` or ``confirm_booking``, is treated as a mutation
    and goes through approval. ``mutation_tools`` and ``destructiveHint``
    always win.
    """
    normalized_name = str(tool_name or "").strip()
    if normalized_name in set(mutation_tools):
        return True
    if normalized_name in set(read_only_tools):
        return False
    metadata = tool_metadata or {}
    annotations = metadata.get("annotations")
    if not isinstance(annotations, dict):
        annotations = {}
    read_only = annotations.get("readOnlyHint", annotations.get("read_only_hint"))
    destructive = annotations.get(
        "destructiveHint", annotations.get("destructive_hint")
    )
    if destructive is True:
        return True
    if read_only is True:
        return False
    if read_only is False:
        return True
    # MCP annotations are advisory and optional. Without one, the name decides:
    # any mutation verb anywhere makes it a mutation; otherwise the first verb
    # found while walking the segments must be a read verb (so a vendor or
    # product prefix such as "notion-search" or "memorizz_list_memories" does
    # not hide the read verb); anything else is treated as a mutation and the
    # host allowlist covers the rest.
    normalized = normalized_name.lower().replace("-", "_").replace(".", "_")
    segments = [part for part in normalized.split("_") if part]
    decisive = None
    for index, segment in enumerate(segments):
        if segment in _MUTATION_VERBS:
            return True
        if segment in _READ_ONLY_PREFIXES:
            decisive = index
            break
    if decisive is None:
        return True
    # "get_and_delete" / "search_then_send": a mutation verb chained to the
    # read verb with a connective is a mutation. Nouns such as "run" in
    # "get_harness_run" are not chained and stay read-only.
    for index in range(decisive + 1, len(segments)):
        if segments[index] in _MUTATION_VERBS and segments[index - 1] in _CONNECTIVES:
            return True
    return False


def enforce_tool_policy(
    server: MCPServerConfig,
    tool_name: str,
    approved: bool = False,
    tool_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Enforce allow/block lists and metadata-aware mutation approval."""
    if server.allowed_tools and tool_name not in server.allowed_tools:
        raise MCPPolicyError(
            f"MCP tool '{tool_name}' is not in the server allowlist",
            server_name=server.name,
        )
    if tool_name in server.blocked_tools:
        raise MCPPolicyError(
            f"MCP tool '{tool_name}' is blocked",
            server_name=server.name,
        )
    mutating = tool_is_mutating(
        tool_name,
        tool_metadata,
        read_only_tools=server.read_only_tools,
        mutation_tools=server.mutation_tools,
    )
    if server.require_approval and mutating and not approved:
        raise MCPPolicyError(
            (
                f"MCP tool '{tool_name}' can change external data and requires "
                "a durable host approval proposal before it can execute."
            ),
            server_name=server.name,
            code="approval_required",
        )


def names_only(mapping: Optional[Dict[str, Any]]) -> Iterable[str]:
    return sorted(str(key) for key in (mapping or {}).keys())
