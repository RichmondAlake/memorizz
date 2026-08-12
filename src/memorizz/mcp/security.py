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

from .errors import MCPConfigurationError, MCPPolicyError
from .models import MCPServerConfig

_MUTATION_PREFIXES = (
    "add",
    "archive",
    "cancel",
    "create",
    "delete",
    "edit",
    "execute",
    "forget",
    "invite",
    "move",
    "publish",
    "remove",
    "respond",
    "send",
    "set",
    "share",
    "store",
    "update",
    "upload",
    "write",
)

_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "bearer_token",
    "client_secret",
    "code_verifier",
    "cookie",
    "password",
    "refresh_token",
    "secret",
    "token",
}


def redact(value: Any) -> Any:
    """Recursively redact known credential-bearing fields."""
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower()
            result[str(key)] = "***" if normalized in _SENSITIVE_KEYS else redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


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
    """Classify side effects using host policy, MCP annotations, then fallback."""
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
    # MCP annotations are advisory and optional. Unknown tools fail safe to a
    # conservative name heuristic unless the host explicitly classifies them.
    normalized = str(tool_name or "").strip().lower().replace("-", "_")
    segments = [part for part in normalized.replace(".", "_").split("_") if part]
    return any(segment in _MUTATION_PREFIXES for segment in segments)


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
