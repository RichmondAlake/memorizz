# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Configuration and static bearer grants for the Memorizz MCP server."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

READ_SCOPE = "memorizz:read"
WRITE_SCOPE = "memorizz:write"
EXECUTE_SCOPE = "memorizz:execute"
ALL_SCOPES = (READ_SCOPE, WRITE_SCOPE, EXECUTE_SCOPE)
_PRINCIPAL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")


def _env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _env_csv(name: str) -> Optional[Set[str]]:
    value = os.environ.get(name)
    if value is None:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class StaticAPIKeyGrant:
    """One static bearer token and its stable tenant principal."""

    principal: str
    token: str
    scopes: tuple[str, ...] = ALL_SCOPES

    def __post_init__(self) -> None:
        if not _PRINCIPAL_PATTERN.fullmatch(self.principal):
            raise ValueError(
                "MCP server API-key principals must start with a letter or number "
                "and contain only letters, numbers, '.', '_', ':', '@', or '-'"
            )
        if self.token != self.token.strip() or any(
            character.isspace() for character in self.token
        ):
            raise ValueError(
                f"MCP server token for '{self.principal}' cannot contain whitespace"
            )
        if len(self.token) < 16:
            raise ValueError(
                f"MCP server token for '{self.principal}' must be at least 16 characters"
            )
        invalid = set(self.scopes).difference(ALL_SCOPES)
        if invalid:
            raise ValueError("Unsupported MCP server scopes: " + ", ".join(invalid))
        if READ_SCOPE not in self.scopes:
            raise ValueError(
                f"MCP server token for '{self.principal}' requires {READ_SCOPE}"
            )


def parse_api_key_grants(raw: Optional[str]) -> List[StaticAPIKeyGrant]:
    """Parse principal-to-token JSON without ever returning secrets publicly."""
    value = str(raw or "").strip()
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("MCP server API keys must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("MCP server API keys must be a JSON object")

    grants: List[StaticAPIKeyGrant] = []
    seen_tokens = set()
    for raw_principal, raw_grant in payload.items():
        principal = str(raw_principal or "").strip()
        if isinstance(raw_grant, str):
            token = raw_grant
            scopes = ALL_SCOPES
        elif isinstance(raw_grant, dict):
            token = str(raw_grant.get("token") or "")
            scopes_value = raw_grant.get("scopes") or ALL_SCOPES
            if not isinstance(scopes_value, list):
                raise ValueError(f"Scopes for '{principal}' must be an array")
            scopes = tuple(
                str(scope).strip() for scope in scopes_value if str(scope).strip()
            )
        else:
            raise ValueError(
                f"API-key grant for '{principal}' must be a token string or object"
            )
        if token in seen_tokens:
            raise ValueError("MCP server bearer tokens must be unique")
        seen_tokens.add(token)
        grants.append(
            StaticAPIKeyGrant(principal=principal, token=token, scopes=tuple(scopes))
        )
    return grants


@dataclass
class MemorizzMCPServerConfig:
    """Runtime policy and transport settings for the first-party server."""

    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8766
    path: str = "/mcp"
    public_url: Optional[str] = None
    stateless_http: bool = True
    allow_anonymous_http: bool = False
    allow_writes: Optional[bool] = None
    allow_trace_queries: bool = False
    allow_agent_execution: Optional[bool] = None
    allow_harness_execution: Optional[bool] = None
    harness_workspace_roots: Optional[Set[str]] = None
    api_key_grants: List[StaticAPIKeyGrant] = field(default_factory=list)
    exposed_agent_ids: Optional[Set[str]] = None
    max_result_items: int = 100
    max_text_chars: int = 50_000
    max_request_body_size: int = 4_194_304
    approval_ttl_seconds: int = 900

    def __post_init__(self) -> None:
        normalized = self.transport.strip().lower().replace("_", "-")
        if normalized in {"http", "streamablehttp"}:
            normalized = "streamable-http"
        if normalized not in {"stdio", "streamable-http"}:
            raise ValueError("MCP server transport must be stdio or streamable-http")
        self.transport = normalized
        if not (1 <= int(self.port) <= 65535):
            raise ValueError("MCP server port must be between 1 and 65535")
        self.port = int(self.port)
        self.path = "/" + self.path.strip().strip("/")
        if self.path == "/":
            raise ValueError("MCP server path cannot be the root path")
        if "\x00" in self.path or any(
            segment in {".", ".."} for segment in self.path.split("/")
        ):
            raise ValueError("MCP server path cannot contain traversal segments")
        self.max_result_items = max(1, min(int(self.max_result_items), 500))
        self.max_text_chars = max(1_024, min(int(self.max_text_chars), 1_000_000))
        self.max_request_body_size = max(
            1_024, min(int(self.max_request_body_size), 16_777_216)
        )
        self.approval_ttl_seconds = max(1, min(int(self.approval_ttl_seconds), 86_400))

        is_stdio = self.transport == "stdio"
        if self.allow_writes is None:
            self.allow_writes = is_stdio
        if self.allow_agent_execution is None:
            self.allow_agent_execution = is_stdio and bool(self.allow_writes)
        if self.allow_harness_execution is None:
            self.allow_harness_execution = is_stdio and bool(self.allow_agent_execution)
        if self.harness_workspace_roots is None:
            self.harness_workspace_roots = {os.getcwd()} if is_stdio else set()
        else:
            self.harness_workspace_roots = {
                os.path.realpath(os.path.expanduser(str(value).strip()))
                for value in self.harness_workspace_roots
                if str(value).strip()
            }

        if self.exposed_agent_ids is not None:
            self.exposed_agent_ids = {
                str(value).strip()
                for value in self.exposed_agent_ids
                if str(value).strip()
            }

        if self.allow_agent_execution and not self.allow_writes:
            raise ValueError(
                "Agent execution requires writes because every turn persists conversation memory"
            )
        if not is_stdio and self.allow_agent_execution and not self.exposed_agent_ids:
            raise ValueError(
                "Remote agent execution requires at least one explicit --agent-id"
            )
        if (
            not is_stdio
            and self.allow_harness_execution
            and not self.harness_workspace_roots
        ):
            raise ValueError(
                "Remote harness execution requires explicit harness workspace roots"
            )

        if not is_stdio and not self.api_key_grants and not self.allow_anonymous_http:
            raise ValueError(
                "Streamable HTTP requires MEMORIZZ_MCP_SERVER_API_KEYS or "
                "the explicit --allow-anonymous flag"
            )
        if not is_stdio and self.api_key_grants and not self.public_url:
            if self.host not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError(
                    "Authenticated non-loopback HTTP requires --public-url"
                )
            display_host = "127.0.0.1" if self.host == "::1" else self.host
            self.public_url = f"http://{display_host}:{self.port}"
        if self.public_url:
            self.public_url = self.public_url.rstrip("/")
            parsed = urlparse(self.public_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(
                    "MCP server public URL must be an absolute HTTP(S) URL"
                )
            if parsed.scheme != "https" and parsed.hostname not in {
                "127.0.0.1",
                "localhost",
                "::1",
            }:
                raise ValueError("Non-loopback MCP server public URLs must use HTTPS")

    @property
    def auth_required(self) -> bool:
        return self.transport != "stdio" and bool(self.api_key_grants)

    def public_dict(self) -> Dict[str, Any]:
        return {
            "transport": self.transport,
            "host": self.host if self.transport != "stdio" else None,
            "port": self.port if self.transport != "stdio" else None,
            "path": self.path if self.transport != "stdio" else None,
            "public_url": self.public_url,
            "stateless_http": self.stateless_http,
            "auth_required": self.auth_required,
            "configured_principals": len(self.api_key_grants),
            "allow_writes": bool(self.allow_writes),
            "allow_trace_queries": bool(self.allow_trace_queries),
            "allow_agent_execution": bool(self.allow_agent_execution),
            "allow_harness_execution": bool(self.allow_harness_execution),
            "harness_workspace_roots": sorted(self.harness_workspace_roots or []),
            "agent_creation": {
                "available": bool(self.transport == "stdio" and self.allow_writes),
                "transport": "stdio",
                "remote_http": False,
            },
            "agent_management": {
                "available": bool(self.transport == "stdio" and self.allow_writes),
                "operations": ["create", "update", "delete"],
                "transport": "stdio",
                "remote_http": False,
            },
            "exposed_agent_ids": sorted(self.exposed_agent_ids or []),
            "max_result_items": self.max_result_items,
            "max_text_chars": self.max_text_chars,
            "durable_approvals": True,
            "approval_ttl_seconds": self.approval_ttl_seconds,
        }

    @classmethod
    def from_env(cls, **overrides: Any) -> "MemorizzMCPServerConfig":
        raw_keys = os.environ.get("MEMORIZZ_MCP_SERVER_API_KEYS")
        grants = parse_api_key_grants(raw_keys)
        values: Dict[str, Any] = {
            "transport": os.environ.get("MEMORIZZ_MCP_SERVER_TRANSPORT", "stdio"),
            "host": os.environ.get("MEMORIZZ_MCP_SERVER_HOST", "127.0.0.1"),
            "port": int(os.environ.get("MEMORIZZ_MCP_SERVER_PORT", "8766")),
            "path": os.environ.get("MEMORIZZ_MCP_SERVER_PATH", "/mcp"),
            "public_url": os.environ.get("MEMORIZZ_MCP_SERVER_PUBLIC_URL"),
            "stateless_http": _env_bool("MEMORIZZ_MCP_SERVER_STATELESS_HTTP"),
            "allow_anonymous_http": _env_bool("MEMORIZZ_MCP_SERVER_ALLOW_ANONYMOUS"),
            "allow_writes": _env_bool("MEMORIZZ_MCP_SERVER_ALLOW_WRITES"),
            "allow_trace_queries": _env_bool("MEMORIZZ_MCP_SERVER_ALLOW_TRACE_QUERIES"),
            "allow_agent_execution": _env_bool(
                "MEMORIZZ_MCP_SERVER_ALLOW_AGENT_EXECUTION"
            ),
            "allow_harness_execution": _env_bool(
                "MEMORIZZ_MCP_SERVER_ALLOW_HARNESS_EXECUTION"
            ),
            "harness_workspace_roots": _env_csv(
                "MEMORIZZ_MCP_SERVER_HARNESS_WORKSPACE_ROOTS"
            ),
            "exposed_agent_ids": _env_csv("MEMORIZZ_MCP_SERVER_AGENT_IDS"),
            "approval_ttl_seconds": int(
                os.environ.get("MEMORIZZ_MCP_SERVER_APPROVAL_TTL", "900")
            ),
            "api_key_grants": grants,
        }
        values = {key: value for key, value in values.items() if value is not None}
        values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        return cls(**values)
