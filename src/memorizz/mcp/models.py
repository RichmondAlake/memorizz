# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Typed, secret-free MCP server configuration models."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .errors import MCPConfigurationError

MCPTransport = Literal["stdio", "streamable_http", "sse"]
MCPAuthType = Literal["none", "bearer", "oauth"]

_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class MCPAuthConfig(BaseModel):
    """Public authentication metadata; secret material lives in a credential store."""

    type: MCPAuthType = "none"
    credential_ref: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    redirect_uri: Optional[str] = None
    client_id: Optional[str] = None
    client_metadata_url: Optional[str] = None
    token_endpoint_auth_method: Optional[
        Literal["none", "client_secret_post", "client_secret_basic"]
    ] = None

    @field_validator("scopes", mode="before")
    @classmethod
    def _normalize_scopes(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item for item in value.split() if item]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("auth.scopes must be a list or space-separated string")


class MCPServerConfig(BaseModel):
    """Validated MCP connection configuration with no embedded credentials."""

    name: str
    transport: MCPTransport = "stdio"
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    cwd: Optional[str] = None
    env_keys: List[str] = Field(default_factory=list)
    header_names: List[str] = Field(default_factory=list)
    timeout: int = Field(default=30, ge=1, le=300)
    connect_timeout: int = Field(default=10, ge=1, le=120)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_result_bytes: int = Field(default=2_000_000, ge=1_024, le=10_000_000)
    enabled: bool = True
    allow_private_network: bool = False
    allowed_tools: List[str] = Field(default_factory=list)
    blocked_tools: List[str] = Field(default_factory=list)
    read_only_tools: List[str] = Field(default_factory=list)
    mutation_tools: List[str] = Field(default_factory=list)
    require_approval: bool = True
    auth: MCPAuthConfig = Field(default_factory=MCPAuthConfig)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not _SERVER_NAME_RE.fullmatch(normalized):
            raise ValueError(
                "name must start with a letter or number and contain only "
                "letters, numbers, '.', '_' or '-' (maximum 128 characters)"
            )
        return normalized

    @field_validator("transport", mode="before")
    @classmethod
    def _normalize_transport(cls, value: Any) -> str:
        normalized = str(value or "stdio").strip().lower().replace("-", "_")
        if normalized in {"http", "streamablehttp"}:
            return "streamable_http"
        return normalized

    @field_validator(
        "args",
        "env_keys",
        "header_names",
        "allowed_tools",
        "blocked_tools",
        "read_only_tools",
        "mutation_tools",
        mode="before",
    )
    @classmethod
    def _normalize_string_lists(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("value must be an array")
        result: List[str] = []
        seen = set()
        for item in value:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport == "stdio":
            if not self.command or not self.command.strip():
                raise ValueError("stdio transport requires command")
            self.command = self.command.strip()
            self.url = None
        else:
            if not self.url or not self.url.strip():
                raise ValueError(f"{self.transport} transport requires url")
            self.url = self.url.strip()
            self.command = None
            self.args = []
            self.cwd = None
        overlap = set(self.allowed_tools).intersection(self.blocked_tools)
        if overlap:
            raise ValueError(
                "tools cannot be both allowed and blocked: "
                + ", ".join(sorted(overlap))
            )
        policy_overlap = set(self.read_only_tools).intersection(self.mutation_tools)
        if policy_overlap:
            raise ValueError(
                "tools cannot be both read-only and mutating: "
                + ", ".join(sorted(policy_overlap))
            )
        return self

    def public_dict(self) -> Dict[str, Any]:
        """Return the persistence-safe representation."""
        return self.model_dump(mode="json", exclude_none=True)


def parse_server_configs(
    servers: Any,
) -> List[MCPServerConfig]:
    """Parse one or many public configurations and reject duplicates."""
    if not servers:
        return []
    values = [servers] if isinstance(servers, dict) else servers
    if not isinstance(values, list):
        raise MCPConfigurationError("MCP servers must be an object or an array")

    parsed: List[MCPServerConfig] = []
    seen = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise MCPConfigurationError(
                f"MCP server entry {index + 1} must be an object"
            )
        try:
            config = MCPServerConfig.model_validate(value)
        except Exception as exc:
            name = str(value.get("name") or f"entry {index + 1}")
            raise MCPConfigurationError(
                f"Invalid MCP server '{name}': {exc}", server_name=name
            ) from exc
        if config.name in seen:
            raise MCPConfigurationError(
                f"MCP server name '{config.name}' is duplicated",
                server_name=config.name,
            )
        seen.add(config.name)
        parsed.append(config)
    return parsed
