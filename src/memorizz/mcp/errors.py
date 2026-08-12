# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Error types and safe serialization for MCP client operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MCPClientError(Exception):
    """Structured MCP failure safe to return through agent tools and APIs."""

    message: str
    code: str = "mcp_error"
    retryable: bool = False
    server_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ok": False,
            "error": self.message,
            "error_code": self.code,
            "retryable": self.retryable,
        }
        if self.server_name:
            payload["server_name"] = self.server_name
        if self.details:
            payload["details"] = self.details
        return payload


class MCPConfigurationError(MCPClientError):
    def __init__(self, message: str, server_name: Optional[str] = None):
        super().__init__(
            message=message,
            code="configuration_error",
            retryable=False,
            server_name=server_name,
        )


class MCPAuthorizationRequired(MCPClientError):
    def __init__(self, message: str, server_name: Optional[str] = None):
        super().__init__(
            message=message,
            code="authorization_required",
            retryable=False,
            server_name=server_name,
        )


class MCPPolicyError(MCPClientError):
    def __init__(
        self,
        message: str,
        server_name: Optional[str] = None,
        code: str = "policy_denied",
    ):
        super().__init__(
            message=message,
            code=code,
            retryable=False,
            server_name=server_name,
        )
