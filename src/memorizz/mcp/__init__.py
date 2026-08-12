# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""First-class MCP client support for Memorizz."""

from .credentials import CredentialStore, EncryptedFileCredentialStore
from .errors import (
    MCPAuthorizationRequired,
    MCPClientError,
    MCPConfigurationError,
    MCPPolicyError,
)
from .manager import MCPClientManager
from .models import MCPAuthConfig, MCPServerConfig

__all__ = [
    "CredentialStore",
    "EncryptedFileCredentialStore",
    "MCPAuthConfig",
    "MCPAuthorizationRequired",
    "MCPClientError",
    "MCPClientManager",
    "MCPConfigurationError",
    "MCPPolicyError",
    "MCPServerConfig",
]
