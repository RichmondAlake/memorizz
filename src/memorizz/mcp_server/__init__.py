# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""First-party MCP server exposing Memorizz agents and memory."""

from importlib import import_module

from .config import MemorizzMCPServerConfig, StaticAPIKeyGrant

# Loaded on first use, so the config and the connected-apps store
# (``oauth_store``) import without the optional MCP SDK.
_LAZY = {
    "MemorizzRuntime": "runtime",
    "create_memorizz_http_app": "server",
    "create_memorizz_mcp_server": "server",
    "run_memorizz_mcp_server": "server",
}

__all__ = [
    "MemorizzMCPServerConfig",
    "MemorizzRuntime",
    "StaticAPIKeyGrant",
    "create_memorizz_http_app",
    "create_memorizz_mcp_server",
    "run_memorizz_mcp_server",
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)
