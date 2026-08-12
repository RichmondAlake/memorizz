# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Public, per-agent MCP configuration used by the CLI and local UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as cfg


def _safe_owner_id(owner_id: str) -> str:
    value = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(owner_id or "cli-default")
    )
    return value[:160] or "cli-default"


def current_owner_id() -> str:
    return str(cfg.load_state().get("agent_id") or "cli-default")


def server_config_path(owner_id: Optional[str] = None) -> Path:
    return (
        cfg.memorizz_home()
        / "mcp_servers"
        / f"{_safe_owner_id(owner_id or current_owner_id())}.json"
    )


def has_server_config(owner_id: Optional[str] = None) -> bool:
    return server_config_path(owner_id).is_file()


def load_servers(owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
    path = server_config_path(owner_id)
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read MCP configuration at {path}: {exc}") from exc
    servers = value.get("servers") if isinstance(value, dict) else value
    if not isinstance(servers, list):
        raise ValueError(f"MCP configuration at {path} must contain a servers array")
    return [dict(server) for server in servers if isinstance(server, dict)]


def save_servers(servers: List[Dict[str, Any]], owner_id: Optional[str] = None) -> Path:
    """Atomically save secret-free MCP configuration with owner-only access."""
    path = server_config_path(owner_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps({"version": 1, "servers": servers}, indent=2) + "\n"
    descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def build_manager(owner_id: Optional[str] = None):
    from ..mcp import MCPClientManager

    resolved_owner = str(owner_id or current_owner_id())
    return MCPClientManager(
        owner_id=resolved_owner,
        servers=load_servers(resolved_owner),
    )
