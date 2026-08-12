# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Privacy-preserving local audit log for MCP tool calls."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .._env_io import memorizz_home

logger = logging.getLogger(__name__)


class MCPAuditLogger:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or memorizz_home() / "mcp_audit.jsonl")
        self._lock = threading.RLock()

    def record(
        self,
        *,
        owner_id: str,
        server_name: str,
        operation: str,
        ok: bool,
        duration_ms: int,
        tool_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ) -> None:
        canonical_arguments = json.dumps(
            arguments or {}, sort_keys=True, separators=(",", ":"), default=str
        )
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "owner_id": owner_id,
            "server_name": server_name,
            "operation": operation,
            "tool_name": tool_name,
            "ok": bool(ok),
            "duration_ms": max(0, int(duration_ms)),
            "arguments_sha256": hashlib.sha256(
                canonical_arguments.encode("utf-8")
            ).hexdigest(),
            "error_code": error_code,
        }
        line = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
                )
                try:
                    os.write(descriptor, line.encode("utf-8"))
                finally:
                    os.close(descriptor)
        except OSError as exc:
            # Auditing is best-effort: a read-only home must not turn a
            # successful MCP operation into a user-visible failure.
            logger.warning("Could not append the MCP audit log: %s", exc)
