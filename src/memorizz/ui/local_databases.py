# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Databases on this computer, for the connect page.

Oracle and MongoDB in Docker containers, running or stopped, found by the
port they publish (1521, 27017), whatever image they were built from; and
the addresses the environment names (``ORACLE_DSN``, ``MONGODB_URI``). Only
container names, states, images and port bindings are read, never a
container's environment, which holds its passwords.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from .connection import mask_uri

# The container port each database listens on.
_PORTS = {"oracle": "1521/tcp", "mongodb": "27017/tcp"}
# Docker is asked at most this often; the page is opened a lot more.
_TTL_SECONDS = 15.0
_cache: Dict[str, Any] = {"at": 0.0, "value": None}
_lock = threading.Lock()


def _service_name(image: str) -> str:
    """The pluggable database an Oracle image starts with."""
    image = image.lower()
    if "enterprise" in image:
        return "ORCLPDB1"
    if "-xe" in image or "/xe" in image or ":xe" in image or image.endswith("xe"):
        return "XEPDB1"
    return "FREEPDB1"


def _short_image(image: str) -> str:
    if image.startswith("sha256:"):
        return "image " + image[7:19]
    return image.rsplit("/", 1)[-1]


def _containers() -> List[Dict[str, Any]]:
    """Containers publishing a database port: name, state, image, port."""
    from .docker_oracle import _run

    code, listed, _ = _run(
        ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.State}}\t{{.Status}}"],
        timeout=3,
    )
    if code != 0 or not listed.strip():
        return []
    rows = {}
    for line in listed.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows[parts[0]] = {
                "state": "running" if parts[1].lower() == "running" else "stopped",
                "status": parts[2] if len(parts) > 2 else "",
            }
    code, inspected, _ = _run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.Name}}\t{{.Config.Image}}\t{{json .HostConfig.PortBindings}}",
            *rows,
        ],
        timeout=3,
    )
    if code != 0:
        return []
    found = []
    for line in inspected.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, image = parts[0].lstrip("/"), parts[1]
        try:
            bindings = json.loads(parts[2]) or {}
        except ValueError:
            continue
        for kind, port in _PORTS.items():
            published = [b for b in bindings.get(port) or [] if b.get("HostPort")]
            if published:
                found.append(
                    {
                        "kind": kind,
                        "name": name,
                        "image": image,
                        "host_port": published[0]["HostPort"],
                        **rows.get(name, {"state": "stopped", "status": ""}),
                    }
                )
    return found


def _discover() -> Dict[str, List[Dict[str, Any]]]:
    databases: Dict[str, List[Dict[str, Any]]] = {"oracle": [], "mongodb": []}
    for container in _containers():
        entry = {
            "label": container["name"],
            "source": "docker",
            "state": container["state"],
            "status": container["status"],
            "image": _short_image(container["image"]),
        }
        if container["kind"] == "oracle":
            entry["dsn"] = (
                f"localhost:{container['host_port']}/"
                f"{_service_name(container['image'])}"
            )
        else:
            entry["uri"] = f"mongodb://localhost:{container['host_port']}"
        if container["state"] != "running":
            entry["hint"] = f"docker start {container['name']}"
        databases[container["kind"]].append(entry)
    for kind in databases:
        databases[kind].sort(key=lambda db: (db["state"] != "running", db["label"]))

    dsn = os.environ.get("ORACLE_DSN", "").strip()
    if dsn and not any(db.get("dsn") == dsn for db in databases["oracle"]):
        databases["oracle"].insert(
            0,
            {
                "label": "ORACLE_DSN",
                "source": "environment",
                "dsn": dsn,
                "user": os.environ.get("ORACLE_USER", "").strip(),
            },
        )
    if os.environ.get("MONGODB_URI", "").strip():
        # The URI can hold a password: show where it points, and leave the
        # field empty so the server reads it from the environment.
        databases["mongodb"].insert(
            0,
            {
                "label": "MONGODB_URI",
                "source": "environment",
                "uri": "",
                "display": mask_uri(os.environ["MONGODB_URI"]),
            },
        )
    return databases


def detect_local_databases(
    *, current: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Oracle and MongoDB databases on this computer; the one the UI is
    connected to (``current``, its connection info) is marked."""
    with _lock:
        if _cache["value"] is None or time.monotonic() - _cache["at"] > _TTL_SECONDS:
            try:
                _cache["value"] = _discover()
            except Exception:
                _cache["value"] = {"oracle": [], "mongodb": []}
            _cache["at"] = time.monotonic()
        databases = {
            kind: [dict(db) for db in rows] for kind, rows in _cache["value"].items()
        }
    connected_dsn = str((current or {}).get("dsn") or "")
    for db in databases["oracle"]:
        db["current"] = bool(connected_dsn) and db.get("dsn") == connected_dsn
    return databases


__all__ = ["detect_local_databases"]
