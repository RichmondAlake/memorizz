"""Which MemoRizz processes on this machine use which memory store.

A swap changes models for everything that shares a store, and a process
reads its embedding model when it starts: after an embedding swap, any other
process on the same store must restart or it searches the re-embedded
vectors with the old model (same size, so nothing fails; results are just
wrong). Each MemoRizz process (UI, MCP server, CLI) records here the store
and embedding model it uses, in a small file in the user's temp folder
(machine-local runtime state, never user data); entries of processes that
have exited are dropped when read. Older MemoRizz versions do not record
themselves; ``unregistered`` lists them from the process table.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_DIR = Path(tempfile.gettempdir()) / "memorizz-clients"


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _unwrap(provider: Any) -> Any:
    """The provider behind a read-only proxy."""
    for _ in range(3):
        inner = getattr(provider, "_provider", None) or getattr(
            provider, "_inner", None
        )
        if inner is None or inner is provider:
            break
        provider = inner
    return provider


def store_identity(provider: Any) -> Dict[str, str]:
    """A stable, non-secret name for a memory store: Oracle user@dsn, a
    filesystem root, MongoDB host and database (credentials removed)."""
    provider = _unwrap(provider)
    config = getattr(provider, "config", None)
    kind = type(provider).__name__.replace("Provider", "").lower() or "unknown"
    if config is None:
        return {"type": kind, "id": kind}
    user, dsn = getattr(config, "user", None), getattr(config, "dsn", None)
    if user and dsn:
        return {"type": "oracle", "id": f"{str(user).upper()}@{dsn}"}
    root = getattr(config, "root_path", None)
    if root:
        return {"type": "filesystem", "id": str(Path(str(root)).expanduser().resolve())}
    uri = getattr(config, "uri", None) or getattr(config, "connection_string", None)
    if uri:
        host = (
            re.sub(r"^[a-z+]+://([^@/]*@)?", "", str(uri)).split("/")[0].split("?")[0]
        )
        database = (
            getattr(config, "db_name", None)
            or getattr(config, "database_name", None)
            or ""
        )
        return {"type": "mongodb", "id": f"{host}/{database}".rstrip("/")}
    return {"type": kind, "id": kind}


def register(kind: str, provider: Any, **extra: Any) -> None:
    """Record that this process uses ``provider`` (call again after a change,
    such as a new connection or an embedding swap). Never raises."""
    try:
        from .._env_io import memorizz_home

        _DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "pid": os.getpid(),
            "kind": kind,
            "command": " ".join(sys.argv[:3])[:200],
            "home": str(memorizz_home()),
            "store": store_identity(provider),
            "embedding": {
                "provider": os.environ.get("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", ""),
                "model": os.environ.get("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", ""),
            },
            "reranker": os.environ.get("MEMORIZZ_RERANKER_MODEL", ""),
            "updated_at": time.time(),
            **extra,
        }
        path = _DIR / f"{os.getpid()}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:  # bookkeeping must never stop a process
        pass


def refresh() -> None:
    """Re-record this process's models (after a swap changed them)."""
    path = _DIR / f"{os.getpid()}.json"
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    entry["embedding"] = {
        "provider": os.environ.get("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", ""),
        "model": os.environ.get("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", ""),
    }
    entry["reranker"] = os.environ.get("MEMORIZZ_RERANKER_MODEL", "")
    entry["updated_at"] = time.time()
    try:
        path.write_text(json.dumps(entry, default=str), encoding="utf-8")
    except OSError:
        pass


def clients(store_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Running MemoRizz processes that recorded their store (optionally one
    store), oldest first."""
    out = []
    for path in sorted(_DIR.glob("*.json")) if _DIR.is_dir() else []:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not _alive(entry.get("pid", -1)):
            path.unlink(missing_ok=True)
            continue
        if store_id is None or entry.get("store", {}).get("id") == store_id:
            out.append(entry)
    return sorted(out, key=lambda e: e.get("updated_at", 0))


def _server(store: Optional[Dict[str, str]]) -> Optional[str]:
    """``host:port`` of a database store (Oracle, MongoDB), if it has one."""
    if not store or store.get("type") not in ("oracle", "mongodb"):
        return None
    address = str(store.get("id", "")).split("@")[-1].split("/")[0]
    return address if re.match(r"^[\w.\-\[\]:]+:\d+$", address) else None


def _connected_to(address: str) -> set:
    """Process ids with an open TCP connection to ``address``."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP@{address}", "-sTCP:ESTABLISHED"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return set()
    return {int(pid) for pid in out.split() if pid.isdigit()}


def unregistered(store: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """MemoRizz processes running on this machine that did not record their
    store (older versions). With a database ``store``, those connected to
    the same database server are marked ``same_server``: they may use it."""
    known = {e["pid"] for e in clients()}
    address = _server(store)
    nearby = _connected_to(address) if address else set()
    try:
        table = subprocess.run(
            ["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return []
    out = []
    for line in table.splitlines():
        pid, _, command = line.strip().partition(" ")
        match = re.search(r"\bmemorizz (ui|mcp serve|mcp|run|chat|serve)\b", command)
        if (
            not match
            or not pid.isdigit()
            or int(pid) in known
            or int(pid) == os.getpid()
        ):
            continue
        port = re.search(r"--port[ =](\d+)", command)
        out.append(
            {
                "pid": int(pid),
                "kind": match.group(1).split()[0],
                "port": int(port.group(1)) if port else None,
                "same_server": int(pid) in nearby,
            }
        )
    return sorted(out, key=lambda p: not p["same_server"])


__all__ = ["clients", "refresh", "register", "store_identity", "unregistered"]
