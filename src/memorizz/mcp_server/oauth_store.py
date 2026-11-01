# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Which cloud apps may use this MemoRizz memory, and their sign-in state.

Pairing codes, approved apps (grants), sign-in requests waiting for approval,
and the apps' OAuth codes and tokens live in one SQLite file next to the
approvals. Codes and tokens are stored only as SHA-256 hashes. This module
needs no MCP SDK, so the CLI and the local UI can manage apps without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .._env_io import memorizz_home

PAIRING_TTL_SECONDS = 600
SIGN_IN_TTL_SECONDS = 900
MAX_PAIRING_ATTEMPTS = 5
# Wrong codes across all sign-ins: past this many in the window, the approval
# page stops checking codes for a while, so guessing one never pays.
MAX_PAIRING_FAILURES = 30
PAIRING_FAILURE_WINDOW_SECONDS = 600
# No 0/O or 1/I, so a code read aloud or off a phone is typed right.
_PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
# last_used_at is written at most this often per app.
_TOUCH_SECONDS = 60


def _digest(secret: str) -> str:
    return hashlib.sha256(str(secret).encode("utf-8")).hexdigest()


def normalize_pairing_code(value: str) -> str:
    """``abcd-efgh`` and ``ABCDEFGH`` are the same code."""
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _scopes(value: Iterable[str]) -> str:
    return " ".join(dict.fromkeys(str(s).strip() for s in value if str(s).strip()))


class OAuthStore:
    """SQLite store for connected apps; safe across processes (WAL)."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path or (memorizz_home() / "mcp_apps.sqlite3")).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS pairing_codes (
                    code_hash TEXT PRIMARY KEY,
                    principal TEXT NOT NULL,
                    scopes TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    used_at REAL
                );
                CREATE TABLE IF NOT EXISTS clients (
                    client_id TEXT PRIMARY KEY,
                    info_json TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sign_in_requests (
                    request_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS grants (
                    grant_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    app_name TEXT NOT NULL,
                    principal TEXT NOT NULL,
                    scopes TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used_at REAL,
                    revoked_at REAL
                );
                CREATE TABLE IF NOT EXISTS codes (
                    code_hash TEXT PRIMARY KEY,
                    grant_id TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tokens (
                    token_hash TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    grant_id TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    scopes TEXT NOT NULL,
                    resource TEXT,
                    expires_at REAL
                );
                CREATE INDEX IF NOT EXISTS tokens_grant_idx ON tokens(grant_id);
                CREATE TABLE IF NOT EXISTS pairing_failures (at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS assertion_ids (
                    client_id TEXT NOT NULL,
                    jti TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (client_id, jti)
                );
                """
            )

    # Pairing codes: the owner's proof on the approval page.

    def create_pairing_code(
        self,
        principal: str,
        scopes: Iterable[str],
        *,
        ttl_seconds: int = PAIRING_TTL_SECONDS,
    ) -> Tuple[str, float]:
        """A single-use code (shown as ``ABCD-EFGH``) and when it expires."""
        raw = "".join(secrets.choice(_PAIRING_ALPHABET) for _ in range(8))
        now = time.time()
        expires_at = now + max(60, int(ttl_seconds))
        with self._lock, closing(self._connect()) as db:
            db.execute("DELETE FROM pairing_codes WHERE expires_at < ?", (now,))
            db.execute(
                "INSERT INTO pairing_codes VALUES (?, ?, ?, ?, ?, NULL)",
                (_digest(raw), principal, _scopes(scopes), now, expires_at),
            )
        return f"{raw[:4]}-{raw[4:]}", expires_at

    def redeem_pairing_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Use a code once; None when it is wrong, used or expired."""
        normalized = normalize_pairing_code(code)
        digest = _digest(normalized)
        now = time.time()
        with self._lock, closing(self._connect()) as db:
            claimed = db.execute(
                "UPDATE pairing_codes SET used_at = ? "
                "WHERE code_hash = ? AND used_at IS NULL AND expires_at > ?",
                (now, digest, now),
            ).rowcount
            if claimed != 1:
                db.execute("INSERT INTO pairing_failures VALUES (?)", (now,))
                return None
            row = db.execute(
                "SELECT principal, scopes FROM pairing_codes WHERE code_hash = ?",
                (digest,),
            ).fetchone()
        return {"principal": row["principal"], "scopes": row["scopes"].split()}

    def pairing_locked(self) -> bool:
        """True after too many wrong codes recently, from any sign-in."""
        since = time.time() - PAIRING_FAILURE_WINDOW_SECONDS
        with self._lock, closing(self._connect()) as db:
            db.execute("DELETE FROM pairing_failures WHERE at < ?", (since,))
            (count,) = db.execute("SELECT COUNT(*) FROM pairing_failures").fetchone()
        return count >= MAX_PAIRING_FAILURES

    # OAuth clients: registered (DCR) or read from a client metadata document.

    def save_client(self, client_id: str, info: Dict[str, Any], kind: str) -> None:
        with self._lock, closing(self._connect()) as db:
            db.execute(
                "INSERT OR REPLACE INTO clients VALUES (?, ?, ?, ?)",
                (client_id, json.dumps(info), kind, time.time()),
            )

    def client(self, client_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT info_json, kind, updated_at FROM clients WHERE client_id = ?",
                (client_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "info": json.loads(row["info_json"]),
            "kind": row["kind"],
            "updated_at": row["updated_at"],
        }

    # Sign-in requests waiting on the approval page.

    def create_sign_in_request(self, client_id: str, params: Dict[str, Any]) -> str:
        request_id = secrets.token_urlsafe(24)
        now = time.time()
        with self._lock, closing(self._connect()) as db:
            db.execute(
                "DELETE FROM sign_in_requests WHERE created_at < ?",
                (now - SIGN_IN_TTL_SECONDS,),
            )
            db.execute(
                "INSERT INTO sign_in_requests VALUES (?, ?, ?, ?, 0)",
                (request_id, client_id, json.dumps(params), now),
            )
        return request_id

    def sign_in_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT * FROM sign_in_requests WHERE request_id = ? AND created_at > ?",
                (str(request_id or ""), time.time() - SIGN_IN_TTL_SECONDS),
            ).fetchone()
        if row is None:
            return None
        return {
            "request_id": row["request_id"],
            "client_id": row["client_id"],
            "params": json.loads(row["params_json"]),
            "attempts": row["attempts"],
        }

    def count_pairing_attempt(self, request_id: str) -> int:
        with self._lock, closing(self._connect()) as db:
            db.execute(
                "UPDATE sign_in_requests SET attempts = attempts + 1 WHERE request_id = ?",
                (request_id,),
            )
            row = db.execute(
                "SELECT attempts FROM sign_in_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return int(row["attempts"]) if row else MAX_PAIRING_ATTEMPTS + 1

    def finish_sign_in_request(self, request_id: str) -> None:
        with self._lock, closing(self._connect()) as db:
            db.execute(
                "DELETE FROM sign_in_requests WHERE request_id = ?", (request_id,)
            )

    # Grants: one per approved app.

    def create_grant(
        self, client_id: str, app_name: str, principal: str, scopes: Iterable[str]
    ) -> str:
        grant_id = "app_" + uuid.uuid4().hex[:16]
        with self._lock, closing(self._connect()) as db:
            db.execute(
                "INSERT INTO grants VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                (
                    grant_id,
                    client_id,
                    app_name,
                    principal,
                    _scopes(scopes),
                    time.time(),
                ),
            )
        return grant_id

    @staticmethod
    def _grant(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "grant_id": row["grant_id"],
            "client_id": row["client_id"],
            "app_name": row["app_name"],
            "principal": row["principal"],
            "scopes": row["scopes"].split(),
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
            "revoked_at": row["revoked_at"],
        }

    def grant(self, grant_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT * FROM grants WHERE grant_id = ?", (grant_id,)
            ).fetchone()
        return self._grant(row) if row else None

    def list_grants(self, *, include_revoked: bool = False) -> List[Dict[str, Any]]:
        query = "SELECT * FROM grants"
        if not include_revoked:
            query += " WHERE revoked_at IS NULL"
        with closing(self._connect()) as db:
            rows = db.execute(query + " ORDER BY created_at DESC").fetchall()
        return [self._grant(row) for row in rows]

    def revoke_grant(self, grant_id: str) -> bool:
        """Cut an app off now: its tokens and unused codes stop working."""
        with self._lock, closing(self._connect()) as db:
            changed = db.execute(
                "UPDATE grants SET revoked_at = ? WHERE grant_id = ? AND revoked_at IS NULL",
                (time.time(), grant_id),
            ).rowcount
            db.execute("DELETE FROM tokens WHERE grant_id = ?", (grant_id,))
            db.execute("DELETE FROM codes WHERE grant_id = ?", (grant_id,))
        return changed == 1

    def touch_grant(self, grant_id: str) -> None:
        now = time.time()
        with self._lock, closing(self._connect()) as db:
            db.execute(
                "UPDATE grants SET last_used_at = ? WHERE grant_id = ? "
                "AND (last_used_at IS NULL OR last_used_at < ?)",
                (now, grant_id, now - _TOUCH_SECONDS),
            )

    # Authorization codes (single use, short-lived).

    def save_code(
        self,
        code: str,
        grant_id: str,
        client_id: str,
        payload: Dict[str, Any],
        expires_at: float,
    ) -> None:
        with self._lock, closing(self._connect()) as db:
            db.execute("DELETE FROM codes WHERE expires_at < ?", (time.time(),))
            db.execute(
                "INSERT INTO codes VALUES (?, ?, ?, ?, ?)",
                (_digest(code), grant_id, client_id, json.dumps(payload), expires_at),
            )

    def code(self, code: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT c.* FROM codes c JOIN grants g ON g.grant_id = c.grant_id "
                "WHERE c.code_hash = ? AND g.revoked_at IS NULL",
                (_digest(code),),
            ).fetchone()
        if row is None:
            return None
        return {
            "grant_id": row["grant_id"],
            "client_id": row["client_id"],
            "payload": json.loads(row["payload_json"]),
            "expires_at": row["expires_at"],
        }

    def take_code(self, code: str) -> bool:
        """Delete a code; True only for the one caller that used it."""
        with self._lock, closing(self._connect()) as db:
            return (
                db.execute(
                    "DELETE FROM codes WHERE code_hash = ?", (_digest(code),)
                ).rowcount
                == 1
            )

    # Access and refresh tokens.

    def save_token(
        self,
        token: str,
        kind: str,
        grant_id: str,
        client_id: str,
        scopes: Iterable[str],
        resource: Optional[str],
        expires_at: Optional[float],
    ) -> None:
        with self._lock, closing(self._connect()) as db:
            db.execute(
                "DELETE FROM tokens WHERE expires_at IS NOT NULL AND expires_at < ?",
                (time.time(),),
            )
            db.execute(
                "INSERT INTO tokens VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _digest(token),
                    kind,
                    grant_id,
                    client_id,
                    _scopes(scopes),
                    resource,
                    expires_at,
                ),
            )

    def token(self, token: str, kind: str) -> Optional[Dict[str, Any]]:
        """A live token of an app that is still approved, with its grant."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT t.*, g.principal, g.app_name FROM tokens t "
                "JOIN grants g ON g.grant_id = t.grant_id "
                "WHERE t.token_hash = ? AND t.kind = ? AND g.revoked_at IS NULL "
                "AND (t.expires_at IS NULL OR t.expires_at > ?)",
                (_digest(token), kind, time.time()),
            ).fetchone()
        if row is None:
            return None
        return {
            "grant_id": row["grant_id"],
            "client_id": row["client_id"],
            "scopes": row["scopes"].split(),
            "resource": row["resource"],
            "expires_at": row["expires_at"],
            "principal": row["principal"],
            "app_name": row["app_name"],
        }

    def delete_token(self, token: str) -> bool:
        with self._lock, closing(self._connect()) as db:
            return (
                db.execute(
                    "DELETE FROM tokens WHERE token_hash = ?", (_digest(token),)
                ).rowcount
                == 1
            )

    # Signed client assertions are single use (RFC 7523 jti).

    def use_assertion_id(self, client_id: str, jti: str, expires_at: float) -> bool:
        now = time.time()
        with self._lock, closing(self._connect()) as db:
            db.execute("DELETE FROM assertion_ids WHERE expires_at < ?", (now,))
            return (
                db.execute(
                    "INSERT OR IGNORE INTO assertion_ids VALUES (?, ?, ?)",
                    (client_id, jti, max(expires_at, now + 60)),
                ).rowcount
                == 1
            )
