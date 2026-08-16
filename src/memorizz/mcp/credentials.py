# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Encrypted MCP credentials and an adapter for the official SDK OAuth store."""

from __future__ import annotations

import base64
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Protocol

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # MCP is an optional ``memorizz[mcp]`` feature.
    Fernet = None  # type: ignore[assignment]

    class InvalidToken(Exception):
        pass


from .._env_io import memorizz_home
from .errors import MCPConfigurationError


def _require_cryptography() -> None:
    if Fernet is None:
        raise MCPConfigurationError(
            "Encrypted MCP credentials require the optional MCP dependencies. "
            "Install them with `pip install 'memorizz[mcp]'`."
        )


class CredentialStore(Protocol):
    def get(self, credential_ref: str) -> Dict[str, Any]:
        ...

    def set(self, credential_ref: str, value: Dict[str, Any]) -> None:
        ...

    def update(self, credential_ref: str, values: Dict[str, Any]) -> None:
        ...

    def delete(self, credential_ref: str) -> bool:
        ...


class EncryptedFileCredentialStore:
    """Small encrypted credential store for local and single-host deployments.

    Production deployments can provide another ``CredentialStore`` backed by a
    cloud secret manager/KMS. The encryption key may be supplied through
    ``MEMORIZZ_MCP_ENCRYPTION_KEY`` or is generated in the Memorizz home with
    owner-only permissions.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        key_path: Optional[Path] = None,
    ) -> None:
        home = memorizz_home()
        self.path = Path(path or home / "mcp_credentials.enc")
        self.key_path = Path(key_path or home / "mcp_credentials.key")
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._thread_lock = threading.RLock()

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        try:
            try:
                os.chmod(self.lock_path, 0o600)
            except OSError:
                pass
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            handle.close()

    def _key(self, create: bool) -> bytes:
        _require_cryptography()
        configured = os.environ.get("MEMORIZZ_MCP_ENCRYPTION_KEY", "").strip()
        if configured:
            raw = configured.encode("ascii", errors="strict")
            try:
                Fernet(raw)
            except Exception as exc:
                raise MCPConfigurationError(
                    "MEMORIZZ_MCP_ENCRYPTION_KEY must be a valid Fernet key"
                ) from exc
            return raw

        if self.key_path.exists():
            raw = self.key_path.read_bytes().strip()
            try:
                Fernet(raw)
            except Exception as exc:
                raise MCPConfigurationError(
                    f"Invalid MCP credential key at {self.key_path}"
                ) from exc
            return raw

        if not create:
            return b""
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        raw = Fernet.generate_key()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(str(self.key_path), flags, 0o600)
        except FileExistsError:
            # Another process created the key while this process was waiting
            # for the credential-store lock.
            return self._key(create=False)
        try:
            os.write(descriptor, raw + b"\n")
        finally:
            os.close(descriptor)
        return raw

    def _load_unlocked(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        key = self._key(create=False)
        if not key:
            raise MCPConfigurationError(
                "MCP credential data exists but its encryption key is missing"
            )
        try:
            cleartext = Fernet(key).decrypt(self.path.read_bytes())
            data = json.loads(cleartext.decode("utf-8"))
        except (InvalidToken, ValueError, json.JSONDecodeError) as exc:
            raise MCPConfigurationError(
                "MCP credential store could not be decrypted; check the encryption key"
            ) from exc
        records = data.get("records", {}) if isinstance(data, dict) else {}
        return records if isinstance(records, dict) else {}

    def _save_unlocked(self, records: Dict[str, Dict[str, Any]]) -> None:
        key = self._key(create=True)
        payload = json.dumps(
            {"version": 1, "records": records},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        encrypted = Fernet(key).encrypt(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        descriptor = os.open(
            str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        try:
            os.write(descriptor, encrypted)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def get(self, credential_ref: str) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with self._thread_lock, self._file_lock():
            return dict(self._load_unlocked().get(credential_ref, {}))

    def set(self, credential_ref: str, value: Dict[str, Any]) -> None:
        with self._thread_lock, self._file_lock():
            records = self._load_unlocked()
            records[credential_ref] = dict(value)
            self._save_unlocked(records)

    def update(self, credential_ref: str, values: Dict[str, Any]) -> None:
        with self._thread_lock, self._file_lock():
            records = self._load_unlocked()
            current = dict(records.get(credential_ref, {}))
            for key, value in values.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value
            records[credential_ref] = current
            self._save_unlocked(records)

    def delete(self, credential_ref: str) -> bool:
        if not self.path.exists():
            return False
        with self._thread_lock, self._file_lock():
            records = self._load_unlocked()
            existed = credential_ref in records
            if existed:
                del records[credential_ref]
                self._save_unlocked(records)
            return existed


class MCPOAuthTokenStorage:
    """Official MCP SDK ``TokenStorage`` backed by encrypted credentials."""

    def __init__(self, store: CredentialStore, credential_ref: str):
        self.store = store
        self.credential_ref = credential_ref

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken

        value = self.store.get(self.credential_ref).get("oauth_tokens")
        return OAuthToken.model_validate(value) if isinstance(value, dict) else None

    async def set_tokens(self, tokens) -> None:
        self.store.update(
            self.credential_ref,
            {"oauth_tokens": tokens.model_dump(mode="json", exclude_none=True)},
        )

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull

        value = self.store.get(self.credential_ref).get("oauth_client_info")
        return (
            OAuthClientInformationFull.model_validate(value)
            if isinstance(value, dict)
            else None
        )

    async def set_client_info(self, client_info) -> None:
        self.store.update(
            self.credential_ref,
            {
                "oauth_client_info": client_info.model_dump(
                    mode="json", exclude_none=True
                )
            },
        )


def environment_fernet_key() -> str:
    """Generate a valid key for operators configuring hosted deployments."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
