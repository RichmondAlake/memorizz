# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Production MCP client manager built on the official Python SDK."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import queue
import random
import re
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .._env_io import memorizz_home
from ..approval import ApprovalStatus, ApprovalStore, default_approval_store
from .audit import MCPAuditLogger
from .credentials import (
    CredentialStore,
    EncryptedFileCredentialStore,
    MCPOAuthTokenStorage,
)
from .errors import MCPAuthorizationRequired, MCPClientError, MCPConfigurationError
from .models import MCPServerConfig, parse_server_configs
from .oauth import PendingOAuthFlow, oauth_flows
from .security import (
    enforce_tool_policy,
    names_only,
    tool_is_mutating,
    validate_remote_url,
    validate_stdio_command,
)

logger = logging.getLogger(__name__)

_SECRET_IN_ERROR_RE = re.compile(
    r"(?i)(authorization|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|api[_ -]?key)"
    r"(\s*[:=]\s*|%3[dD])([^\s,;&]+)"
)


def _exception_message(exc: BaseException) -> str:
    """Flatten task-group errors so authorization/network causes stay useful."""
    children = list(getattr(exc, "exceptions", None) or [])
    if children:
        messages = [_exception_message(child) for child in children]
        return "; ".join(message for message in messages if message)
    message = str(exc).strip()
    cause = getattr(exc, "__cause__", None)
    if cause and (not message or "taskgroup" in message.replace(" ", "").lower()):
        return _exception_message(cause)
    return message or type(exc).__name__


def _find_mcp_error(exc: BaseException) -> Optional[MCPClientError]:
    """Find a structured MCP error inside AnyIO/asyncio exception groups."""
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, MCPClientError):
            return current
        pending.extend(getattr(current, "exceptions", None) or [])
        for linked in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return None


async def _reject_auth_response(response: Any, server_name: str) -> None:
    """Preserve 401/403 before the MCP SDK flattens them to INTERNAL_ERROR."""
    status_code = int(getattr(response, "status_code", 0) or 0)
    if status_code in {401, 403}:
        raise MCPAuthorizationRequired(
            f"Authorization is required for MCP server '{server_name}'",
            server_name,
        )


@functools.lru_cache(maxsize=1)
def _scope_pinned_metadata_class():
    """OAuth client metadata whose configured scopes the SDK cannot widen.

    Given no scope challenge, the MCP SDK requests every scope the server
    advertises (for Gmail that includes full mailbox access). When a
    connection lists its scopes, those are what the user signs in to.
    """
    from mcp.shared.auth import OAuthClientMetadata

    class ScopePinnedClientMetadata(OAuthClientMetadata):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "scope" and self.__dict__.get("scope"):
                return
            super().__setattr__(name, value)

    return ScopePinnedClientMetadata


def _same_origin_issuer(issuer: str, expected: str) -> bool:
    """Whether two issuers differ only by the slash after a bare origin."""
    if issuer.rstrip("/") != expected.rstrip("/"):
        return False
    return urlparse(issuer).path in ("", "/") and urlparse(expected).path in ("", "/")


def _tolerate_origin_issuer_slash() -> None:
    """Accept ``https://host`` and ``https://host/`` as the same issuer.

    The MCP SDK parses protected-resource metadata with pydantic URLs, which
    can add a slash to a bare origin, then compares the authorization server's
    ``issuer`` as an exact string. Google's MCP servers (Gmail, Calendar) fail
    that check. Only this case is relaxed: same scheme, host and port, and no
    path on either side.
    """
    from mcp.client.auth import oauth2

    current = oauth2.validate_metadata_issuer
    if getattr(current, "_memorizz_origin_slash", False):
        return

    def validate(oauth_metadata, expected_issuer: str) -> None:
        if _same_origin_issuer(str(oauth_metadata.issuer), str(expected_issuer)):
            return
        current(oauth_metadata, expected_issuer)

    validate._memorizz_origin_slash = True
    oauth2.validate_metadata_issuer = validate


def _sign_in_probe(tools: List[Any]) -> Optional[str]:
    """A read-only tool that takes no arguments, to trigger OAuth sign-in."""
    candidates = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        annotations = tool.get("annotations") or {}
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        if annotations.get("readOnlyHint") is not True or schema.get("required"):
            continue
        name = str(tool["name"])
        candidates.append((not name.startswith("list_"), len(name), name))
    return min(candidates)[2] if candidates else None


MAX_CACHED_TOOLS = 100

AuthorizationURLHandler = Callable[[str], None]
AuthorizationCallbackReader = Callable[[], str]


def _safe_ref_part(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value)
    )[:160]


def _model_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _model_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dict(item) for item in value]
    return value


def _describe_schema_error(error: Any) -> str:
    """A schema violation phrased so a model can fix the call."""
    schema = error.schema if isinstance(error.schema, dict) else {}
    if error.validator == "additionalProperties" and isinstance(error.instance, dict):
        properties = schema.get("properties") or {}
        notes = []
        for key in sorted(set(error.instance) - set(properties)):
            # A field documented under another property belongs inside it,
            # e.g. a Notion page "title" goes in "properties".
            home = next(
                (
                    name
                    for name, sub in properties.items()
                    if isinstance(sub, dict)
                    and re.search(
                        rf"[\"'`]{re.escape(key)}[\"'`]",
                        str(sub.get("description") or ""),
                    )
                ),
                None,
            )
            notes.append(
                f"'{key}' is not allowed here; put it inside '{home}', "
                f'like {{"{home}": {{"{key}": ...}}}}'
                if home
                else f"'{key}' is not allowed here"
            )
        allowed = ", ".join(sorted(properties))
        return "; ".join(notes) + (f" (allowed: {allowed})" if allowed else "")
    if error.validator in {"anyOf", "oneOf"} and isinstance(
        error.validator_value, list
    ):
        kinds = []
        for option in error.validator_value:
            if not isinstance(option, dict) or "type" not in option:
                continue
            kind = option["type"]
            if kind == "array" and isinstance(option.get("items"), dict):
                kind = f"array of {option['items'].get('type', 'values')}s"
            kinds.append(str(kind))
        if kinds:
            got = type(error.instance).__name__.replace("dict", "object")
            message = f"expected {' or '.join(kinds)}, got {got}"
            # Models used to REST APIs wrap text, e.g. {"text": {"content":
            # "Title"}}. Quote the text they meant as the value to send.
            text = _first_text(error.instance) if "string" in kinds else None
            if text is not None:
                message += f"; use the plain value {json.dumps(text)}"
            return message
    return str(error.message)


def _texts(value: Any, depth: int = 0) -> List[str]:
    """Every non-empty string inside a nested object or list."""
    if depth > 6:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    items = (
        value.values()
        if isinstance(value, dict)
        else value
        if isinstance(value, list)
        else []
    )
    return [text for item in items for text in _texts(item, depth + 1)]


def _allows_string(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    kind = schema.get("type")
    if kind == "string" or (isinstance(kind, list) and "string" in kind):
        return True
    return any(
        _allows_string(option)
        for key in ("anyOf", "oneOf")
        for option in schema.get(key) or []
    )


def _documented_home(key: str, properties: Dict[str, Any]) -> Optional[str]:
    """The object property whose description names ``key`` in quotes."""
    pattern = rf"[\"'`]{re.escape(key)}[\"'`]"
    for name, sub in properties.items():
        if (
            isinstance(sub, dict)
            and sub.get("type") == "object"
            and re.search(pattern, str(sub.get("description") or ""))
        ):
            return name
    return None


def _repair_schema_error(root: Any, error: Any, notes: List[str]) -> bool:
    """Apply one certain repair for ``error`` inside ``root``; True if changed."""
    path = list(error.absolute_path)
    where = "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}" for part in path
    ).lstrip(".")
    schema = error.schema if isinstance(error.schema, dict) else {}
    instance = error.instance
    if error.validator == "additionalProperties" and isinstance(instance, dict):
        properties = schema.get("properties") or {}
        for key in sorted(set(instance) - set(properties)):
            home = _documented_home(key, properties)
            if home is None:
                continue
            target = instance.setdefault(home, {})
            if not isinstance(target, dict):
                continue
            if key not in target:
                target[key] = instance.pop(key)
                notes.append(f"moved {where or 'arguments'}.{key} into {home}")
                return True
            if _texts(target[key])[:1] == _texts(instance[key])[:1]:
                instance.pop(key)
                notes.append(f"dropped {where or 'arguments'}.{key}, already in {home}")
                return True
        return False
    if (
        error.validator in {"anyOf", "oneOf", "type"}
        and isinstance(instance, (dict, list))
        and _allows_string(schema)
        and path
    ):
        texts = list(dict.fromkeys(_texts(instance)))
        if len(texts) != 1:
            return False
        parent = root
        for part in path[:-1]:
            parent = parent[part]
        parent[path[-1]] = texts[0]
        notes.append(f"used the text of {where} as its value")
        return True
    return False


def _first_text(value: Any, depth: int = 0) -> Optional[str]:
    """The first non-empty string inside a nested object or list."""
    if depth > 6:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    items = value.values() if isinstance(value, dict) else value
    if isinstance(value, (dict, list)):
        for item in items:
            found = _first_text(item, depth + 1)
            if found is not None:
                return found
    return None


class MCPClientManager:
    """Own MCP configurations, credentials, protocol calls, and safety policy."""

    def __init__(
        self,
        owner_id: str,
        servers: Any = None,
        credential_store: Optional[CredentialStore] = None,
        audit_logger: Optional[MCPAuditLogger] = None,
        approval_store: Optional[ApprovalStore] = None,
    ) -> None:
        self.owner_id = str(owner_id or "default")
        self.credential_store = credential_store or EncryptedFileCredentialStore()
        self.audit_logger = audit_logger or MCPAuditLogger()
        self.approval_store = approval_store
        self._servers: Dict[str, MCPServerConfig] = {}
        self._tool_metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._metrics_lock = threading.RLock()
        self._metrics: Dict[str, Dict[str, int]] = {}
        self.configure_servers(servers or [])

    # ------------------------------------------------------------------ config

    def _default_credential_ref(self, server_name: str) -> str:
        return f"mcp:{_safe_ref_part(self.owner_id)}:{_safe_ref_part(server_name)}"

    def configure_servers(self, servers: Any) -> List[Dict[str, Any]]:
        """Validate configs and move inline secrets into encrypted storage."""
        if not servers:
            self._servers = {}
            return []
        values = [servers] if isinstance(servers, dict) else servers
        if not isinstance(values, list):
            raise MCPConfigurationError("MCP servers must be an object or an array")

        public_values: List[Dict[str, Any]] = []
        for index, raw_value in enumerate(values):
            if not isinstance(raw_value, dict):
                raise MCPConfigurationError(
                    f"MCP server entry {index + 1} must be an object"
                )
            raw = dict(raw_value)
            name = str(raw.get("name") or "").strip()
            if not name:
                raise MCPConfigurationError(
                    f"MCP server entry {index + 1} is missing 'name'"
                )

            auth_value = raw.get("auth")
            if isinstance(auth_value, str):
                auth_raw: Dict[str, Any] = {"type": auth_value}
            elif isinstance(auth_value, dict):
                auth_raw = dict(auth_value)
            else:
                auth_raw = {}

            credential_ref = str(
                auth_raw.get("credential_ref")
                or raw.get("credential_ref")
                or self._default_credential_ref(name)
            ).strip()
            auth_raw["credential_ref"] = credential_ref

            secrets: Dict[str, Any] = {}
            raw_env = raw.pop("env", None)
            if isinstance(raw_env, dict) and raw_env:
                secrets["stdio_env"] = {
                    str(key): str(value)
                    for key, value in raw_env.items()
                    if str(key).strip()
                }
                raw["env_keys"] = list(names_only(secrets["stdio_env"]))

            raw_headers = raw.pop("headers", None)
            if isinstance(raw_headers, dict) and raw_headers:
                secrets["http_headers"] = {
                    str(key): str(value)
                    for key, value in raw_headers.items()
                    if str(key).strip()
                }
                raw["header_names"] = list(names_only(secrets["http_headers"]))

            bearer_token = (
                auth_raw.pop("token", None)
                or auth_raw.pop("bearer_token", None)
                or raw.pop("bearer_token", None)
            )
            if bearer_token:
                secrets["bearer_token"] = str(bearer_token)
                auth_raw["type"] = "bearer"

            client_secret = auth_raw.pop("client_secret", None)
            if client_secret:
                secrets["oauth_client_secret"] = str(client_secret)
                auth_raw.setdefault("type", "oauth")

            oauth_tokens = auth_raw.pop("tokens", None)
            if isinstance(oauth_tokens, dict) and oauth_tokens:
                secrets["oauth_tokens"] = oauth_tokens
                auth_raw.setdefault("type", "oauth")

            raw["auth"] = auth_raw
            public_values.append(raw)

            if secrets:
                self.credential_store.update(credential_ref, secrets)

            client_id = str(auth_raw.get("client_id") or "").strip()
            if client_id:
                record = self.credential_store.get(credential_ref)
                secret = record.get("oauth_client_secret")
                redirect_uri = str(auth_raw.get("redirect_uri") or "").strip()
                scopes_value = auth_raw.get("scopes") or []
                if isinstance(scopes_value, str):
                    scopes_value = scopes_value.split()
                client_info = {
                    "client_id": client_id,
                    "client_secret": secret,
                    "redirect_uris": [redirect_uri] if redirect_uri else None,
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "scope": " ".join(str(value) for value in scopes_value) or None,
                    "token_endpoint_auth_method": auth_raw.get(
                        "token_endpoint_auth_method"
                    )
                    or ("client_secret_post" if secret else "none"),
                }
                self.credential_store.update(
                    credential_ref,
                    {
                        "oauth_client_info": {
                            key: value
                            for key, value in client_info.items()
                            if value is not None
                        }
                    },
                )

        parsed = parse_server_configs(public_values)
        self._servers = {server.name: server for server in parsed}
        return self.server_dicts()

    def server_dicts(self) -> List[Dict[str, Any]]:
        return [server.public_dict() for server in self._servers.values()]

    def get_server(self, server_name: str) -> MCPServerConfig:
        server = self._servers.get(str(server_name or "").strip())
        if not server:
            raise MCPConfigurationError(
                f"Unknown MCP server '{server_name}'", server_name=server_name
            )
        if not server.enabled:
            raise MCPConfigurationError(
                f"MCP server '{server_name}' is disabled", server_name=server_name
            )
        return server

    def upsert_server(self, server: Dict[str, Any]) -> Dict[str, Any]:
        values = self.server_dicts()
        name = str(server.get("name") or "").strip()
        values = [value for value in values if value.get("name") != name]
        values.append(dict(server))
        self.configure_servers(values)
        return self.get_server(name).public_dict()

    def remove_server(self, server_name: str, delete_credentials: bool = True) -> bool:
        server = self._servers.pop(server_name, None)
        if not server:
            return False
        if delete_credentials and server.auth.credential_ref:
            self.credential_store.delete(server.auth.credential_ref)
        return True

    # -------------------------------------------------------------- credentials

    def disconnect(self, server_name: str) -> bool:
        server = self.get_server(server_name)
        if not server.auth.credential_ref:
            return False
        return self.credential_store.delete(server.auth.credential_ref)

    def _credential_record(self, server: MCPServerConfig) -> Dict[str, Any]:
        if not server.auth.credential_ref:
            return {}
        return self.credential_store.get(server.auth.credential_ref)

    def connection_status(self, server_name: Optional[str] = None) -> Dict[str, Any]:
        names = [server_name] if server_name else list(self._servers)
        statuses = []
        for name in names:
            try:
                server = self.get_server(str(name))
                record = self._credential_record(server)
                authenticated = (
                    server.auth.type == "none"
                    or bool(record.get("bearer_token"))
                    or bool(record.get("oauth_tokens"))
                )
                statuses.append(
                    {
                        "name": server.name,
                        "transport": server.transport,
                        "url": server.url,
                        "command": server.command,
                        "enabled": server.enabled,
                        "auth_type": server.auth.type,
                        "authenticated": authenticated,
                        "credential_ref": server.auth.credential_ref,
                        "state": "configured" if authenticated else "reauth_required",
                        "require_approval": server.require_approval,
                    }
                )
            except MCPClientError as exc:
                statuses.append(exc.to_dict())
        return {"ok": True, "servers": statuses}

    # --------------------------------------------------------------- transports

    def _oauth_metadata(self, server: MCPServerConfig):
        from .. import __version__

        redirect_uri = server.auth.redirect_uri
        if not redirect_uri:
            raise MCPAuthorizationRequired(
                "OAuth server requires auth.redirect_uri before authorization can begin",
                server.name,
            )
        application_type = (
            "native"
            if (urlparse(redirect_uri).hostname or "")
            in {"127.0.0.1", "localhost", "::1"}
            else "web"
        )
        record = self._credential_record(server)
        return _scope_pinned_metadata_class()(
            redirect_uris=[redirect_uri],
            token_endpoint_auth_method=server.auth.token_endpoint_auth_method
            or ("client_secret_post" if record.get("oauth_client_secret") else "none"),
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=" ".join(server.auth.scopes) or None,
            client_name="Memorizz MCP Client",
            software_version=__version__,
            application_type=application_type,
        )

    def _oauth_provider(
        self,
        server: MCPServerConfig,
        redirect_handler=None,
        callback_handler=None,
    ):
        from mcp.client.auth import OAuthClientProvider

        _tolerate_origin_issuer_slash()

        if not server.url or not server.auth.credential_ref:
            raise MCPConfigurationError("OAuth requires a URL and credential reference")
        return OAuthClientProvider(
            server_url=server.url,
            client_metadata=self._oauth_metadata(server),
            storage=MCPOAuthTokenStorage(
                self.credential_store, server.auth.credential_ref
            ),
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            client_metadata_url=server.auth.client_metadata_url,
        )

    @asynccontextmanager
    async def _client(
        self,
        server: MCPServerConfig,
        redirect_handler=None,
        callback_handler=None,
    ):
        try:
            import httpx2
            from mcp import Client, StdioServerParameters
            from mcp.client.sse import sse_client
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise MCPConfigurationError(
                "MCP connections require the optional MCP dependencies. "
                "Install them with `pip install 'memorizz[mcp]'`."
            ) from exc

        transport = None
        managed_http_client = None
        if server.transport == "stdio":
            resolved_command = validate_stdio_command(server.command or "")
            record = self._credential_record(server)
            stdio_env = record.get("stdio_env")
            if stdio_env is not None and not isinstance(stdio_env, dict):
                stdio_env = None
            parameters = StdioServerParameters(
                command=resolved_command,
                args=list(server.args),
                env={str(k): str(v) for k, v in (stdio_env or {}).items()} or None,
                cwd=server.cwd,
            )
            # Pytest, notebooks, and some web runners replace ``sys.stderr``
            # with objects that cannot be passed to a child process. The real
            # interpreter stream remains safe for server diagnostics.
            transport = stdio_client(
                parameters, errlog=getattr(sys, "__stderr__", None) or sys.stderr
            )
        else:
            if not server.url:
                raise MCPConfigurationError("Remote MCP server is missing a URL")
            record = self._credential_record(server)
            headers = {
                str(key): str(value)
                for key, value in (record.get("http_headers") or {}).items()
                if str(key).lower()
                not in {"authorization", "content-length", "host", "transfer-encoding"}
            }
            auth = None
            if server.auth.type == "bearer":
                bearer = record.get("bearer_token")
                if not bearer:
                    raise MCPAuthorizationRequired(
                        f"MCP server '{server.name}' requires a bearer token",
                        server.name,
                    )
                headers["Authorization"] = f"Bearer {bearer}"
            elif server.auth.type == "oauth":
                if not record.get("oauth_tokens") and (
                    redirect_handler is None or callback_handler is None
                ):
                    raise MCPAuthorizationRequired(
                        f"MCP server '{server.name}' must be authorized before use",
                        server.name,
                    )
                auth = self._oauth_provider(
                    server,
                    redirect_handler=redirect_handler,
                    callback_handler=callback_handler,
                )

            validate_remote_url(server.url, server.allow_private_network)

            if server.transport == "streamable_http":
                timeout = httpx2.Timeout(
                    float(server.timeout), connect=float(server.connect_timeout)
                )

                async def validate_request(request) -> None:
                    # Re-check every redirect target to prevent a trusted host
                    # from redirecting the client into a private network.
                    validate_remote_url(str(request.url), server.allow_private_network)

                async def validate_response(response) -> None:
                    await _reject_auth_response(response, server.name)

                # Response hooks run before httpx hands a 401 to the auth
                # flow, so with OAuth the provider must see it first to start
                # sign-in or refresh tokens.
                managed_http_client = httpx2.AsyncClient(
                    headers=headers,
                    auth=auth,
                    timeout=timeout,
                    follow_redirects=True,
                    trust_env=False,
                    event_hooks={
                        "request": [validate_request],
                        "response": [] if auth is not None else [validate_response],
                    },
                )
                await managed_http_client.__aenter__()
                transport = streamable_http_client(
                    server.url, http_client=managed_http_client
                )
            else:

                def sse_http_client_factory(headers=None, timeout=None, auth=None):
                    async def validate_request(request) -> None:
                        validate_remote_url(
                            str(request.url), server.allow_private_network
                        )

                    async def validate_response(response) -> None:
                        await _reject_auth_response(response, server.name)

                    return httpx2.AsyncClient(
                        headers=headers,
                        timeout=timeout,
                        auth=auth,
                        follow_redirects=True,
                        trust_env=False,
                        event_hooks={
                            "request": [validate_request],
                            "response": (
                                [] if auth is not None else [validate_response]
                            ),
                        },
                    )

                transport = sse_client(
                    server.url,
                    headers=headers,
                    auth=auth,
                    timeout=float(server.connect_timeout),
                    sse_read_timeout=float(server.timeout),
                    httpx_client_factory=sse_http_client_factory,
                )

        client = Client(
            transport,
            read_timeout_seconds=float(server.timeout),
            raise_exceptions=True,
            mode="auto",
        )
        try:
            async with client:
                yield client
        finally:
            if managed_http_client is not None:
                await managed_http_client.__aexit__(None, None, None)

    # --------------------------------------------------------------- operations

    @staticmethod
    def _run_async(coro: Awaitable[Any]) -> Any:
        """Run async SDK operations safely from sync tools and async UI threads."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            running_loop = False
        else:
            running_loop = True

        if not running_loop:
            return asyncio.run(coro)

        result_queue: queue.Queue = queue.Queue(maxsize=1)

        def runner() -> None:
            try:
                result_queue.put((True, asyncio.run(coro)))
            except BaseException as exc:  # propagate to the caller thread
                result_queue.put((False, exc))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        ok, value = result_queue.get()
        if ok:
            return value
        raise value

    def _record_metric(self, server_name: str, operation: str, ok: bool) -> None:
        with self._metrics_lock:
            metrics = self._metrics.setdefault(
                server_name, {"requests": 0, "successes": 0, "failures": 0}
            )
            metrics["requests"] += 1
            metrics["successes" if ok else "failures"] += 1
            metrics[f"operation:{operation}"] = (
                metrics.get(f"operation:{operation}", 0) + 1
            )

    def diagnostics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            return {"ok": True, "metrics": json.loads(json.dumps(self._metrics))}

    @staticmethod
    def _ensure_payload_size(server: MCPServerConfig, value: Any) -> Any:
        size = len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        if size > server.max_result_bytes:
            raise MCPClientError(
                message=(
                    f"MCP server '{server.name}' returned {size} bytes, exceeding "
                    f"the configured max_result_bytes={server.max_result_bytes}"
                ),
                code="response_too_large",
                retryable=False,
                server_name=server.name,
            )
        return value

    def _safe_error(
        self, exc: BaseException, server: MCPServerConfig
    ) -> MCPClientError:
        structured = _find_mcp_error(exc)
        if structured is not None:
            return structured
        message = _exception_message(exc)
        message = _SECRET_IN_ERROR_RE.sub(r"\1\2***", message)
        lowered = message.lower()
        if "registration failed" in lowered and not server.auth.client_id:
            return MCPClientError(
                message=(
                    f"MCP server '{server.name}' signs in through a provider that "
                    "does not register clients automatically. Create an OAuth "
                    f"client there with the redirect URI {server.auth.redirect_uri}, "
                    "then edit this connection and add its client ID and secret."
                ),
                code="oauth_client_required",
                retryable=False,
                server_name=server.name,
            )
        if any(
            marker in lowered
            for marker in (
                "authorization code",
                "no redirect handler",
                "no callback handler",
                "401",
                "unauthorized",
                "invalid_grant",
            )
        ):
            return MCPAuthorizationRequired(
                f"Authorization is required for MCP server '{server.name}'",
                server.name,
            )
        retryable = any(
            marker in lowered
            for marker in (
                "timeout",
                "temporarily",
                "connection",
                "reset",
                "503",
                "429",
            )
        )
        return MCPClientError(
            message=f"MCP server '{server.name}' request failed: {message}",
            code="connection_error" if retryable else "protocol_error",
            retryable=retryable,
            server_name=server.name,
        )

    async def _with_retry(
        self,
        server: MCPServerConfig,
        operation: Callable[[Any], Awaitable[Any]],
        *,
        allow_retry: bool,
        redirect_handler=None,
        callback_handler=None,
    ) -> Any:
        attempts = 1 + (server.max_retries if allow_retry else 0)
        last_error: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                async with self._client(
                    server,
                    redirect_handler=redirect_handler,
                    callback_handler=callback_handler,
                ) as client:
                    return await operation(client)
            except BaseException as exc:
                last_error = exc
                safe = self._safe_error(exc, server)
                if not allow_retry or not safe.retryable or attempt + 1 >= attempts:
                    raise safe from exc
                await asyncio.sleep(
                    min(2.0, 0.2 * (2**attempt)) + random.random() * 0.1
                )
        raise self._safe_error(last_error or RuntimeError("MCP request failed"), server)

    async def _list_tools_async(
        self,
        server: MCPServerConfig,
        redirect_handler=None,
        callback_handler=None,
    ) -> Dict[str, Any]:
        async def operation(client):
            tools: List[Any] = []
            cursor = None
            while True:
                result = await client.list_tools(cursor=cursor, cache_mode="refresh")
                tools.extend(result.tools)
                cursor = getattr(result, "next_cursor", None)
                if not cursor:
                    break
            return self._ensure_payload_size(
                server,
                {
                    "ok": True,
                    "server_name": server.name,
                    "protocol_version": client.protocol_version,
                    "server_info": _model_dict(client.server_info),
                    "tools": [_model_dict(tool) for tool in tools],
                },
            )

        return await self._with_retry(
            server,
            operation,
            allow_retry=True,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

    def list_tools(self, server_name: str) -> Dict[str, Any]:
        server = self.get_server(server_name)
        started = time.monotonic()
        try:
            result = self._run_async(self._list_tools_async(server))
            self._tool_metadata[server.name] = {
                str(item.get("name")): dict(item)
                for item in (result.get("tools") or [])
                if isinstance(item, dict) and item.get("name")
            }
            self._remember_tool_listing(server, result.get("tools") or [])
            self._record_metric(server.name, "tools/list", True)
            return result
        except BaseException as exc:
            safe = self._safe_error(exc, server)
            self._record_metric(server.name, "tools/list", False)
            return safe.to_dict()
        finally:
            self.audit_logger.record(
                owner_id=self.owner_id,
                server_name=server.name,
                operation="tools/list",
                ok="result" in locals() and bool(result.get("ok")),
                duration_ms=int((time.monotonic() - started) * 1000),
                error_code=(safe.code if "safe" in locals() else None),
            )

    async def _call_tool_async(
        self,
        server: MCPServerConfig,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        async def operation(client):
            from ..streaming import current_cancellation, current_stream

            token = current_cancellation.get()
            session = current_stream.get()

            async def progress(value, total=None, message=None):
                # External tool output is tool activity, never our answer text.
                if session is not None:
                    await asyncio.to_thread(
                        session.emit,
                        "status",
                        stage="mcp_tool_progress",
                        tool_name=tool_name,
                        progress=value,
                        total=total,
                    )

            task = asyncio.current_task()
            loop = asyncio.get_running_loop()
            unregister = (
                token.register(lambda: loop.call_soon_threadsafe(task.cancel))
                if token
                else lambda: None
            )
            try:
                result = await client.call_tool(
                    tool_name, arguments, progress_callback=progress
                )
            finally:
                unregister()
            payload = _model_dict(result)
            is_error = bool(getattr(result, "is_error", False))
            return self._ensure_payload_size(
                server,
                {
                    "ok": not is_error,
                    "server_name": server.name,
                    "tool_name": tool_name,
                    "result": payload,
                    **(
                        {"error": "MCP tool returned an error result"}
                        if is_error
                        else {}
                    ),
                },
            )

        # Mutations are deliberately never retried.
        return await self._with_retry(server, operation, allow_retry=False)

    def _approval_store(self) -> ApprovalStore:
        if self.approval_store is None:
            self.approval_store = default_approval_store()
        return self.approval_store

    def tool_requires_approval(self, server_name: str, tool_name: str) -> bool:
        """Return the host/MCP-annotation-aware mutation decision."""
        server = self.get_server(server_name)
        metadata = self._known_tool(server.name, str(tool_name))
        return bool(server.require_approval) and tool_is_mutating(
            tool_name,
            metadata,
            read_only_tools=server.read_only_tools,
            mutation_tools=server.mutation_tools,
        )

    def _call_tool_authorized(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a policy-validated call after host authorization."""
        server = self.get_server(server_name)
        values = arguments or {}
        started = time.monotonic()
        try:
            normalized_name = str(tool_name or "").strip()
            if not normalized_name:
                raise MCPConfigurationError("MCP tool name cannot be empty")
            if not isinstance(values, dict):
                raise MCPConfigurationError("MCP tool arguments must be an object")
            enforce_tool_policy(
                server,
                normalized_name,
                approved=True,
                tool_metadata=self._tool_metadata.get(server.name, {}).get(
                    normalized_name
                ),
            )
            result = self._run_async(
                self._call_tool_async(server, normalized_name, values)
            )
            self._record_metric(server.name, "tools/call", bool(result.get("ok")))
            return result
        except BaseException as exc:
            safe = self._safe_error(exc, server)
            self._record_metric(server.name, "tools/call", False)
            return safe.to_dict()
        finally:
            self.audit_logger.record(
                owner_id=self.owner_id,
                server_name=server.name,
                operation="tools/call",
                tool_name=str(tool_name or ""),
                arguments=values if isinstance(values, dict) else {},
                ok="result" in locals() and bool(result.get("ok")),
                duration_ms=int((time.monotonic() - started) * 1000),
                error_code=(safe.code if "safe" in locals() else None),
            )

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a read-only tool or create a durable mutation proposal."""
        server = self.get_server(server_name)
        values = arguments or {}
        try:
            tool_name = str(tool_name or "").strip()
            if not tool_name:
                raise MCPConfigurationError("MCP tool name cannot be empty")
            if not isinstance(values, dict):
                raise MCPConfigurationError("MCP tool arguments must be an object")
            values, _ = self.repair_arguments(server.name, tool_name, values)
            problems = self._argument_problems(server.name, tool_name, values)
            if problems:
                # Caught here, the call never reaches the server or becomes an
                # approval request, and the model gets the exact fix.
                return {
                    "ok": False,
                    "error_code": "invalid_arguments",
                    "error": (
                        f"The arguments do not match {tool_name}'s input schema: "
                        + "; ".join(problems)
                        + ". Fix them and call the tool again."
                    ),
                }
            mutation_requires_approval = self.tool_requires_approval(
                server.name, tool_name
            )
            enforce_tool_policy(
                server,
                tool_name,
                approved=mutation_requires_approval,
                tool_metadata=self._tool_metadata.get(server.name, {}).get(tool_name),
            )
            if mutation_requires_approval:
                proposal = self._approval_store().propose(
                    owner_id=self.owner_id,
                    tool_name=f"mcp:{server.name}:{tool_name}",
                    arguments=values,
                    policy_reason=(
                        f"MCP tool {server.name}.{tool_name} may change external data"
                    ),
                    checkpoint={
                        "version": 1,
                        "kind": "mcp_tool_call",
                        "server_name": server.name,
                        "tool_name": tool_name,
                        "arguments": values,
                    },
                )
                return {
                    "ok": False,
                    "status": "approval_required",
                    "error_code": "approval_required",
                    "proposal": proposal.to_dict(include_arguments=True),
                }
            return self._call_tool_authorized(server.name, tool_name, values)
        except BaseException as exc:
            safe = self._safe_error(exc, server)
            self._record_metric(server.name, "tools/call", False)
            return safe.to_dict()

    def _known_tool(self, server_name: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """This process's listing of a tool, else the cached one."""
        listed = self._tool_metadata.get(server_name, {}).get(tool_name)
        if listed:
            return listed
        try:
            cached = self.cached_tools(server_name)
        except Exception:
            return None
        return next((item for item in cached if item.get("name") == tool_name), None)

    def _input_validator(self, server_name: str, tool_name: str) -> Any:
        """A validator for the tool's advertised input schema, if usable."""
        tool = self._known_tool(server_name, tool_name)
        schema = (tool or {}).get("inputSchema") or (tool or {}).get("input_schema")
        if not isinstance(schema, dict) or not schema:
            return None
        try:
            import jsonschema

            validator_class = jsonschema.validators.validator_for(
                schema, default=jsonschema.Draft202012Validator
            )
            validator_class.check_schema(schema)
            return validator_class(schema)
        except Exception as exc:  # an unusable schema is the server's to judge
            logger.debug("Skipping argument check for %s: %s", tool_name, exc)
            return None

    def repair_arguments(
        self, server_name: str, tool_name: str, values: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Fit the model's own values to the schema where the fix is certain.

        Two repairs, both guided by the schema and never inventing content: a
        field the schema documents under a sibling object moves into it (a
        Notion page "title" into "properties"), and an object holding a single
        piece of text becomes that text where only a plain value is allowed.
        Returns the arguments and a note per change.
        """
        validator = self._input_validator(server_name, tool_name)
        if validator is None or not isinstance(values, dict):
            return values, []
        fixed = json.loads(json.dumps(values, default=str))
        notes: List[str] = []
        for _ in range(8):
            errors = sorted(
                validator.iter_errors(fixed),
                key=lambda error: len(error.absolute_path),
                reverse=True,
            )
            if not any(_repair_schema_error(fixed, error, notes) for error in errors):
                break
        return (fixed, notes) if notes else (values, [])

    def _argument_problems(
        self, server_name: str, tool_name: str, values: Dict[str, Any]
    ) -> List[str]:
        """Where ``values`` break the tool's advertised input schema, if known."""
        validator = self._input_validator(server_name, tool_name)
        if validator is None:
            return []
        try:
            errors = list(validator.iter_errors(values))
        except Exception as exc:
            logger.debug("Skipping argument check for %s: %s", tool_name, exc)
            return []
        problems = []
        for error in sorted(errors, key=lambda e: list(e.absolute_path))[:5]:
            where = "".join(
                f"[{part}]" if isinstance(part, int) else f".{part}"
                for part in error.absolute_path
            ).lstrip(".")
            problems.append(f"{where or 'arguments'}: {_describe_schema_error(error)}")
        return problems

    def approve_tool_call(
        self, proposal_id: str, *, approver_id: str, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        proposal = self._approval_store().approve(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        )
        return proposal.to_dict(include_arguments=True)

    def list_tool_call_approvals(
        self, *, status: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return [
            proposal.to_dict(include_arguments=True)
            for proposal in self._approval_store().list(
                owner_id=self.owner_id, status=status, limit=limit
            )
            if str(proposal.tool_name).startswith("mcp:")
        ]

    def reject_tool_call(
        self, proposal_id: str, *, approver_id: str, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        proposal = self._approval_store().reject(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        )
        return proposal.to_dict(include_arguments=True)

    def cancel_tool_call(
        self, proposal_id: str, *, approver_id: str, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel a pending call (operator-facing alias for rejection)."""
        return self.reject_tool_call(
            proposal_id,
            approver_id=approver_id,
            reason=reason or "Cancelled by host",
        )

    def resume_tool_call(self, proposal_id: str) -> Dict[str, Any]:
        """Consume and execute the exact MCP call stored in a proposal."""
        store = self._approval_store()
        proposal = store.get(proposal_id)
        if proposal is None:
            return {
                "ok": False,
                "error_code": "approval_not_found",
                "error": f"Unknown approval proposal '{proposal_id}'",
            }
        if proposal.owner_id != self.owner_id:
            return {
                "ok": False,
                "error_code": "approval_owner_mismatch",
                "error": "This proposal belongs to another MCP owner",
            }
        if proposal.status != ApprovalStatus.APPROVED:
            return {
                "ok": False,
                "error_code": "invalid_approval_state",
                "status": proposal.status.value,
                "error": "Only an approved proposal can be resumed",
            }
        checkpoint = dict(proposal.checkpoint or {})
        if checkpoint.get("kind") != "mcp_tool_call":
            return {
                "ok": False,
                "error_code": "invalid_approval_checkpoint",
                "error": "Proposal is not an MCP tool-call checkpoint",
            }
        server_name = str(checkpoint.get("server_name") or "")
        tool_name = str(checkpoint.get("tool_name") or "")
        arguments = checkpoint.get("arguments") or {}
        store.consume(
            proposal_id,
            expected_tool_name=f"mcp:{server_name}:{tool_name}",
            expected_arguments=arguments,
        )
        return self._call_tool_authorized(server_name, tool_name, arguments)

    async def _list_capability_async(
        self, server: MCPServerConfig, capability: str
    ) -> Dict[str, Any]:
        async def operation(client):
            values: List[Any] = []
            cursor = None
            while True:
                if capability == "resources":
                    response = await client.list_resources(
                        cursor=cursor, cache_mode="refresh"
                    )
                    values.extend(response.resources)
                elif capability == "resource_templates":
                    response = await client.list_resource_templates(
                        cursor=cursor, cache_mode="refresh"
                    )
                    values.extend(response.resource_templates)
                else:
                    response = await client.list_prompts(
                        cursor=cursor, cache_mode="refresh"
                    )
                    values.extend(response.prompts)
                cursor = getattr(response, "next_cursor", None)
                if not cursor:
                    break
            return self._ensure_payload_size(
                server,
                {
                    "ok": True,
                    "server_name": server.name,
                    capability: [_model_dict(value) for value in values],
                },
            )

        return await self._with_retry(server, operation, allow_retry=True)

    def list_resources(self, server_name: str) -> Dict[str, Any]:
        server = self.get_server(server_name)
        try:
            return self._run_async(self._list_capability_async(server, "resources"))
        except BaseException as exc:
            return self._safe_error(exc, server).to_dict()

    def list_resource_templates(self, server_name: str) -> Dict[str, Any]:
        server = self.get_server(server_name)
        try:
            return self._run_async(
                self._list_capability_async(server, "resource_templates")
            )
        except BaseException as exc:
            return self._safe_error(exc, server).to_dict()

    def list_prompts(self, server_name: str) -> Dict[str, Any]:
        server = self.get_server(server_name)
        try:
            return self._run_async(self._list_capability_async(server, "prompts"))
        except BaseException as exc:
            return self._safe_error(exc, server).to_dict()

    def read_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        server = self.get_server(server_name)
        uri = str(uri or "").strip()
        if not uri:
            return MCPConfigurationError(
                "MCP resource URI cannot be empty", server.name
            ).to_dict()

        async def run():
            async def operation(client):
                result = await client.read_resource(uri, cache_mode="refresh")
                return self._ensure_payload_size(
                    server,
                    {
                        "ok": True,
                        "server_name": server.name,
                        "uri": uri,
                        "result": _model_dict(result),
                    },
                )

            return await self._with_retry(server, operation, allow_retry=True)

        try:
            return self._run_async(run())
        except BaseException as exc:
            return self._safe_error(exc, server).to_dict()

    def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        server = self.get_server(server_name)
        prompt_name = str(prompt_name or "").strip()
        if not prompt_name:
            return MCPConfigurationError(
                "MCP prompt name cannot be empty", server.name
            ).to_dict()

        async def run():
            async def operation(client):
                result = await client.get_prompt(prompt_name, arguments or {})
                return self._ensure_payload_size(
                    server,
                    {
                        "ok": True,
                        "server_name": server.name,
                        "prompt_name": prompt_name,
                        "result": _model_dict(result),
                    },
                )

            return await self._with_retry(server, operation, allow_retry=True)

        try:
            return self._run_async(run())
        except BaseException as exc:
            return self._safe_error(exc, server).to_dict()

    # ------------------------------------------------------------ tool catalog

    def _tool_catalog_path(self) -> Path:
        owner = _safe_ref_part(self.owner_id or "default") or "default"
        return memorizz_home() / "mcp_tools" / f"{owner}.json"

    @staticmethod
    def _server_fingerprint(server: MCPServerConfig) -> str:
        """Identifies where a server's tools came from, not its credentials."""
        raw = json.dumps(
            [server.transport, server.url, server.command, list(server.args)],
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _read_tool_catalog(self) -> Dict[str, Any]:
        try:
            data = json.loads(self._tool_catalog_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _remember_tool_listing(self, server: MCPServerConfig, tools: List[Any]) -> None:
        """Keep the last tool list so agents can expose tools without a call."""
        rows = []
        for tool in tools[:MAX_CACHED_TOOLS]:
            if not isinstance(tool, dict) or not tool.get("name"):
                continue
            rows.append(
                {
                    "name": str(tool["name"]),
                    "description": str(tool.get("description") or "")[:1000],
                    "inputSchema": tool.get("inputSchema")
                    or tool.get("input_schema")
                    or {},
                    "annotations": tool.get("annotations") or {},
                }
            )
        path = self._tool_catalog_path()
        try:
            catalog = self._read_tool_catalog()
            catalog[server.name] = {
                "fingerprint": self._server_fingerprint(server),
                "listed_at": datetime.now(timezone.utc).isoformat(),
                "tools": rows,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(catalog), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as exc:
            logger.debug("Could not cache MCP tools for %s: %s", server.name, exc)

    def cached_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """Tools from the last successful listing of this server, if unchanged.

        Blocked tools and tools outside an allowlist are left out.
        """
        server = self.get_server(server_name)
        entry = self._read_tool_catalog().get(server.name)
        if not isinstance(entry, dict):
            return []
        if entry.get("fingerprint") != self._server_fingerprint(server):
            return []
        return [
            dict(tool)
            for tool in entry.get("tools") or []
            if isinstance(tool, dict)
            and tool.get("name")
            and (not server.allowed_tools or tool["name"] in server.allowed_tools)
            and tool["name"] not in server.blocked_tools
        ]

    def test_connection(self, server_name: str) -> Dict[str, Any]:
        result = self.list_tools(server_name)
        if result.get("ok"):
            result["state"] = "ready"
            result["tool_count"] = len(result.get("tools") or [])
        return result

    # ------------------------------------------------------------------- OAuth

    def begin_oauth(
        self,
        server_name: str,
        wait_seconds: float = 20.0,
        return_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Begin browser OAuth for the UI and return the authorization URL."""
        server = self.get_server(server_name)
        if server.auth.type != "oauth":
            raise MCPConfigurationError(
                f"MCP server '{server.name}' is not configured for OAuth",
                server.name,
            )
        flow = PendingOAuthFlow(
            owner_id=self.owner_id, server_name=server.name, return_to=return_to
        )

        async def redirect_handler(url: str) -> None:
            flow.authorization_url = url
            parsed = urlparse(url)
            flow.state = (parse_qs(parsed.query).get("state") or [None])[0]
            if not flow.state:
                flow.error = "OAuth authorization URL did not contain state"
            else:
                oauth_flows.add(flow)
            flow.url_ready.set()

        async def callback_handler():
            from mcp.shared.auth import AuthorizationCodeResult

            ready = await asyncio.to_thread(flow.callback_ready.wait, 600)
            if not ready:
                raise MCPAuthorizationRequired("OAuth callback timed out", server.name)
            if flow.error:
                raise MCPAuthorizationRequired(flow.error, server.name)
            if not flow.code:
                raise MCPAuthorizationRequired(
                    "OAuth callback did not contain an authorization code", server.name
                )
            return AuthorizationCodeResult(
                code=flow.code,
                state=flow.callback_state,
                iss=flow.issuer,
            )

        def worker() -> None:
            try:
                flow.result = asyncio.run(
                    self._list_tools_async(
                        server,
                        redirect_handler=redirect_handler,
                        callback_handler=callback_handler,
                    )
                )
                # Some servers (Google's) list tools anonymously and ask for
                # sign-in only on a call; a read-only call starts it.
                probe = (
                    None
                    if flow.authorization_url
                    or self._credential_record(server).get("oauth_tokens")
                    else _sign_in_probe(flow.result.get("tools") or [])
                )
                if probe:
                    asyncio.run(
                        self._with_retry(
                            server,
                            lambda client: client.call_tool(probe, {}),
                            allow_retry=False,
                            redirect_handler=redirect_handler,
                            callback_handler=callback_handler,
                        )
                    )
                if not flow.authorization_url:
                    flow.url_ready.set()
            except BaseException as exc:
                safe = self._safe_error(exc, server)
                flow.error = safe.message
                flow.result = safe.to_dict()
                flow.url_ready.set()
            finally:
                flow.completed.set()

        threading.Thread(target=worker, daemon=True).start()
        if not flow.url_ready.wait(wait_seconds):
            raise MCPClientError(
                f"Timed out starting OAuth for MCP server '{server.name}'",
                code="oauth_start_timeout",
                retryable=True,
                server_name=server.name,
            )
        if flow.error and not flow.authorization_url:
            code = (
                (flow.result or {}).get("error_code")
                if isinstance(flow.result, dict)
                else None
            )
            if code and code != "authorization_required":
                raise MCPClientError(
                    message=flow.error,
                    code=code,
                    retryable=False,
                    server_name=server.name,
                )
            raise MCPAuthorizationRequired(flow.error, server.name)
        if (
            not flow.authorization_url
            and isinstance(flow.result, dict)
            and flow.result.get("ok")
        ):
            signed_in = bool(self._credential_record(server).get("oauth_tokens"))
            return {
                "ok": True,
                "server_name": server.name,
                "already_authenticated": signed_in,
                "message": (
                    None
                    if signed_in
                    else "The server answered without asking for sign-in."
                ),
            }
        return {
            "ok": True,
            "server_name": server.name,
            "authorization_url": flow.authorization_url,
            "state": flow.state,
        }

    @staticmethod
    def complete_oauth_callback(
        *,
        state: str,
        code: Optional[str],
        issuer: Optional[str] = None,
        error: Optional[str] = None,
        wait_seconds: float = 20.0,
    ) -> Optional[Dict[str, Any]]:
        flow = oauth_flows.complete_callback(
            state=state, code=code, issuer=issuer, error=error
        )
        if not flow:
            return None
        if wait_seconds > 0:
            flow.completed.wait(wait_seconds)
        completed = flow.completed.is_set()
        exchange_ok = bool(isinstance(flow.result, dict) and flow.result.get("ok"))
        return {
            "ok": not bool(error) and (exchange_ok if completed else True),
            "owner_id": flow.owner_id,
            "server_name": flow.server_name,
            "return_to": flow.return_to,
            "completed": completed,
            "pending": not completed,
            **(
                {"error": flow.result.get("error")}
                if completed
                and isinstance(flow.result, dict)
                and not exchange_ok
                and flow.result.get("error")
                else {}
            ),
        }

    def authorize_interactive(
        self,
        server_name: str,
        *,
        on_authorization_url: Optional[AuthorizationURLHandler] = None,
        callback_reader: Optional[AuthorizationCallbackReader] = None,
    ) -> Dict[str, Any]:
        """Complete OAuth in a terminal using a browser and pasted callback URL."""
        server = self.get_server(server_name)
        if server.auth.type != "oauth":
            raise MCPConfigurationError(
                f"MCP server '{server.name}' is not configured for OAuth",
                server.name,
            )

        def default_url_handler(url: str) -> None:
            print(f"Open this URL to authorize {server.name}:\n{url}\n")
            try:
                webbrowser.open(url)
            except Exception:
                pass

        def default_callback_reader() -> str:
            return input("Paste the full callback URL: ").strip()

        url_handler = on_authorization_url or default_url_handler
        reader = callback_reader or default_callback_reader

        async def redirect_handler(url: str) -> None:
            await asyncio.to_thread(url_handler, url)

        async def callback_handler():
            from mcp.shared.auth import AuthorizationCodeResult

            callback_url = await asyncio.to_thread(reader)
            parsed = urlparse(callback_url)
            params = parse_qs(parsed.query)
            error = (params.get("error") or [None])[0]
            if error:
                raise MCPAuthorizationRequired(
                    f"OAuth authorization failed: {error}", server.name
                )
            code = (params.get("code") or [None])[0]
            state = (params.get("state") or [None])[0]
            issuer = (params.get("iss") or [None])[0]
            if not code:
                raise MCPAuthorizationRequired(
                    "Callback URL did not contain an authorization code", server.name
                )
            return AuthorizationCodeResult(code=code, state=state, iss=issuer)

        try:
            result = self._run_async(
                self._list_tools_async(
                    server,
                    redirect_handler=redirect_handler,
                    callback_handler=callback_handler,
                )
            )
            probe = (
                None
                if self._credential_record(server).get("oauth_tokens")
                else _sign_in_probe(result.get("tools") or [])
            )
            if probe:
                self._run_async(
                    self._with_retry(
                        server,
                        lambda client: client.call_tool(probe, {}),
                        allow_retry=False,
                        redirect_handler=redirect_handler,
                        callback_handler=callback_handler,
                    )
                )
            return result
        except BaseException as exc:
            return self._safe_error(exc, server).to_dict()

    # --------------------------------------------------------------- safe export
