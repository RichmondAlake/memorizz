# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Publish a local MemAgent definition to a hosted MemoRizz Simuground.

Typical workshop usage::

    from memorizz import cloud

    cloud.connect("https://platform.memorizz.dev", key="wk_...")
    agent.publish()

Only the agent's definition travels: its name, instruction, and persona
(role, goals, background). Its model choice, local tools, MCP servers,
sandbox, voice, and local memory stay on your machine. The workshop key is
kept in memory only and is never written to the agent, a store, or a trace.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from . import __version__

MAX_NAME_CHARS = 40
MAX_INSTRUCTION_CHARS = 4_000
MAX_PERSONA_FIELD_CHARS = 600

_lock = threading.RLock()
_connection: Dict[str, Optional[str]] = {"url": None, "key": None}


class CloudError(RuntimeError):
    """Actionable failure when connecting to or publishing to MemoRizz Cloud."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "cloud_error",
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _normalize_url(raw: str) -> str:
    text = str(raw or "").strip().rstrip("/")
    if not text:
        raise CloudError(
            "Pass the MemoRizz server URL to cloud.connect(url, key=...) "
            "or set MEMORIZZ_CLOUD_URL.",
            code="missing_url",
        )
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CloudError(
            f"Invalid MemoRizz Cloud URL {text!r}: expected https://...",
            code="invalid_url",
        )
    if parsed.username or parsed.password:
        raise CloudError(
            "Do not put credentials in the MemoRizz Cloud URL; pass key=... instead.",
            code="invalid_url",
        )
    return text


def connect(
    url: Optional[str] = None,
    key: Optional[str] = None,
) -> Dict[str, str]:
    """Configure the MemoRizz server URL and workshop key for ``agent.publish()``.

    Falls back to ``MEMORIZZ_CLOUD_URL`` and ``MEMORIZZ_CLOUD_KEY`` when either
    argument is omitted. Returns ``{"url": ...}`` (never the key).
    """
    resolved_url = _normalize_url(
        url if url is not None else os.environ.get("MEMORIZZ_CLOUD_URL", "")
    )
    resolved_key = str(
        key if key is not None else os.environ.get("MEMORIZZ_CLOUD_KEY", "")
    ).strip()
    if not resolved_key:
        raise CloudError(
            "Pass your workshop key to cloud.connect(url, key=...) "
            "or set MEMORIZZ_CLOUD_KEY.",
            code="missing_key",
        )
    with _lock:
        _connection["url"] = resolved_url
        _connection["key"] = resolved_key
    return {"url": resolved_url}


def disconnect() -> None:
    """Clear the in-memory MemoRizz Cloud connection settings."""
    with _lock:
        _connection["url"] = None
        _connection["key"] = None


def _resolve_connection(
    url: Optional[str] = None, key: Optional[str] = None
) -> Tuple[str, str]:
    with _lock:
        saved_url = _connection.get("url")
        saved_key = _connection.get("key")
    raw_url = (
        url
        if url is not None
        else (saved_url or os.environ.get("MEMORIZZ_CLOUD_URL", ""))
    )
    raw_key = (
        key
        if key is not None
        else (saved_key or os.environ.get("MEMORIZZ_CLOUD_KEY", ""))
    )
    resolved_url = _normalize_url(raw_url)
    resolved_key = str(raw_key or "").strip()
    if not resolved_key:
        raise CloudError(
            "No workshop key configured. Call cloud.connect(url, key=...) "
            "or set MEMORIZZ_CLOUD_KEY.",
            code="missing_key",
        )
    return resolved_url, resolved_key


def _extract_persona(agent: Any) -> Optional[Dict[str, str]]:
    persona = None
    if isinstance(agent, Mapping):
        persona = agent.get("persona")
    else:
        manager = getattr(agent, "persona_manager", None)
        if manager is not None:
            persona = getattr(manager, "configuration_persona", None) or getattr(
                manager, "current_persona", None
            )
        if persona is None:
            persona = getattr(agent, "persona", None)
    if persona is None:
        return None

    def _get(field: str) -> str:
        val = (
            persona.get(field)
            if isinstance(persona, Mapping)
            else getattr(persona, field, None)
        )
        if hasattr(val, "value"):
            val = val.value
        return str(val or "").strip()

    extracted: Dict[str, str] = {}
    for field in ("role", "goals", "background"):
        text = _get(field)
        if text:
            extracted[field] = text
    return extracted or None


def definition(agent: Any) -> Dict[str, Any]:
    """Extract the uploadable definition of a MemAgent (no tools, model, voice, or memory)."""
    if isinstance(agent, Mapping):
        raw_name = agent.get("name")
        raw_instruction = agent.get("instruction")
        persona_obj = agent.get("persona")
        persona_name = (
            persona_obj.get("name")
            if isinstance(persona_obj, Mapping)
            else getattr(persona_obj, "name", None)
        )
    else:
        raw_name = getattr(agent, "name", None)
        raw_instruction = getattr(agent, "instruction", None)
        manager = getattr(agent, "persona_manager", None)
        persona_obj = (
            getattr(manager, "configuration_persona", None)
            if manager is not None
            else getattr(agent, "persona", None)
        )
        persona_name = (
            getattr(persona_obj, "name", None) if persona_obj is not None else None
        )

    name = " ".join(str(raw_name or persona_name or "").split()).strip()
    if not name:
        raise CloudError(
            "Give your agent a name (or a Persona with a name) before publishing.",
            code="name_required",
        )

    instruction = str(raw_instruction or "").strip()
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        raise CloudError(
            f"Agent instruction is {len(instruction):,} characters; "
            f"shorten it to at most {MAX_INSTRUCTION_CHARS:,} characters before publishing.",
            code="instruction_too_long",
        )

    persona = _extract_persona(agent)
    if persona:
        for field, value in persona.items():
            if len(value) > MAX_PERSONA_FIELD_CHARS:
                raise CloudError(
                    f"Persona {field} is {len(value):,} characters; "
                    f"shorten it to at most {MAX_PERSONA_FIELD_CHARS:,} characters before publishing.",
                    code="persona_field_too_long",
                )

    payload: Dict[str, Any] = {
        "name": name[:MAX_NAME_CHARS].strip(),
        "instruction": instruction,
        "client": f"memorizz {__version__}",
    }
    if persona:
        payload["persona"] = persona
    return payload


def dropped_capabilities(agent: Any) -> List[str]:
    """Capabilities present on ``agent`` locally that are intentionally not uploaded."""
    if isinstance(agent, Mapping):
        return []
    dropped: List[str] = []
    tm = getattr(agent, "tool_manager", None)
    if tm is not None and callable(getattr(tm, "list_tools", None)):
        own_tools = [
            name
            for name in (tm.list_tools() or [])
            if not (
                isinstance(tm.get_tool_metadata(name), dict)
                and tm.get_tool_metadata(name).get("source") == "environment"
            )
        ]
        if own_tools:
            dropped.append("local tools")
    if getattr(agent, "mcp_servers", None):
        dropped.append("MCP connections")
    if getattr(agent, "voice", False):
        dropped.append("voice")
    if getattr(agent, "knowledge_base_ids", None):
        dropped.append("knowledge bases")
    if getattr(agent, "memory_ids", None):
        dropped.append("existing memory")
    if getattr(agent, "delegates", None):
        dropped.append("delegates")
    if getattr(agent, "meta_harness", False):
        dropped.append("harness")
    return dropped


def _friendly_http_error(status_code: int, body: Any, default_url: str) -> CloudError:
    error_code = ""
    detail = ""
    if isinstance(body, Mapping):
        error_code = str(body.get("error") or "").strip()
        detail = str(body.get("detail") or "").strip()

    if error_code == "invalid_key" or error_code == "missing_key":
        return CloudError(
            "Your workshop key was rejected. Check that you copied the full wk_... key and try cloud.connect() again.",
            code=error_code or "invalid_key",
            status_code=status_code,
        )
    if error_code == "key_expired":
        return CloudError(
            detail or "Your workshop key has expired. Ask the host for a fresh key.",
            code="key_expired",
            status_code=status_code,
        )
    if error_code == "key_revoked":
        return CloudError(
            "Your workshop key has been revoked by the host.",
            code="key_revoked",
            status_code=status_code,
        )
    if error_code == "rate_limited":
        return CloudError(
            detail
            or "Too many publish requests in one minute. Wait a few seconds and try again.",
            code="rate_limited",
            status_code=status_code,
        )
    if error_code == "world_full":
        return CloudError(
            detail or "The Simuground is full right now. Ask the host to make space.",
            code="world_full",
            status_code=status_code,
        )
    if error_code in {
        "instruction_too_long",
        "persona_field_too_long",
        "name_required",
        "unsupported_field",
        "payload_too_large",
    }:
        return CloudError(
            detail or f"Publish rejected ({error_code}).",
            code=error_code,
            status_code=status_code,
        )
    if status_code == 401:
        return CloudError(
            "Authentication failed (HTTP 401). Check your workshop key in cloud.connect().",
            code="unauthorized",
            status_code=401,
        )
    if status_code == 404:
        return CloudError(
            f"Publish endpoint not found at {default_url}. Check the server URL in cloud.connect().",
            code="not_found",
            status_code=404,
        )
    if status_code >= 500:
        return CloudError(
            detail
            or f"The MemoRizz server returned HTTP {status_code}. Try again in a moment.",
            code=error_code or "server_error",
            status_code=status_code,
        )
    return CloudError(
        detail or f"Publish failed with HTTP {status_code}.",
        code=error_code or "http_error",
        status_code=status_code,
    )


def _request_json(
    method: str,
    endpoint: str,
    *,
    key: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    import requests

    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": f"memorizz/{__version__}",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"

    try:
        response = requests.request(
            method,
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise CloudError(
            f"Timed out waiting for {endpoint}. Check your connection and try again.",
            code="timeout",
        ) from exc
    except requests.ConnectionError as exc:
        raise CloudError(
            f"Could not reach {endpoint}. Check the URL and your network connection.",
            code="connection_error",
        ) from exc
    except requests.RequestException as exc:
        raise CloudError(
            f"Network request to {endpoint} failed: {type(exc).__name__}.",
            code="network_error",
        ) from exc

    try:
        body: Any = response.json()
    except ValueError:
        body = None

    if not response.ok:
        raise _friendly_http_error(response.status_code, body, endpoint)
    if not isinstance(body, dict):
        raise CloudError(
            f"Unexpected response from {endpoint} (expected JSON object).",
            code="invalid_response",
            status_code=response.status_code,
        )
    return body


def publish(
    agent: Any,
    *,
    url: Optional[str] = None,
    key: Optional[str] = None,
    timeout: float = 30.0,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Publish ``agent``'s definition to the configured MemoRizz Simuground.

    Publishing a second time with the same workshop key updates your existing
    agent in place rather than adding a duplicate.
    """
    base_url, resolved_key = _resolve_connection(url=url, key=key)
    payload = definition(agent)
    endpoint = f"{base_url}/simuground/api/publish"
    result = _request_json(
        "POST", endpoint, key=resolved_key, payload=payload, timeout=timeout
    )
    if not quiet:
        verb = "Updated" if result.get("updated") else "Published"
        name = result.get("name") or payload["name"]
        agent_id = result.get("agent_id") or ""
        watch = result.get("watch") or f"{base_url}/simuground/watch"
        print(f"{verb} {name} ({agent_id}) in the Simuground.")
        print(f"Watch live: {watch}")
        dropped = dropped_capabilities(agent)
        if dropped:
            print(
                "Note: "
                + ", ".join(dropped)
                + " stay local; your uploaded agent runs on the workshop model with the space's tools."
            )
    return result


def status(
    *,
    url: Optional[str] = None,
    key: Optional[str] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Check the status of the agent bound to your workshop key."""
    base_url, resolved_key = _resolve_connection(url=url, key=key)
    endpoint = f"{base_url}/simuground/api/publish"
    return _request_json("GET", endpoint, key=resolved_key, timeout=timeout)


__all__ = [
    "CloudError",
    "MAX_INSTRUCTION_CHARS",
    "MAX_NAME_CHARS",
    "MAX_PERSONA_FIELD_CHARS",
    "connect",
    "definition",
    "disconnect",
    "dropped_capabilities",
    "publish",
    "status",
]
