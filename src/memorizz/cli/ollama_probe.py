# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tiny stdlib-only client for probing a local Ollama daemon.

Mirrors the request shapes used by the web UI (``ui/app.py`` ``/api/tags`` and
``/api/pull``) so the CLI's zero-config detection and ``/ollama`` slash command
behave identically. Uses only ``urllib`` — no extra dependencies, no import of
the heavy ``ollama`` SDK just to check reachability.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

DEFAULT_HOST = "http://localhost:11434"


def resolve_host(override: Optional[str] = None) -> str:
    """Resolve the Ollama base URL (explicit override > ``OLLAMA_HOST`` > default)."""
    host = override or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    return host.rstrip("/")


def probe(host: Optional[str] = None, timeout: int = 3) -> Dict[str, object]:
    """Probe the daemon's ``/api/tags`` endpoint.

    Returns a dict shaped like the UI's response::

        {"reachable": bool, "host": str, "models": [str, ...], "error": str|None}

    ``models`` is empty (but ``reachable`` True) when the daemon is up with no
    pulled models. ``reachable`` is False on any connection/timeout error.
    """
    import urllib.error
    import urllib.request

    base = resolve_host(host)
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {
            "reachable": False,
            "host": base,
            "models": [],
            "error": str(exc.reason),
        }
    except (TimeoutError, OSError, ValueError) as exc:
        return {"reachable": False, "host": base, "models": [], "error": str(exc)}

    models: List[str] = []
    for entry in payload.get("models", []) or []:
        name = entry.get("name") or entry.get("model")
        if name:
            models.append(name)
    return {"reachable": True, "host": base, "models": models, "error": None}


def reachable(host: Optional[str] = None, timeout: int = 2) -> bool:
    """Return True if the Ollama daemon answers ``/api/tags``."""
    return bool(probe(host, timeout=timeout)["reachable"])


def list_models(host: Optional[str] = None) -> Optional[List[str]]:
    """Return the list of pulled model names, or ``None`` if the daemon is down."""
    result = probe(host)
    if not result["reachable"]:
        return None
    return list(result["models"])  # type: ignore[arg-type]


def pull(
    name: str, host: Optional[str] = None, timeout: int = 1800
) -> Tuple[bool, str]:
    """Pull a model via ``POST /api/pull`` (blocking). Returns ``(ok, message)``."""
    import urllib.error
    import urllib.request

    base = resolve_host(host)
    body = json.dumps({"name": name, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/pull",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return False, str(exc.reason)
    except (TimeoutError, OSError, ValueError) as exc:
        return False, str(exc)

    if payload.get("status") == "success":
        return True, f"Pulled {name}"
    return False, payload.get("error") or json.dumps(payload)
