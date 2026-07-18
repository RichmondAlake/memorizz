# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Ollama daemon endpoints (list installed / pull / delete models).

Extracted verbatim from ui/app.py — behavior is unchanged. These talk to the
local Ollama HTTP API via stdlib urllib (no SDK), reading the daemon host from
``OLLAMA_HOST`` (default ``http://localhost:11434``).
"""

import json
import os
from typing import List

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/ollama", tags=["ollama"])


def _ollama_host() -> str:
    return (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")


@router.post("/pull")
async def ollama_pull(name: str = Form(...)):
    """Pull an Ollama model. Synchronous — large pulls can take 5-15 min.

    We pass ``stream: false`` so the daemon buffers progress on its side and
    only returns when the operation finishes.
    """
    import urllib.error
    import urllib.request

    host = _ollama_host()
    body = json.dumps({"name": name, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/pull",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return JSONResponse(
            {"ok": False, "error": f"HTTP {exc.code}: {exc.reason}"},
            status_code=exc.code,
        )
    except urllib.error.URLError as exc:
        return JSONResponse({"ok": False, "error": str(exc.reason)}, status_code=502)
    except (TimeoutError, OSError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=504)

    if payload.get("status") == "success":
        return JSONResponse({"ok": True, "message": f"Pulled {name}"})
    return JSONResponse(
        {"ok": False, "error": payload.get("error") or json.dumps(payload)},
        status_code=500,
    )


@router.delete("/models/{name:path}")
async def ollama_delete(name: str):
    """Remove a locally pulled Ollama model via Ollama's /api/delete."""
    import urllib.error
    import urllib.request

    host = _ollama_host()
    body = json.dumps({"name": name}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/delete",
        data=body,
        method="DELETE",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
    except urllib.error.HTTPError as exc:
        detail = (
            exc.read().decode("utf-8", "ignore") if hasattr(exc, "read") else exc.reason
        )
        return JSONResponse(
            {"ok": False, "error": f"HTTP {exc.code}: {detail}"},
            status_code=exc.code,
        )
    except urllib.error.URLError as exc:
        return JSONResponse({"ok": False, "error": str(exc.reason)}, status_code=502)
    except (TimeoutError, OSError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=504)

    return JSONResponse({"ok": True, "message": f"Removed {name}"})


@router.get("/installed")
async def ollama_installed():
    """Report which models the local Ollama daemon has pulled.

    Powers the "Not installed yet" banner under the Default Model picker in
    Settings, so the user sees a copy-paste ``ollama pull <tag>`` before a model
    is ever sent to the LLM.
    """
    import urllib.error
    import urllib.request

    host = _ollama_host()
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return JSONResponse(
            {"reachable": False, "host": host, "error": str(exc.reason)}
        )
    except (TimeoutError, OSError, ValueError) as exc:
        return JSONResponse({"reachable": False, "host": host, "error": str(exc)})

    models: List[str] = []
    for entry in payload.get("models", []) or []:
        name = entry.get("name") or entry.get("model")
        if name:
            models.append(name)
    return JSONResponse({"reachable": True, "host": host, "models": models})
