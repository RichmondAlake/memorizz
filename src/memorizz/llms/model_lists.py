"""The newest models each provider lists, for model pickers.

Asks the provider's own model-list endpoint (no tokens are spent) and keeps
the answer for a few hours. A provider without a key, or one that can't be
reached, gives an empty list, so callers can fall back to a built-in catalog.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Tuple

CACHE_SECONDS = 6 * 3600
_cache: Dict[str, Tuple[float, List[str]]] = {}
_lock = threading.Lock()

# OpenAI lists every model it serves; keep the ones that chat or run agents.
_OPENAI_KEEP = re.compile(r"^(gpt-\d|o\d|codex|chatgpt-)", re.IGNORECASE)
_OPENAI_SKIP = re.compile(
    r"audio|realtime|transcribe|tts|image|search|embedding|moderation|"
    r"instruct|vision|dall-e|whisper|-\d{4}-\d{2}-\d{2}$|-\d{4}$",
    re.IGNORECASE,
)


def _get_json(url: str, headers: Dict[str, str], timeout: float = 4.0) -> dict:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _anthropic() -> List[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return []
    body = _get_json(
        "https://api.anthropic.com/v1/models?limit=100",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    rows = [
        row
        for row in body.get("data") or []
        if str(row.get("id", "")).startswith("claude-")
    ]
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return [str(row["id"]) for row in rows]


def _openai() -> List[str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return []
    body = _get_json(
        "https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"}
    )
    rows = [
        row
        for row in body.get("data") or []
        if _OPENAI_KEEP.search(str(row.get("id", "")))
        and not _OPENAI_SKIP.search(str(row.get("id", "")))
    ]
    rows.sort(key=lambda row: int(row.get("created") or 0), reverse=True)
    return [str(row["id"]) for row in rows]


def _deepseek() -> List[str]:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return []
    body = _get_json(
        "https://api.deepseek.com/models", {"Authorization": f"Bearer {key}"}
    )
    return [str(row["id"]) for row in body.get("data") or [] if row.get("id")]


def _ollama() -> List[str]:
    host = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    if not host.startswith("http"):
        host = f"http://{host}"
    body = _get_json(f"{host.rstrip('/')}/api/tags", {}, timeout=2.0)
    rows = [
        row
        for row in body.get("models") or []
        # Embedding models can't chat.
        if "embed" not in str(row.get("name", "")).lower()
    ]
    rows.sort(key=lambda row: str(row.get("modified_at") or ""), reverse=True)
    return [str(row["name"]) for row in rows]


SOURCES: Dict[str, Callable[[], List[str]]] = {
    "anthropic": _anthropic,
    "openai": _openai,
    "deepseek": _deepseek,
    "ollama": _ollama,
}


def latest_models(provider: str, *, limit: int = 10) -> List[str]:
    """Up to ``limit`` of the provider's newest models, or [] when unknown."""
    name = str(provider or "").strip().lower()
    source = SOURCES.get(name)
    if source is None:
        return []
    now = time.monotonic()
    with _lock:
        cached = _cache.get(name)
        if cached and now - cached[0] < CACHE_SECONDS:
            return cached[1][:limit]
    try:
        models = source()
    except Exception:
        models = []
    with _lock:
        # A failed lookup is kept briefly, so pages don't wait on it again.
        _cache[name] = (now if models else now - CACHE_SECONDS + 300, models)
    return models[:limit]


def latest_models_for(
    providers: Iterable[str], *, limit: int = 10
) -> Dict[str, List[str]]:
    """Several providers at once, asked in parallel."""
    names = [str(name).lower() for name in dict.fromkeys(providers) if name]
    if not names:
        return {}
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        found = list(pool.map(lambda name: latest_models(name, limit=limit), names))
    return dict(zip(names, found))


def available_providers() -> List[str]:
    """Providers that can be listed here: a key is set, or Ollama is local."""
    keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    found = [name for name, env in keys.items() if os.environ.get(env)]
    return [*found, "ollama"]


def clear_cache(provider: Optional[str] = None) -> None:
    with _lock:
        if provider:
            _cache.pop(provider.lower(), None)
        else:
            _cache.clear()


__all__ = [
    "available_providers",
    "clear_cache",
    "latest_models",
    "latest_models_for",
]
