"""Discover account-visible or installed models without exposing credentials."""

from __future__ import annotations

import copy
import hashlib
import os
import re
import time
from urllib.parse import urlsplit

import requests

from .measurement import token_price

_cache: dict[tuple, tuple[float, dict]] = {}


def _text_model(name: str) -> bool:
    """Conservative text-family filter; the list API does not certify endpoints."""
    return bool(re.match(r"^(gpt-|chatgpt-|o[134](?:-|$)|ft:)", name)) and not any(
        word in name
        for word in (
            "audio",
            "realtime",
            "transcribe",
            "tts",
            "image",
            "search",
            "instruct",
            "codex",
        )
    )


def discover_models(
    provider: str, *, host: str = "http://localhost:11434", refresh: bool = False
) -> dict:
    key_name = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai_decisions": "OPENAI_API_KEY",
    }.get(provider)
    key = os.environ.get(key_name, "") if key_name else ""
    cache_key = (provider, host, hashlib.sha256(key.encode()).hexdigest())
    cached = _cache.get(cache_key)
    if not refresh and cached and time.monotonic() - cached[0] < 60:
        return copy.deepcopy(cached[1])
    data = {
        "provider": provider,
        "models": [],
        "source": "unavailable",
        "message": "",
        "fetched_at": time.time(),
    }
    try:
        entries = []
        if key_name and not key:
            data["message"] = f"Configure {key_name} on the server to list your models."
            return data
        if provider == "openai":
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=12,
            )
            response.raise_for_status()
            entries = [
                (r["id"], r["id"])
                for r in response.json()["data"]
                if _text_model(r["id"])
            ]
            data.update(
                source="provider_api",
                message="Account-visible text-model families. Endpoint compatibility and quota are checked when a run starts.",
            )
        elif provider == "anthropic":
            after = None
            for _ in range(10):
                params = {"limit": 1000}
                if after:
                    params["after_id"] = after
                response = requests.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                    params=params,
                    timeout=12,
                )
                response.raise_for_status()
                payload = response.json()
                entries.extend(
                    (r["id"], r.get("display_name") or r["id"]) for r in payload["data"]
                )
                if not payload.get("has_more"):
                    break
                next_id = payload.get("last_id")
                if not next_id or next_id == after:
                    raise ValueError("Invalid model pagination")
                after = next_id
            data.update(
                source="provider_api",
                message="Models returned by your Anthropic account. Quota is checked when a run starts.",
            )
        elif provider == "ollama":
            url = urlsplit(host)
            if (
                url.scheme not in {"http", "https"}
                or not url.netloc
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError("Invalid Ollama host")
            response = requests.get(host.rstrip("/") + "/api/tags", timeout=5)
            response.raise_for_status()
            entries = [
                (r.get("name") or r["model"], r.get("name") or r["model"])
                for r in response.json()["models"]
                if not any(
                    w in (r.get("name") or r.get("model", "")).lower()
                    for w in ("embed", "rerank")
                )
            ]
            data.update(
                source="installed",
                message="Installed Ollama generation models. Embedding models are excluded.",
            )
        elif provider in {"huggingface", "mlx", "cross_encoder"}:
            from huggingface_hub import scan_cache_dir

            entries = [
                (r.repo_id, r.repo_id)
                for r in scan_cache_dir().repos
                if r.repo_type == "model"
            ]
            data.update(
                source="local_cache",
                message="Locally cached model repositories; verify compatibility with this adapter.",
            )
        elif provider == "openai_decisions":
            entries = [("gpt-6-luna", "GPT-6 Luna · Decisions API")]
            data.update(
                source="documented_catalog",
                message="Dedicated /v1/decisions endpoint. Typed predicates, choices and scores; account access is checked on the live request.",
            )
        elif provider in {"jev", "voyage"}:
            names = (
                ["jev-1.13.0", "jev-latest"]
                if provider == "jev"
                else ["rerank-2.5", "rerank-3"]
            )
            entries = [(name, name) for name in names]
            data.update(
                source="documented_catalog",
                message="Documented model IDs; account access is verified on the live request. Pin a version for comparisons.",
            )
        elif provider == "azure":
            deployment = os.environ.get(
                "AZURE_OPENAI_DEPLOYMENT_NAME"
            ) or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            entries = [(deployment, deployment)] if deployment else []
            data.update(
                source="configuration",
                message="Configured Azure deployment. Listing every deployment requires Azure management credentials; custom deployment names remain available.",
            )
        else:
            data[
                "message"
            ] = "This adapter has no model discovery endpoint configured. Use a custom model name."
        for model, label in sorted(set(entries)):
            data["models"].append(
                {"id": model, "name": label, "pricing": token_price(provider, model)}
            )
        if not entries and not data["message"]:
            data[
                "message"
            ] = "No compatible models were returned. Use a custom model name or check provider configuration."
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 502
        data.update(
            source="unavailable",
            message=f"Model discovery returned HTTP {code}. Check the provider key and permissions.",
        )
    except Exception:
        data.update(
            source="unavailable",
            message="Could not discover models. Check the provider connection or use a custom model name.",
        )
    _cache[cache_key] = (time.monotonic(), data)
    return copy.deepcopy(data)
