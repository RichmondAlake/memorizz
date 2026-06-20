# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""HuggingFace model-cache endpoints (pull / delete / list installed).

Extracted verbatim from ui/app.py — behavior is unchanged. These use
``huggingface_hub`` (an optional dependency: ``pip install memorizz[huggingface]``)
and degrade gracefully when the SDK isn't installed in the current env.
"""

from typing import List

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/huggingface", tags=["huggingface"])


@router.post("/pull")
async def huggingface_pull(repo_id: str = Form(...)):
    """Download a HuggingFace repo into the local cache.

    Uses snapshot_download so all repo files (config + tokenizer + weights) are
    pulled together. Honors HF_TOKEN from the environment for gated repos.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return JSONResponse(
            {"ok": False, "error": "huggingface_hub not installed"},
            status_code=503,
        )
    try:
        path = snapshot_download(repo_id=repo_id)
    except Exception as exc:  # pragma: no cover - depends on network
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse(
        {"ok": True, "message": f"Downloaded {repo_id}", "path": str(path)}
    )


@router.delete("/models/{repo_id:path}")
async def huggingface_delete(repo_id: str):
    """Delete every revision of a cached HuggingFace model repo."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return JSONResponse(
            {"ok": False, "error": "huggingface_hub not installed"},
            status_code=503,
        )
    try:
        cache_info = scan_cache_dir()
    except Exception as exc:  # pragma: no cover - defensive
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    revisions = []
    for repo in cache_info.repos:
        if repo.repo_id == repo_id and repo.repo_type == "model":
            for rev in repo.revisions:
                revisions.append(rev.commit_hash)
    if not revisions:
        return JSONResponse(
            {"ok": False, "error": f"{repo_id} not found in HF cache"},
            status_code=404,
        )

    try:
        strategy = cache_info.delete_revisions(*revisions)
        strategy.execute()
    except Exception as exc:  # pragma: no cover - defensive
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return JSONResponse(
        {
            "ok": True,
            "message": f"Removed {repo_id}",
            "freed_bytes": getattr(strategy, "expected_freed_size", None),
        }
    )


@router.get("/installed")
async def huggingface_installed():
    """Report which HuggingFace repos are cached locally.

    Powers the same "Not cached locally" / "Available offline" banner as Ollama,
    but for the ``huggingface`` provider — by walking the standard HF cache
    directory via huggingface_hub.scan_cache_dir(). Falls back to
    ``{available: false}`` when the SDK isn't installed so the UI can render an
    actionable warning.
    """
    try:
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError as exc:
        return JSONResponse(
            {
                "available": False,
                "error": (
                    "huggingface_hub not installed — run "
                    "`pip install memorizz[huggingface]` to enable cache scanning."
                ),
                "detail": str(exc),
            }
        )

    try:
        cache_info = scan_cache_dir()
    except Exception as exc:  # pragma: no cover - defensive
        return JSONResponse({"available": False, "error": str(exc)})

    models: List[str] = []
    for repo in getattr(cache_info, "repos", []) or []:
        # Only model repos count toward "available LLMs". Datasets and spaces
        # show up here too but aren't valid HF LLM provider IDs.
        if getattr(repo, "repo_type", "model") != "model":
            continue
        repo_id = getattr(repo, "repo_id", None)
        if repo_id:
            models.append(repo_id)
    models.sort()

    return JSONResponse(
        {
            "available": True,
            "cache_dir": str(HF_HUB_CACHE or ""),
            "models": models,
        }
    )
