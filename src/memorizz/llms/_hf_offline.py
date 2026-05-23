# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Offline detection for HuggingFace-backed providers.

When the user runs the desktop UI without network access, the default
``transformers.pipeline`` / ``SentenceTransformer`` constructors still
issue HEAD requests to ``huggingface.co`` to revalidate cached files.
``huggingface_hub`` retries each failed request five times with
exponential backoff (1s, 2s, 4s, 8s, 8s ≈ 23s per file), turning a
single agent load into a 60–90 second hang and a wall of stack traces.

This module gives the HF providers a fast yes/no answer for "should we
even try the network?" — by checking the standard ``HF_HUB_OFFLINE`` /
``TRANSFORMERS_OFFLINE`` env vars and, as a fallback, doing a single
~1.5s TCP probe of ``huggingface.co``. The probe result is cached for
the lifetime of the process so we never repeat the wait.
"""

import logging
import os
import socket
from typing import Optional

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_PROBE_HOST = "huggingface.co"
_PROBE_PORT = 443
_PROBE_TIMEOUT_S = 1.5

_offline_cache: Optional[bool] = None


def _env_offline() -> bool:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MEMORIZZ_OFFLINE"):
        value = os.environ.get(key, "").strip().lower()
        if value in _TRUTHY:
            return True
    return False


def _probe_huggingface() -> bool:
    """Return True if huggingface.co is reachable within the probe timeout."""
    try:
        with socket.create_connection(
            (_PROBE_HOST, _PROBE_PORT), timeout=_PROBE_TIMEOUT_S
        ):
            return True
    except OSError:
        return False


def is_hf_offline(refresh: bool = False) -> bool:
    """Return True when HuggingFace Hub should be treated as offline.

    True if any of HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / MEMORIZZ_OFFLINE
    is set, or when a quick TCP probe to huggingface.co fails. The probe
    result is cached per-process; pass ``refresh=True`` to force a re-check
    (e.g. after the user reconnects).
    """
    global _offline_cache
    if _env_offline():
        return True
    if not refresh and _offline_cache is not None:
        return _offline_cache
    _offline_cache = not _probe_huggingface()
    if _offline_cache:
        logger.info(
            "huggingface.co unreachable — switching HF providers to "
            "local-files-only mode for this process."
        )
    return _offline_cache


def enable_hf_offline_env() -> None:
    """Force-enable HF offline mode in this process.

    Sets the standard env vars so any subprocess inherits the setting,
    and patches ``huggingface_hub.constants`` directly because that
    module reads the env var once at import time.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as _hf_const

        _hf_const.HF_HUB_OFFLINE = True
    except ImportError:
        pass
