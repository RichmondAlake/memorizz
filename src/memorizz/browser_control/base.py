# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Provider interface and registry for browser control."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .._registry import ProviderRegistry
from .models import BrowserControlResult

logger = logging.getLogger(__name__)


class BrowserControlProvider(ABC):
    """Interface for a provider that completes bounded browser tasks."""

    provider_name = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = dict(config or {})

    def get_provider_name(self) -> str:
        return str(getattr(self, "provider_name", self.__class__.__name__)).lower()

    def get_config(self) -> Dict[str, Any]:
        """Return secret-free, JSON-serializable persistence config."""
        return dict(self._config)

    def validate_configuration(self) -> Optional[str]:
        """Return an actionable configuration error, if any."""
        return None

    @abstractmethod
    def run_task(
        self,
        task: str,
        *,
        max_steps: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> BrowserControlResult:
        """Complete one browser task under provider policy."""

    def close(self) -> None:
        """Release provider resources."""
        return None


def _provider_key(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "").replace("_", "")


_REGISTRY: ProviderRegistry[BrowserControlProvider] = ProviderRegistry(
    logger=logger,
    unknown_message="Unknown browser-control provider: %s",
    normalize=_provider_key,
)
_PROVIDER_REGISTRY: Dict[str, type[BrowserControlProvider]] = _REGISTRY.providers


def register_provider(name: str, provider_cls: type[BrowserControlProvider]) -> None:
    _REGISTRY.register(name, provider_cls)


def get_provider_class(name: str) -> Optional[type[BrowserControlProvider]]:
    return _REGISTRY.get(name)


def create_browser_control_provider(
    name: str, config: Optional[Dict[str, Any]] = None
) -> Optional[BrowserControlProvider]:
    """Instantiate and validate a registered browser-control provider."""
    return _REGISTRY.create(name, config)
