# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Base classes for internet access providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import InternetPageContent, InternetSearchResult

logger = logging.getLogger(__name__)


class InternetAccessProvider(ABC):
    """Interface for providers that offer internet search / browsing."""

    provider_name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return getattr(self, "provider_name", self.__class__.__name__).lower()

    def get_config(self) -> Dict[str, Any]:
        """Return serializable config information."""
        return dict(self._config)

    @abstractmethod
    def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> List[InternetSearchResult]:
        """Search the internet and return normalized results."""

    @abstractmethod
    def fetch_url(self, url: str, **kwargs) -> InternetPageContent:
        """Fetch and parse the contents of a specific URL."""

    def close(self) -> None:
        """Cleanup resources (override when necessary)."""
        return None


_PROVIDER_REGISTRY: Dict[str, type[InternetAccessProvider]] = {}


def register_provider(name: str, provider_cls: type[InternetAccessProvider]) -> None:
    """Register an internet access provider by name."""
    _PROVIDER_REGISTRY[name.lower()] = provider_cls


def get_provider_class(name: str) -> Optional[type[InternetAccessProvider]]:
    """Return the provider class for a given name."""
    if not name:
        return None
    return _PROVIDER_REGISTRY.get(name.lower())


def create_internet_access_provider(
    name: str, config: Optional[Dict[str, Any]] = None
) -> Optional[InternetAccessProvider]:
    """Instantiate a provider from the registry."""
    provider_cls = get_provider_class(name)
    if not provider_cls:
        logger.warning("Unknown internet access provider: %s", name)
        return None

    config = config or {}
    try:
        return provider_cls(**config)
    except TypeError:
        try:
            return provider_cls(config=config)  # type: ignore[arg-type]
        except TypeError as exc:
            logger.error(
                "Failed to initialize provider '%s' with config keys: %s",
                name,
                list(config.keys()),
            )
            raise exc
