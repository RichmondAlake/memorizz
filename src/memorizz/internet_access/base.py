# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Base classes for internet access providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .._registry import ProviderRegistry
from .models import InternetPageContent, InternetSearchResult

logger = logging.getLogger(__name__)

# Config keys whose values are credentials. They never leave ``get_config()``
# and are dropped from any saved config on restore.
_SECRET_CONFIG_KEYS = frozenset(
    {"api_key", "apikey", "api_token", "token", "secret", "password", "authorization"}
)
# Keys that describe a saved config and are never constructor arguments.
_DERIVED_CONFIG_KEYS = frozenset({"api_key_set"})


def scrub_secret_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A copy of ``config`` without credential values.

    ``api_key_set`` records that a key was present, so a restored provider
    knows to resolve it from the environment instead.
    """
    source = dict(config or {})
    had_key = any(
        str(key).lower() in _SECRET_CONFIG_KEYS and bool(value)
        for key, value in source.items()
    )
    safe = {
        key: value
        for key, value in source.items()
        if str(key).lower() not in _SECRET_CONFIG_KEYS
    }
    if had_key or "api_key_set" in source:
        safe["api_key_set"] = bool(had_key or source.get("api_key_set"))
    return safe


class InternetAccessProvider(ABC):
    """Interface for providers that offer internet search / browsing."""

    provider_name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return getattr(self, "provider_name", self.__class__.__name__).lower()

    def get_config(self) -> Dict[str, Any]:
        """Return serializable config information; never the API key itself.

        Providers that hold a key report ``api_key_set`` instead. The saved
        agent record carries this config, so the key must come back from the
        environment (``TAVILY_API_KEY`` and the like) on restore.
        """
        safe = scrub_secret_config(self._config)
        if hasattr(self, "api_key"):
            safe["api_key_set"] = bool(
                getattr(self, "api_key", None) or safe.get("api_key_set")
            )
        return safe

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

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        """A positive int from a setting, or None."""
        if value is None:
            return None
        try:
            result = int(value)
        except (TypeError, ValueError):
            return None
        return result if result > 0 else None

    @staticmethod
    def _truncate_text(
        value: Optional[str], limit: Optional[int]
    ) -> Tuple[Optional[str], bool, Optional[int]]:
        """``(text, was_truncated, original_length)``, cut to ``limit`` chars."""
        if not value:
            return value, False, None
        if not limit or limit <= 0:
            return value, False, len(value)
        original_length = len(value)
        if original_length <= limit:
            return value, False, original_length
        return value[:limit], True, original_length


_REGISTRY: ProviderRegistry[InternetAccessProvider] = ProviderRegistry(
    logger=logger,
    unknown_message="Unknown internet access provider: %s",
    init_error_message="Failed to initialize provider '%s' with config keys: %s",
    # ``api_key_set`` only describes a saved config; constructors never take it.
    drop_config_keys=_DERIVED_CONFIG_KEYS,
    validate=False,
)
_PROVIDER_REGISTRY: Dict[str, type[InternetAccessProvider]] = _REGISTRY.providers


def register_provider(name: str, provider_cls: type[InternetAccessProvider]) -> None:
    """Register an internet access provider by name."""
    _REGISTRY.register(name, provider_cls)


def get_provider_class(name: str) -> Optional[type[InternetAccessProvider]]:
    """Return the provider class for a given name."""
    return _REGISTRY.get(name)


def create_internet_access_provider(
    name: str, config: Optional[Dict[str, Any]] = None
) -> Optional[InternetAccessProvider]:
    """Instantiate a provider from the registry."""
    return _REGISTRY.create(name, config)
