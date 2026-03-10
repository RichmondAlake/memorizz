# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Internet access provider interfaces and implementations."""

import logging
import os

from .base import (
    InternetAccessProvider,
    create_internet_access_provider,
    get_provider_class,
    register_provider,
)
from .models import InternetPageContent, InternetSearchResult
from .providers.firecrawl import FirecrawlProvider
from .providers.offline import OfflineInternetProvider
from .providers.tavily import TavilyProvider

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER_ENV = "MEMORIZZ_DEFAULT_INTERNET_PROVIDER"
DEFAULT_PROVIDER_API_KEY_ENV = "MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY"

__all__ = [
    "InternetAccessProvider",
    "InternetPageContent",
    "InternetSearchResult",
    "FirecrawlProvider",
    "TavilyProvider",
    "OfflineInternetProvider",
    "create_internet_access_provider",
    "register_provider",
    "get_provider_class",
    "get_default_internet_access_provider",
]


def get_default_internet_access_provider() -> InternetAccessProvider:
    """
    Return a usable internet provider for Deep Research agents.

    Preference order:
    1. Explicit provider via MEMORIZZ_DEFAULT_INTERNET_PROVIDER.
    2. Tavily (TAVILY_API_KEY).
    3. Firecrawl (FIRECRAWL_API_KEY).
    4. Offline provider placeholder so the tool still responds.
    """

    provider_name = os.getenv(DEFAULT_PROVIDER_ENV)
    provider_config = {}
    if provider_name:
        api_key = os.getenv(DEFAULT_PROVIDER_API_KEY_ENV)
        if api_key:
            provider_config["api_key"] = api_key
        try:
            provider = create_internet_access_provider(provider_name, provider_config)
            if provider:
                return provider
        except Exception as exc:  # pragma: no cover - best effort fallback
            logger.warning(
                "Failed to initialize provider '%s' from env: %s", provider_name, exc
            )

    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            return TavilyProvider(api_key=tavily_key)
        except Exception as exc:  # pragma: no cover - best effort fallback
            logger.warning("Failed to initialize Tavily provider: %s", exc)

    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    if firecrawl_key:
        try:
            return FirecrawlProvider(api_key=firecrawl_key)
        except Exception as exc:  # pragma: no cover - best effort fallback
            logger.warning("Failed to initialize Firecrawl provider: %s", exc)

    reason = (
        "Set TAVILY_API_KEY, FIRECRAWL_API_KEY, or MEMORIZZ_DEFAULT_INTERNET_PROVIDER "
        "to enable live internet access."
    )
    return OfflineInternetProvider(reason=reason)
