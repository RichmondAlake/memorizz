# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Offline fallback provider for environments without external access."""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import InternetAccessProvider, register_provider
from ..models import InternetPageContent, InternetSearchResult


class OfflineInternetProvider(InternetAccessProvider):
    """Provider that returns informative placeholders when internet access is disabled."""

    provider_name = "offline"

    def __init__(self, reason: str = "Internet access provider is not configured"):
        super().__init__({"reason": reason})
        self.reason = reason

    def search(
        self, query: str, max_results: int = 5, **kwargs: Any
    ) -> List[InternetSearchResult]:
        message = (
            f"Internet access unavailable: {self.reason}. Configure FIRECRAWL_API_KEY, "
            "TAVILY_API_KEY, or MEMORIZZ_DEFAULT_INTERNET_PROVIDER to enable live search."
        )
        return [
            InternetSearchResult(
                url="",
                title="Internet access unavailable",
                snippet=message,
                metadata={"status": "offline"},
            )
        ]

    def fetch_url(self, url: str, **kwargs: Any) -> InternetPageContent:
        message = (
            f"Cannot fetch '{url}' because internet access is disabled. "
            "Configure FIRECRAWL_API_KEY, TAVILY_API_KEY, or MEMORIZZ_DEFAULT_INTERNET_PROVIDER "
            "to enable browsing."
        )
        return InternetPageContent(
            url=url,
            title="Internet access unavailable",
            content=message,
            metadata={"status": "offline"},
        )


register_provider(OfflineInternetProvider.provider_name, OfflineInternetProvider)
