# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Manager responsible for routing internet access actions to providers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...internet_access import InternetAccessProvider
from ...internet_access.models import InternetPageContent, InternetSearchResult
from ._provider_slot import ProviderSlot


class InternetAccessManager(ProviderSlot):
    """Wrapper over InternetAccessProvider implementations."""

    provider_kind = "internet"

    def __init__(self, provider: Optional[InternetAccessProvider] = None):
        self.provider = provider

    def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute a search query using the provider."""
        if not self.provider:
            raise ValueError("Internet access provider is not configured")
        results = self.provider.search(query=query, max_results=max_results, **kwargs)
        return [self._result_to_dict(item) for item in results]

    def fetch_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """Fetch a URL using the provider."""
        if not self.provider:
            raise ValueError("Internet access provider is not configured")
        page = self.provider.fetch_url(url=url, **kwargs)
        return self._page_to_dict(page)

    # Serialization helpers -------------------------------------------------
    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, InternetSearchResult):
            return result.to_dict()
        if isinstance(result, dict):
            return result
        return {"value": result}

    def _page_to_dict(self, page: Any) -> Dict[str, Any]:
        if isinstance(page, InternetPageContent):
            return page.to_dict()
        if isinstance(page, dict):
            return page
        return {"content": page}
