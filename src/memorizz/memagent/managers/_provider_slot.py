"""The optional provider a manager uses (internet access, a sandbox)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ProviderSlot:
    """Attach, detach and describe one optional provider. Managers set
    ``provider`` in ``__init__`` and name the kind for log messages."""

    provider_kind = "provider"
    provider: Optional[Any] = None

    def set_provider(self, provider: Optional[Any]) -> Optional[Any]:
        """Attach or detach a provider; the previous one is closed."""
        previous = self.provider
        if previous and previous is not provider:
            try:
                previous.close()
            except Exception as exc:
                logger.debug(
                    "Failed to close previous %s provider: %s", self.provider_kind, exc
                )
        self.provider = provider
        return previous

    def is_enabled(self) -> bool:
        """Return True if a provider is available."""
        return self.provider is not None

    def get_provider_name(self) -> Optional[str]:
        if not self.provider:
            return None
        return self.provider.get_provider_name()

    def get_provider_config(self) -> Optional[Dict[str, Any]]:
        if not self.provider:
            return None
        return self.provider.get_config()
