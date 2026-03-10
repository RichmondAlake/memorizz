# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Factory for resolving an AutomationStore from a configured MemoryProvider."""

from __future__ import annotations

from typing import Any, Optional

from .base import AutomationStore


def get_automation_store(memory_provider: Any) -> Optional[AutomationStore]:
    if memory_provider is None:
        return None

    provider_cls = getattr(memory_provider, "__class__", None)
    provider_name = getattr(provider_cls, "__name__", "")
    provider_module = getattr(provider_cls, "__module__", "")

    if provider_name == "OracleProvider" and provider_module.endswith(
        "memorizz.memory_provider.oracle.provider"
    ):
        from .oracle import OracleAutomationStore

        return OracleAutomationStore(memory_provider)

    return None
