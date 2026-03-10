# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from enum import Enum


class SemanticCacheScope(Enum):
    """Scope for semantic cache searches."""

    LOCAL = "local"  # Search only this agent's cache entries (filtered by agent_id)
    GLOBAL = "global"  # Search across all cache entries (no agent_id filter)
