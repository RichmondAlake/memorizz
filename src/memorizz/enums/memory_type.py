# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from enum import Enum


class MemoryType(Enum):
    """Enum for different types of memory stores."""

    PERSONAS = "personas"
    TOOLBOX = "toolbox"
    ENTITY_MEMORY = "entity_memory"
    SHORT_TERM_MEMORY = "short_term_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    CONVERSATION_MEMORY = "conversation_memory"
    WORKFLOW_MEMORY = "workflow_memory"
    MEMAGENT = "agents"
    SHARED_MEMORY = "shared_memory"
    SUMMARIES = "summaries"
    SEMANTIC_CACHE = "semantic_cache"
