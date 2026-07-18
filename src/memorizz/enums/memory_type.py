# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from enum import Enum, EnumMeta

# Legacy value/name aliases. Records written before the ``LONG_TERM_MEMORY``
# → ``KNOWLEDGE_BASE`` rename still carry the old string; accept them
# transparently so older rows keep loading.
_MEMORY_TYPE_VALUE_ALIASES = {
    "long_term_memory": "knowledge_base",
}
_MEMORY_TYPE_NAME_ALIASES = {
    "LONG_TERM_MEMORY": "KNOWLEDGE_BASE",
}


class _MemoryTypeMeta(EnumMeta):
    """Metaclass that lets ``MemoryType["LONG_TERM_MEMORY"]`` resolve to
    ``MemoryType.KNOWLEDGE_BASE`` for backwards compatibility."""

    def __getitem__(cls, name):  # type: ignore[override]
        try:
            return super().__getitem__(name)
        except KeyError:
            if isinstance(name, str) and name in _MEMORY_TYPE_NAME_ALIASES:
                return super().__getitem__(_MEMORY_TYPE_NAME_ALIASES[name])
            raise


class MemoryType(Enum, metaclass=_MemoryTypeMeta):
    """Enum for different types of memory stores."""

    PERSONAS = "personas"
    TOOLBOX = "toolbox"
    ENTITY_MEMORY = "entity_memory"
    SHORT_TERM_MEMORY = "short_term_memory"
    KNOWLEDGE_BASE = "knowledge_base"
    CONVERSATION_MEMORY = "conversation_memory"
    WORKFLOW_MEMORY = "workflow_memory"
    SKILLBOX = "skillbox"
    MEMAGENT = "agents"
    SHARED_MEMORY = "shared_memory"
    SUMMARIES = "summaries"
    SEMANTIC_CACHE = "semantic_cache"
    TOOL_LOG = "tool_log"

    @classmethod
    def _missing_(cls, value):
        """Map legacy value strings (e.g. ``long_term_memory``) to current members."""
        if isinstance(value, str):
            if value in _MEMORY_TYPE_VALUE_ALIASES:
                return cls(_MEMORY_TYPE_VALUE_ALIASES[value])
            lowered = value.lower()
            if lowered in _MEMORY_TYPE_VALUE_ALIASES:
                return cls(_MEMORY_TYPE_VALUE_ALIASES[lowered])
        return None
