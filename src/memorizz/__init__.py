# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from .coordination import SharedMemory
from .internet_access import (
    FirecrawlProvider,
    InternetAccessProvider,
    TavilyProvider,
    create_internet_access_provider,
)
from .long_term.procedural.toolbox import Toolbox
from .long_term.semantic import KnowledgeBase
from .long_term.semantic.persona import Persona, RoleType
from .memagent import MemAgent
from .memory_provider import MemoryProvider, MemoryType
from .short_term_memory.working_memory.cwm import CWM
from .tool_context import get_tool_context, reset_tool_context, set_tool_context


# Lazy import MongoDB to avoid requiring pymongo when not needed
def __getattr__(name):
    if name == "MongoDBProvider":
        from .memory_provider.mongodb import MongoDBProvider

        return MongoDBProvider
    if name in ("FileSystemProvider", "FileSystemConfig"):
        from .memory_provider.filesystem import FileSystemConfig, FileSystemProvider

        return FileSystemProvider if name == "FileSystemProvider" else FileSystemConfig
    if name in ("AutomationJob", "AutomationRun", "AutomationDelivery"):
        from .automation import AutomationDelivery, AutomationJob, AutomationRun

        _map = {
            "AutomationJob": AutomationJob,
            "AutomationRun": AutomationRun,
            "AutomationDelivery": AutomationDelivery,
        }
        return _map[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "MemoryProvider",
    "MongoDBProvider",
    "FileSystemProvider",
    "FileSystemConfig",
    "MemoryType",
    "Persona",
    "RoleType",
    "Toolbox",
    "KnowledgeBase",
    "CWM",
    "SharedMemory",
    "MemAgent",
    "InternetAccessProvider",
    "FirecrawlProvider",
    "TavilyProvider",
    "create_internet_access_provider",
    "AutomationJob",
    "AutomationRun",
    "AutomationDelivery",
    "get_tool_context",
    "set_tool_context",
    "reset_tool_context",
]
