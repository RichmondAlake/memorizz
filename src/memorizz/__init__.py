# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .conversation_history import is_trace_bundle_entry, strip_trace_bundles
from .coordination import SharedMemory
from .enums import ApplicationMode
from .internet_access import (
    FirecrawlProvider,
    InternetAccessProvider,
    TavilyProvider,
    create_internet_access_provider,
)
from .long_term.procedural.toolbox import Toolbox
from .long_term.semantic import KnowledgeBase
from .long_term.semantic.persona import Persona, RoleType
from .memagent import MemAgent, MemAgentModel
from .memagent.builders import (
    MemAgentBuilder,
    create_assistant,
    create_chatbot,
    create_deep_research_agent,
    create_task_agent,
)
from .memory_provider import MemoryProvider, MemoryType
from .short_term_memory.working_memory.cwm import CWM
from .tool_context import get_tool_context, reset_tool_context, set_tool_context

try:
    __version__ = _pkg_version("memorizz")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"


# Lazy imports for optional-dependency / heavy-SDK surfaces, so a plain
# `import memorizz` stays lean (no pymongo/oracledb/anthropic/ollama import
# unless the corresponding symbol is actually used).
def __getattr__(name):
    if name == "MongoDBProvider":
        from .memory_provider.mongodb import MongoDBProvider

        return MongoDBProvider
    if name == "MongoDBConfig":
        try:
            from .memory_provider.mongodb.provider import MongoDBConfig
        except ImportError as exc:
            raise ImportError(
                'MongoDB support requires pymongo. Install with: pip install "memorizz[mongodb]"'
            ) from exc

        return MongoDBConfig
    if name == "OracleProvider":
        from .memory_provider.oracle import OracleProvider

        return OracleProvider
    if name == "OracleConfig":
        from .memory_provider.oracle.provider import OracleConfig

        return OracleConfig
    if name in ("FileSystemProvider", "FileSystemConfig"):
        from .memory_provider.filesystem import FileSystemConfig, FileSystemProvider

        return FileSystemProvider if name == "FileSystemProvider" else FileSystemConfig
    if name in ("OpenAI", "AzureOpenAI", "Anthropic", "OllamaLLM", "HuggingFaceLLM"):
        from . import llms

        return getattr(llms, name)
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
    "__version__",
    # Agent
    "MemAgent",
    "MemAgentModel",
    "MemAgentBuilder",
    "create_assistant",
    "create_chatbot",
    "create_task_agent",
    "create_deep_research_agent",
    "ApplicationMode",
    # Memory providers + configs
    "MemoryProvider",
    "MemoryType",
    "MongoDBProvider",
    "MongoDBConfig",
    "OracleProvider",
    "OracleConfig",
    "FileSystemProvider",
    "FileSystemConfig",
    # LLM providers (lazy)
    "OpenAI",
    "AzureOpenAI",
    "Anthropic",
    "OllamaLLM",
    "HuggingFaceLLM",
    # Memory primitives
    "Persona",
    "RoleType",
    "Toolbox",
    "KnowledgeBase",
    "CWM",
    "SharedMemory",
    # Internet access
    "InternetAccessProvider",
    "FirecrawlProvider",
    "TavilyProvider",
    "create_internet_access_provider",
    # Automation (lazy)
    "AutomationJob",
    "AutomationRun",
    "AutomationDelivery",
    # Tool context + trace helpers
    "get_tool_context",
    "set_tool_context",
    "reset_tool_context",
    "is_trace_bundle_entry",
    "strip_trace_bundles",
]
