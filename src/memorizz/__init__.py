# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""memorizz — a memory framework for AI agents.

The entire public API is resolved lazily (PEP 562 ``__getattr__``) so that a bare
``import memorizz`` — and therefore ``memorizz --help`` / ``--version`` and any
tool that merely imports the package — costs almost nothing: no numpy, no
pydantic model building, no provider SDKs. The heavy modules load only when a
symbol is actually accessed (``from memorizz import MemAgent`` triggers the core
import at that point, exactly as before).
"""

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("memorizz")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"


# name -> (submodule, attribute). Imported on first attribute access.
_LAZY = {
    # Agent
    "MemAgent": (".memagent", "MemAgent"),
    "MemAgentModel": (".memagent", "MemAgentModel"),
    "MemAgentBuilder": (".memagent.builders", "MemAgentBuilder"),
    "create_assistant": (".memagent.builders", "create_assistant"),
    "create_chatbot": (".memagent.builders", "create_chatbot"),
    "create_task_agent": (".memagent.builders", "create_task_agent"),
    "create_deep_research_agent": (".memagent.builders", "create_deep_research_agent"),
    "ApplicationMode": (".enums", "ApplicationMode"),
    # Memory providers + configs
    "MemoryProvider": (".memory_provider", "MemoryProvider"),
    "MemoryType": (".memory_provider", "MemoryType"),
    "MongoDBProvider": (".memory_provider.mongodb", "MongoDBProvider"),
    "OracleProvider": (".memory_provider.oracle", "OracleProvider"),
    "OracleConfig": (".memory_provider.oracle.provider", "OracleConfig"),
    "FileSystemProvider": (".memory_provider.filesystem", "FileSystemProvider"),
    "FileSystemConfig": (".memory_provider.filesystem", "FileSystemConfig"),
    # LLM providers
    "OpenAI": (".llms", "OpenAI"),
    "AzureOpenAI": (".llms", "AzureOpenAI"),
    "Anthropic": (".llms", "Anthropic"),
    "OllamaLLM": (".llms", "OllamaLLM"),
    "HuggingFaceLLM": (".llms", "HuggingFaceLLM"),
    # Memory primitives
    "Persona": (".long_term.semantic.persona", "Persona"),
    "RoleType": (".long_term.semantic.persona", "RoleType"),
    "Toolbox": (".long_term.procedural.toolbox", "Toolbox"),
    "KnowledgeBase": (".long_term.semantic", "KnowledgeBase"),
    "SharedMemory": (".coordination", "SharedMemory"),
    # Internet access
    "InternetAccessProvider": (".internet_access", "InternetAccessProvider"),
    "FirecrawlProvider": (".internet_access", "FirecrawlProvider"),
    "TavilyProvider": (".internet_access", "TavilyProvider"),
    "create_internet_access_provider": (
        ".internet_access",
        "create_internet_access_provider",
    ),
    # Automation
    "AutomationJob": (".automation", "AutomationJob"),
    "AutomationRun": (".automation", "AutomationRun"),
    "AutomationDelivery": (".automation", "AutomationDelivery"),
    # Tool context + trace helpers
    "get_tool_context": (".tool_context", "get_tool_context"),
    "set_tool_context": (".tool_context", "set_tool_context"),
    "reset_tool_context": (".tool_context", "reset_tool_context"),
    "is_trace_bundle_entry": (".conversation_history", "is_trace_bundle_entry"),
    "strip_trace_bundles": (".conversation_history", "strip_trace_bundles"),
}


def __getattr__(name):
    # MongoDBConfig pulls pymongo/bson transitively; give the same actionable
    # message as the other optional-dependency surfaces instead of a raw
    # ModuleNotFoundError('bson').
    if name == "MongoDBConfig":
        try:
            from .memory_provider.mongodb.provider import MongoDBConfig
        except ImportError as exc:
            raise ImportError(
                'MongoDB support requires pymongo. Install with: pip install "memorizz[mongodb]"'
            ) from exc
        return MongoDBConfig

    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = importlib.import_module(spec[0], __name__)
    return getattr(module, spec[1])


def __dir__():
    return sorted(__all__)


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
    # LLM providers
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
    "SharedMemory",
    # Internet access
    "InternetAccessProvider",
    "FirecrawlProvider",
    "TavilyProvider",
    "create_internet_access_provider",
    # Automation
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
