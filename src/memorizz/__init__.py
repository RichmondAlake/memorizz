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

# Kept in lockstep with pyproject.toml and asserted by release tests.
__version__ = "0.8.0"

# ``capabilities`` intentionally uses an eager, tiny import.  A lazy export
# with the same name as its submodule is not stable under Python's from-list
# import machinery: ``from memorizz import capabilities`` can otherwise bind
# the ``memorizz.capabilities`` module instead of the documented callable.
from .capabilities import capabilities as capabilities

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
    "SubTask": (".task_decomposition", "SubTask"),
    "ApplicationMode": (".enums", "ApplicationMode"),
    "SemanticCatalog": (".semantic_layer", "SemanticCatalog"),
    "SemanticModel": (".semantic_layer", "SemanticModel"),
    "SemanticQuery": (".semantic_layer", "SemanticQuery"),
    "ApprovalProposal": (".approval", "ApprovalProposal"),
    "ApprovalResumeResult": (".approval", "ApprovalResumeResult"),
    "ApprovalStatus": (".approval", "ApprovalStatus"),
    "SQLiteApprovalStore": (".approval", "SQLiteApprovalStore"),
    "ToolPolicy": (".tooling", "ToolPolicy"),
    "ToolOutcome": (".tool_outcomes", "ToolOutcome"),
    "ToolOutcomeStatus": (".tool_outcomes", "ToolOutcomeStatus"),
    "ToolResult": (".tool_outcomes", "ToolResult"),
    "ToolResultPolicy": (".tooling", "ToolResultPolicy"),
    "ContextPolicy": (".tooling", "ContextPolicy"),
    "CompletionCandidate": (".completion", "CompletionCandidate"),
    "CompletionDecision": (".completion", "CompletionDecision"),
    "CompletionPolicy": (".completion", "CompletionPolicy"),
    "CompletionRejectedError": (".completion", "CompletionRejectedError"),
    "StreamEvent": (".streaming", "StreamEvent"),
    "EventStream": (".streaming", "EventStream"),
    "CancellationToken": (".streaming", "CancellationToken"),
    "RetrievalPolicy": (".retrieval", "RetrievalPolicy"),
    "PersonalizationContext": (".personalization", "PersonalizationContext"),
    "PersonalizationContextBuilder": (
        ".personalization",
        "PersonalizationContextBuilder",
    ),
    "PersonalizationPolicy": (".personalization", "PersonalizationPolicy"),
    "build_personalization_context": (
        ".personalization",
        "build_personalization_context",
    ),
    "LearningControlPlane": (".learning", "LearningControlPlane"),
    "LearningControlPlaneConfig": (".learning", "LearningControlPlaneConfig"),
    "LearningEvent": (".learning", "LearningEvent"),
    "LearningEventType": (".learning", "LearningEventType"),
    "EvidencePack": (".learning", "EvidencePack"),
    "OutcomeEvidence": (".learning", "OutcomeEvidence"),
    "ForgettingReport": (".learning", "ForgettingReport"),
    "SemanticCacheInspection": (
        ".short_term_memory.semantic_cache",
        "SemanticCacheInspection",
    ),
    "governed_tool": (".tooling", "governed_tool"),
    # Meta-harness
    "MetaHarness": (".metaharness", "MetaHarness"),
    "AgentHarness": (".metaharness", "AgentHarness"),
    "HarnessTask": (".metaharness", "HarnessTask"),
    "HarnessResult": (".metaharness", "HarnessResult"),
    "HarnessRun": (".metaharness", "HarnessRun"),
    "HarnessStatus": (".metaharness", "HarnessStatus"),
    "HarnessEvent": (".metaharness", "HarnessEvent"),
    "HarnessEventType": (".metaharness", "HarnessEventType"),
    "HarnessPermissions": (".metaharness", "HarnessPermissions"),
    "HarnessBudget": (".metaharness", "HarnessBudget"),
    "VerificationSpec": (".metaharness", "VerificationSpec"),
    "HarnessPlan": (".metaharness", "HarnessPlan"),
    "HarnessStage": (".metaharness", "HarnessStage"),
    "CodexHarness": (".metaharness", "CodexHarness"),
    "ClaudeCodeHarness": (".metaharness", "ClaudeCodeHarness"),
    "OpenHandsHarness": (".metaharness", "OpenHandsHarness"),
    "NativeMemAgentHarness": (".metaharness", "NativeMemAgentHarness"),
    "PersistedMemAgentHarness": (".metaharness", "PersistedMemAgentHarness"),
    # Memory providers + configs
    "MemoryProvider": (".memory_provider", "MemoryProvider"),
    "MemoryType": (".memory_provider", "MemoryType"),
    "MongoDBProvider": (".memory_provider.mongodb", "MongoDBProvider"),
    "OracleProvider": (".memory_provider.oracle", "OracleProvider"),
    "OracleConfig": (".memory_provider.oracle.provider", "OracleConfig"),
    "LocalOracleRuntime": (".memory_provider.oracle", "LocalOracleRuntime"),
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
    "EntityMemory": (".long_term.semantic.entity_memory", "EntityMemory"),
    "SharedMemory": (".coordination", "SharedMemory"),
    # Internet access
    "InternetAccessProvider": (".internet_access", "InternetAccessProvider"),
    "FirecrawlProvider": (".internet_access", "FirecrawlProvider"),
    "TavilyProvider": (".internet_access", "TavilyProvider"),
    "create_internet_access_provider": (
        ".internet_access",
        "create_internet_access_provider",
    ),
    # Browser control
    "BrowserControlProvider": (".browser_control", "BrowserControlProvider"),
    "BrowserControlResult": (".browser_control", "BrowserControlResult"),
    "BrowserUseProvider": (".browser_control", "BrowserUseProvider"),
    "create_browser_control_provider": (
        ".browser_control",
        "create_browser_control_provider",
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
    "SubTask",
    "ApplicationMode",
    "capabilities",
    "SemanticCatalog",
    "SemanticModel",
    "SemanticQuery",
    "ApprovalProposal",
    "ApprovalResumeResult",
    "ApprovalStatus",
    "SQLiteApprovalStore",
    "ToolPolicy",
    "ToolOutcome",
    "ToolOutcomeStatus",
    "ToolResult",
    "ToolResultPolicy",
    "ContextPolicy",
    "CompletionCandidate",
    "CompletionDecision",
    "CompletionPolicy",
    "CompletionRejectedError",
    "RetrievalPolicy",
    "PersonalizationContext",
    "PersonalizationContextBuilder",
    "PersonalizationPolicy",
    "build_personalization_context",
    "LearningControlPlane",
    "LearningControlPlaneConfig",
    "LearningEvent",
    "LearningEventType",
    "EvidencePack",
    "OutcomeEvidence",
    "ForgettingReport",
    "SemanticCacheInspection",
    "governed_tool",
    # Meta-harness
    "MetaHarness",
    "AgentHarness",
    "HarnessTask",
    "HarnessResult",
    "HarnessRun",
    "HarnessStatus",
    "HarnessEvent",
    "HarnessEventType",
    "HarnessPermissions",
    "HarnessBudget",
    "VerificationSpec",
    "HarnessPlan",
    "HarnessStage",
    "CodexHarness",
    "ClaudeCodeHarness",
    "OpenHandsHarness",
    "NativeMemAgentHarness",
    "PersistedMemAgentHarness",
    # Memory providers + configs
    "MemoryProvider",
    "MemoryType",
    "MongoDBProvider",
    "MongoDBConfig",
    "OracleProvider",
    "OracleConfig",
    "LocalOracleRuntime",
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
    "EntityMemory",
    "SharedMemory",
    # Internet access
    "InternetAccessProvider",
    "FirecrawlProvider",
    "TavilyProvider",
    "create_internet_access_provider",
    # Browser control
    "BrowserControlProvider",
    "BrowserControlResult",
    "BrowserUseProvider",
    "create_browser_control_provider",
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
