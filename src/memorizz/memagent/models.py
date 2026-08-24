# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Data models for MemAgent configuration and state."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .constants import DEFAULT_INSTRUCTION, DEFAULT_MAX_STEPS, DEFAULT_TOOL_ACCESS


class MemAgentModel(BaseModel):
    """Data model for persisting and loading MemAgent configuration."""

    model: Optional[Any] = None
    llm_config: Optional[Dict[str, Any]] = None  # Configuration for the LLM
    agent_id: Optional[str] = None
    application_id: Optional[str] = None
    name: Optional[str] = None
    tools: Optional[Union[List, Any]] = None
    persona: Optional[Any] = None
    instruction: Optional[str] = Field(default=DEFAULT_INSTRUCTION)
    application_mode: Optional[str] = "assistant"
    memory_types: Optional[
        List[str]
    ] = None  # Custom memory types that override application_mode defaults
    max_steps: int = Field(default=DEFAULT_MAX_STEPS)
    memory_ids: Optional[List[str]] = None
    tool_access: Optional[str] = Field(default=DEFAULT_TOOL_ACCESS)
    knowledge_base_ids: Optional[List[str]] = None
    delegates: Optional[List[str]] = None  # Store delegate agent IDs
    embedding_config: Optional[Dict[str, Any]] = None
    semantic_cache: Optional[bool] = False  # Enable semantic cache
    semantic_cache_config: Optional[
        Union[Any, Dict[str, Any]]
    ] = None  # Semantic cache configuration
    tool_result_policy: Optional[Dict[str, Any]] = None
    context_policy: Optional[Dict[str, Any]] = None
    completion_policy: Optional[Dict[str, Any]] = None
    retrieval_policy: Optional[Dict[str, Any]] = None
    delegation_config: Optional[Dict[str, Any]] = None
    skill_retrieval: bool = False
    skill_retrieval_config: Optional[Dict[str, Any]] = None
    semantic_layer_config: Optional[Dict[str, Any]] = None
    context_window_tokens: Optional[int] = None
    is_favorite: bool = False
    internet_access_provider: Optional[str] = None
    internet_access_config: Optional[Dict[str, Any]] = None
    skills_marketplace_provider: Optional[str] = None
    skills_marketplace_config: Optional[Dict[str, Any]] = None
    sandbox_provider: Optional[Union[str, Dict[str, Any]]] = None
    browser_control: Optional[Union[str, Dict[str, Any]]] = None
    meta_harness: bool = False
    meta_harness_mode: Optional[str] = None
    default_harness: str = "auto"
    harness_config: Optional[Dict[str, Any]] = None
    skill_paths: Optional[List[str]] = None
    mcp_servers: Optional[List[Dict[str, Any]]] = None
    self_aware: bool = False
    self_aware_config: Optional[Dict[str, Any]] = None
    continual_learning: bool = False
    continual_learning_config: Optional[Dict[str, Any]] = None
    learning_control_plane: bool = False
    learning_control_plane_config: Optional[Dict[str, Any]] = None
    automations_enabled: bool = True
    default_timezone: Optional[str] = None
    whatsapp_enabled: bool = False
    whatsapp_config: Optional[Dict[str, Any]] = None

    model_config = {
        "arbitrary_types_allowed": True  # Allow arbitrary types like Toolbox
    }


class MemAgentConfig:
    """Configuration helper for MemAgent initialization."""

    def __init__(
        self,
        instruction: str = DEFAULT_INSTRUCTION,
        max_steps: int = DEFAULT_MAX_STEPS,
        tool_access: str = DEFAULT_TOOL_ACCESS,
        semantic_cache: bool = False,
        **kwargs,
    ):
        self.instruction = instruction
        self.max_steps = max_steps
        self.tool_access = tool_access
        self.semantic_cache = semantic_cache

        # Store additional configuration
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
