# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from typing import List

# from ..memagent import MemAgent
from ...enums.memory_type import MemoryType


# Can take in an agent and then return a prompt that informs the agent on how to manage the context window
class CWM:
    # def __init__(self, agent: MemAgent):
    #     self.agent = agent

    @staticmethod
    def get_prompt_from_memory_types(memory_types: List[MemoryType]):
        prompt = "You are an AI Agent endowed with a powerful, multi-tiered memory augmentation system. Your mission is to use all available memory modalities to deliver consistent, accurate, and context-rich responses. The aim is to esure that through augmented memory, you become belivable, capable, and reliable."

        for memory_type in memory_types:
            prompt += CWM._generate_prompt_for_memory_type(memory_type)

        return prompt

    @staticmethod
    def _generate_prompt_for_memory_type(memory_type: MemoryType):
        # Define memory type prompts in a dictionary for better maintainability
        memory_prompts = {
            MemoryType.CONVERSATION_MEMORY: {
                "description": "This is a memory type that stores the conversation history between the agent and the user.",
                "usage": "Use this to provide continuity, avoid repeating yourself, and reference prior turns.",
            },
            MemoryType.WORKFLOW_MEMORY: {
                "description": "This is a memory type that stores the workflow history between the agent and the user.",
                "usage": "Use this to provide continuity, avoid repeating yourself, and reference prior turns.",
            },
            MemoryType.SHARED_MEMORY: {
                "description": "This is a memory type that stores shared blackboard information for multi-agent coordination.",
                "usage": "Use this to coordinate with other agents, understand your role in the agent hierarchy, and access shared coordination activities and context.",
            },
            MemoryType.SUMMARIES: {
                "description": "This is a memory type that stores compressed summaries of past conversations and interactions to preserve important context while managing memory efficiently.",
                "usage": "Use these summaries to understand the broader context of your interactions with the user, recall important topics, preferences, and past decisions. This helps you provide more personalized and context-aware responses even when specific conversations are no longer in active memory.",
            },
        }

        # Get the prompt configuration for this memory type
        prompt_config = memory_prompts.get(memory_type)

        if prompt_config:
            prompt = f"\n\nMemory Type: {memory_type.value}\n"
            prompt += f"Memory Type Description: {prompt_config['description']}\n"
            prompt += f"Memory Type Usage: {prompt_config['usage']}\n"
            return prompt
        else:
            # Handle unknown memory types gracefully
            return f"\n\nMemory Type: {memory_type.value}\n"


# Can take in an array of memory stores and then return a prompt that informs the agent on how to manage the context window
