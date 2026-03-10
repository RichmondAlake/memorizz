# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
import time
from typing import Optional

import numpy as np

from ..embeddings import get_embedding
from ..enums.application_mode import ApplicationModeConfig
from ..enums.memory_type import MemoryType
from ..llms.llm_provider import LLMProvider
from ..memory_provider import MemoryProvider
from .conversational_memory_unit import ConversationMemoryUnit


# Use lazy initialization for OpenAI
def get_openai_llm():
    from ..llms.openai import OpenAI

    return OpenAI()


class MemoryUnit:
    def __init__(
        self,
        application_mode: str,
        memory_provider: MemoryProvider = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        # Validate and set the application mode
        if isinstance(application_mode, str):
            self.application_mode = ApplicationModeConfig.validate_mode(
                application_mode
            )
        else:
            self.application_mode = application_mode

        self.memory_provider = memory_provider
        self.query_embedding = None
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            self.llm_provider = get_openai_llm()
        # Get the memory types for this application mode
        self.active_memory_types = ApplicationModeConfig.get_memory_types(
            self.application_mode
        )

    def generate_memory_unit(self, content: dict):
        """
        Generate the memory unit based on the application mode.
        The memory unit type is determined by the content and active memory types.
        """

        # Generate the embedding of the memory unit
        content["embedding"] = get_embedding(content["content"])

        # Determine the appropriate memory unit type based on content and active memory types
        if (
            MemoryType.CONVERSATION_MEMORY in self.active_memory_types
            and "role" in content
        ):
            return self._generate_conversational_memory_unit(content)
        elif MemoryType.WORKFLOW_MEMORY in self.active_memory_types:
            return self._generate_workflow_memory_unit(content)
        elif MemoryType.LONG_TERM_MEMORY in self.active_memory_types:
            return self._generate_knowledge_base_unit(content)
        else:
            # Default to conversational if available, otherwise use the first active memory type
            if MemoryType.CONVERSATION_MEMORY in self.active_memory_types:
                return self._generate_conversational_memory_unit(content)
            else:
                raise ValueError(
                    f"No suitable memory unit type for application mode: {self.application_mode.value}"
                )

    def _generate_conversational_memory_unit(
        self, content: dict
    ) -> ConversationMemoryUnit:
        """
        Generate the conversational memory unit.

        Parameters:
            content (dict): The content of the memory unit.

        Returns:
            ConversationMemoryUnit: The conversational memory unit.
        """
        memory_unit = ConversationMemoryUnit(
            role=content["role"],
            content=content["content"],
            timestamp=content["timestamp"],
            conversation_id=content["conversation_id"],
            memory_id=content["memory_id"],
            embedding=content["embedding"],
        )

        # Save the memory unit to the memory provider
        self._save_memory_unit(memory_unit, MemoryType.CONVERSATION_MEMORY)

        return memory_unit

    def _generate_workflow_memory_unit(self, content: dict):
        """
        Generate a workflow memory unit.

        Parameters:
            content (dict): The content of the memory unit.

        Returns:
            dict: The workflow memory unit.
        """
        workflow_component = {
            "content": content["content"],
            "timestamp": content.get("timestamp", time.time()),
            "memory_id": content["memory_id"],
            "embedding": content["embedding"],
            "component_type": "workflow",
            "workflow_step": content.get("workflow_step", "unknown"),
            "task_id": content.get("task_id"),
        }

        # Save the memory unit to the memory provider
        self._save_memory_unit(workflow_component, MemoryType.WORKFLOW_MEMORY)

        return workflow_component

    def _generate_knowledge_base_unit(self, content: dict):
        """
        Generate a knowledge base (long-term memory) component.

        Parameters:
            content (dict): The content of the memory unit.

        Returns:
            dict: The knowledge base memory unit.
        """
        knowledge_component = {
            "content": content["content"],
            "timestamp": content.get("timestamp", time.time()),
            "memory_id": content["memory_id"],
            "embedding": content["embedding"],
            "component_type": "knowledge",
            "category": content.get("category", "general"),
            "importance": content.get("importance", 0.5),
        }

        # Save the memory unit to the memory provider
        self._save_memory_unit(knowledge_component, MemoryType.LONG_TERM_MEMORY)

        return knowledge_component

    def _save_memory_unit(self, memory_unit: any, memory_type: MemoryType = None):
        """
        Save the memory unit to the memory provider.

        Parameters:
            memory_unit: The memory unit to save
            memory_type: Specific memory type to save to (optional)
        """

        # Remove the score(vector similarity score calculated by the vector search of the memory provider) from the memory unit if it exists
        if isinstance(memory_unit, dict) and "score" in memory_unit:
            memory_unit.pop("score", None)

        # Convert Pydantic model to dictionary if needed
        if hasattr(memory_unit, "model_dump"):
            memory_unit_dict = memory_unit.model_dump()
        elif hasattr(memory_unit, "dict"):
            memory_unit_dict = memory_unit.dict()
        else:
            # If it's already a dictionary, use it as is
            memory_unit_dict = memory_unit

        # If memory_type is not specified, determine from the component or use conversation as default
        if memory_type is None:
            if MemoryType.CONVERSATION_MEMORY in self.active_memory_types:
                memory_type = MemoryType.CONVERSATION_MEMORY
            else:
                # Use the first available memory type from active types
                memory_type = (
                    self.active_memory_types[0]
                    if self.active_memory_types
                    else MemoryType.CONVERSATION_MEMORY
                )

        # Validate that the memory type is active for this application mode
        if memory_type not in self.active_memory_types:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Memory type {memory_type.value} not active for application mode {self.application_mode.value}"
            )

        logger = logging.getLogger(__name__)

        logger.info(
            f"Storing memory unit of type {memory_type.value} in memory provider"
        )
        logger.debug(f"Memory component data: {memory_unit_dict}")
        stored_id = self.memory_provider.store(memory_unit_dict, memory_type)
        logger.info(f"Stored memory unit with ID: {stored_id}")
        return stored_id

    def retrieve_memory_units_by_memory_id(
        self, memory_id: str, memory_type: MemoryType
    ):
        """
        Retrieve the memory units by memory id.

        Parameters:
            memory_id (str): The id of the memory to retrieve the memory units for.
            memory_type (MemoryType): The type of the memory to retrieve the memory units for.

        Returns:
            List[MemoryUnit]: The memory units.
        """
        if memory_type == MemoryType.CONVERSATION_MEMORY:
            return (
                self.memory_provider.retrieve_conversation_history_ordered_by_timestamp(
                    memory_id
                )
            )

        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            return self.memory_provider.retrieve_workflow_history_ordered_by_timestamp(
                memory_id
            )
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")

    def retrieve_memory_units_by_conversation_id(self, conversation_id: str):
        pass

    def retrieve_memory_units_by_query(
        self, query: str, memory_id: str, memory_type: MemoryType, limit: int = 5
    ):
        """
        Retrieve the memory units by query.

        Parameters:
            query (str): The query to use for retrieval.
            memory_id (str): The id of the memory to retrieve the memory units for.
            memory_type (MemoryType): The type of the memory to retrieve the memory units for.
            limit (int): The limit of the memory units to return.

        Returns:
            List[MemoryUnit]: The memory units.
        """

        # Create the query embedding here so that it is not created for each memory unit
        self.query_embedding = get_embedding(query)

        # Get the memory units by query
        memory_units = self.memory_provider.retrieve_memory_units_by_query(
            query, self.query_embedding, memory_id, memory_type, limit
        )

        # Get the surronding conversation ids from each of the memory units
        # Handle cases where conversation_id might be missing or _id is used instead
        surrounding_conversation_ids = []
        for memory_unit in memory_units:
            surrounding_conversation_ids.append(memory_unit["_id"])

        # Before returning the memory units, we need to update the memory signals within the memory units
        for memory_unit in memory_units:
            self.update_memory_signals_within_memory_unit(
                memory_unit, memory_type, surrounding_conversation_ids
            )

        # Calculate the memory signal for each of the memory units
        for memory_unit in memory_units:
            memory_unit["memory_signal"] = self.calculate_memory_signal(
                memory_unit, query
            )

        # Sort the memory units by the memory signal
        memory_units.sort(key=lambda x: x["memory_signal"], reverse=True)

        # Return the memory units
        return memory_units

    def update_memory_signals_within_memory_unit(
        self,
        memory_unit: any,
        memory_type: MemoryType,
        surrounding_conversation_ids: list[str],
    ):
        """
        Update the memory signal within the memory unit.

        Parameters:
            memory_unit (dict): The memory unit to update the memory signal within.
            memory_type (MemoryType): The type of the memory to update the memory signal within.
            surrounding_conversation_ids (list[str]): The list of surrounding conversation ids.
        """

        # Update the recall_recency field (how recently the memory unit was recalled), this is the current timestamp
        memory_unit["recall_recency"] = time.time()

        if memory_type == MemoryType.CONVERSATION_MEMORY:
            # Update the importance field with a list of calling ID and surronding conversation ID's
            memory_unit["associated_conversation_ids"] = surrounding_conversation_ids

        # Save the memory unit to the memory provider
        self._save_memory_unit(memory_unit)

    def calculate_memory_signal(self, memory_unit: any, query: str):
        """
        Calculate the memory signal within the memory unit.

        Parameters:
            memory_unit (any): The memory unit to calculate the memory signal within.
            query (str): The query to use for calculation.

        Returns:
            float: The memory signal between 0 and 1.
        """
        # Detect the gap between the current timestamp and the recall_recency field
        recency = time.time() - memory_unit["recall_recency"]

        # Get the number of associated memory ids (this is used to calcualte the importance of the memory unit)
        number_of_associated_conversation_ids = len(
            memory_unit["associated_conversation_ids"]
        )

        # If the score exists, use it as the relevance score (this is the vector similarity score calculated by the vector search of the memory provider)
        if "score" in memory_unit:
            relevance = memory_unit["score"]
        else:
            # Calculate the relevance of the memory unit which is a vector score between the memory unit and the query
            relevance = self.calculate_relevance(query, memory_unit)

        # Calulate importance of the memory unit
        importance = self.calculate_importance(memory_unit["content"], query)

        # Calculate the normalized memory signal
        memory_signal = (
            recency * number_of_associated_conversation_ids * relevance * importance
        )

        # Normalize the memory signal between 0 and 1
        memory_signal = memory_signal / 100

        # Return the memory signal
        return memory_signal

    def calculate_relevance(self, query: str, memory_unit: any) -> float:
        """
        Calculate the relevance of the query with the memory unit.

        Parameters:
            query (str): The query to use for calculation.
            memory_unit (any): The memory unit to calculate the relevance within.

        Returns:
            float: The relevance between 0 and 1.
        """
        # Get embedding of the query
        if self.query_embedding is None:
            self.query_embedding = get_embedding(query)

        # Get embedding of the memory unit if it is not already embedded
        if "embedding" not in memory_unit or memory_unit["embedding"] is None:
            memory_unit_embedding = get_embedding(memory_unit["content"])
        else:
            memory_unit_embedding = memory_unit["embedding"]

        # Calculate the cosine similarity between the query embedding and the memory unit embedding
        relevance = self.cosine_similarity(self.query_embedding, memory_unit_embedding)

        # Return the relevance
        return relevance

    # We might not need this as the memory compoennt should have a score from retrieval
    def cosine_similarity(
        self, query_embedding: list[float], memory_unit_embedding: list[float]
    ) -> float:
        """
        Calculate the cosine similarity between two embeddings.

        Parameters:
            query_embedding (list[float]): The query embedding.
            memory_unit_embedding (list[float]): The memory unit embedding.

        Returns:
            float: The cosine similarity between the two embeddings.
        """
        # Calculate the dot product of the two embeddings
        dot_product = np.dot(query_embedding, memory_unit_embedding)

        # Calculate the magnitude of the two embeddings
        magnitude_query_embedding = np.linalg.norm(query_embedding)
        magnitude_memory_unit_embedding = np.linalg.norm(memory_unit_embedding)

        # Calculate the cosine similarity
        cosine_similarity = dot_product / (
            magnitude_query_embedding * magnitude_memory_unit_embedding
        )

        # Return the cosine similarity
        return cosine_similarity

    def calculate_importance(self, memory_unit_content: str, query: str) -> float:
        """
        Calculate the importance of the memory unit.
        Using an LLM to calculate the importance of the memory unit.

        Parameters:
            memory_unit_content (str): The content of the memory unit to calculate the importance within.
            query (str): The query to use for calculation.

        Returns:
            float: The importance between 0 and 1.
        """

        importance_prompt = f"""
        Calculate the importance of the following memory unit:
        {memory_unit_content}
        in relation to the following query and rate the likely poignancy of the memory unit:
        {query}
        Return the importance of the memory unit as a number between 0 and 1.
        """

        # Get the importance of the memory unit
        importance = self.llm_provider.generate_text(
            importance_prompt,
            instructions="Return the importance of the memory unit as a number between 0 and 1. No other text or comments, just the number. For example: 0.5",
        )

        # Return the importance
        return float(importance)
