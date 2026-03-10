# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...embeddings import get_embedding
from ...enums.memory_type import MemoryType
from ...memory_provider import MemoryProvider


class KnowledgeBase:
    """
    KnowledgeBase class that implements a specialized form of long-term memory.

    Instead of writing documents to a separate "knowledge_base" collection,
    each ingestion is stored directly in the agent's `long_term_memory` store.
    """

    def __init__(self, memory_provider: Optional[MemoryProvider] = None):
        """
        Initialize a new KnowledgeBase instance.

        Parameters:
        -----------
        memory_provider : Optional[MemoryProvider]
            The memory provider to use for storage and retrieval.
            If not provided, a default MemoryProvider will be used.
        """
        self.memory_provider = memory_provider or MemoryProvider()

    def ingest_knowledge(self, corpus: str, namespace: str) -> str:
        """
        Embed and save text content to the memory provider under the given namespace.

        Parameters:
        -----------
        corpus : str
            The text content to be ingested and stored.
        namespace : str
            A namespace to organize and categorize the knowledge.

        Returns:
        --------
        str
            A unique long_term_memory_id that can be attached to an agent to scope its long-term knowledge.
        """
        long_term_memory_id = str(uuid.uuid4())

        # Generate embedding for the corpus
        embedding = get_embedding(corpus)

        # Create the knowledge entry
        knowledge_entry = {
            "content": corpus,
            "embedding": embedding,
            "namespace": namespace,
            "long_term_memory_id": long_term_memory_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Store the knowledge entry in the long_term_memory collection
        self.memory_provider.store(
            knowledge_entry, memory_store_type=MemoryType.LONG_TERM_MEMORY
        )

        return long_term_memory_id

    def retrieve_knowledge(self, long_term_memory_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all knowledge entries associated with a given long_term_memory_id.

        Parameters:
        -----------
        long_term_memory_id : str
            The unique ID to retrieve knowledge for.

        Returns:
        --------
        List[Dict[str, Any]]
            A list of knowledge documents, each containing the original content,
            its embedding, and the memory ID.
        """
        # Use list_all to get all documents and filter manually
        all_entries = self.memory_provider.list_all(
            memory_store_type=MemoryType.LONG_TERM_MEMORY
        )

        # Filter entries with the matching long_term_memory_id
        knowledge_entries = [
            entry
            for entry in all_entries
            if entry.get("long_term_memory_id") == long_term_memory_id
        ]

        return knowledge_entries

    def retrieve_knowledge_by_query(
        self, query: str, namespace: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge entries that are semantically similar to the query.

        Parameters:
        -----------
        query : str
            The query to retrieve relevant knowledge for.
        namespace : Optional[str]
            If provided, limit the search to knowledge within this namespace.
        limit : int
            Maximum number of entries to return.

        Returns:
        --------
        List[Dict[str, Any]]
            A list of knowledge documents that are semantically similar to the query.
        """
        # Generate embedding for the query
        query_embedding = get_embedding(query)

        # Create a query object for semantic search
        query_obj = {"embedding": query_embedding, "limit": limit}

        # If namespace is provided, add it to the query
        if namespace:
            query_obj["namespace"] = namespace

        # Use the retrieve_by_query method for semantics search
        results = self.memory_provider.retrieve_by_query(
            query_obj, memory_store_type=MemoryType.LONG_TERM_MEMORY, limit=limit
        )

        # If results is a single dict, wrap it in a list
        if results and isinstance(results, dict):
            results = [results]

        # If no results, return empty list
        return results or []

    def delete_knowledge(self, long_term_memory_id: str) -> bool:
        """
        Delete all knowledge entries associated with a given long_term_memory_id.

        Parameters:
        -----------
        long_term_memory_id : str
            The unique ID of the knowledge to delete.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        # Get all entries with this memory ID
        entries = self.retrieve_knowledge(long_term_memory_id)

        # Delete each entry
        success = True
        for entry in entries:
            entry_id = entry.get("_id")
            if entry_id:
                if not self.memory_provider.delete_by_id(
                    entry_id, memory_store_type=MemoryType.LONG_TERM_MEMORY
                ):
                    success = False

        return success

    def update_knowledge(self, long_term_memory_id: str, corpus: str) -> bool:
        """
        Update the content and embedding of a knowledge entry.

        Parameters:
        -----------
        long_term_memory_id : str
            The unique ID of the knowledge to update.
        corpus : str
            The new text content.

        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        # Retrieve the existing entries to get their metadata
        entries = self.retrieve_knowledge(long_term_memory_id)
        if not entries:
            return False

        # Generate new embedding for the updated corpus
        embedding = get_embedding(corpus)

        # Update each entry with new content and embedding
        success = True
        for entry in entries:
            entry_id = entry.get("_id")
            if entry_id:
                # Update the entry with new content and embedding
                entry["content"] = corpus
                entry["embedding"] = embedding
                entry["updated_at"] = datetime.now().isoformat()

                # Update the entry in the memory provider
                if not self.memory_provider.update_by_id(
                    entry_id, entry, memory_store_type=MemoryType.LONG_TERM_MEMORY
                ):
                    success = False

        return success

    def attach_to_agent(self, agent, long_term_memory_id: str) -> bool:
        """
        Attach the long-term knowledge to a MemAgent.

        This updates the agent's configuration to include the long-term memory ID.

        Parameters:
        -----------
        agent : MemAgent
            The agent to attach the knowledge to.
        long_term_memory_id : str
            The unique ID of the knowledge to attach.

        Returns:
        --------
        bool
            True if the attachment was successful, False otherwise.
        """
        try:
            # Verify that the knowledge exists
            entries = self.retrieve_knowledge(long_term_memory_id)
            if not entries:
                return False

            # Store the long_term_memory_id in the agent's attributes if it doesn't exist
            if not hasattr(agent, "long_term_memory_ids"):
                agent.long_term_memory_ids = []

            # Add the ID if it's not already there
            if long_term_memory_id not in agent.long_term_memory_ids:
                agent.long_term_memory_ids.append(long_term_memory_id)

                # Save the updated agent
                agent.update()

            return True
        except Exception as e:
            print(f"Error attaching knowledge to agent: {str(e)}")
            return False
