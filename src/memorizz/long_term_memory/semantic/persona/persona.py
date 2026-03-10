# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import uuid
from datetime import datetime
from typing import Union

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....memory_provider import MemoryProvider
from .role_type import PREDEFINED_INFO, RoleType


class Persona:
    def __init__(
        self,
        name: str,
        role: Union[RoleType, str] = RoleType.GENERAL,
        goals: str = "",
        background: str = "",
        persona_id: str = None,
    ):
        """
        Initialize a Persona instance for an AI agent with a deterministic role.

        Parameters:
        -----------
        name : str
            The name of the persona.
        role : Union[RoleType, str], optional
            A predefined role for the agent (e.g., GENERAL, ASSISTANT, etc.) or the string value of a role.
            If not provided, the default role of GENERAL will be used.
        goals : str, optional
            Custom goals for the persona. If provided, these are appended to the predefined goals.
        background : str, optional
            Custom background for the persona. If provided, these are appended to the predefined background.
        persona_id : str, optional
            A unique identifier for the persona. If not provided, one will be generated.
        """
        self.name = name

        # Handle both RoleType enum and string role values
        if isinstance(role, str):
            role_enum = None
            # Try to match the string to a RoleType enum
            for role_type in RoleType:
                if role_type.value == role:
                    role_enum = role_type
                    break
            # If no match is found, default to GENERAL
            if role_enum is None:
                role_enum = RoleType.GENERAL
                # Store the original string value
                self.role = role
            else:
                self.role = role_enum.value
        else:
            self.role = role.value
            role_enum = role

        # Retrieve default goals and background of the role
        default_goals = PREDEFINED_INFO[role_enum]["goals"]
        default_background = PREDEFINED_INFO[role_enum]["background"]
        # Append custom goals and background to the default ones
        self.goals = f"{default_goals} {goals}".strip() if goals else default_goals
        self.background = (
            f"{default_background} {background}".strip()
            if background
            else default_background
        )

        # Generate or assign persona_id
        self.persona_id = persona_id if persona_id else self.generate_persona_id()

        # Generate the embedding based on the concatenated persona details.
        self.embedding = self._generate_embedding()

        # Timestamp for creation
        self.created_at = datetime.now().isoformat()

    @staticmethod
    def generate_persona_id() -> str:
        """Generate a unique persona ID optimized for MongoDB."""
        # Generate MongoDB ObjectId for better performance
        return str(uuid.uuid4())

    def _generate_embedding(self):
        """
        Generate an embedding vector for the persona based on its attributes.

        Returns:
        --------
        list or numpy.array: The embedding vector representing the persona.
        """
        embedding_input = f"{self.name} {self.role} {self.goals} {self.background}"
        return get_embedding(embedding_input)

    def to_dict(self) -> dict:
        """
        Serialize the Persona into a dictionary format, including the embedding.

        Returns:
        --------
        dict: A dictionary representation of the persona.
        """
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "role": self.role,
            "goals": self.goals,
            "background": self.background,
            "embedding": self.embedding,
            "created_at": self.created_at,
        }

    def store_persona(self, provider: MemoryProvider) -> str:
        """
        Store the persona's JSON structure and embedding into the memory provider.

        Parameters:
        -----------
        provider : MemoryProvider
            The memory provider to use for storage.

        Returns:
        --------
        str
            The ID of the stored persona.
        """
        print(
            f"Storing persona: {self.name} in the memory provider, in the {MemoryType.PERSONAS.value} collection"
        )
        persona_data = self.to_dict()
        return provider.store(persona_data, memory_store_type=MemoryType.PERSONAS)

    @staticmethod
    def retrieve_persona(persona_id: str, provider: MemoryProvider) -> dict:
        """
        Retrieve a persona from the memory provider by persona_id.

        Parameters:
        -----------
        persona_id : str
            The unique identifier of the persona to retrieve.
        provider : MemoryProvider
            The memory provider to use for retrieval.

        Returns:
        --------
        dict: The persona's JSON structure including its embedding, or None if not found.
        """
        print(
            f"Retrieving persona: {persona_id} from the memory provider, in the {MemoryType.PERSONAS.value} collection"
        )
        return provider.retrieve_by_id(
            persona_id, memory_store_type=MemoryType.PERSONAS
        )

    @staticmethod
    def delete_persona(persona_id: str, provider: MemoryProvider) -> bool:
        """
        Delete a persona from the memory provider using its persona_id.

        Parameters:
        -----------
        persona_id : str
            The unique identifier of the persona to delete.
        provider : MemoryProvider
            The memory provider to use for deletion.

        Returns:
        --------
        bool: True if deletion was successful, False otherwise.
        """
        return provider.delete_by_id(persona_id, memory_store_type=MemoryType.PERSONAS)

    @staticmethod
    def list_personas(provider: MemoryProvider) -> list:
        """
        List all personas within the memory provider.

        Parameters:
        -----------
        provider : MemoryProvider
            The memory provider to use for listing personas.

        Returns:
        --------
        list: A list of all personas in the memory provider.
        """
        print(f"Listing all personas in the {MemoryType.PERSONAS.value} collection")
        return provider.list_all(memory_store_type=MemoryType.PERSONAS)

    @staticmethod
    def get_most_similar_persona(
        input: str, provider: MemoryProvider, limit: int = 1
    ) -> dict:
        """
        Get the persona with the most similar embedding to the query.

        Parameters:
        -----------
        input : str
            The input to search for.
        provider : MemoryProvider
            The memory provider to use for retrieval.
        limit : int, optional
            The number of personas to return.

        Returns:
        --------
        list: A list of the most similar personas.
        """
        return provider.retrieve_by_query(
            input, memory_store_type=MemoryType.PERSONAS, limit=limit
        )

    def generate_system_prompt_input(self) -> str:
        """
        Generate a system prompt input based on the persona's goals and background.
        """
        return f"""
            You are {self.name}, and you are a {self.role}. You have the following goals: {self.goals}. Your background is: {self.background}.
        """

    def __repr__(self) -> str:
        return (
            f"Persona(persona_id='{self.persona_id}', name='{self.name}', role='{self.role}', "
            f"goals='{self.goals}', background='{self.background}', embedding=[...])"
        )


# Example usage:
if __name__ == "__main__":
    # Create a Persona instance
    persona = Persona(
        name="Alex",
        role=RoleType.ASSISTANT,
        goals="Focus on proactive notifications and user engagement.",
        background="Alex also has experience in scheduling and reminders.",
    )
    print("Initialized Persona:", persona)

    # Store the Persona in MongoDB
    stored_id = persona.store_persona()
    print("Stored Persona ID:", stored_id)

    # Retrieve the Persona from MongoDB using persona_id
    retrieved_persona = Persona.retrieve_persona(persona.persona_id)
    print("Retrieved Persona:", retrieved_persona)

    # Optionally, delete the Persona from MongoDB
    deleted = Persona.delete_persona(persona.persona_id)
    print("Deletion successful:", deleted)
