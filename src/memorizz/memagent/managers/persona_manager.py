# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Persona management functionality for MemAgent."""

import logging
from typing import Any, Dict, List, Optional

from ...long_term_memory.semantic.persona.persona import Persona

logger = logging.getLogger(__name__)


class PersonaManager:
    """
    Manages persona configuration and updates for MemAgent.

    This class encapsulates persona-related functionality that was
    previously embedded in the main MemAgent class.
    """

    def __init__(self, memory_provider=None):
        """
        Initialize the persona manager.

        Args:
            memory_provider: Optional memory provider for persona storage.
        """
        self.memory_provider = memory_provider
        self.current_persona = None
        self._persona_cache = {}

    def set_persona(self, persona: Persona, agent_id: str, save: bool = True) -> bool:
        """
        Set the agent's persona.

        Args:
            persona: The Persona instance to set.
            agent_id: The agent ID.
            save: Whether to persist the persona.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.current_persona = persona

            # Cache the persona
            self._persona_cache[agent_id] = persona

            # Persist if requested
            if save and self.memory_provider:
                self._persist_persona(agent_id, persona)

            logger.info(f"Set persona for agent {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set persona: {e}")
            return False

    def load_persona(self, persona_id: str) -> Optional[Persona]:
        """
        Load a persona from storage.

        Args:
            persona_id: The persona ID to load.

        Returns:
            The loaded Persona instance, or None if not found.
        """
        try:
            # Check cache first
            if persona_id in self._persona_cache:
                return self._persona_cache[persona_id]

            # Load from storage
            if self.memory_provider:
                persona_data = self.memory_provider.retrieve_by_id(persona_id)
                if persona_data:
                    persona = self._deserialize_persona(persona_data)
                    self._persona_cache[persona_id] = persona
                    return persona

            logger.warning(f"Persona not found: {persona_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to load persona: {e}")
            return None

    def update_persona_from_summaries(
        self, summaries: List[str], llm_provider=None
    ) -> Dict[str, str]:
        """
        Update persona based on memory summaries.

        Args:
            summaries: List of memory summaries.
            llm_provider: LLM provider for generating updates.

        Returns:
            Dictionary with updated persona attributes.
        """
        try:
            if not self.current_persona:
                logger.warning("No current persona to update")
                return {}

            # Generate persona updates using LLM
            updates = self._generate_persona_updates(summaries, llm_provider)

            # Apply updates to current persona
            if updates:
                self._apply_persona_updates(updates)

            return updates

        except Exception as e:
            logger.error(f"Failed to update persona from summaries: {e}")
            return {}

    def export_persona(self) -> Optional[Dict[str, Any]]:
        """
        Export the current persona as a dictionary.

        Returns:
            Dictionary representation of the persona.
        """
        if not self.current_persona:
            return None

        try:
            return {
                "name": getattr(self.current_persona, "name", "Unknown"),
                "role": getattr(self.current_persona, "role", "Assistant"),
                "personality_traits": getattr(
                    self.current_persona, "personality_traits", []
                ),
                "expertise": getattr(self.current_persona, "expertise", []),
                "background": getattr(self.current_persona, "background", ""),
                "goals": getattr(self.current_persona, "goals", []),
                "constraints": getattr(self.current_persona, "constraints", []),
            }
        except Exception as e:
            logger.error(f"Failed to export persona: {e}")
            return None

    def delete_persona(self, agent_id: str, save: bool = True) -> bool:
        """
        Delete the agent's persona.

        Args:
            agent_id: The agent ID.
            save: Whether to persist the deletion.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Clear current persona
            self.current_persona = None

            # Remove from cache
            if agent_id in self._persona_cache:
                del self._persona_cache[agent_id]

            # Persist deletion if requested
            if save and self.memory_provider:
                # Update agent to remove persona reference
                pass  # Implementation depends on memory provider interface

            logger.info(f"Deleted persona for agent {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete persona: {e}")
            return False

    def get_persona_prompt(self) -> str:
        """
        Generate a prompt string from the current persona.

        Returns:
            Formatted persona prompt string.
        """
        if not self.current_persona:
            return ""

        try:
            prompt_parts = []

            if hasattr(self.current_persona, "name"):
                prompt_parts.append(f"You are {self.current_persona.name}")

            if hasattr(self.current_persona, "role"):
                prompt_parts.append(f"Your role is: {self.current_persona.role}")

            if hasattr(self.current_persona, "personality_traits"):
                traits = self.current_persona.personality_traits
                if traits:
                    prompt_parts.append(f"Personality traits: {', '.join(traits)}")

            if hasattr(self.current_persona, "expertise"):
                expertise = self.current_persona.expertise
                if expertise:
                    prompt_parts.append(f"Areas of expertise: {', '.join(expertise)}")

            if hasattr(self.current_persona, "background"):
                if self.current_persona.background:
                    prompt_parts.append(
                        f"Background: {self.current_persona.background}"
                    )

            return "\n".join(prompt_parts)

        except Exception as e:
            logger.error(f"Failed to generate persona prompt: {e}")
            return ""

    def _persist_persona(self, agent_id: str, persona: Persona) -> bool:
        """Persist persona to storage."""
        try:
            if not self.memory_provider:
                return False

            # Serialize and store persona
            persona_data = self._serialize_persona(persona)
            self.memory_provider.store(
                memory_id=f"persona_{agent_id}", memory_unit=persona_data
            )

            return True

        except Exception as e:
            logger.error(f"Failed to persist persona: {e}")
            return False

    def _serialize_persona(self, persona: Persona) -> Dict[str, Any]:
        """Serialize a Persona instance to dictionary."""
        return self.export_persona() or {}

    def _deserialize_persona(self, data: Dict[str, Any]) -> Persona:
        """Deserialize dictionary to Persona instance."""
        # This would need proper implementation based on Persona class
        return Persona(**data)

    def _generate_persona_updates(
        self, summaries: List[str], llm_provider
    ) -> Dict[str, str]:
        """Generate persona updates using LLM."""
        # Placeholder for LLM-based persona update generation
        return {}

    def _apply_persona_updates(self, updates: Dict[str, str]):
        """Apply updates to the current persona."""
        if not self.current_persona:
            return

        for key, value in updates.items():
            if hasattr(self.current_persona, key):
                setattr(self.current_persona, key, value)
