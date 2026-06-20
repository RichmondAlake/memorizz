# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Persona management for MemAgent.

Wraps the :class:`~memorizz.long_term.semantic.persona.Persona` lifecycle for
a single agent, including:

- Accepting either a ``Persona`` instance or a stored dict via
  :meth:`PersonaManager.set_persona` (dicts are rehydrated through
  ``Persona.from_dict`` so legacy documents load cleanly).
- Rendering the persona block for the agent system prompt, with optional
  evolution-history context so the agent can reason about continuity when
  calling ``update_persona``.
- Applying traceable updates through :meth:`PersonaManager.apply_update`,
  which persists to the PERSONAS collection and returns the history entry.
"""

import logging
from typing import Any, Dict, Optional, Union

from ...long_term.semantic.persona.persona import Persona

logger = logging.getLogger(__name__)


class PersonaManager:
    """
    Manages persona configuration and updates for MemAgent.
    """

    def __init__(self, memory_provider=None):
        """
        Initialize the persona manager.

        Args:
            memory_provider: Optional memory provider for persona storage.
        """
        self.memory_provider = memory_provider
        self.current_persona: Optional[Persona] = None
        self._persona_cache: Dict[str, Persona] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_persona(
        self,
        persona: Union[Persona, Dict[str, Any], None],
        agent_id: str,
        save: bool = True,
    ) -> bool:
        """
        Set the agent's active persona.

        ``persona`` may be a :class:`Persona` instance or a stored dict. Dicts
        are rehydrated via :meth:`Persona.from_dict` (no default-merging, no
        re-embedding) so round-tripped state is preserved.

        When ``save=True`` and a memory_provider is configured, the persona is
        also stored in the PERSONAS collection so it becomes discoverable via
        the saved-personas picker in the UI.
        """
        try:
            if persona is None:
                self.current_persona = None
                if agent_id in self._persona_cache:
                    del self._persona_cache[agent_id]
                return True

            if isinstance(persona, Persona):
                instance = persona
            elif isinstance(persona, dict):
                instance = Persona.from_dict(persona)
            else:
                logger.error(
                    "set_persona received unsupported type %s; ignoring.",
                    type(persona).__name__,
                )
                return False

            self.current_persona = instance
            if agent_id:
                self._persona_cache[agent_id] = instance

            if save and self.memory_provider is not None:
                try:
                    instance.store_persona(self.memory_provider)
                except Exception as exc:
                    logger.warning(
                        "Failed to persist persona to PERSONAS collection: %s", exc
                    )

            logger.info("Set persona for agent %s", agent_id)
            return True

        except Exception as exc:
            logger.error("Failed to set persona: %s", exc, exc_info=True)
            return False

    def load_persona(self, persona_id: str) -> Optional[Persona]:
        """
        Load a persona from the PERSONAS collection by storage id.

        Returns a :class:`Persona` instance or None if not found.
        """
        if not persona_id:
            return None
        if persona_id in self._persona_cache:
            return self._persona_cache[persona_id]

        if self.memory_provider is None:
            return None

        try:
            persona_data = Persona.retrieve_persona(persona_id, self.memory_provider)
            if not persona_data:
                logger.warning("Persona not found: %s", persona_id)
                return None
            persona = Persona.from_dict(persona_data)
            self._persona_cache[persona_id] = persona
            return persona
        except Exception as exc:
            logger.error("Failed to load persona %s: %s", persona_id, exc)
            return None

    def delete_persona(self, agent_id: str, save: bool = False) -> bool:
        """Clear the agent's active persona (does not delete from storage).

        ``save`` is accepted for call-site symmetry with ``set_persona``; clearing
        is in-memory only, so it is currently a no-op.
        """
        try:
            self.current_persona = None
            if agent_id in self._persona_cache:
                del self._persona_cache[agent_id]
            return True
        except Exception as exc:
            logger.error("Failed to clear persona: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def apply_update(
        self,
        updates: Dict[str, Any],
        change_trigger: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply a traceable update to the active persona.

        Delegates to :meth:`Persona.update`. The persona document in the
        PERSONAS collection is updated in place (or re-stored if the storage
        id is unknown); the caller is responsible for refreshing any
        downstream snapshot (e.g. MemAgentModel.persona) via
        :attr:`current_persona` afterwards.
        """
        if self.current_persona is None:
            return {
                "updated": False,
                "error": (
                    "No active persona set on this agent. Attach a persona via "
                    "MemAgentBuilder().with_persona(...) or agent.set_persona(...)."
                ),
            }
        if self.memory_provider is None:
            return {
                "updated": False,
                "error": (
                    "No memory_provider configured; persona updates cannot be persisted."
                ),
            }
        try:
            return self.current_persona.update(
                updates=updates,
                change_trigger=change_trigger,
                provider=self.memory_provider,
            )
        except ValueError as exc:
            return {"updated": False, "error": str(exc)}
        except Exception as exc:
            logger.error("persona.update failed: %s", exc, exc_info=True)
            return {"updated": False, "error": f"Internal error: {exc}"}

    # ------------------------------------------------------------------
    # Export / prompt
    # ------------------------------------------------------------------

    def export_persona(self) -> Optional[Dict[str, Any]]:
        """Return the current persona as a dict, including history."""
        if self.current_persona is None:
            return None
        try:
            return self.current_persona.to_dict()
        except Exception as exc:
            logger.error("Failed to export persona: %s", exc)
            return None

    def get_persona_prompt(
        self,
        include_history: bool = True,
        history_limit: int = 5,
    ) -> str:
        """
        Render the persona block for the agent system prompt.

        When ``include_history=True`` and evolution_history is non-empty, the
        block also summarizes recent changes so the agent can preserve
        continuity when deciding whether to call ``update_persona``.
        """
        if self.current_persona is None:
            return ""
        try:
            return self.current_persona.generate_system_prompt_input(
                include_history=include_history,
                history_limit=history_limit,
            )
        except Exception as exc:
            logger.error("Failed to generate persona prompt: %s", exc)
            return ""
