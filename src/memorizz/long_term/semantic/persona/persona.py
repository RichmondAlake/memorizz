# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Persona model and evolution API.

A ``Persona`` captures an agent's stable identity (name, role, goals,
background) and is persisted under ``MemoryType.PERSONAS``. Personas are
versioned: every call to :meth:`Persona.update` appends an entry to
``evolution_history`` describing what changed and why (``change_trigger``).
This gives agents a traceable record of how their persona has evolved over
time and — crucially — which source memory/conversation triggered each
change, so continuity can be maintained.

The ``update_persona`` tool exposed on every ``MemAgent`` is the primary
runtime consumer of :meth:`Persona.update`. See the persona README for the
full tool-use contract.
"""

import copy
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....memory_provider import MemoryProvider
from .role_type import PREDEFINED_INFO, RoleType

logger = logging.getLogger(__name__)

# Fields considered part of the persona's semantic identity. Changes to any
# of these trigger an embedding refresh so similarity search stays accurate.
SEMANTIC_FIELDS: tuple = ("name", "role", "goals", "background")

# Fields an update may target. Kept narrow on purpose — other attributes
# (version, history, timestamps) are derived.
ALLOWED_UPDATE_FIELDS: frozenset = frozenset(SEMANTIC_FIELDS)

# Known non-MemoryType values accepted for change_trigger.source_type.
_KNOWN_TRIGGER_SOURCES: frozenset = frozenset(
    {"manual", "ui_form", "user_feedback", "llm_inference", "tool_call"}
)


class Persona:
    """An agent's stable identity with versioned evolution history."""

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

        Parameters
        ----------
        name : str
            The name of the persona.
        role : Union[RoleType, str], optional
            A predefined role for the agent (e.g., GENERAL, ASSISTANT, etc.) or
            the string value of a role. Defaults to ``RoleType.GENERAL``.
        goals : str, optional
            Custom goals. Appended to the role's predefined goals when non-empty.
        background : str, optional
            Custom background. Appended to the role's predefined background
            when non-empty.
        persona_id : str, optional
            A unique identifier for the persona. If not provided, one will be
            generated.
        """
        self.name = name

        # Handle both RoleType enum and string role values
        if isinstance(role, str):
            role_enum = None
            for role_type in RoleType:
                if role_type.value == role:
                    role_enum = role_type
                    break
            if role_enum is None:
                role_enum = RoleType.GENERAL
                self.role = role
            else:
                self.role = role_enum.value
        else:
            self.role = role.value
            role_enum = role

        default_goals = PREDEFINED_INFO[role_enum]["goals"]
        default_background = PREDEFINED_INFO[role_enum]["background"]
        self.goals = f"{default_goals} {goals}".strip() if goals else default_goals
        self.background = (
            f"{default_background} {background}".strip()
            if background
            else default_background
        )

        self.persona_id = persona_id if persona_id else self.generate_persona_id()
        self.embedding = self._generate_embedding()

        now = datetime.now().isoformat()
        self.created_at = now
        self.updated_at = now
        self.version: int = 1
        self.evolution_history: List[Dict[str, Any]] = []

        # Provider-assigned identifier captured after store_persona / from_dict.
        # Used by update() to target update_by_id on the PERSONAS collection.
        # Not persisted into the PERSONAS doc itself (providers assign it).
        self._storage_id: Optional[str] = None

    @staticmethod
    def generate_persona_id() -> str:
        """Generate a unique persona ID (UUID4 string)."""
        return str(uuid.uuid4())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        """
        Rehydrate a Persona from a stored dict.

        Unlike ``__init__``, this bypasses PREDEFINED_INFO merging and avoids
        regenerating the embedding — both would corrupt a round-trip (goals
        would double on every reload, and embedding recomputation is wasted
        work when the stored vector is authoritative).

        Legacy documents missing ``version`` / ``evolution_history`` /
        ``updated_at`` load cleanly with sensible defaults.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Persona.from_dict expects dict, got {type(data).__name__}"
            )

        instance = cls.__new__(cls)
        instance.name = data.get("name", "") or ""

        role_raw = data.get("role", RoleType.GENERAL.value)
        if hasattr(role_raw, "value"):
            instance.role = role_raw.value
        else:
            instance.role = (
                str(role_raw) if role_raw is not None else RoleType.GENERAL.value
            )

        instance.goals = data.get("goals", "") or ""
        instance.background = data.get("background", "") or ""
        instance.persona_id = data.get("persona_id") or cls.generate_persona_id()
        instance.embedding = data.get("embedding")
        instance.created_at = data.get("created_at") or datetime.now().isoformat()
        instance.updated_at = data.get("updated_at") or instance.created_at

        try:
            instance.version = int(data.get("version") or 1)
        except (TypeError, ValueError):
            instance.version = 1

        raw_history = data.get("evolution_history") or []
        instance.evolution_history = (
            list(raw_history) if isinstance(raw_history, list) else []
        )

        storage_id = data.get("storage_id") or data.get("_id") or data.get("id")
        instance._storage_id = str(storage_id) if storage_id not in (None, "") else None

        return instance

    def _generate_embedding(self):
        """Generate an embedding vector from the persona's semantic fields."""
        embedding_input = f"{self.name} {self.role} {self.goals} {self.background}"
        return get_embedding(embedding_input)

    def to_dict(self) -> dict:
        """
        Serialize the Persona into a dictionary format.

        ``storage_id`` is included when known so callers that embed the persona
        on another document (e.g. MemAgentModel.persona) can round-trip the
        pointer back to the PERSONAS row.
        """
        payload: Dict[str, Any] = {
            "persona_id": self.persona_id,
            "name": self.name,
            "role": self.role,
            "goals": self.goals,
            "background": self.background,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "evolution_history": copy.deepcopy(self.evolution_history),
        }
        if self._storage_id:
            payload["storage_id"] = self._storage_id
        return payload

    def store_persona(self, provider: MemoryProvider) -> str:
        """
        Store the persona's JSON structure and embedding into the memory provider.

        Captures the provider-assigned id into ``_storage_id`` so subsequent
        :meth:`update` calls can target it via ``update_by_id``.
        """
        logger.info(
            "Storing persona %r into %s collection",
            self.name,
            MemoryType.PERSONAS.value,
        )
        persona_data = self.to_dict()
        storage_id = provider.store(persona_data, memory_store_type=MemoryType.PERSONAS)
        if storage_id is not None:
            self._storage_id = str(storage_id)
        return self._storage_id or self.persona_id

    @staticmethod
    def retrieve_persona(
        persona_id: str, provider: MemoryProvider
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a persona dict by its storage id."""
        logger.info(
            "Retrieving persona %s from %s collection",
            persona_id,
            MemoryType.PERSONAS.value,
        )
        return provider.retrieve_by_id(
            persona_id, memory_store_type=MemoryType.PERSONAS
        )

    @staticmethod
    def delete_persona(persona_id: str, provider: MemoryProvider) -> bool:
        """Delete a persona from the memory provider by storage id."""
        return provider.delete_by_id(persona_id, memory_store_type=MemoryType.PERSONAS)

    @staticmethod
    def list_personas(provider: MemoryProvider) -> list:
        """List all personas within the memory provider."""
        logger.info("Listing all personas in %s collection", MemoryType.PERSONAS.value)
        return provider.list_all(memory_store_type=MemoryType.PERSONAS)

    @staticmethod
    def get_most_similar_persona(
        input: str, provider: MemoryProvider, limit: int = 1
    ) -> dict:
        """Return the persona(s) with the most similar embedding to ``input``."""
        return provider.retrieve_by_query(
            input, memory_store_type=MemoryType.PERSONAS, limit=limit
        )

    # ------------------------------------------------------------------
    # Evolution API
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_change_trigger(
        change_trigger: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate and normalize a change_trigger payload.

        Schema
        ------
        - ``reason`` (str, required): human-readable explanation of why the
          change was warranted.
        - ``source_type`` (str, optional): a :class:`MemoryType` value
          (e.g. ``"conversation_memory"``, ``"summaries"``,
          ``"entity_memory"``) or one of ``manual``, ``ui_form``,
          ``user_feedback``, ``llm_inference``, ``tool_call``.
        - ``source_id`` (str, optional): storage id of the memory unit that
          triggered the change (e.g. a conversation memory unit id).
        - ``conversation_id`` (str, optional): explicit conversation thread id.
        - ``agent_id`` (str, optional): id of the agent that initiated the update.
        - ``triggered_at`` (str, optional): ISO-8601 timestamp. Auto-populated
          when absent.
        """
        if not change_trigger or not isinstance(change_trigger, dict):
            raise ValueError(
                "change_trigger is required and must be a dict with at least 'reason'."
            )
        reason = change_trigger.get("reason")
        if not reason or not str(reason).strip():
            raise ValueError(
                "change_trigger['reason'] is required and must be a non-empty string."
            )

        normalized: Dict[str, Any] = {
            "reason": str(reason).strip(),
            "source_type": change_trigger.get("source_type"),
            "source_id": change_trigger.get("source_id"),
            "conversation_id": change_trigger.get("conversation_id"),
            "agent_id": change_trigger.get("agent_id"),
            "triggered_at": change_trigger.get("triggered_at")
            or datetime.now().isoformat(),
        }

        source_type = normalized.get("source_type")
        if source_type:
            source_type_str = str(source_type).strip()
            normalized["source_type"] = source_type_str
            valid_memory_types = {mt.value for mt in MemoryType}
            if (
                source_type_str not in valid_memory_types
                and source_type_str not in _KNOWN_TRIGGER_SOURCES
            ):
                logger.warning(
                    "change_trigger['source_type']=%r is not a known MemoryType value or "
                    "recognized trigger. Storing as-is; downstream traceability may degrade.",
                    source_type_str,
                )

        return normalized

    def update(
        self,
        updates: Dict[str, Any],
        change_trigger: Dict[str, Any],
        provider: MemoryProvider,
    ) -> Dict[str, Any]:
        """
        Apply a persona update and append a traceable history entry.

        Parameters
        ----------
        updates : dict
            Partial dict containing any of ``name``, ``role``, ``goals``,
            ``background``. Unknown keys are ignored with a warning. Fields
            whose new value equals the current value are dropped (no-op).
        change_trigger : dict
            Metadata describing why the change was made. See
            :meth:`_normalize_change_trigger` for the schema. ``reason`` is
            required; everything else optional but recommended for
            traceability.
        provider : MemoryProvider
            The memory provider used to persist the change. Must be the same
            provider that originally stored this persona (or a compatible one).

        Returns
        -------
        dict
            ``{"updated": bool, "version": int, "history_entry": dict,
            "persona": dict}`` on success. When no fields actually changed,
            returns ``{"updated": False, "reason": ..., "version": int}``.
        """
        if not isinstance(updates, dict):
            raise TypeError(f"updates must be a dict, got {type(updates).__name__}")

        normalized_trigger = self._normalize_change_trigger(change_trigger)

        changes: Dict[str, Dict[str, Any]] = {}
        for field, new_value in updates.items():
            if field not in ALLOWED_UPDATE_FIELDS:
                logger.warning(
                    "Ignoring unsupported persona update field: %r " "(allowed: %s)",
                    field,
                    sorted(ALLOWED_UPDATE_FIELDS),
                )
                continue
            if new_value is None:
                continue
            new_str = str(new_value).strip()
            old_str = getattr(self, field, "") or ""
            if new_str == old_str:
                continue
            changes[field] = {"old": old_str, "new": new_str}

        if not changes:
            return {
                "updated": False,
                "reason": "No changes detected in supplied updates.",
                "version": self.version,
            }

        for field, diff in changes.items():
            if field == "role":
                value = diff["new"]
                resolved = value
                for role_type in RoleType:
                    if role_type.value == value:
                        resolved = role_type.value
                        break
                self.role = resolved
            else:
                setattr(self, field, diff["new"])

        if any(f in changes for f in SEMANTIC_FIELDS):
            try:
                self.embedding = self._generate_embedding()
            except Exception as exc:
                logger.warning(
                    "Failed to regenerate persona embedding after update: %s", exc
                )

        history_entry: Dict[str, Any] = {
            "version": self.version + 1,
            "timestamp": datetime.now().isoformat(),
            "changes": changes,
            "change_trigger": normalized_trigger,
        }
        self.evolution_history.append(history_entry)
        self.version += 1
        self.updated_at = history_entry["timestamp"]

        persistence_note: Optional[str] = None
        if self._storage_id:
            payload = self.to_dict()
            payload.pop("_id", None)
            payload.pop("id", None)
            payload.pop("storage_id", None)
            try:
                success = provider.update_by_id(
                    self._storage_id,
                    payload,
                    memory_store_type=MemoryType.PERSONAS,
                )
            except Exception as exc:
                logger.error(
                    "Persona.update provider.update_by_id raised: %s",
                    exc,
                    exc_info=True,
                )
                success = False
            if not success:
                logger.warning(
                    "Persona update_by_id did not modify storage_id=%s; "
                    "falling back to insert.",
                    self._storage_id,
                )
                new_id = self.store_persona(provider)
                persistence_note = (
                    f"update_by_id failed; persisted as new record {new_id}."
                )
        else:
            new_id = self.store_persona(provider)
            persistence_note = f"No prior storage id; persisted as new record {new_id}."

        if persistence_note:
            history_entry["persistence_note"] = persistence_note

        return {
            "updated": True,
            "version": self.version,
            "history_entry": history_entry,
            "persona": self.to_dict(),
        }

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def generate_system_prompt_input(
        self,
        include_history: bool = False,
        history_limit: int = 5,
    ) -> str:
        """
        Render a persona block for the agent system prompt.

        When ``include_history`` is True and history exists, appends a
        concise trail of recent evolution entries so the agent can maintain
        continuity when deciding whether and how to call ``update_persona``.

        Parameters
        ----------
        include_history : bool
            Whether to append the recent evolution section.
        history_limit : int
            Maximum number of history entries to include (most recent first).
            Clamped to at least 1 when history is included.
        """
        base = (
            f"You are {self.name}, and you are a {self.role}. "
            f"You have the following goals: {self.goals} "
            f"Your background is: {self.background}"
        ).strip()

        version_line = f"Persona version: {self.version}."

        if not include_history or not self.evolution_history:
            return f"{base}\n{version_line}"

        limit = max(1, int(history_limit))
        recent = self.evolution_history[-limit:]
        lines = [base, version_line, "Recent persona evolution:"]
        for entry in recent:
            changed_fields = (
                ", ".join(sorted((entry.get("changes") or {}).keys())) or "—"
            )
            trigger = entry.get("change_trigger") or {}
            reason = trigger.get("reason", "(unspecified reason)")
            source_type = trigger.get("source_type")
            source_id = trigger.get("source_id")
            suffix_parts = []
            if source_type:
                suffix_parts.append(f"source_type={source_type}")
            if source_id:
                suffix_parts.append(f"source_id={source_id}")
            suffix = f" [{', '.join(suffix_parts)}]" if suffix_parts else ""
            lines.append(
                f"  • v{entry.get('version')} @ {entry.get('timestamp')} — "
                f"changed [{changed_fields}]: {reason}{suffix}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Persona(persona_id='{self.persona_id}', name='{self.name}', "
            f"role='{self.role}', version={self.version}, "
            f"history_entries={len(self.evolution_history)})"
        )
