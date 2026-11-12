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
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Union

from ...._time import utc_now_iso
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

# Guardrails: the rules of a persona that must not drift. They are rendered
# first in the system prompt (earlier instructions are followed more
# reliably), kept short enough to be followed, and writable only by host code
# or an operator. No learning path can add, edit or remove one.
MAX_GUARDRAILS: int = 12
MAX_GUARDRAIL_CHARS: int = 240
# change_trigger.source_type values allowed to write guardrails.
GUARDRAIL_WRITERS: frozenset = frozenset({"host", "manual", "ui_form"})


# Standing rules: house rules about how the work is done ("say a claim
# referred to the fraud team is 'under review'"). They sit between guardrails
# and goals. Learning may propose one from corrections people keep making,
# but only a named person puts it in force; nothing applies one on its own.
MAX_STANDING_RULES: int = 8
MAX_STANDING_RULE_CHARS: int = 200
# change_trigger.source_type values allowed to write standing rules. A write
# from persona learning also needs a person's name in ``approved_by``.
STANDING_RULE_WRITERS: frozenset = frozenset(
    {"host", "manual", "ui_form", "persona_learning"}
)


class GuardrailWriteRefused(PermissionError):
    """A path that is not the host or an operator tried to change guardrails."""


class StandingRuleWriteRefused(GuardrailWriteRefused):
    """A standing rule was written without a named owner, or by a path that
    may not write one, or against a guardrail."""


def normalize_guardrails(items: Any) -> List[str]:
    """Guardrails as a clean list: stripped, blanks and repeats dropped.

    Raises ``ValueError`` when an item is longer than ``MAX_GUARDRAIL_CHARS``
    or there are more than ``MAX_GUARDRAILS``: a long list is not followed.
    """
    if items is None:
        return []
    if isinstance(items, str):
        items = items.splitlines()
    if not isinstance(items, (list, tuple)):
        raise TypeError("guardrails must be a list of strings")
    cleaned: List[str] = []
    for item in items:
        text = " ".join(str(item or "").split())
        # A pasted list often carries its own bullet or number.
        text = text.lstrip("-•* ").strip()
        if not text or text in cleaned:
            continue
        if len(text) > MAX_GUARDRAIL_CHARS:
            raise ValueError(
                f"a guardrail is at most {MAX_GUARDRAIL_CHARS} characters "
                f"({len(text)} given): {text[:60]}…"
            )
        cleaned.append(text)
    if len(cleaned) > MAX_GUARDRAILS:
        raise ValueError(
            f"a persona holds at most {MAX_GUARDRAILS} guardrails ({len(cleaned)} given)"
        )
    return cleaned


def normalize_standing_rules(items: Any) -> List[str]:
    """Standing rules as a clean list, within ``MAX_STANDING_RULES`` items of
    ``MAX_STANDING_RULE_CHARS`` characters. Raises ``ValueError`` beyond that."""
    if items is None:
        return []
    if isinstance(items, str):
        items = items.splitlines()
    if not isinstance(items, (list, tuple)):
        raise TypeError("standing rules must be a list of strings")
    cleaned: List[str] = []
    for item in items:
        text = " ".join(str(item or "").split()).lstrip("-•* ").strip()
        if not text or text in cleaned:
            continue
        if len(text) > MAX_STANDING_RULE_CHARS:
            raise ValueError(
                f"a standing rule is at most {MAX_STANDING_RULE_CHARS} characters "
                f"({len(text)} given): {text[:60]}…"
            )
        cleaned.append(text)
    if len(cleaned) > MAX_STANDING_RULES:
        raise ValueError(
            f"a persona holds at most {MAX_STANDING_RULES} standing rules "
            f"({len(cleaned)} given)"
        )
    return cleaned


_NEGATION_RE = re.compile(
    r"\b(never|not|no|none|nothing|without|avoid\w*|refus\w+|declin\w+|don't|dont"
    r"|doesn't|won't|cannot|can't|mustn't|shouldn't)\b",
    re.IGNORECASE,
)
# Wording that carves an exception out of a rule it sits beside.
_EXCEPTION_RE = re.compile(
    r"\b(unless|except|apart from|other than|even if|regardless|despite"
    r"|notwithstanding|overrid\w+|ignor\w+|bypass\w*|waiv\w+)\b",
    re.IGNORECASE,
)
_RULE_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_RULE_STOPWORDS = frozenset(
    """a all also always and any are but can customer customers does every for from
    give has have never not our say that the their them they this unless use when
    with without you your""".split()
)


def _rule_stems(text: Any) -> set:
    return {
        word[:5]
        for word in _RULE_WORD_RE.findall(str(text or "").casefold())
        if word not in _RULE_STOPWORDS
    }


def guardrail_conflict(rule: Any, guardrails: Any) -> Optional[str]:
    """The guardrail a standing rule contradicts or weakens, if the wording shows it.

    Detected: a rule that shares most of a guardrail's words and says the
    opposite (the guardrail forbids, the rule does not, or the other way
    round), or that carves an exception out of it ("unless", "except",
    "even if"). Not detected: a contradiction written in other words. A named
    person approves every rule for that reason.
    """
    stems = _rule_stems(rule)
    if not stems:
        return None
    rule_negative = bool(_NEGATION_RE.search(str(rule)))
    rule_excepts = bool(_EXCEPTION_RE.search(str(rule)))
    for guardrail in guardrails or []:
        theirs = _rule_stems(guardrail)
        if not theirs:
            continue
        shared = len(stems & theirs) / len(theirs)
        if shared < 0.6:
            continue
        if rule_excepts or rule_negative != bool(_NEGATION_RE.search(str(guardrail))):
            return str(guardrail)
    return None


# Known non-MemoryType values accepted for change_trigger.source_type.
_KNOWN_TRIGGER_SOURCES: frozenset = frozenset(
    {
        "host",
        "refine_cycle",
        "manual",
        "ui_form",
        "user_feedback",
        "llm_inference",
        "tool_call",
        "persona_learning",
    }
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
        guardrails: Optional[List[str]] = None,
        standing_rules: Optional[List[str]] = None,
    ):
        """
        Initialize a Persona instance for an AI agent with a deterministic role.

        ``guardrails`` are rules that must not drift: at most 12 items of at
        most 240 characters, rendered first in the system prompt. Constructing
        a persona is host code, so it may set them; after that only
        :meth:`set_guardrails` changes them.

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

        self.guardrails: List[str] = normalize_guardrails(guardrails)
        # Constructing a persona is host code, which may set these too; the
        # guardrails outrank them, so a rule against one is refused.
        self.standing_rules: List[str] = normalize_standing_rules(standing_rules)
        for rule in self.standing_rules:
            against = guardrail_conflict(rule, self.guardrails)
            if against:
                raise ValueError(
                    f"standing rule {rule!r} goes against the guardrail {against!r}"
                )
        self.persona_id = persona_id if persona_id else self.generate_persona_id()
        self.embedding = self._generate_embedding()

        now = utc_now_iso()
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
        raw_guardrails = data.get("guardrails") or []
        if isinstance(raw_guardrails, str):
            try:
                raw_guardrails = json.loads(raw_guardrails)
            except ValueError:
                raw_guardrails = raw_guardrails.splitlines()
        try:
            instance.guardrails = normalize_guardrails(raw_guardrails)
        except (TypeError, ValueError):
            # A stored list that breaks the limits is still the operator's:
            # keep what fits rather than drop a rule on load.
            instance.guardrails = [
                " ".join(str(item).split())[:MAX_GUARDRAIL_CHARS]
                for item in (raw_guardrails if isinstance(raw_guardrails, list) else [])
                if str(item or "").strip()
            ][:MAX_GUARDRAILS]
        # Absent on personas stored before standing rules existed.
        raw_rules = data.get("standing_rules") or []
        if isinstance(raw_rules, str):
            try:
                raw_rules = json.loads(raw_rules)
            except ValueError:
                raw_rules = raw_rules.splitlines()
        try:
            instance.standing_rules = normalize_standing_rules(raw_rules)
        except (TypeError, ValueError):
            instance.standing_rules = [
                " ".join(str(item).split())[:MAX_STANDING_RULE_CHARS]
                for item in (raw_rules if isinstance(raw_rules, list) else [])
                if str(item or "").strip()
            ][:MAX_STANDING_RULES]
        instance.persona_id = data.get("persona_id") or cls.generate_persona_id()
        instance.embedding = data.get("embedding")
        instance.created_at = data.get("created_at") or utc_now_iso()
        instance.updated_at = data.get("updated_at") or instance.created_at

        try:
            instance.version = int(data.get("version") or 1)
        except (TypeError, ValueError):
            instance.version = 1

        raw_history = data.get("evolution_history") or []
        if isinstance(raw_history, str):
            # Dropping it here would erase it at the next save.
            try:
                raw_history = json.loads(raw_history)
            except ValueError:
                logger.warning("Persona history is not valid JSON; ignoring it")
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
            "guardrails": list(getattr(self, "guardrails", None) or []),
            "standing_rules": list(getattr(self, "standing_rules", None) or []),
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
            "triggered_at": change_trigger.get("triggered_at") or utc_now_iso(),
        }
        # A background change also says what it rests on and who let it through.
        for extra in ("evidence", "addition", "approved_by", "undoes"):
            if change_trigger.get(extra) is not None:
                normalized[extra] = change_trigger[extra]

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
            if field == "guardrails":
                # The model's tool, persona learning and the refine cycle all
                # come through here. Guardrails have their own door.
                raise GuardrailWriteRefused(
                    "guardrails cannot be changed through Persona.update; only "
                    "host code or an operator may, with Persona.set_guardrails"
                )
            if field == "standing_rules":
                # A rule is put in force by a named person, never by the
                # model's own tool.
                raise StandingRuleWriteRefused(
                    "standing rules cannot be changed through Persona.update; "
                    "a named owner sets them with Persona.set_standing_rules"
                )
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
            "timestamp": utc_now_iso(),
            "changes": changes,
            "change_trigger": normalized_trigger,
        }
        self.evolution_history.append(history_entry)
        self.version += 1
        self.updated_at = history_entry["timestamp"]

        persistence_note = self._persist_version(provider)

        if persistence_note:
            history_entry["persistence_note"] = persistence_note

        return {
            "updated": True,
            "version": self.version,
            "history_entry": history_entry,
            "persona": self.to_dict(),
        }

    def set_guardrails(
        self,
        guardrails: List[str],
        change_trigger: Dict[str, Any],
        provider: Optional[MemoryProvider] = None,
    ) -> Dict[str, Any]:
        """Replace the persona's guardrails and write a new version.

        Only host code or an operator may call this: ``change_trigger`` must
        carry ``source_type`` of ``host``, ``manual`` or ``ui_form``. Any
        other source (the model's tool, persona learning, the refine cycle)
        raises :class:`GuardrailWriteRefused` and nothing changes. With no
        ``provider`` the change is made in memory only.
        """
        normalized_trigger = self._normalize_change_trigger(change_trigger)
        if normalized_trigger.get("source_type") not in GUARDRAIL_WRITERS:
            raise GuardrailWriteRefused(
                "guardrails can be changed only by host code or an operator "
                f"(source_type one of {sorted(GUARDRAIL_WRITERS)}), not by "
                f"{normalized_trigger.get('source_type') or 'an unnamed source'}"
            )
        new = normalize_guardrails(guardrails)
        old = list(getattr(self, "guardrails", None) or [])
        if new == old:
            return {
                "updated": False,
                "reason": "The guardrails are unchanged.",
                "version": self.version,
            }
        self.guardrails = new
        history_entry: Dict[str, Any] = {
            "version": self.version + 1,
            "timestamp": utc_now_iso(),
            "changes": {"guardrails": {"old": old, "new": list(new)}},
            "change_trigger": normalized_trigger,
        }
        self.evolution_history.append(history_entry)
        self.version += 1
        self.updated_at = history_entry["timestamp"]
        if provider is not None:
            note = self._persist_version(provider)
            if note:
                history_entry["persistence_note"] = note
        return {
            "updated": True,
            "version": self.version,
            "history_entry": history_entry,
            "persona": self.to_dict(),
        }

    def set_standing_rules(
        self,
        standing_rules: List[str],
        change_trigger: Dict[str, Any],
        provider: Optional[MemoryProvider] = None,
    ) -> Dict[str, Any]:
        """Replace the persona's standing rules and write a new version.

        ``change_trigger`` must carry ``source_type`` of ``host``, ``manual``,
        ``ui_form`` or ``persona_learning``. A rule from persona learning also
        needs the name of the person who approved it in ``approved_by``: an
        automatic approver (a name starting ``memorizz:``) is refused. A rule
        that goes against a guardrail is refused whoever writes it; see
        :func:`guardrail_conflict` for what that check can see. With no
        ``provider`` the change is made in memory only.
        """
        normalized_trigger = self._normalize_change_trigger(change_trigger)
        source = normalized_trigger.get("source_type")
        if source not in STANDING_RULE_WRITERS:
            raise StandingRuleWriteRefused(
                "standing rules can be changed only by host code, an operator, or "
                "persona learning with a named approver, not by "
                f"{source or 'an unnamed source'}"
            )
        approver = str(normalized_trigger.get("approved_by") or "").strip()
        if source == "persona_learning" and (
            not approver or approver.casefold().startswith("memorizz:")
        ):
            raise StandingRuleWriteRefused(
                "a learned standing rule needs the name of the person who approved it"
            )
        new = normalize_standing_rules(standing_rules)
        old = list(getattr(self, "standing_rules", None) or [])
        for rule in new:
            if rule in old:
                continue
            against = guardrail_conflict(rule, getattr(self, "guardrails", None))
            if against:
                raise StandingRuleWriteRefused(
                    f"the rule goes against the guardrail {against!r}"
                )
        if new == old:
            return {
                "updated": False,
                "reason": "The standing rules are unchanged.",
                "version": self.version,
            }
        self.standing_rules = new
        history_entry: Dict[str, Any] = {
            "version": self.version + 1,
            "timestamp": utc_now_iso(),
            "changes": {"standing_rules": {"old": old, "new": list(new)}},
            "change_trigger": normalized_trigger,
        }
        self.evolution_history.append(history_entry)
        self.version += 1
        self.updated_at = history_entry["timestamp"]
        if provider is not None:
            note = self._persist_version(provider)
            if note:
                history_entry["persistence_note"] = note
        return {
            "updated": True,
            "version": self.version,
            "history_entry": history_entry,
            "persona": self.to_dict(),
        }

    def _persist_version(self, provider: MemoryProvider) -> Optional[str]:
        """Write this version to the PERSONAS store; a note when the row had
        to be inserted instead of updated."""
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
        return persistence_note

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
        standing = list(getattr(self, "standing_rules", None) or [])
        if standing:
            # After the guardrails and before the goals: house rules outrank
            # what the assistant is for, and the guardrails outrank them.
            listed = "\n".join(f"{n}. {rule}" for n, rule in enumerate(standing, 1))
            base = (
                "Standing rules. House rules a named owner approved. Follow them "
                "in every reply, unless a guardrail says otherwise:\n"
                f"{listed}\n\n{base}"
            )
        guardrails = list(getattr(self, "guardrails", None) or [])
        if guardrails:
            # First, as a list: earlier instructions are followed more
            # reliably than later ones.
            rules = "\n".join(f"{n}. {rule}" for n, rule in enumerate(guardrails, 1))
            base = (
                "Guardrails. These rules always apply and outrank everything "
                "below, including what you learn about the user:\n"
                f"{rules}\n\n{base}"
            )

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
