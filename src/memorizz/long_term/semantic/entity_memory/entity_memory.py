# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
import weakref
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ...._time import to_epoch, utc_now_iso
from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_ENTITY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SELF_ENTITY_ALIASES = frozenset(
    {"user", "current user", "self", "profile", "user profile"}
)
_SELF_IDENTITY_KEYS = frozenset(
    {"authenticated_user", "current_user", "user", "self", "user_profile"}
)
_PROFILE_QUERY_TERMS = frozenset(
    {
        "about",
        "audience",
        "goal",
        "know",
        "me",
        "my",
        "name",
        "preference",
        "profile",
        "project",
        "role",
        "user",
    }
)
_IDENTITY_KEY_RE = re.compile(r"[^a-z0-9_.:-]+")

# -- vectors for large entities ----------------------------------------------

#: Where a record says its vector does not cover all of it. Absent when it
#: does. ``embedding_status`` is ``"partial"`` or ``"missing"``.
_EMBEDDING_STATUS_KEYS = (
    "embedding_status",
    "embedding_reason",
    "embedding_attributes_covered",
)
#: Token estimate: one token per three characters. English runs nearer four,
#: so this overcounts, which is the safe side of an input limit.
_CHARS_PER_TOKEN = 3
#: Share of the model's input limit an entity's text may use.
_EMBEDDING_BUDGET_SHARE = 0.9
#: Input limits in tokens, by the start of the embedding model's name.
_MODEL_INPUT_TOKENS = (
    ("text-embedding-", 8191),
    ("voyage-2", 4000),
    ("voyage-large-2", 16000),
    ("voyage-code-2", 16000),
    ("voyage-", 32000),
    ("nomic-embed", 8192),
    ("bge-m3", 8192),
    ("mxbai-embed", 512),
    ("all-minilm", 256),
    ("snowflake-arctic-embed", 512),
    ("embeddinggemma", 2048),
    ("google/embeddinggemma", 2048),
    ("qwen3-embedding", 32000),
    ("qwen/qwen3-embedding", 32000),
)
#: Ollama's builds of some models take less than the original (it reports
#: nomic-embed-text's context as 2,048 tokens). Checked before the list above.
_OLLAMA_MODEL_INPUT_TOKENS = (
    ("nomic-embed-text-v2", 512),
    ("nomic-embed", 2048),
)
#: Input limits by embedding provider when the model is not listed above.
_PROVIDER_INPUT_TOKENS = {
    "openai": 8191,
    "azure": 8191,
    "voyageai": 32000,
    "ollama": 2048,
    "huggingface": 512,
}
_DEFAULT_INPUT_TOKENS = 8191
_LENGTH_ERROR_RE = re.compile(
    r"context.length|maximum (context|input|sequence)|max(imum)?[ _]tokens"
    r"|too many tokens|too long|token limit|input length|exceeds? .{0,40}tokens"
    r"|reduce (the )?(length|input|prompt)",
    re.IGNORECASE,
)

#: Earlier values an attribute keeps, newest first. Forgetting a conversation
#: restores the newest value another conversation (or the user) gave.
ATTRIBUTE_HISTORY_LIMIT = 20
#: What a history entry keeps of the value it replaced.
_HISTORY_FIELDS = (
    "value",
    "confidence",
    "source",
    "thread_id",
    "pinned",
    "level",
    "updated_at",
    "confirmations",
    "last_seen",
)
#: Keys only a history entry carries: when the value stopped being true
#: (``valid_to``) and whether it ever was current (``applied``: a weaker write
#: that disagreed is kept, not applied); when the system learned it
#: (``recorded_at``) and stopped believing it (``superseded_at``); and, for a
#: value that turned out to be wrong rather than out of date, ``corrected``
#: with ``corrected_at``. ``backfilled`` marks an earlier value the host
#: supplied after the fact.
_HISTORY_ONLY_FIELDS = (
    "valid_from",
    "valid_to",
    "applied",
    "recorded_at",
    "superseded_at",
    "corrected",
    "corrected_at",
    "backfilled",
)
#: The same value from the same place counts as a new confirmation only
#: after this long, so an agent that repeats a fact every turn does not
#: inflate its count (or cause a write per turn).
REINFORCE_QUIET_SECONDS = 3600.0
#: The newest confirmations an attribute keeps the detail of.
CONFIRMED_BY_LIMIT = 5

#: How much a write's source is trusted, strongest first. A write replaces
#: the current value only when its level is the same or higher.
SOURCE_LEVEL_HOST = 3  # pinned, or written by the host through the SDK
SOURCE_LEVEL_USER = 2  # stated by a user in conversation
SOURCE_LEVEL_AGENT = 1  # inferred by the agent, from a tool, or by consolidation
SOURCE_LEVEL_LABELS = {
    SOURCE_LEVEL_HOST: "Host or pinned",
    SOURCE_LEVEL_USER: "Stated by the user",
    SOURCE_LEVEL_AGENT: "Inferred, tool or consolidation",
}
_USER_SOURCES = ("turn", "entity_capture", "conversation", "user", "mcp")
_HOST_SOURCES = ("host", "sdk", "operator", "ui")


def source_level(attribute: Mapping[str, Any]) -> int:
    """The level of a stored attribute value (or a history entry).

    A value written since levels existed carries its own ``level``. An older
    one is ranked from what it does say: pinned is the host's level, a source
    that names the conversation is the user's, and anything else (a tool, an
    inference, consolidation, or no source at all) is the lowest, so it stays
    as replaceable as it was before the rule.
    """
    if attribute.get("pinned"):
        return SOURCE_LEVEL_HOST
    try:
        stored = int(attribute.get("level") or 0)
    except (TypeError, ValueError):
        stored = 0
    if stored in SOURCE_LEVEL_LABELS:
        return stored
    source = str(attribute.get("source") or "").strip().casefold()
    head = source.split(":", 1)[0]
    tail = source.split(":", 1)[1] if ":" in source else ""
    if head in _HOST_SOURCES:
        return SOURCE_LEVEL_HOST
    if head in _USER_SOURCES or (head == "memory_routing" and tail == "turn"):
        return SOURCE_LEVEL_USER
    return SOURCE_LEVEL_AGENT


def _moment(value: Any) -> Optional[float]:
    """Epoch seconds of a stored or caller-supplied time (a date alone is
    midnight UTC); None when there is none."""
    return to_epoch(value)


def attribute_timeline(attribute: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """An attribute's values, current first, each with its dates.

    Every row has ``value``, ``valid_from``, ``valid_to`` (None while
    current), ``applied``, ``current``, ``level`` and what provenance was
    stored. For values written before dates were kept, ``valid_from`` falls
    back to ``updated_at``, so nothing has to be migrated.

    Two clocks. ``valid_from`` and ``valid_to`` say when the value was true
    in the world. ``recorded_at`` says when the system learned it and
    ``superseded_at`` when it stopped believing it (None while current). A
    value that was *corrected* (it was wrong, not out of date) has
    ``corrected: True`` and no ``valid_to``: it was believed from
    ``recorded_at`` to ``superseded_at`` and was never true.

    ``confirmations`` counts how often the value was stated again after it
    was first stored, ``last_seen`` is when that last happened and
    ``confirmed_by`` lists the newest of them.
    """
    rows: List[Dict[str, Any]] = []
    current = {
        key: attribute.get(key)
        for key in ("value", "confidence", "source", "thread_id", "pinned")
        if attribute.get(key) not in (None, "")
    }
    current.update(
        valid_from=attribute.get("valid_from")
        or attribute.get("created_at")
        or attribute.get("updated_at"),
        valid_to=None,
        applied=True,
        current=True,
        level=source_level(attribute),
        recorded_at=attribute.get("created_at") or attribute.get("updated_at"),
        superseded_at=None,
        corrected=False,
        confirmations=_count(attribute.get("confirmations")),
        last_seen=attribute.get("last_seen"),
    )
    if attribute.get("confirmed_by"):
        current["confirmed_by"] = [
            dict(item)
            for item in attribute.get("confirmed_by") or []
            if isinstance(item, dict)
        ]
    rows.append(current)
    for item in attribute.get("history") or []:
        if not isinstance(item, dict):
            continue
        row = {
            key: item.get(key)
            for key in ("value", "confidence", "source", "thread_id", "pinned")
            if item.get(key) not in (None, "")
        }
        applied = item.get("applied") is not False
        corrected = bool(item.get("corrected")) and applied
        row.update(
            valid_from=item.get("valid_from") or item.get("updated_at"),
            valid_to=item.get("valid_to") if applied and not corrected else None,
            applied=applied,
            current=False,
            level=source_level(item),
            recorded_at=item.get("recorded_at")
            or item.get("valid_from")
            or item.get("updated_at"),
            superseded_at=(
                item.get("superseded_at")
                or item.get("corrected_at")
                or item.get("valid_to")
            )
            if applied
            else None,
            corrected=corrected,
            confirmations=_count(item.get("confirmations")),
            last_seen=item.get("last_seen"),
        )
        if corrected:
            row["corrected_at"] = item.get("corrected_at") or item.get("superseded_at")
        if item.get("backfilled"):
            row["backfilled"] = True
        if not applied:
            row["offered_at"] = item.get("updated_at")
        rows.append(row)
    return rows


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def entity_attribute_timeline(
    record: Mapping[str, Any], attribute: str
) -> List[Dict[str, Any]]:
    """Every value an entity has held for ``attribute``, current first.

    :func:`attribute_timeline` of the live attribute, followed by the values
    a consolidation pass retired (they sit in the record's metadata): each of
    those ends, on both clocks, when it was retired. Empty when the entity
    never had the attribute.
    """
    wanted = str(attribute or "").strip().casefold()
    rows: List[Dict[str, Any]] = []
    for item in (record or {}).get("attributes") or []:
        if isinstance(item, dict) and str(item.get("name") or "").casefold() == wanted:
            rows.extend(attribute_timeline(item))
            break
    metadata = (record or {}).get("metadata")
    retired = (metadata or {}).get("retired_attributes") if metadata else None
    for item in reversed(list(retired or [])):
        if (
            not isinstance(item, dict)
            or str(item.get("name") or "").casefold() != wanted
        ):
            continue
        ended = item.get("retired_at")
        for index, row in enumerate(attribute_timeline(item)):
            if index == 0:
                row.update(
                    current=False,
                    valid_to=ended,
                    superseded_at=ended,
                    retired=True,
                    retired_by=item.get("retired_by"),
                )
            rows.append(row)
    return rows


def value_as_of(
    rows: Sequence[Mapping[str, Any]],
    when: Any = None,
    *,
    known_at: Any = None,
) -> Optional[Dict[str, Any]]:
    """The timeline row that answers a point-in-time question, or None.

    ``rows`` is :func:`attribute_timeline` or :func:`entity_attribute_timeline`.

    ``when`` asks by valid time: the value that was true in the world then,
    as the system understands it today. A corrected value was never true, so
    it is not returned. ``known_at`` asks by recorded time: only what the
    system had learned by then counts, and a value corrected or replaced
    after ``known_at`` is still what it believed. Give both for "what did the
    system believe on ``known_at`` about ``when``"; give only ``known_at``
    for the value it believed then. A weaker write that was never applied is
    never returned.
    """
    at = _moment(when)
    known = _moment(known_at)
    if at is None and known is None:
        return next((dict(row) for row in rows if row.get("current")), None)
    best: Optional[Tuple[float, float, Dict[str, Any]]] = None
    for row in rows:
        if not row.get("applied"):
            continue
        recorded = _moment(row.get("recorded_at")) or 0.0
        superseded = _moment(row.get("superseded_at"))
        if known is not None:
            if recorded > known:
                continue
            still_believed = superseded is None or superseded > known
        else:
            still_believed = superseded is None
        corrected = bool(row.get("corrected")) and not still_believed
        if corrected:
            continue
        started = _moment(row.get("valid_from")) or recorded
        ended = None if still_believed else _moment(row.get("valid_to"))
        if at is None:
            # What the system believed at ``known_at``.
            if not still_believed:
                continue
        elif started > at or (ended is not None and at >= ended):
            continue
        rank = (started, recorded)
        if best is None or rank > best[:2]:
            best = (started, recorded, dict(row))
    return best[2] if best else None


def entity_aliases(record: Mapping[str, Any]) -> List[str]:
    """Other names this entity answers to: the names of records merged into
    it, and any an operator added. Stored in the record's metadata."""
    metadata = record.get("metadata") if isinstance(record, Mapping) else None
    raw = (metadata or {}).get("aliases") if isinstance(metadata, Mapping) else None
    seen: List[str] = []
    for item in raw or []:
        text = " ".join(str(item or "").split())
        if text and text.casefold() not in {name.casefold() for name in seen}:
            seen.append(text)
    return seen


def _estimate_tokens(text: str) -> int:
    return -(-len(text or "") // _CHARS_PER_TOKEN)


def _is_length_error(exc: BaseException) -> bool:
    """Whether an embedding error says the input was too long."""
    code = str(getattr(exc, "code", "") or "")
    if code in {"context_length_exceeded", "string_above_max_length"}:
        return True
    return bool(_LENGTH_ERROR_RE.search(str(exc) or ""))


def _active_embedding_model() -> Tuple[str, str]:
    """The global embedder's provider and model, without creating one."""
    from .... import embeddings as embeddings_module

    manager = getattr(embeddings_module, "_global_embedding_manager", None)
    if manager is not None:
        provider = str(getattr(getattr(manager, "provider_type", None), "value", ""))
        model = str((getattr(manager, "config", None) or {}).get("model") or "")
        if not model:
            try:
                model = str(manager.get_default_model() or "")
            except Exception:
                model = ""
        return provider, model
    try:
        provider, config = embeddings_module.embedding_settings_from_env()
    except Exception:
        return "", ""
    return str(provider or ""), str((config or {}).get("model") or "")


def embedding_input_tokens() -> int:
    """The input limit, in tokens, of the model that embeds entities.

    ``MEMORIZZ_ENTITY_EMBEDDING_MAX_TOKENS`` overrides it; otherwise it comes
    from the configured embedding model, and 8,191 (OpenAI's) when unknown.
    """
    raw = str(os.environ.get("MEMORIZZ_ENTITY_EMBEDDING_MAX_TOKENS") or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    provider, model = _active_embedding_model()
    name = model.casefold()
    table = _MODEL_INPUT_TOKENS
    if provider.casefold() == "ollama":
        table = _OLLAMA_MODEL_INPUT_TOKENS + table
    for prefix, tokens in table:
        if name.startswith(prefix):
            return tokens
    return _PROVIDER_INPUT_TOKENS.get(provider.casefold(), _DEFAULT_INPUT_TOKENS)


# -- does a scope hold any entity? -------------------------------------------

#: How long a scope's "no entities" answer is trusted. A write through
#: EntityMemory in this process clears it at once; this bounds how long a
#: write from another process (the UI, the MCP server) goes unseen.
EMPTY_SCOPE_TTL_SECONDS = 30.0
#: How long a scope's "has entities" answer is trusted.
FOUND_SCOPE_TTL_SECONDS = 300.0
_PRESENCE_PROBE_ROWS = 20
# provider -> {(memory_id, user_id): (found, when)}
_PRESENCE: "weakref.WeakKeyDictionary[Any, Dict[Tuple[Any, Any], Tuple[bool, float]]]" = (
    weakref.WeakKeyDictionary()
)
_PRESENCE_LOCK = threading.Lock()


def _presence_get(provider: Any, scope: Tuple[Any, Any]) -> Optional[bool]:
    with _PRESENCE_LOCK:
        try:
            entry = (_PRESENCE.get(provider) or {}).get(scope)
        except TypeError:  # a provider that cannot be weakly referenced
            return None
    if entry is None:
        return None
    found, at = entry
    ttl = FOUND_SCOPE_TTL_SECONDS if found else EMPTY_SCOPE_TTL_SECONDS
    return found if time.monotonic() - at < ttl else None


def _presence_set(provider: Any, scope: Tuple[Any, Any], found: bool) -> None:
    with _PRESENCE_LOCK:
        try:
            _PRESENCE.setdefault(provider, {})[scope] = (bool(found), time.monotonic())
        except TypeError:
            pass


_LISTENERS_ATTRIBUTE = "_memorizz_supersession_listeners"


def add_supersession_listener(provider: Any, listener: Any) -> None:
    """Call ``listener(record, changes)`` whenever an entity in ``provider``
    has a current value replaced by a different one.

    ``changes`` is a list of ``{"attribute", "old_value", "new_value",
    "at"}``. Every :class:`EntityMemory` on the provider reports to it,
    whoever made the write. A listener that returns False is dropped, and one
    that raises is logged and kept: a listener never fails a write.
    """
    listeners = getattr(provider, _LISTENERS_ATTRIBUTE, None)
    if not isinstance(listeners, list):
        listeners = []
        try:
            setattr(provider, _LISTENERS_ATTRIBUTE, listeners)
        except Exception:
            return
    if listener not in listeners:
        listeners.append(listener)


def superseded_values(
    before: Optional[Mapping[str, Any]], after: Mapping[str, Any], at: str
) -> List[Dict[str, Any]]:
    """The attributes whose current value differs between two versions of a
    record (one that is gone from ``after`` is not a supersession)."""

    def _values(record: Optional[Mapping[str, Any]]) -> Dict[str, str]:
        return {
            str(item.get("name") or "").casefold(): str(item.get("value") or "")
            for item in (record or {}).get("attributes") or []
            if isinstance(item, dict) and item.get("name")
        }

    old, new = _values(before), _values(after)
    names = {
        str(item.get("name") or "").casefold(): str(item.get("name"))
        for item in after.get("attributes") or []
        if isinstance(item, dict) and item.get("name")
    }
    changes = []
    for key, value in new.items():
        earlier = old.get(key)
        if earlier and " ".join(earlier.casefold().split()) != " ".join(
            value.casefold().split()
        ):
            changes.append(
                {
                    "attribute": names[key],
                    "old_value": earlier,
                    "new_value": value,
                    "at": at,
                }
            )
    return changes


class EntityAttributeInput(BaseModel):
    """Model-facing contract for a single entity-memory fact."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        validation_alias=AliasChoices("name", "attribute", "attribute_name", "key"),
        description="Stable attribute name, for example role or preferred_language.",
    )
    value: str = Field(description="The factual value to store for this attribute.")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence from 0.0 to 1.0.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Optional non-secret provenance for the fact.",
    )


class EntityRelationInput(BaseModel):
    """Model-facing contract for a relation between two entities."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(description="Target entity identifier.")
    relation_type: str = Field(description="Relationship label, for example works_on.")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityAttribute(BaseModel):
    """Represents a single attribute associated with an entity."""

    name: str = Field(
        validation_alias=AliasChoices("name", "attribute", "attribute_name", "key")
    )
    value: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: Optional[str] = None
    #: The conversation that taught this value, so forgetting that
    #: conversation (:meth:`EntityMemory.forget_thread`) takes it out.
    thread_id: Optional[str] = None
    #: Set or confirmed by the user. A write that is not pinned (the chat
    #: model, entity capture, memory routing) never replaces a pinned value.
    pinned: bool = False
    #: What kind of memory capture took it for: ``preference``, ``fact`` or
    #: ``event``. None when it was not said.
    kind: Optional[str] = None
    #: How much the write is trusted: 3 host or pinned, 2 stated by the user,
    #: 1 inferred, from a tool or by consolidation. See :func:`source_level`.
    level: Optional[int] = None
    #: Set only by a host that knows when the value became true; otherwise
    #: ``created_at`` (when it became the current value here) is used.
    #: ``created_at`` is also the recorded time: when the system learned it.
    valid_from: Optional[str] = None
    #: On a write only, never stored: this value replaces one that was wrong
    #: rather than one that stopped being true. The replaced value is kept
    #: in the history as ``corrected`` and the new one takes its valid-from.
    correction: bool = False
    #: How often this value was stated again after it was first stored, when
    #: that last happened, and the newest few (``source``, ``thread_id``,
    #: ``level``, ``at``). Stating a value again never changes it.
    confirmations: Optional[int] = None
    last_seen: Optional[str] = None
    confirmed_by: List[Dict[str, Any]] = Field(default_factory=list)
    #: Earlier values of this attribute, newest first, each with its
    #: ``value``, ``confidence``, ``source``, ``thread_id``, ``updated_at``,
    #: ``valid_from`` and ``valid_to``. A weaker write that disagreed with
    #: the current value is kept here with ``applied: False``. Prompts and
    #: vectors never read it.
    history: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class EntityRelation(BaseModel):
    """Represents a labeled relationship between two entities."""

    entity_id: str
    relation_type: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class EntityMemoryRecord(BaseModel):
    """Structured entity record persisted in the memory provider."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str
    name: Optional[str] = None
    entity_type: Optional[str] = None
    attributes: List[EntityAttribute] = Field(default_factory=list)
    relations: List[EntityRelation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_id: Optional[str] = None
    user_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: str
    updated_at: str


class EntityMemory:
    """High-level helper for storing and retrieving structured entity facts."""

    def __init__(
        self,
        memory_provider: MemoryProvider,
        *,
        max_embedding_tokens: Optional[int] = None,
        attribute_history_limit: Optional[int] = None,
    ):
        """``max_embedding_tokens`` caps the text an entity's vector is made
        from; by default it follows the embedding model's input limit (see
        :func:`embedding_input_tokens`). ``attribute_history_limit`` is how
        many replaced values each attribute keeps (20 by default)."""
        if memory_provider is None:
            raise ValueError("EntityMemory requires a MemoryProvider instance")
        self.memory_provider = memory_provider
        self.attribute_history_limit = max(
            0,
            int(
                ATTRIBUTE_HISTORY_LIMIT
                if attribute_history_limit is None
                else attribute_history_limit
            ),
        )
        self._max_embedding_tokens = (
            int(max_embedding_tokens) if max_embedding_tokens else None
        )

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def upsert_entity(
        self,
        *,
        entity_id: Optional[str] = None,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        attributes: Optional[Sequence[Union[EntityAttribute, Dict[str, Any]]]] = None,
        relations: Optional[Sequence[Union[EntityRelation, Dict[str, Any]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        identity_key: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        source_level: Optional[int] = None,
    ) -> str:
        """
        Create or update an entity record.

        ``source_level`` says how much this write is trusted (3 host, 2 stated
        by the user, 1 inferred, tool or consolidation). A direct call is host
        code, so it defaults to 3; the agent's own write paths pass theirs.

        Returns the entity_id used to store the record (generated when absent).
        :meth:`upsert_entity_with_report` also says how much of the record its
        vector covers.
        """
        return self.upsert_entity_with_report(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes,
            relations=relations,
            metadata=metadata,
            identity_key=identity_key,
            memory_id=memory_id,
            user_id=user_id,
            source_level=source_level,
        )["entity_id"]

    def upsert_entity_with_report(
        self,
        *,
        entity_id: Optional[str] = None,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        attributes: Optional[Sequence[Union[EntityAttribute, Dict[str, Any]]]] = None,
        relations: Optional[Sequence[Union[EntityRelation, Dict[str, Any]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        identity_key: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        source_level: Optional[int] = None,
    ) -> Dict[str, Any]:
        """:meth:`upsert_entity`, returning ``entity_id``, ``unchanged`` (no new
        information, nothing written) and ``embedding``: the vector's
        ``status`` (``full``, ``partial``, ``missing`` or ``provider`` when the
        store embeds), its ``reason`` code and how many attributes it covers.

        An entity whose text is past the embedding model's input limit gets a
        vector of its name, type and newest attributes; the record notes it
        in ``metadata.embedding_status``. See :func:`embedding_input_tokens`.

        A replaced attribute value moves to that attribute's ``history``. A
        pinned attribute (``pinned=True``, set by the user) is only replaced
        by another pinned write; ``kept_pinned`` names the attributes a write
        left as they were for that reason.

        A write also replaces the current value only when its source level
        is the same or higher (:func:`source_level`). A weaker write that
        disagrees is kept in the attribute's history with ``applied: False``
        and named in ``kept_stronger``; the current value stays.
        """
        now = self._timestamp()
        normalized_identity = self._normalize_identity_key(identity_key)
        supplied_metadata = dict(metadata or {})
        if "identity_key" in supplied_metadata:
            # Only the host's ``identity_key`` argument binds a canonical
            # identity. A model- or caller-supplied value in metadata is
            # stored verbatim otherwise, and one invalid key used to break
            # every scoped read for the tenant.
            supplied_metadata.pop("identity_key")
            logger.info(
                "Ignoring caller-supplied metadata.identity_key; pass identity_key "
                "as an argument instead"
            )
        if normalized_identity:
            supplied_metadata["identity_key"] = normalized_identity
        # Without a canonical identity or an explicit entity_id, the record is
        # keyed by its normalised (scope, type, name) so racing first writes
        # and name variants converge on one row.
        name_key = (
            self._name_identity_key(name, entity_type)
            if not entity_id and not normalized_identity
            else ""
        )
        name_bound_id = (
            self._deterministic_name_id(name_key, memory_id=memory_id, user_id=user_id)
            if name_key
            else ""
        )

        existing = None
        # A canonical identity is host/application authority. When both an
        # identity key and a stale/model-supplied entity_id are present, reuse
        # the canonical identity rather than reviving a duplicate record.
        if normalized_identity:
            existing = self._find_by_identity_key(
                normalized_identity, memory_id=memory_id, user_id=user_id
            )
        if not existing and entity_id:
            existing = self._fetch_one(
                {"entity_id": entity_id}, memory_id=memory_id, user_id=user_id
            )
        if not existing and name:
            existing = self._fetch_one(
                {"name": name}, memory_id=memory_id, user_id=user_id
            )
        if not existing and name_bound_id:
            # A case or spacing variant of a name this scope already holds
            # merges into that record instead of overwriting its row.
            existing = self._fetch_one(
                {"entity_id": name_bound_id}, memory_id=memory_id, user_id=user_id
            )
        if not existing and name and not entity_id and not normalized_identity:
            # A name a merged record used to carry ("CFU") finds the record
            # it was merged into, after the identity key and the exact name.
            existing = self.get_entity_by_alias(
                name, memory_id=memory_id, user_id=user_id
            )
            if existing:
                # The alias must not rename the record it resolved to.
                name = None
        if existing and name and existing.get("name"):
            # Another spelling of the name the record has ("counter fraud
            # unit" for "Counter Fraud Unit") is the same name, not a rename.
            def _spelling(value: Any) -> str:
                return " ".join(str(value).split()).casefold()

            if _spelling(name) == _spelling(existing["name"]):
                name = None
        if existing and entity_type and existing.get("entity_type"):
            # Likewise "organization" for "organisation" is not a new type.
            def _type_spelling(value: Any) -> str:
                return re.sub(
                    r"is(ation|e|ed|ing)\b", r"iz\1", str(value).casefold().strip()
                )

            if _type_spelling(entity_type) == _type_spelling(existing["entity_type"]):
                entity_type = None
        if existing:
            entity_id = str(existing.get("entity_id") or entity_id or uuid.uuid4())
        elif normalized_identity:
            entity_id = self._deterministic_entity_id(
                normalized_identity, memory_id=memory_id, user_id=user_id
            )
        elif name_bound_id:
            entity_id = name_bound_id
        else:
            entity_id = entity_id or str(uuid.uuid4())

        kept_pinned: List[str] = []
        kept_stronger: List[str] = []
        record = self._merge_record(
            existing=existing,
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes,
            relations=relations,
            metadata=supplied_metadata,
            memory_id=memory_id,
            user_id=user_id,
            timestamp=now,
            kept_pinned=kept_pinned,
            kept_stronger=kept_stronger,
            write_level=source_level,
        )

        # Write-path dedup (NOOP guard): when the merge produced no new
        # information — the agent re-asserted facts already stored verbatim —
        # skip the re-embed (an embedding API call) and the re-store
        # entirely. Timestamp-only churn does not count as new information.
        if existing and self._stable_snapshot(record) == self._stable_snapshot(
            existing
        ):
            _presence_set(self.memory_provider, (memory_id, user_id), True)
            return {
                "entity_id": record["entity_id"],
                "unchanged": True,
                "embedding": self.embedding_report(existing),
                "kept_pinned": kept_pinned,
                "kept_stronger": kept_stronger,
            }

        # A change the vector does not read (provenance, history, confidence,
        # pinning) keeps the stored vector: no embedding call. Otherwise embed;
        # without an embedder the fact is still worth keeping, and lookups by
        # name and the exact scoped fallback still find it.
        report = self._reuse_embedding(existing, record)
        if report is None:
            report = self._embed_record(record)

        if existing and existing.get("_id") is not None:
            record["_id"] = existing["_id"]
        elif normalized_identity:
            # Canonical identities need a deterministic storage key as well as
            # a deterministic logical entity_id. Without this, two concurrent
            # first writes can both miss the lookup and MongoDB can insert two
            # documents with different generated _id values. Filesystem uses
            # this as its document path and Oracle accepts the UUID as its row
            # id, so retries become idempotent across all built-in providers.
            record["_id"] = record["entity_id"]
        elif not existing and name_bound_id:
            # Name-only first writes were check-then-insert: two concurrent
            # writers both missed the lookup and inserted two rows. The same
            # deterministic storage key makes them converge on one row;
            # ``consolidate_duplicate_entities`` still merges legacy pairs.
            record["_id"] = record["entity_id"]
        self.memory_provider.store(
            data=record,
            memory_store_type=MemoryType.ENTITY_MEMORY,
        )
        _presence_set(self.memory_provider, (memory_id, user_id), True)
        superseded = superseded_values(existing, record, now)
        self.notify_superseded(record, superseded)
        # Mongo returns _id for updates; we prefer our stable entity_id
        return {
            "entity_id": record["entity_id"],
            "unchanged": False,
            "embedding": report,
            "kept_pinned": kept_pinned,
            "kept_stronger": kept_stronger,
            "superseded": superseded,
        }

    def notify_superseded(
        self, record: Mapping[str, Any], changes: Sequence[Mapping[str, Any]]
    ) -> None:
        """Tell the provider's supersession listeners that ``record`` had
        current values replaced (see :func:`add_supersession_listener`)."""
        listeners = getattr(self.memory_provider, _LISTENERS_ATTRIBUTE, None)
        if not changes or not isinstance(listeners, list):
            return
        for listener in list(listeners):
            try:
                if listener(dict(record), [dict(item) for item in changes]) is False:
                    listeners.remove(listener)
            except Exception as exc:
                logger.debug("Supersession listener failed: %s", exc)

    def reinforce_attribute(
        self,
        attribute: str,
        *,
        entity_id: Optional[str] = None,
        name: Optional[str] = None,
        value: Optional[str] = None,
        source: Optional[str] = None,
        thread_id: Optional[str] = None,
        source_level: Optional[int] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Record that an entity's current value was stated again.

        Adds one to the attribute's ``confirmations``, sets ``last_seen`` and
        notes who said it (``source``, ``thread_id``, ``source_level``). The
        value, its source level and its valid-from stay as they are, and the
        stored vector is kept. With ``value``, nothing is counted unless
        that is still the current value. A ``source`` that already confirmed
        the value is not counted twice. Returns whether a confirmation was
        recorded.
        """
        record = self._resolve(entity_id, name, memory_id=memory_id, user_id=user_id)
        if not record:
            return False
        wanted = str(attribute or "").strip().casefold()
        now = self._timestamp()
        changed = False
        attributes: List[Any] = []
        for item in record.get("attributes") or []:
            if (
                not isinstance(item, dict)
                or str(item.get("name") or "").casefold() != wanted
                or changed
            ):
                attributes.append(item)
                continue
            if value is not None and not self._same_value(item, {"value": value}):
                attributes.append(item)
                continue
            if source and any(
                isinstance(seen, dict) and seen.get("source") == source
                for seen in item.get("confirmed_by") or []
            ):
                attributes.append(item)
                continue
            offered = {"source": source, "thread_id": thread_id}
            if source_level in SOURCE_LEVEL_LABELS:
                offered["level"] = int(source_level)
            attributes.append(self._reinforced(item, offered, now, always=True))
            changed = True
        if changed:
            self._store_rewritten(
                {**record, "attributes": attributes, "updated_at": now},
                previous=record,
            )
        return changed

    def withdraw_confirmations(
        self,
        source: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> int:
        """Take back the confirmations one ``source`` recorded, in this
        memory and user. Returns how many were removed."""
        removed = 0
        for record in self.list_entities(
            memory_id=memory_id, user_id=user_id, include_superseded=True
        ):
            attributes, count = self._without_confirmations(
                record.get("attributes") or [],
                lambda seen: str(seen.get("source") or "") == source,
            )
            if count:
                removed += count
                self._store_rewritten(
                    {
                        **record,
                        "attributes": attributes,
                        "updated_at": self._timestamp(),
                    },
                    previous=record,
                )
        return removed

    @staticmethod
    def _without_confirmations(
        attributes: Iterable[Any], matches: Any
    ) -> Tuple[List[Any], int]:
        """``attributes`` without the confirmations ``matches`` accepts, and
        how many that was."""
        kept: List[Any] = []
        removed = 0
        for item in attributes:
            confirmed = item.get("confirmed_by") if isinstance(item, dict) else None
            if not confirmed:
                kept.append(item)
                continue
            staying = [
                seen
                for seen in confirmed
                if not (isinstance(seen, dict) and matches(seen))
            ]
            gone = len(confirmed) - len(staying)
            if not gone:
                kept.append(item)
                continue
            removed += gone
            updated = dict(item)
            updated["confirmations"] = max(0, _count(item.get("confirmations")) - gone)
            if staying:
                updated["confirmed_by"] = staying
                updated["last_seen"] = staying[0].get("at")
            else:
                updated.pop("confirmed_by", None)
            if not updated["confirmations"]:
                for field in ("confirmations", "last_seen", "confirmed_by"):
                    updated.pop(field, None)
            kept.append(updated)
        return kept, removed

    def value_as_of(
        self,
        attribute: str,
        when: Any = None,
        *,
        entity_id: Optional[str] = None,
        name: Optional[str] = None,
        known_at: Any = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """An entity's value for ``attribute`` at a point in time, as a
        timeline row (``value``, ``valid_from``, ``valid_to``,
        ``recorded_at``, ``level``, ...), or None.

        ``when`` is the date in the world ("the limit in September");
        ``known_at`` is the date of the system's knowledge ("what did we
        believe in October"). See :func:`value_as_of`. The entity is found
        by ``entity_id``, or by ``name`` or an alias.
        """
        record = self._resolve(entity_id, name, memory_id=memory_id, user_id=user_id)
        if not record:
            return None
        return value_as_of(
            entity_attribute_timeline(record, attribute), when, known_at=known_at
        )

    def _resolve(
        self,
        entity_id: Optional[str],
        name: Optional[str],
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if entity_id:
            return self._fetch_one_in_scope(
                {"entity_id": entity_id}, memory_id=memory_id, user_id=user_id
            )
        if name:
            return self.get_entity_by_name(
                name, memory_id=memory_id, user_id=user_id
            ) or self.get_entity_by_alias(name, memory_id=memory_id, user_id=user_id)
        return None

    def record_attribute(
        self,
        *,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        attribute_name: str,
        attribute_value: str,
        confidence: float = 0.85,
        source: Optional[str] = None,
        entity_type: Optional[str] = None,
        identity_key: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Convenience helper to upsert a single attribute for an entity."""
        timestamp = self._timestamp()
        attribute = EntityAttribute(
            name=attribute_name,
            value=attribute_value,
            confidence=confidence,
            source=source,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return self.upsert_entity(
            entity_id=entity_id,
            name=entity_name,
            entity_type=entity_type,
            identity_key=identity_key,
            attributes=[attribute],
            memory_id=memory_id,
            user_id=user_id,
        )

    def get_entity(
        self,
        entity_id: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the stored record for a specific entity_id.

        ``metadata.embedding_status`` is ``"partial"`` when the record's vector
        covers only its name, type and newest attributes (its text was past
        the embedding model's input limit) and ``"missing"`` when it has no
        vector; ``metadata.embedding_reason`` says why. Neither is there when
        the vector covers the whole record. :meth:`embedding_report` reads them.
        """
        return self._fetch_one(
            {"entity_id": entity_id}, memory_id=memory_id, user_id=user_id
        )

    def has_entities(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Whether this memory/user scope holds any live entity.

        One bounded query, remembered per scope for a short while: a "no"
        for :data:`EMPTY_SCOPE_TTL_SECONDS`, a "yes" for
        :data:`FOUND_SCOPE_TTL_SECONDS`. A write through ``EntityMemory`` in
        this process turns the answer to "yes" at once. When the store cannot
        tell, the answer is "yes".
        """
        scope = (memory_id, user_id)
        cached = _presence_get(self.memory_provider, scope)
        if cached is not None:
            return cached
        try:
            rows = self._ensure_list(
                self.memory_provider.retrieve_by_query(
                    {},
                    memory_type=MemoryType.ENTITY_MEMORY,
                    limit=_PRESENCE_PROBE_ROWS,
                    memory_id=memory_id,
                    user_id=user_id,
                )
            )
        except Exception as exc:
            logger.debug("Entity presence check failed (%s)", type(exc).__name__)
            return True
        found = (
            any(
                self._record_in_scope(row, memory_id=memory_id, user_id=user_id)
                and not self._is_superseded(row)
                for row in rows
            )
            or len(rows) >= _PRESENCE_PROBE_ROWS
        )  # a full page of others: unsure
        _presence_set(self.memory_provider, scope, found)
        return found

    def get_by_identity_key(
        self,
        identity_key: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """The record a host identity key names (for example the user's own
        profile), found by its deterministic id first: one keyed read, no
        embedding call and no vector search."""
        key = self._normalize_identity_key(identity_key)
        if not key:
            return None
        record = self.get_entity(
            self._deterministic_entity_id(key, memory_id=memory_id, user_id=user_id),
            memory_id=memory_id,
            user_id=user_id,
        )
        if record and self._stored_identity_key(record.get("metadata")) == key:
            return record
        # Records keyed before ids were derived from the identity.
        return self._find_by_identity_key(key, memory_id=memory_id, user_id=user_id)

    def get_entity_by_name(
        self,
        name: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the stored record matching a given name."""
        return self._fetch_one({"name": name}, memory_id=memory_id, user_id=user_id)

    def get_entity_by_alias(
        self,
        alias: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """The live record that lists ``alias`` among its other names."""
        wanted = " ".join(str(alias or "").casefold().split())
        if not wanted:
            return None
        for record in self.list_entities(memory_id=memory_id, user_id=user_id):
            if wanted in {name.casefold() for name in entity_aliases(record)}:
                return record
        return None

    def set_aliases(
        self,
        entity_id: str,
        aliases: Sequence[str],
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """Replace an entity's alias list (an operator's edit). Returns the
        list as stored: trimmed, without repeats or the entity's own name."""
        record = self._fetch_one(
            {"entity_id": entity_id}, memory_id=memory_id, user_id=user_id
        )
        if not record:
            raise ValueError("entity not found")
        own = " ".join(str(record.get("name") or "").casefold().split())
        cleaned = [
            name
            for name in entity_aliases({"metadata": {"aliases": list(aliases or [])}})
            if name.casefold() != own
        ][:50]
        metadata = dict(record.get("metadata") or {})
        if cleaned:
            metadata["aliases"] = cleaned
        else:
            metadata.pop("aliases", None)
        self._store_rewritten(
            {**record, "metadata": metadata, "updated_at": self._timestamp()}
        )
        return cleaned

    def list_entities(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_superseded: bool = False,
    ) -> List[Dict[str, Any]]:
        """List entities in exactly one memory/user scope.

        ``user_id=None`` is the anonymous/legacy scope. Authenticated callers
        never inherit those rows implicitly; legacy adoption is an explicit
        administrative migration so request-serving reads stay fail-closed.
        """
        try:
            rows = self.memory_provider.list_all(
                MemoryType.ENTITY_MEMORY, user_id=user_id
            )
        except TypeError:
            # Backward compatibility for third-party providers written before
            # list_all accepted a user_id keyword. Isolation is still enforced
            # below against every returned document.
            rows = self.memory_provider.list_all(MemoryType.ENTITY_MEMORY)

        filtered: List[Dict[str, Any]] = []
        seen = set()
        for entity in self._ensure_list(rows):
            if not self._record_in_scope(
                entity,
                memory_id=memory_id,
                user_id=user_id,
            ):
                continue
            if not include_superseded and self._is_superseded(entity):
                continue
            key = str(
                entity.get("entity_id")
                or entity.get("_id")
                or self._stable_snapshot(entity)
            )
            if key in seen:
                continue
            seen.add(key)
            filtered.append(entity)
        return filtered

    def consolidate_duplicate_entities(
        self,
        *,
        memory_id: str,
        user_id: Optional[str],
        apply: bool = False,
        allow_legacy_scope: bool = False,
        entity_ids: Optional[Sequence[str]] = None,
        canonical_identity_key: Optional[str] = None,
        provenance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plan or safely apply duplicate consolidation inside one exact scope.

        The default is a read-only plan.  ``apply=True`` first writes a merged
        canonical record and then marks duplicate rows as ``superseded_by``;
        it never deletes data.  Superseded rows stay recoverable for audit and
        rollback but are excluded from normal lookup and prompt retrieval.
        Anonymous legacy scope requires an additional explicit opt-in.
        ``provenance`` (a consolidation plan id) is recorded on each
        superseded row so :meth:`restore_consolidation` can undo the merge.
        """
        resolved_memory_id = str(memory_id or "").strip()
        if not resolved_memory_id:
            raise ValueError("memory_id is required for entity consolidation")
        if user_id is None and not allow_legacy_scope:
            raise ValueError(
                "user_id is required; set allow_legacy_scope=True to inspect "
                "anonymous legacy rows"
            )

        rows = self.list_entities(
            memory_id=resolved_memory_id,
            user_id=user_id,
            include_superseded=False,
        )
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        requested_ids = {
            str(value).strip() for value in (entity_ids or []) if str(value).strip()
        }
        normalized_identity = self._normalize_identity_key(canonical_identity_key)
        if requested_ids:
            selected = [
                row
                for row in rows
                if str(row.get("entity_id") or "").strip() in requested_ids
            ]
            found_ids = {str(row.get("entity_id") or "").strip() for row in selected}
            missing_ids = sorted(requested_ids - found_ids)
            if missing_ids:
                raise ValueError(
                    "Every entity_id must exist in the exact memory/user scope; "
                    f"missing {len(missing_ids)} requested record(s)"
                )
            if len(selected) < 2:
                raise ValueError(
                    "Explicit consolidation requires at least two entity_ids"
                )
            grouped["explicit_selection"] = selected
        else:
            for row in rows:
                key = self._duplicate_group_key(row)
                if key:
                    grouped.setdefault(key, []).append(row)

        plans: List[Dict[str, Any]] = []
        for group_key, members in sorted(grouped.items()):
            if len(members) < 2:
                continue
            canonical = self._choose_canonical(members)
            duplicate_rows = [row for row in members if row is not canonical]
            merged = self._merge_duplicate_group(canonical, duplicate_rows)
            if normalized_identity:
                merged_metadata = dict(merged.get("metadata") or {})
                merged_metadata["identity_key"] = normalized_identity
                merged["metadata"] = merged_metadata
            canonical_id = str(merged.get("entity_id") or "")
            duplicate_ids = sorted(
                str(row.get("entity_id") or "")
                for row in duplicate_rows
                if row.get("entity_id")
            )
            attribute_names = sorted(
                {
                    str(item.get("name"))
                    for item in merged.get("attributes", [])
                    if item.get("name")
                }
            )
            plan = {
                "group_key": group_key,
                "canonical_entity_id": canonical_id,
                "duplicate_entity_ids": duplicate_ids,
                "record_count": len(members),
                "attribute_names": attribute_names,
                "relation_count": len(merged.get("relations") or []),
                "conflicting_attribute_names": self._attribute_conflicts(members),
                "canonical_identity_bound": bool(normalized_identity),
                "applied": False,
            }
            if apply:
                if provenance:
                    # What the surviving record held before the merge, so
                    # restore_consolidation can put it back exactly.
                    merged_metadata = dict(merged.get("metadata") or {})
                    merged_metadata["merge_undo"] = {
                        **dict(merged_metadata.get("merge_undo") or {}),
                        provenance: {
                            "attributes": list(canonical.get("attributes") or []),
                            "relations": list(canonical.get("relations") or []),
                            "merged_at": merged.get("updated_at"),
                            # The alias list before this merge added to it.
                            "aliases": entity_aliases(canonical),
                        },
                    }
                    merged["metadata"] = merged_metadata
                self._store_consolidated_group(
                    merged,
                    duplicate_rows,
                    memory_id=resolved_memory_id,
                    user_id=user_id,
                    provenance=provenance,
                )
                plan["applied"] = True
            plans.append(plan)

        return {
            "schema_version": 1,
            "dry_run": not apply,
            "memory_id": resolved_memory_id,
            "user_scope_bound": user_id is not None,
            "groups": plans,
            "duplicate_group_count": len(plans),
            "records_to_supersede": sum(
                len(item["duplicate_entity_ids"]) for item in plans
            ),
            "deleted_count": 0,
            "strategy": "merge_then_soft_supersede",
            "explicit_selection": bool(requested_ids),
        }

    def retire_attributes(
        self,
        entity_id: str,
        *,
        memory_id: str,
        user_id: Optional[str],
        retirements: Sequence[Mapping[str, Any]],
        retired_by: str,
        include_pinned: bool = False,
    ) -> int:
        """Take facts out of an entity without losing them.

        Each ``retirements`` item names an attribute (``name``) and may give a
        ``reason`` and the attribute that ``kept`` its place. The attribute
        moves to ``metadata.retired_attributes``, where prompts and searches
        never read it, and :meth:`restore_consolidation` puts it back.
        A pinned attribute (the user's own) stays unless ``include_pinned``,
        which a user's own removal passes. Returns how many attributes were
        retired.
        """
        record = self._fetch_one_in_scope(
            {"entity_id": entity_id}, memory_id=memory_id, user_id=user_id
        )
        if not record:
            raise ValueError("entity not found in the requested scope")
        wanted = {
            str(item.get("name") or "").casefold(): item
            for item in retirements
            if str(item.get("name") or "").strip()
        }
        now = self._timestamp()
        kept: List[Dict[str, Any]] = []
        retired: List[Dict[str, Any]] = []
        for attribute in record.get("attributes") or []:
            item = wanted.get(str((attribute or {}).get("name") or "").casefold())
            if (
                not isinstance(attribute, dict)
                or item is None
                or (attribute.get("pinned") and not include_pinned)
            ):
                kept.append(attribute)
                continue
            retired.append(
                {
                    **attribute,
                    "retired_by": retired_by,
                    "retired_at": now,
                    "retired_reason": str(item.get("reason") or "")[:500],
                    "kept_attribute": item.get("kept"),
                }
            )
        if not retired:
            return 0
        metadata = dict(record.get("metadata") or {})
        metadata["retired_attributes"] = [
            *(metadata.get("retired_attributes") or []),
            *retired,
        ]
        updated = {**record, "attributes": kept, "metadata": metadata}
        updated["updated_at"] = now
        self._store_rewritten(updated)
        return len(retired)

    def forget_attributes(
        self,
        entity_id: str,
        names: Sequence[str],
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> int:
        """Delete attributes for good: their value, history and any retired
        copy. For a user removing a fact about themselves; consolidation uses
        :meth:`retire_attributes`, which can be undone. An entity left with
        nothing is deleted. Returns how many attributes were removed.
        """
        record = self._fetch_one_in_scope(
            {"entity_id": entity_id}, memory_id=memory_id, user_id=user_id
        )
        if not record:
            raise ValueError("entity not found in the requested scope")
        wanted = {str(name or "").strip().casefold() for name in names}
        wanted.discard("")
        attributes = [
            item
            for item in record.get("attributes") or []
            if not (
                isinstance(item, dict)
                and str(item.get("name") or "").casefold() in wanted
            )
        ]
        removed = len(record.get("attributes") or []) - len(attributes)
        metadata = dict(record.get("metadata") or {})
        retired = metadata.get("retired_attributes")
        if isinstance(retired, list):
            metadata["retired_attributes"] = [
                item
                for item in retired
                if not (
                    isinstance(item, dict)
                    and str(item.get("name") or "").casefold() in wanted
                )
            ]
        if not removed and metadata == (record.get("metadata") or {}):
            return 0
        if not attributes and not record.get("relations"):
            storage_id = record.get("_id") or record.get("entity_id")
            self.memory_provider.delete_by_id(str(storage_id), MemoryType.ENTITY_MEMORY)
            return removed
        updated = {
            **record,
            "attributes": attributes,
            "metadata": metadata,
            "updated_at": self._timestamp(),
        }
        self._store_rewritten(updated, previous=record)
        return removed

    def forget_thread(
        self,
        thread_id: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
        sources: Sequence[str] = (),
    ) -> Dict[str, int]:
        """Take out what one conversation taught, keeping what others did.

        An attribute whose value came from ``thread_id`` (its ``thread_id``,
        or a ``source`` listed in ``sources`` for values written before
        attributes carried a thread) goes back to the newest earlier value
        from somewhere else, or goes away when there is none. That thread's
        entries leave every history and the retired facts too, as do
        relations whose ``metadata.thread_id`` names it. An entity left with
        no attributes and no relations is deleted. Pinned values are the
        user's and stay unless the conversation itself pinned them.

        Returns ``{"entities_changed", "entities_deleted", "attributes_removed",
        "attributes_restored"}``.
        """
        thread = str(thread_id or "").strip()
        if not thread:
            raise ValueError("thread_id is required")
        legacy = {str(item) for item in sources or () if str(item or "").strip()}

        def _from_thread(item: Mapping[str, Any]) -> bool:
            return str(item.get("thread_id") or "") == thread or (
                bool(legacy) and str(item.get("source") or "") in legacy
            )

        return self._forget_where(
            _from_thread, relation_thread=thread, memory_id=memory_id, user_id=user_id
        )

    def forget_sources(
        self,
        sources: Sequence[str],
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> Dict[str, int]:
        """Take out what particular sources taught (``source`` on each
        attribute, for example ``"doc_comment:<id>"`` when that comment is
        deleted), the way :meth:`forget_thread` does for a conversation:
        each value goes back to its newest earlier value from elsewhere, or
        goes away. Returns the same counts."""
        wanted = {
            str(item).strip() for item in sources or () if str(item or "").strip()
        }
        if not wanted:
            raise ValueError("at least one source is required")
        return self._forget_where(
            lambda item: str(item.get("source") or "") in wanted,
            relation_thread=None,
            memory_id=memory_id,
            user_id=user_id,
        )

    def _forget_where(
        self,
        _from_thread: Any,
        *,
        relation_thread: Optional[str],
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> Dict[str, int]:
        """Rewrite every record in scope without the attribute values (and
        history entries and retired copies) ``_from_thread`` matches."""
        thread = relation_thread or ""
        counts = {
            "entities_changed": 0,
            "entities_deleted": 0,
            "attributes_removed": 0,
            "attributes_restored": 0,
        }
        for record in self.list_entities(
            memory_id=memory_id, user_id=user_id, include_superseded=True
        ):
            changed = False
            attributes: List[Dict[str, Any]] = []
            for attribute in record.get("attributes") or []:
                if not isinstance(attribute, dict):
                    attributes.append(attribute)
                    continue
                history = [
                    dict(item)
                    for item in attribute.get("history") or []
                    if isinstance(item, dict)
                ]
                remaining = [item for item in history if not _from_thread(item)]
                (attribute,), unconfirmed = self._without_confirmations(
                    [attribute], _from_thread
                )
                changed = changed or bool(unconfirmed)
                if not _from_thread(attribute):
                    if len(remaining) != len(history):
                        attribute = {**attribute, "history": remaining}
                        if not remaining:
                            attribute.pop("history")
                        changed = True
                    attributes.append(attribute)
                    continue
                changed = True
                # Only a value that was once current can come back: a weaker
                # write that was never applied stays in the history.
                earlier = next(
                    (
                        index
                        for index, item in enumerate(remaining)
                        if item.get("applied") is not False
                    ),
                    None,
                )
                if earlier is None:
                    counts["attributes_removed"] += 1
                    continue
                restored = {
                    key: value
                    for key, value in attribute.items()
                    if key
                    not in {*_HISTORY_FIELDS, "history", "valid_from", "confirmed_by"}
                }
                returning = dict(remaining[earlier])
                for key in _HISTORY_ONLY_FIELDS:
                    if key != "valid_from":
                        returning.pop(key, None)
                restored.update(returning)
                restored.setdefault("confidence", 0.8)
                rest = remaining[:earlier] + remaining[earlier + 1 :]
                if rest:
                    restored["history"] = rest
                attributes.append(restored)
                counts["attributes_restored"] += 1
            relations = [
                relation
                for relation in record.get("relations") or []
                if not (
                    thread
                    and isinstance(relation, dict)
                    and str((relation.get("metadata") or {}).get("thread_id") or "")
                    == thread
                )
            ]
            changed = changed or len(relations) != len(record.get("relations") or [])
            metadata = dict(record.get("metadata") or {})
            retired = metadata.get("retired_attributes")
            if isinstance(retired, list):
                kept_retired = [
                    item
                    for item in retired
                    if not (isinstance(item, dict) and _from_thread(item))
                ]
                if len(kept_retired) != len(retired):
                    metadata["retired_attributes"] = kept_retired
                    changed = True
            if not changed:
                continue
            if not attributes and not relations:
                storage_id = record.get("_id") or record.get("entity_id")
                if storage_id is not None and self.memory_provider.delete_by_id(
                    str(storage_id), MemoryType.ENTITY_MEMORY
                ):
                    counts["entities_deleted"] += 1
                    continue
            updated = {
                **record,
                "attributes": attributes,
                "relations": relations,
                "metadata": metadata,
                "updated_at": self._timestamp(),
            }
            self._store_rewritten(updated, previous=record)
            counts["entities_changed"] += 1
        return counts

    def restore_consolidation(
        self, plan_id: str, *, memory_id: str, user_id: Optional[str]
    ) -> Dict[str, int]:
        """Undo what one consolidation plan changed in this memory and user.

        Merged duplicates come back, retired attributes return to their
        entity, and facts the plan added are removed.
        """
        source = f"consolidation:{plan_id}"
        counts = {"entities": 0, "attributes": 0, "facts_removed": 0}
        for record in self.list_entities(
            memory_id=memory_id, user_id=user_id, include_superseded=True
        ):
            metadata = dict(record.get("metadata") or {})
            changed = False
            undo = dict(metadata.get("merge_undo") or {})
            snapshot = undo.pop(plan_id, None)
            if isinstance(snapshot, dict):
                # The record that absorbed duplicates goes back to what it
                # held, plus any fact written since the merge.
                merged_at = str(snapshot.get("merged_at") or "")
                kept_names = {
                    str(item.get("name") or "").casefold()
                    for item in snapshot.get("attributes") or []
                    if isinstance(item, dict)
                }
                later = [
                    item
                    for item in record.get("attributes") or []
                    if isinstance(item, dict)
                    and str(item.get("updated_at") or "") > merged_at
                    and str(item.get("name") or "").casefold() not in kept_names
                ]
                record = {
                    **record,
                    "attributes": [*(snapshot.get("attributes") or []), *later],
                    "relations": list(snapshot.get("relations") or []),
                }
                if undo:
                    metadata["merge_undo"] = undo
                else:
                    metadata.pop("merge_undo", None)
                metadata.pop("merged_entity_ids", None)
                metadata.pop("consolidated_at", None)
                if "aliases" in snapshot:
                    # Take back the aliases this merge added.
                    if snapshot.get("aliases"):
                        metadata["aliases"] = list(snapshot["aliases"])
                    else:
                        metadata.pop("aliases", None)
                changed = True
            if metadata.get("superseded_by_plan") == plan_id:
                for key in ("superseded_by", "superseded_at", "superseded_by_plan"):
                    metadata.pop(key, None)
                counts["entities"] += 1
                changed = True
            attributes = [
                attribute
                for attribute in record.get("attributes") or []
                if not (
                    isinstance(attribute, dict) and attribute.get("source") == source
                )
            ]
            removed = len(record.get("attributes") or []) - len(attributes)
            if removed:
                counts["facts_removed"] += removed
                changed = True
            attributes, unconfirmed = self._without_confirmations(
                attributes, lambda seen: str(seen.get("source") or "") == source
            )
            if unconfirmed:
                counts["confirmations_removed"] = (
                    counts.get("confirmations_removed", 0) + unconfirmed
                )
                changed = True
            retired = list(metadata.get("retired_attributes") or [])
            returning = [
                item
                for item in retired
                if isinstance(item, dict) and item.get("retired_by") == plan_id
            ]
            if returning:
                metadata["retired_attributes"] = [
                    item for item in retired if item not in returning
                ]
                live = {
                    str(attribute.get("name") or "").casefold()
                    for attribute in attributes
                    if isinstance(attribute, dict)
                }
                for item in returning:
                    attribute = {
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "retired_by",
                            "retired_at",
                            "retired_reason",
                            "kept_attribute",
                        }
                    }
                    if str(attribute.get("name") or "").casefold() not in live:
                        attributes.append(attribute)
                        counts["attributes"] += 1
                changed = True
            if changed:
                self._store_rewritten(
                    {
                        **record,
                        "attributes": attributes,
                        "metadata": metadata,
                        "updated_at": self._timestamp(),
                    }
                )
        return counts

    def _store_rewritten(
        self, record: Dict[str, Any], *, previous: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a changed record over its row, with a fresh embedding (or
        ``previous``'s vector when the text it embeds did not change)."""
        payload = dict(record)
        payload.pop("score", None)
        if self._reuse_embedding(previous, payload) is None:
            self._embed_record(payload)
        self.memory_provider.store(
            data=payload, memory_store_type=MemoryType.ENTITY_MEMORY
        )

    def _reuse_embedding(
        self, existing: Optional[Dict[str, Any]], record: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Give ``record`` ``existing``'s vector when the text it embeds is
        unchanged and that vector is complete and the current embedder's size.
        Returns the embedding report, or None when ``record`` needs embedding.
        """
        from ....memory_provider.base import provider_manages_embeddings

        vector = (existing or {}).get("embedding")
        if not vector or provider_manages_embeddings(self.memory_provider):
            return None
        report = self.embedding_report(existing)
        if report.get("status") != "full":
            return None

        def _text(item: Dict[str, Any]) -> str:
            metadata = {
                key: value
                for key, value in (item.get("metadata") or {}).items()
                if key not in _EMBEDDING_STATUS_KEYS
            }
            return self._build_embedding_text({**item, "metadata": metadata})

        if _text(existing) != _text(record):
            return None
        try:
            from ....embeddings import get_embedding_dimensions

            if len(vector) != int(get_embedding_dimensions()):
                return None
        except Exception:
            return None
        record["embedding"] = vector
        return report

    # -- vectors ---------------------------------------------------------------

    def _embedding_budget(self) -> int:
        """Tokens an entity's embedding text may use."""
        limit = self._max_embedding_tokens or embedding_input_tokens()
        return max(16, int(limit * _EMBEDDING_BUDGET_SHARE))

    @staticmethod
    def _bounded_embedding_text(
        record: Dict[str, Any], budget_tokens: int
    ) -> Tuple[str, int]:
        """The entity's name, type and newest attributes, in ``budget_tokens``.

        Attributes go newest first (by ``updated_at``, then confidence), and
        relations follow while room is left. Returns the text and how many
        attributes it holds.
        """
        budget = max(1, int(budget_tokens)) * _CHARS_PER_TOKEN
        lines: List[str] = []
        if record.get("name"):
            lines.append(f"Name: {record['name']}")
        if record.get("entity_type"):
            lines.append(f"Type: {record['entity_type']}")
        used = sum(len(line) + 1 for line in lines)

        def _rank(item: Tuple[int, Dict[str, Any]]) -> Tuple[str, float, int]:
            index, attribute = item
            try:
                confidence = float(attribute.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            when = attribute.get("updated_at") or attribute.get("created_at") or ""
            # Facts written together share a timestamp; the later one is newer.
            return str(when), confidence, index

        attributes = [
            item for item in record.get("attributes") or [] if isinstance(item, dict)
        ]
        covered = 0
        ranked = sorted(enumerate(attributes), key=_rank, reverse=True)
        for _index, attribute in ranked:
            line = f"{attribute.get('name')}: {attribute.get('value')}"
            room = budget - used - 1
            if len(line) > room:
                if covered or room < 32:
                    continue
                line = line[:room]  # one huge fact: keep its start
            lines.append(line)
            used += len(line) + 1
            covered += 1
        for relation in record.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            line = f"Relation ({relation.get('relation_type')})->{relation.get('entity_id')}"
            if used + len(line) + 1 <= budget:
                lines.append(line)
                used += len(line) + 1
        return "\n".join(lines)[:budget], covered

    def _embed_record(
        self, record: Dict[str, Any], *, strict: bool = False
    ) -> Dict[str, Any]:
        """Embed ``record`` in place and note in its metadata what the vector
        leaves out. Returns the report :meth:`upsert_entity_with_report` gives.
        ``strict`` raises an embedding error other than a too-long input.
        """
        from ....memory_provider.base import provider_manages_embeddings

        metadata = {
            key: value
            for key, value in (record.get("metadata") or {}).items()
            if key not in _EMBEDDING_STATUS_KEYS
        }
        total = self._attribute_count(record)
        text = self._build_embedding_text({**record, "metadata": metadata})
        covered: Optional[int] = total
        if provider_manages_embeddings(self.memory_provider):
            status, reason = "provider", None
        elif not text:
            status, reason, covered = "missing", "nothing_to_embed", 0
        else:
            status, reason, covered = self._embed_within_limit(
                record, text, total, strict=strict
            )
        if status in {"partial", "missing"}:
            metadata["embedding_status"] = status
            metadata["embedding_reason"] = reason
            if covered is not None:
                metadata["embedding_attributes_covered"] = covered
        if metadata or "metadata" in record:
            record["metadata"] = metadata
        return self._embedding_summary(status, reason, covered, total)

    def _embed_within_limit(
        self, record: Dict[str, Any], text: str, total: int, *, strict: bool = False
    ) -> Tuple[str, Optional[str], Optional[int]]:
        """Embed ``text``, or a bounded version of it when it is past the
        model's input limit (estimated, or as the provider's length error
        says). A failed embed keeps the previous vector when there is one.
        Returns the status, its reason and how many attributes are covered.
        """
        budget = self._embedding_budget()
        reason: Optional[str] = None
        covered = total
        if _estimate_tokens(text) > budget:
            text, covered = self._bounded_embedding_text(record, budget)
            reason = "over_token_limit"
        for attempt in range(3):
            try:
                record["embedding"] = get_embedding(text)
            except Exception as exc:
                if attempt < 2 and _is_length_error(exc):
                    # The provider counts tokens its own way: halve and retry.
                    budget = max(16, min(budget, _estimate_tokens(text)) // 2)
                    text, covered = self._bounded_embedding_text(record, budget)
                    reason = "provider_length_error"
                    continue
                if strict and not _is_length_error(exc):
                    raise
                kept = record.get("embedding") is not None
                logger.warning(
                    "Storing entity %s %s (%s)",
                    record.get("entity_id"),
                    "with its previous embedding" if kept else "without an embedding",
                    type(exc).__name__,
                )
                if kept:
                    return "partial", "previous_vector_kept", None
                record.pop("embedding", None)
                return (
                    "missing",
                    "over_token_limit" if _is_length_error(exc) else "embedding_failed",
                    0,
                )
            break
        if reason:
            logger.info(
                "Entity %s is past the embedding input limit; its vector covers "
                "its name and %s of %s attributes",
                record.get("entity_id"),
                covered,
                total,
            )
            return "partial", reason, covered
        return "full", None, total

    @staticmethod
    def _attribute_count(record: Dict[str, Any]) -> int:
        return sum(
            1 for item in record.get("attributes") or [] if isinstance(item, dict)
        )

    @staticmethod
    def _embedding_summary(
        status: str, reason: Optional[str], covered: Optional[int], total: int
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "status": status,
            "reason": reason,
            "attributes_total": total,
        }
        if covered is not None:
            summary["attributes_covered"] = covered
        return summary

    @classmethod
    def embedding_report(cls, record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """What a stored record's vector covers, read from its metadata:
        ``status`` is ``partial`` or ``missing`` when ``metadata.embedding_status``
        says so, else ``full``."""
        record = record or {}
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        total = cls._attribute_count(record)
        status = str(metadata.get("embedding_status") or "")
        if status not in {"partial", "missing"}:
            return cls._embedding_summary("full", None, total, total)
        covered = metadata.get("embedding_attributes_covered")
        return cls._embedding_summary(
            status,
            metadata.get("embedding_reason"),
            covered if isinstance(covered, int) else None,
            total,
        )

    def migrate_legacy_scope(self, *, memory_id: str, user_id: str) -> int:
        """Explicitly assign anonymous entity rows to one authenticated user.

        This is an operator-only migration primitive and is never registered as
        an agent tool. Runtime reads remain strictly scoped before and after the
        migration.
        """
        if not str(memory_id or "").strip():
            raise ValueError("memory_id is required for legacy entity migration")
        if not str(user_id or "").strip():
            raise ValueError("A non-empty target user_id is required")
        migrate = getattr(
            self.memory_provider, "migrate_entity_memory_user_scope", None
        )
        if not callable(migrate):
            raise NotImplementedError(
                "This memory provider does not support managed legacy entity migration"
            )
        return int(migrate(memory_id=str(memory_id), user_id=str(user_id)) or 0)

    def search_entities_with_diagnostics(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return scoped entity matches plus content-free retrieval metadata."""
        safe_limit = max(1, min(int(limit or 5), 100))
        diagnostics: Dict[str, Any] = {
            "selector": "query",
            "retrieval_mode": "semantic",
            "semantic_attempted": False,
            "semantic_match_count": 0,
            "fallback_used": False,
            "fallback_candidate_count": 0,
            "legacy_scope_match_count": 0,
            "match_count": 0,
            "degraded": False,
            "degraded_reason": None,
            "scope": {
                "memory_id_bound": memory_id is not None,
                "user_id_bound": user_id is not None,
            },
        }
        if not str(query or "").strip():
            diagnostics.update(
                {
                    "retrieval_mode": "exact_fallback",
                    "fallback_used": True,
                    "degraded": True,
                    "degraded_reason": "empty_query",
                }
            )
            return [], diagnostics

        vector_status = self._provider_vector_search_status()
        if vector_status:
            diagnostics["vector_search"] = vector_status

        semantic_results: List[Dict[str, Any]] = []
        semantic_available = vector_status.get("available") if vector_status else None
        if semantic_available is not False:
            diagnostics["semantic_attempted"] = True
            try:
                raw_results = self.memory_provider.retrieve_by_query(
                    query,
                    memory_type=MemoryType.ENTITY_MEMORY,
                    limit=safe_limit,
                    memory_id=memory_id,
                    user_id=user_id,
                )
                semantic_results = [
                    record
                    for record in self._ensure_list(raw_results)
                    if self._record_in_scope(
                        record,
                        memory_id=memory_id,
                        user_id=user_id,
                    )
                    and not self._is_superseded(record)
                ]
            except Exception as exc:
                diagnostics["semantic_error_type"] = type(exc).__name__
                semantic_error_reason = getattr(exc, "reason", None)
                if semantic_error_reason in {
                    "semantic_embedding_failed",
                    "vector_query_failed",
                    "vector_search_unavailable",
                }:
                    diagnostics["semantic_error_reason"] = semantic_error_reason
                logger.warning(
                    "Entity semantic search failed; using exact scoped fallback (%s)",
                    type(exc).__name__,
                )

        diagnostics["semantic_match_count"] = len(semantic_results)
        if semantic_results:
            _presence_set(self.memory_provider, (memory_id, user_id), True)
            matches = semantic_results[:safe_limit]
            diagnostics["match_count"] = len(matches)
            return matches, diagnostics

        fallback_candidate_limit = min(500, max(50, safe_limit * 20))
        scoped_entities = self._fallback_candidates(
            memory_id=memory_id,
            user_id=user_id,
            limit=fallback_candidate_limit,
        )
        fallback_matches = self._rank_exact_fallback(
            scoped_entities, query=query, limit=safe_limit
        )
        reason = "semantic_empty"
        if semantic_available is False:
            reason = str(vector_status.get("reason") or "vector_search_unavailable")
        elif diagnostics.get("semantic_error_type"):
            reason = str(diagnostics.get("semantic_error_reason") or "semantic_error")

        diagnostics.update(
            {
                "retrieval_mode": "exact_fallback",
                "fallback_used": True,
                "fallback_candidate_count": len(scoped_entities),
                "fallback_candidate_limit": fallback_candidate_limit,
                "fallback_candidate_limit_reached": len(scoped_entities)
                >= fallback_candidate_limit,
                "legacy_scope_match_count": 0,
                "match_count": len(fallback_matches),
                "degraded": semantic_available is False
                or bool(diagnostics.get("semantic_error_type")),
                "degraded_reason": reason,
            }
        )
        logger.info(
            "Entity memory retrieval used exact fallback "
            "(reason=%s, candidates=%s, matches=%s, legacy=%s)",
            reason,
            len(scoped_entities),
            len(fallback_matches),
            0,
        )
        return fallback_matches, diagnostics

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _normalize_identity_key(value: Optional[str]) -> str:
        text = str(value or "").strip().casefold()
        if not text:
            return ""
        normalized = _IDENTITY_KEY_RE.sub("-", text).strip("-._:")
        if not normalized:
            raise ValueError("identity_key must contain a letter or number")
        if len(normalized) > 160:
            raise ValueError("identity_key must be 160 characters or fewer")
        return normalized

    @staticmethod
    def _name_identity_key(name: Optional[str], entity_type: Optional[str]) -> str:
        """The normalised ``name:<type>:<name>`` key a name-only write converges on."""
        normalized_name = " ".join(str(name or "").casefold().split())
        if not normalized_name:
            return ""
        normalized_type = " ".join(str(entity_type or "unknown").casefold().split())
        return f"name:{normalized_type}:{normalized_name}"

    @staticmethod
    def _deterministic_name_id(
        name_key: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> str:
        """A stable id for a name-keyed entity in one scope.

        The scope document uses ``name_key`` rather than ``identity_key`` so it
        can never collide with an identity-bound id.
        """
        scope = json.dumps(
            {"memory_id": memory_id, "name_key": name_key, "user_id": user_id},
            sort_keys=True,
            separators=(",", ":"),
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"memorizz:entity:{scope}"))

    _invalid_identity_key_logged = False

    @classmethod
    def _stored_identity_key(cls, metadata: Any) -> str:
        """The normalised identity key on a stored record, or ``""``.

        An invalid stored value (one an older release accepted verbatim) is
        treated as absent rather than raising out of every read, and is logged
        once per process.
        """
        if not isinstance(metadata, dict):
            return ""
        raw = metadata.get("identity_key")
        if raw is None or raw == "":
            return ""
        try:
            return cls._normalize_identity_key(raw)
        except ValueError as exc:
            if not EntityMemory._invalid_identity_key_logged:
                EntityMemory._invalid_identity_key_logged = True
                logger.warning(
                    "Ignoring an invalid stored entity identity_key (%s); the record "
                    "is treated as unbound. Further occurrences are logged at debug.",
                    exc,
                )
            else:
                logger.debug("Ignoring an invalid stored entity identity_key (%s)", exc)
            return ""

    @staticmethod
    def _deterministic_entity_id(
        identity_key: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> str:
        scope = json.dumps(
            {
                "identity_key": identity_key,
                "memory_id": memory_id,
                "user_id": user_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"memorizz:entity:{scope}"))

    def _find_by_identity_key(
        self,
        identity_key: str,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        for row in self.list_entities(memory_id=memory_id, user_id=user_id):
            if self._stored_identity_key(row.get("metadata")) == identity_key:
                return row
        return None

    @staticmethod
    def _is_superseded(record: Dict[str, Any]) -> bool:
        metadata = record.get("metadata") or {}
        return bool(
            isinstance(metadata, dict)
            and str(metadata.get("superseded_by") or "").strip()
        )

    @classmethod
    def _duplicate_group_key(cls, record: Dict[str, Any]) -> str:
        identity_key = cls._stored_identity_key(record.get("metadata"))
        if identity_key:
            return f"identity:{identity_key}"
        name = " ".join(str(record.get("name") or "").casefold().split())
        entity_type = " ".join(
            str(record.get("entity_type") or "unknown").casefold().split()
        )
        return f"name:{entity_type}:{name}" if name else ""

    @staticmethod
    def _choose_canonical(members: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        def _rank(record: Dict[str, Any]) -> Tuple[int, int, str, str]:
            metadata = record.get("metadata") or {}
            identity_bound = int(
                isinstance(metadata, dict) and bool(metadata.get("identity_key"))
            )
            return (
                identity_bound,
                len(record.get("attributes") or []),
                str(record.get("updated_at") or ""),
                str(record.get("entity_id") or ""),
            )

        return max(members, key=_rank)

    @classmethod
    def _attribute_conflicts(cls, members: Sequence[Dict[str, Any]]) -> List[str]:
        values: Dict[str, set[str]] = {}
        for record in members:
            for item in record.get("attributes") or []:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                name = str(item["name"]).casefold()
                values.setdefault(name, set()).add(str(item.get("value") or ""))
        return sorted(
            name for name, candidates in values.items() if len(candidates) > 1
        )

    @classmethod
    def _merge_duplicate_group(
        cls,
        canonical: Dict[str, Any],
        duplicates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = dict(canonical)
        members = [canonical, *duplicates]
        attributes: Dict[str, Dict[str, Any]] = {}
        for record in members:
            for raw in record.get("attributes") or []:
                if not isinstance(raw, dict) or not raw.get("name"):
                    continue
                item = dict(raw)
                key = str(item["name"]).casefold()
                current = attributes.get(key)
                try:
                    confidence = float(item.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                try:
                    current_confidence = float((current or {}).get("confidence") or 0.0)
                except (TypeError, ValueError):
                    current_confidence = 0.0
                # The stronger source wins first: a merge must not swap a
                # value the host wrote for the copy a conversation left.
                item_rank = (
                    source_level(item),
                    confidence,
                    str(item.get("updated_at") or ""),
                )
                current_rank = (
                    source_level(current) if current else 0,
                    current_confidence,
                    str((current or {}).get("updated_at") or ""),
                )
                if current is None or item_rank > current_rank:
                    attributes[key] = item

        relations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in members:
            for raw in record.get("relations") or []:
                if not isinstance(raw, dict):
                    continue
                target = str(raw.get("entity_id") or "").strip()
                relation_type = str(raw.get("relation_type") or "").strip()
                if target and relation_type:
                    relations[(target, relation_type.casefold())] = dict(raw)

        metadata: Dict[str, Any] = {}
        for record in reversed(members):
            raw_metadata = record.get("metadata") or {}
            if isinstance(raw_metadata, dict):
                metadata.update(
                    {
                        key: value
                        for key, value in raw_metadata.items()
                        if key not in {"superseded_by", "superseded_at"}
                    }
                )
        alias_ids = sorted(
            {
                str(record.get("entity_id"))
                for record in duplicates
                if record.get("entity_id")
            }
        )
        metadata["merged_entity_ids"] = sorted(
            set(metadata.get("merged_entity_ids") or []).union(alias_ids)
        )
        metadata["consolidated_at"] = cls._timestamp()
        # The names of the records merged away become the survivor's aliases,
        # so a later mention of one resolves here.
        own = " ".join(str(canonical.get("name") or "").casefold().split())
        names = entity_aliases(canonical)
        for record in duplicates:
            for name in [record.get("name"), *entity_aliases(record)]:
                text = " ".join(str(name or "").split())
                if (
                    text
                    and text.casefold() != own
                    and text.casefold() not in {item.casefold() for item in names}
                ):
                    names.append(text)
        if names:
            metadata["aliases"] = names
        else:
            metadata.pop("aliases", None)
        merged["attributes"] = list(attributes.values())
        merged["relations"] = list(relations.values())
        merged["metadata"] = metadata
        merged["updated_at"] = cls._timestamp()
        return merged

    def _store_consolidated_group(
        self,
        canonical: Dict[str, Any],
        duplicates: Sequence[Dict[str, Any]],
        *,
        memory_id: str,
        user_id: Optional[str],
        provenance: Optional[str] = None,
    ) -> None:
        if not self._record_in_scope(canonical, memory_id=memory_id, user_id=user_id):
            raise ValueError("Canonical entity escaped the requested scope")
        canonical_payload = dict(canonical)
        # An embedder outage still stops the merge, so duplicates are kept;
        # a merged record past the input limit gets a bounded vector.
        self._embed_record(canonical_payload, strict=True)
        self.memory_provider.store(
            data=canonical_payload,
            memory_store_type=MemoryType.ENTITY_MEMORY,
        )

        canonical_id = str(canonical_payload.get("entity_id") or "")
        verified = self._fetch_one_in_scope(
            {"entity_id": canonical_id},
            memory_id=memory_id,
            user_id=user_id,
            include_superseded=True,
        )
        if not verified:
            raise RuntimeError("Canonical entity verification failed; duplicates kept")

        now = self._timestamp()
        for duplicate in duplicates:
            if not self._record_in_scope(
                duplicate, memory_id=memory_id, user_id=user_id
            ):
                raise ValueError("Duplicate entity escaped the requested scope")
            archived = dict(duplicate)
            metadata = dict(archived.get("metadata") or {})
            metadata.update({"superseded_by": canonical_id, "superseded_at": now})
            if provenance:
                metadata["superseded_by_plan"] = provenance
            archived["metadata"] = metadata
            archived["updated_at"] = now
            self.memory_provider.store(
                data=archived,
                memory_store_type=MemoryType.ENTITY_MEMORY,
            )

    def _fetch_one(
        self,
        query: Dict[str, Any],
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve one record without ever crossing a memory/user boundary."""
        return self._fetch_one_in_scope(query, memory_id=memory_id, user_id=user_id)

    def _fetch_one_in_scope(
        self,
        query: Dict[str, Any],
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
        include_superseded: bool = False,
    ) -> Optional[Dict[str, Any]]:
        scoped_query = dict(query)
        if memory_id is not None:
            scoped_query["memory_id"] = memory_id
        scoped_query["user_id"] = user_id
        try:
            results = self.memory_provider.retrieve_by_query(
                scoped_query,
                memory_type=MemoryType.ENTITY_MEMORY,
                limit=5,
                include_embedding=True,
                memory_id=memory_id,
                user_id=user_id,
            )
        except TypeError:
            results = self.memory_provider.retrieve_by_query(
                scoped_query,
                memory_type=MemoryType.ENTITY_MEMORY,
                limit=5,
                memory_id=memory_id,
            )
        for item in self._ensure_list(results):
            if self._record_in_scope(
                item,
                memory_id=memory_id,
                user_id=user_id,
            ) and (include_superseded or not self._is_superseded(item)):
                return item
        return None

    def _fallback_candidates(
        self,
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Load a bounded candidate window for exact fallback ranking.

        All built-in providers support an empty filter query with a server-side
        limit. Third-party providers retain a compatibility path, followed by
        the same strict client-side scope check.
        """
        safe_limit = max(1, min(int(limit), 500))
        scoped_query = False
        try:
            rows = self.memory_provider.retrieve_by_query(
                {},
                memory_type=MemoryType.ENTITY_MEMORY,
                limit=safe_limit,
                memory_id=memory_id,
                user_id=user_id,
            )
            scoped_query = True
        except TypeError:
            try:
                rows = self.memory_provider.retrieve_by_query(
                    {},
                    memory_store_type=MemoryType.ENTITY_MEMORY,
                    limit=safe_limit,
                    memory_id=memory_id,
                )
            except Exception as exc:
                logger.warning(
                    "Legacy entity fallback query failed; using scoped list (%s)",
                    type(exc).__name__,
                )
                rows = self.list_entities(memory_id=memory_id, user_id=user_id)
        except Exception as exc:
            logger.warning(
                "Bounded entity fallback query failed; using scoped list (%s)",
                type(exc).__name__,
            )
            rows = self.list_entities(memory_id=memory_id, user_id=user_id)

        candidates: List[Dict[str, Any]] = []
        rows = self._ensure_list(rows)
        for row in rows:
            if not self._record_in_scope(row, memory_id=memory_id, user_id=user_id):
                continue
            if self._is_superseded(row):
                continue
            candidates.append(row)
            if len(candidates) >= safe_limit:
                break
        # The per-turn recall already answers has_entities for this scope.
        if candidates:
            _presence_set(self.memory_provider, (memory_id, user_id), True)
        elif scoped_query and not rows:
            _presence_set(self.memory_provider, (memory_id, user_id), False)
        return candidates

    @staticmethod
    def _record_in_scope(
        record: Dict[str, Any],
        *,
        memory_id: Optional[str],
        user_id: Optional[str],
    ) -> bool:
        if not isinstance(record, dict):
            return False
        if memory_id is not None and record.get("memory_id") != memory_id:
            return False
        return record.get("user_id") == user_id

    def _provider_vector_search_status(self) -> Dict[str, Any]:
        status_fn = getattr(self.memory_provider, "get_vector_search_status", None)
        if not callable(status_fn):
            return {}
        try:
            raw = status_fn(MemoryType.ENTITY_MEMORY)
        except Exception as exc:
            return {
                "available": None,
                "reason": "status_check_failed",
                "error_type": type(exc).__name__,
            }
        if not isinstance(raw, dict):
            return {}
        # Provider errors and raw index definitions can contain credentials or
        # topology details. Only persist this bounded diagnostic allowlist.
        return {
            key: raw.get(key)
            for key in ("available", "queryable", "status", "reason")
            if raw.get(key) is not None
        }

    @classmethod
    def _rank_exact_fallback(
        cls,
        records: List[Dict[str, Any]],
        *,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        query_tokens = cls._tokens(query)
        profile_intent = bool(query_tokens.intersection(_PROFILE_QUERY_TERMS))
        ranked = []
        for record in records:
            record_tokens = cls._tokens(cls._build_embedding_text(record))
            overlap = len(query_tokens.intersection(record_tokens))
            name = str(record.get("name") or "").strip().casefold()
            metadata = record.get("metadata") or {}
            explicit_self = bool(
                name in _SELF_ENTITY_ALIASES
                or metadata.get("is_self") is True
                or cls._stored_identity_key(metadata) in _SELF_IDENTITY_KEYS
                or str(metadata.get("subject") or "").casefold()
                in {"current_user", "user", "self"}
            )
            person_profile = bool(
                profile_intent
                and str(record.get("entity_type") or "").casefold()
                in {"person", "user"}
            )
            if not (explicit_self or person_profile or overlap):
                continue
            priority = 2 if explicit_self else (1 if person_profile else 0)
            ranked.append(
                (
                    priority,
                    overlap,
                    str(record.get("updated_at") or ""),
                    record,
                )
            )
        ranked.sort(key=lambda item: item[:3], reverse=True)
        return [item[3] for item in ranked[:limit]]

    @staticmethod
    def _tokens(value: Any) -> set:
        tokens = set(_ENTITY_TOKEN_RE.findall(str(value or "").casefold()))
        normalized = set(tokens)
        for token in tokens:
            if len(token) > 3 and token.endswith("s"):
                normalized.add(token[:-1])
        return normalized

    def _ensure_list(self, result: Any) -> List[Dict[str, Any]]:
        """Normalize Mongo cursors or single dicts into a list."""
        if result is None:
            return []
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        try:
            return list(result)
        except TypeError:
            return []

    @staticmethod
    def _stable_snapshot(record: Dict[str, Any]) -> str:
        """Serialize a record with volatile fields stripped, for NOOP checks.

        Timestamps, embeddings, and storage ids churn on every write even
        when the facts themselves are unchanged; they must not defeat the
        duplicate-write guard in :meth:`upsert_entity`.
        """
        volatile = {"embedding", "_id", "id", "created_at", "updated_at", "score"}

        def _strip(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: _strip(val)
                    for key, val in sorted(value.items())
                    if key not in volatile
                }
            if isinstance(value, (list, tuple)):
                return [_strip(item) for item in value]
            return value

        try:
            return json.dumps(_strip(record), sort_keys=True, default=str)
        except Exception:
            return str(record)

    def _merge_record(
        self,
        *,
        existing: Optional[Dict[str, Any]],
        entity_id: str,
        name: Optional[str],
        entity_type: Optional[str],
        attributes: Optional[Sequence[Union[EntityAttribute, Dict[str, Any]]]],
        relations: Optional[Sequence[Union[EntityRelation, Dict[str, Any]]]],
        metadata: Optional[Dict[str, Any]],
        memory_id: Optional[str],
        timestamp: str,
        user_id: Optional[str] = None,
        kept_pinned: Optional[List[str]] = None,
        kept_stronger: Optional[List[str]] = None,
        write_level: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Merge dictionaries for upsert operations."""
        record = existing.copy() if existing else {}
        record["entity_id"] = record.get("entity_id") or entity_id
        if name:
            record["name"] = name
        # A record keeps the type it has unless the host says otherwise: a
        # model that files a fact about a "team" as an "organization" has
        # relabelled it, not learned something, and records of two types
        # are never merged as duplicates.
        if entity_type and (
            not record.get("entity_type")
            or write_level not in SOURCE_LEVEL_LABELS
            or int(write_level) >= SOURCE_LEVEL_HOST
        ):
            record["entity_type"] = entity_type
        if memory_id:
            record["memory_id"] = memory_id
        # Persist user_id explicitly, including the anonymous ``None`` scope.
        record["user_id"] = user_id
        if metadata:
            record["metadata"] = {**record.get("metadata", {}), **metadata}
        stored_metadata = record.get("metadata")
        if (
            isinstance(stored_metadata, dict)
            and "identity_key" in stored_metadata
            and not self._stored_identity_key(stored_metadata)
        ):
            # Heal a poisoned record on its next write.
            record["metadata"] = {
                key: value
                for key, value in stored_metadata.items()
                if key != "identity_key"
            }

        record_attributes = self._merge_attributes(
            record.get("attributes", []),
            attributes,
            timestamp,
            kept_pinned,
            kept_stronger=kept_stronger,
            write_level=write_level,
        )
        record_relations = self._merge_relations(
            record.get("relations", []), relations, timestamp
        )

        record["attributes"] = record_attributes
        record["relations"] = record_relations
        record["created_at"] = record.get("created_at", timestamp)
        record["updated_at"] = timestamp
        return record

    def _merge_attributes(
        self,
        existing: Iterable[Dict[str, Any]],
        updates: Optional[Sequence[Union[EntityAttribute, Dict[str, Any]]]],
        timestamp: str,
        kept_pinned: Optional[List[str]] = None,
        *,
        kept_stronger: Optional[List[str]] = None,
        write_level: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Apply ``updates`` by attribute name, keeping what each replaced
        value said, and when it held, in the attribute's ``history``.

        A write replaces the current value only when its source level is the
        same or higher (:func:`source_level`); at the same level the newer
        write wins. A pinned value is the strongest and is replaced only by a
        pinned write; those attributes go into ``kept_pinned``. A weaker
        write that disagrees is kept in the history with ``applied: False``
        and its attribute goes into ``kept_stronger``.

        The same value said again from the same place changes nothing, so the
        write stays a no-op; said from another conversation, the earlier
        provenance joins the history, where forgetting either conversation
        still leaves the fact.
        """
        attributes = {attr["name"].lower(): attr for attr in existing if "name" in attr}
        for attr in updates or []:
            normalized = self._stamped(EntityAttribute, attr, timestamp)
            if normalized.get("pinned"):
                level = SOURCE_LEVEL_HOST
            elif normalized.get("level") in SOURCE_LEVEL_LABELS:
                level = int(normalized["level"])
            elif write_level in SOURCE_LEVEL_LABELS:
                level = int(write_level)
            else:
                # A direct call with nothing said is host code.
                level = SOURCE_LEVEL_HOST
            normalized["level"] = level
            correction = bool(normalized.pop("correction", False))
            key = normalized["name"].lower()
            previous = attributes.get(key)
            if isinstance(previous, dict):
                same_value = self._same_value(previous, normalized)
                pinned_kept = previous.get("pinned") and not normalized.get("pinned")
                weaker = level < source_level(previous)
                if pinned_kept or weaker:
                    if pinned_kept and kept_pinned is not None:
                        kept_pinned.append(previous["name"])
                    if same_value:
                        # A weaker source saying the same thing confirms the
                        # value; it changes nothing else about it.
                        attributes[key] = self._reinforced(
                            previous, normalized, timestamp
                        )
                    else:
                        if not pinned_kept and kept_stronger is not None:
                            kept_stronger.append(previous["name"])
                        attributes[key] = self._with_unapplied(
                            previous, normalized, timestamp
                        )
                    continue
                if same_value and all(
                    previous.get(field) == normalized.get(field)
                    for field in ("source", "thread_id", "pinned")
                ):
                    # Re-asserted verbatim. A higher confidence still counts,
                    # and so does saying it again after a while.
                    kept = dict(previous)
                    if float(normalized.get("confidence") or 0) > float(
                        previous.get("confidence") or 0
                    ):
                        kept.update(
                            confidence=normalized["confidence"], updated_at=timestamp
                        )
                    attributes[key] = self._reinforced(kept, normalized, timestamp)
                    continue
                if same_value:
                    # The same value from somewhere else: the earlier
                    # provenance joins the history, and it is a confirmation.
                    history = self._attribute_history(previous, replaced_at=timestamp)
                    if history:
                        for field in ("confirmations", "last_seen"):
                            history[0].pop(field, None)
                        normalized["history"] = history
                    normalized["created_at"] = previous.get("created_at") or timestamp
                    if not normalized.get("valid_from") and previous.get("valid_from"):
                        normalized["valid_from"] = previous["valid_from"]
                    counted = self._reinforced(
                        previous, normalized, timestamp, always=True
                    )
                    for field in ("confirmations", "last_seen", "confirmed_by"):
                        if counted.get(field):
                            normalized[field] = counted[field]
                    attributes[key] = normalized
                    continue
                started = (
                    previous.get("valid_from")
                    or previous.get("created_at")
                    or previous.get("updated_at")
                )
                offered = _moment(normalized.get("valid_from"))
                held = _moment(previous.get("valid_from"))
                if not correction and offered is not None and held is not None:
                    if offered < held:
                        # The host is filling in an earlier period: the
                        # current value stays, and this one takes its place
                        # in the past.
                        attributes[key] = self._with_backfilled(
                            previous, normalized, timestamp
                        )
                        continue
                    correction = offered == held
                history = self._attribute_history(
                    previous,
                    replaced_at=timestamp,
                    valid_to=None if correction else normalized.get("valid_from"),
                    corrected=correction,
                )
                normalized["created_at"] = timestamp
                if correction and not normalized.get("valid_from") and started:
                    # A correction is about the same period the wrong value
                    # claimed.
                    normalized["valid_from"] = started
                if history:
                    normalized["history"] = history
                for field in ("confirmations", "last_seen", "confirmed_by"):
                    normalized.pop(field, None)
            # ``created_at`` is when this value became current (a value said
            # again keeps it): the recorded time, and the valid-from date
            # too unless the host gave one. A second copy is not stored.
            attributes[key] = normalized
        return list(attributes.values())

    def _with_unapplied(
        self,
        previous: Mapping[str, Any],
        offered: Mapping[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """``previous`` with ``offered`` noted in its history as a write that
        was not applied. Offering the same value again adds nothing."""
        history = [
            dict(item)
            for item in previous.get("history") or []
            if isinstance(item, dict)
        ]
        for item in history:
            if item.get("applied") is False and self._same_value(item, offered):
                return dict(previous)
        entry = {
            field: offered.get(field)
            for field in _HISTORY_FIELDS
            if offered.get(field) not in (None, "", False)
        }
        entry.update(updated_at=timestamp, applied=False)
        return {
            **previous,
            "history": ([entry] + history)[: self.attribute_history_limit],
        }

    @staticmethod
    def _reinforced(
        previous: Mapping[str, Any],
        offered: Mapping[str, Any],
        timestamp: str,
        *,
        always: bool = False,
    ) -> Dict[str, Any]:
        """``previous`` with one more confirmation: its count, when it was
        last seen and who said it. Nothing else about the value changes.

        The same source in the same conversation counts again only after
        :data:`REINFORCE_QUIET_SECONDS`, unless ``always``.
        """
        entry = {
            key: value
            for key, value in (
                ("source", offered.get("source")),
                ("thread_id", offered.get("thread_id")),
                ("level", offered.get("level")),
            )
            if value not in (None, "")
        }
        origin = (entry.get("source"), entry.get("thread_id"))
        confirmed = [
            dict(item)
            for item in previous.get("confirmed_by") or []
            if isinstance(item, dict)
        ]
        if not always:
            seen = [
                item.get("at")
                for item in confirmed
                if (item.get("source"), item.get("thread_id")) == origin
            ]
            if (previous.get("source"), previous.get("thread_id")) == origin:
                seen.append(
                    previous.get("last_seen")
                    or previous.get("updated_at")
                    or previous.get("created_at")
                )
            now = _moment(timestamp) or 0.0
            if any(
                now - (_moment(at) or 0.0) < REINFORCE_QUIET_SECONDS
                for at in seen
                if at
            ):
                return dict(previous)
        entry["at"] = timestamp
        return {
            **previous,
            "confirmations": _count(previous.get("confirmations")) + 1,
            "last_seen": timestamp,
            "confirmed_by": ([entry] + confirmed)[:CONFIRMED_BY_LIMIT],
        }

    def _with_backfilled(
        self,
        previous: Mapping[str, Any],
        offered: Mapping[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """``previous`` with ``offered`` placed in its history as a value
        that was true before it: from ``offered``'s valid-from until the
        next value the history knows began."""
        history = [
            dict(item)
            for item in previous.get("history") or []
            if isinstance(item, dict)
        ]
        begins = _moment(offered.get("valid_from")) or 0.0
        for item in history:
            if (
                item.get("applied") is not False
                and self._same_value(item, offered)
                and _moment(item.get("valid_from")) == begins
            ):
                return dict(previous)
        later = [
            (moment, value)
            for moment, value in (
                (_moment(row.get("valid_from")), row.get("valid_from"))
                for row in [previous, *history]
                if row.get("applied") is not False and not row.get("corrected")
            )
            if moment is not None and moment > begins
        ]
        entry = {
            field: offered.get(field)
            for field in _HISTORY_FIELDS
            if offered.get(field) not in (None, "", False)
        }
        entry.update(
            updated_at=timestamp,
            valid_from=offered.get("valid_from"),
            valid_to=min(later)[1] if later else None,
            recorded_at=timestamp,
            superseded_at=timestamp,
            backfilled=True,
        )
        index = next(
            (
                position
                for position, item in enumerate(history)
                if (_moment(item.get("valid_from") or item.get("updated_at")) or 0.0)
                < begins
            ),
            len(history),
        )
        history.insert(index, entry)
        return {**previous, "history": history[: self.attribute_history_limit]}

    @staticmethod
    def _same_value(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return " ".join(str(left.get("value") or "").casefold().split()) == " ".join(
            str(right.get("value") or "").casefold().split()
        )

    def _attribute_history(
        self,
        previous: Mapping[str, Any],
        replaced_at: Optional[str] = None,
        *,
        valid_to: Optional[str] = None,
        corrected: bool = False,
    ) -> List[Dict[str, Any]]:
        """``previous``'s own history with ``previous`` itself in front.

        It was true from when it began until ``valid_to`` (when the value
        replacing it began, if the host said; otherwise ``replaced_at``), and
        believed from when it was recorded until ``replaced_at``. A
        ``corrected`` value gets no ``valid_to``: it was never true.
        """
        entry = {
            field: previous.get(field)
            for field in _HISTORY_FIELDS
            if previous.get(field) not in (None, "", False)
        }
        if entry:
            started = (
                previous.get("valid_from")
                or previous.get("created_at")
                or previous.get("updated_at")
            )
            if started:
                entry["valid_from"] = started
            recorded = previous.get("created_at") or previous.get("updated_at")
            if recorded and recorded != started:
                entry["recorded_at"] = recorded
            if corrected and replaced_at:
                entry["corrected"] = True
                entry["corrected_at"] = replaced_at
            elif replaced_at:
                ends = valid_to or replaced_at
                if (
                    valid_to
                    and started
                    and (_moment(valid_to) or 0.0) <= (_moment(started) or 0.0)
                ):
                    ends = replaced_at
                entry["valid_to"] = ends
                if ends != replaced_at:
                    entry["superseded_at"] = replaced_at
        older = [
            dict(item)
            for item in previous.get("history") or []
            if isinstance(item, dict)
        ]
        limit = self.attribute_history_limit
        return ([entry] + older)[:limit] if entry else older[:limit]

    def _merge_relations(
        self,
        existing: Iterable[Dict[str, Any]],
        updates: Optional[Sequence[Union[EntityRelation, Dict[str, Any]]]],
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        relations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for relation in existing:
            entity_id = str(relation.get("entity_id") or "").strip()
            relation_type = str(relation.get("relation_type") or "").strip()
            if not entity_id or not relation_type:
                continue
            relations[(entity_id, relation_type.casefold())] = dict(relation)
        for relation in updates or []:
            normalized = self._stamped(EntityRelation, relation, timestamp)
            key = (
                str(normalized["entity_id"]).strip(),
                str(normalized["relation_type"]).strip().casefold(),
            )
            previous = relations.get(key)
            if previous and not normalized.get("created_at"):
                normalized["created_at"] = previous.get("created_at")
            relations[key] = normalized
        return list(relations.values())

    @staticmethod
    def _stamped(
        model: Union[Type[EntityAttribute], Type[EntityRelation]],
        value: Union[EntityAttribute, EntityRelation, Dict[str, Any]],
        timestamp: str,
    ) -> Dict[str, Any]:
        """An attribute or relation as a validated dict with its timestamps."""
        payload = (value if isinstance(value, model) else model(**value)).model_dump()
        payload["created_at"] = payload.get("created_at") or timestamp
        payload["updated_at"] = timestamp
        if model is EntityAttribute:
            # Records written before these fields existed keep their shape
            # (and their no-op checks) until a value actually uses one.
            for field, empty in (
                ("thread_id", None),
                ("pinned", False),
                ("kind", None),
                ("level", None),
                ("valid_from", None),
                ("history", []),
                ("confirmations", None),
                ("last_seen", None),
                ("confirmed_by", []),
            ):
                if payload.get(field) == empty:
                    payload.pop(field, None)
        return payload

    @staticmethod
    def _build_embedding_text(record: Dict[str, Any]) -> str:
        """Create a summary string describing the entity for embeddings."""
        lines = []
        if record.get("name"):
            lines.append(f"Name: {record['name']}")
        if record.get("entity_type"):
            lines.append(f"Type: {record['entity_type']}")
        for attribute in record.get("attributes", []):
            lines.append(f"{attribute.get('name')}: {attribute.get('value')}")
        for relation in record.get("relations", []):
            lines.append(
                f"Relation ({relation.get('relation_type')})->{relation.get('entity_id')}"
            )
        if record.get("metadata"):
            # Retired facts stay out of the vector, so searches stop finding them.
            metadata_parts = [
                f"{key}: {value}"
                for key, value in record["metadata"].items()
                if key not in {"retired_attributes", "merge_undo"}
                and key not in _EMBEDDING_STATUS_KEYS
            ]
            lines.extend(metadata_parts)
        return "\n".join(lines)

    @staticmethod
    def _timestamp() -> str:
        return utc_now_iso()
