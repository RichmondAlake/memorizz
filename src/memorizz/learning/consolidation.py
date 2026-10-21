# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Memory consolidation: tidy an agent's memory between conversations.

An agent writes memory one turn at a time, so over many conversations its
store collects duplicate entities, the same fact under two names, values the
user has since changed, and things the user said that the agent never wrote
down. Letta runs a "sleep-time" agent for this and Anthropic's Managed Agents
call it "dreaming". MemoRizz runs one consolidation pass, usually on the
agent's background model and off the user's turn:

1. **Duplicate entities** (no model): records for the same entity are merged
   with ``EntityMemory.consolidate_duplicate_entities``.
2. **Duplicate rows** (no model): summaries, and optionally knowledge-base
   chunks, whose text is the same as another row's are suppressed.
3. **Attribute review** (model): within an entity, a fact stored under two
   names, or one a newer fact contradicts, is retired.
4. **New facts** (model): durable facts the user stated in recent
   conversations, and the agent never stored, are added to entity memory.
   Every new fact must quote the user's own words from the transcript.

A pass is a dry-run plan first. Applying it needs a named approver, and
nothing is deleted: rows are suppressed (``retention_state``), retired
attributes move to the entity's metadata, and added facts carry the plan id,
so :meth:`MemoryConsolidator.restore` undoes a plan.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .._time import utc_now_iso
from ..enums.memory_type import MemoryType
from ..memagent.utils.context_dedup import (
    _parse_timestamp,
    is_suppressed,
    naive_time_convention,
)
from ..retrieval import SEALED_STATE
from .models import canonical_json

logger = logging.getLogger(__name__)

MERGE_ENTITIES = "merge_entities"
SUPPRESS_DUPLICATE = "suppress_duplicate"
RETIRE_ATTRIBUTE = "retire_attribute"
ADD_FACT = "add_fact"

AUTOMATIC_APPROVER = "memorizz:consolidation"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ConsolidationConfig:
    """When and how a background consolidation pass runs.

    ``every_n_turns`` and ``every_hours`` schedule a pass after a turn: the
    first one reached wins. ``auto_apply`` applies a scheduled plan as
    ``memorizz:consolidation``; otherwise the plan waits for review.
    """

    enabled: bool = False
    every_n_turns: int = 20
    every_hours: float = 24.0
    auto_apply: bool = True
    review_attributes: bool = True
    extract_facts: bool = True
    max_entities: int = 40
    max_threads: int = 5
    max_messages_per_thread: int = 40

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _truthy(self.enabled))
        object.__setattr__(self, "auto_apply", _truthy(self.auto_apply))
        object.__setattr__(self, "review_attributes", _truthy(self.review_attributes))
        object.__setattr__(self, "extract_facts", _truthy(self.extract_facts))
        object.__setattr__(self, "every_n_turns", max(1, int(self.every_n_turns)))
        object.__setattr__(self, "every_hours", max(0.0, float(self.every_hours)))
        for name, ceiling in (
            ("max_entities", 500),
            ("max_threads", 50),
            ("max_messages_per_thread", 400),
        ):
            object.__setattr__(
                self, name, max(1, min(int(getattr(self, name)), ceiling))
            )

    @classmethod
    def from_value(cls, value: Any) -> "ConsolidationConfig":
        if isinstance(value, cls):
            return value
        if value is None or value is False:
            return cls()
        if value is True:
            return cls(enabled=True)
        if not isinstance(value, Mapping):
            raise TypeError("memory_consolidation must be a bool or a mapping")
        known = {item for item in cls.__dataclass_fields__}
        data = {key: value[key] for key in value if key in known}
        data.setdefault("enabled", True)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ConsolidationOp:
    """One change a plan proposes."""

    kind: str
    memory_type: str
    target_id: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "memory_type": self.memory_type,
            "target_id": self.target_id,
            "reason": self.reason,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConsolidationOp":
        return cls(
            kind=str(value["kind"]),
            memory_type=str(value.get("memory_type") or ""),
            target_id=str(value.get("target_id") or ""),
            reason=str(value.get("reason") or ""),
            details=dict(value.get("details") or {}),
        )


@dataclass(frozen=True)
class ConsolidationPlan:
    """A dry run: what a consolidation pass would change, and why."""

    plan_id: str
    memory_id: str
    user_id: Optional[str]
    ops: Tuple[ConsolidationOp, ...]
    model: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)
    warnings: Tuple[str, ...] = ()

    @staticmethod
    def make_id(
        memory_id: str, user_id: Optional[str], ops: Sequence[ConsolidationOp]
    ) -> str:
        payload = canonical_json(
            {
                "memory_id": memory_id,
                "user_id": user_id,
                "ops": [op.to_dict() for op in ops],
            }
        )
        return "consolidate-" + hashlib.sha256(payload.encode()).hexdigest()[:28]

    @property
    def counts(self) -> Dict[str, int]:
        counts = {
            MERGE_ENTITIES: 0,
            SUPPRESS_DUPLICATE: 0,
            RETIRE_ATTRIBUTE: 0,
            ADD_FACT: 0,
        }
        for op in self.ops:
            counts[op.kind] = counts.get(op.kind, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "dry_run": True,
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "model": self.model,
            "created_at": self.created_at,
            "counts": self.counts,
            "ops": [op.to_dict() for op in self.ops],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConsolidationPlan":
        return cls(
            plan_id=str(value["plan_id"]),
            memory_id=str(value.get("memory_id") or ""),
            user_id=value.get("user_id"),
            ops=tuple(
                ConsolidationOp.from_dict(item)
                for item in value.get("ops") or []
                if isinstance(item, Mapping)
            ),
            model=value.get("model"),
            created_at=str(value.get("created_at") or utc_now_iso()),
            warnings=tuple(str(item) for item in value.get("warnings") or []),
        )


@dataclass(frozen=True)
class ConsolidationResult:
    plan_id: str
    approved_by: str
    applied: Dict[str, int]
    errors: Tuple[str, ...] = ()
    applied_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "dry_run": False,
            "approved_by": self.approved_by,
            "applied": dict(self.applied),
            "errors": list(self.errors),
            "applied_at": self.applied_at,
            "ok": not self.errors,
        }


def text_generator(model: Any) -> Optional[Callable[[str], str]]:
    """A one-prompt text call on any MemoRizz model, or None without one."""
    if model is None:
        return None
    text_fn = getattr(model, "generate_text", None)
    if callable(text_fn):
        return lambda prompt: str(text_fn(prompt) or "")
    generate = getattr(model, "generate", None)
    if not callable(generate):
        return None

    def _call(prompt: str) -> str:
        response = generate([{"role": "user", "content": prompt}])
        if isinstance(response, str):
            return response
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            return str(getattr(message, "content", "") or "")
        return str(response or "")

    return _call


def model_for_agent(record: Any) -> Any:
    """The model a saved agent's consolidation runs on: its background
    model, else its chat model (None when it has neither)."""
    from ..llms.llm_factory import create_llm_provider

    data = record if isinstance(record, Mapping) else getattr(record, "__dict__", {})
    config = data.get("background_llm_config") or data.get("llm_config")
    return create_llm_provider(dict(config)) if config else None


def _model_label(model: Any) -> Optional[str]:
    if model is None:
        return None
    try:
        config = model.get_config()
    except Exception:
        config = None
    if isinstance(config, Mapping) and config.get("model"):
        provider = config.get("provider")
        return f"{provider}/{config['model']}" if provider else str(config["model"])
    return getattr(model, "model", None) or type(model).__name__


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """The JSON object in a model reply, forgiving the slips small models
    make: code fences, a trailing comma, a stray ``)`` after a list."""
    cleaned = _FENCE_RE.sub("", str(text or "").strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    repaired = re.sub(r",\s*([\]}])", r"\1", re.sub(r"\]\s*\)", "]", candidate))
    for attempt in (candidate, repaired):
        try:
            value = json.loads(attempt)
        except ValueError:
            continue
        return value if isinstance(value, dict) else None
    return None


def _normalized(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


def _row_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "summary_content", "content", "excerpt"):
        value = row.get(key)
        if isinstance(value, Mapping):
            value = value.get("content") or value.get("text")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("_id") or row.get("id") or row.get("record_id") or "")


def _row_thread(row: Mapping[str, Any]) -> str:
    content = row.get("content")
    nested = content if isinstance(content, Mapping) else {}
    return str(
        row.get("thread_id")
        or row.get("conversation_id")
        or nested.get("thread_id")
        or nested.get("conversation_id")
        or ""
    )


def _row_time(row: Mapping[str, Any], memory_type: Any = None) -> float:
    """Epoch seconds a row was written (0.0 when it records none)."""
    for key in ("created_at", "timestamp", "updated_at"):
        value = _parse_timestamp(row.get(key), naive_time_convention(memory_type, key))
        if value is not None:
            return value
    return 0.0


def _op_label(op: "ConsolidationOp") -> str:
    """What a change did, for the memory history journal."""
    details = op.details
    if op.kind == MERGE_ENTITIES:
        names = [name for name in details.get("names") or [] if name]
        return (
            "Merged " + " and ".join(dict.fromkeys(names))
            if len(set(names)) > 1
            else "Merged duplicates of " + str(names[0] if names else op.target_id)
        )
    if op.kind == SUPPRESS_DUPLICATE:
        kind = "summary" if op.memory_type == "summaries" else "knowledge-base chunk"
        return f"Hid a duplicate {kind}"
    if op.kind == ADD_FACT:
        verb = "Updated" if details.get("replaces") is not None else "Added"
        return f"{verb} {details.get('entity')}.{details.get('attribute')}"
    return op.reason


def _state_before(row: Mapping[str, Any]) -> str:
    """The ``retention_state`` a row had before it was suppressed."""
    meta = (
        row.get("retention_meta")
        if isinstance(row.get("retention_meta"), Mapping)
        else {}
    )
    previous = row.get("suppressed_from") or meta.get("suppressed_from")
    return str(previous or "active")


def _flag(row: Mapping[str, Any], name: str) -> bool:
    if _truthy(row.get(name)):
        return True
    metadata = row.get("metadata")
    return isinstance(metadata, Mapping) and _truthy(metadata.get(name))


def _cosine(left: Any, right: Any) -> float:
    try:
        a = [float(x) for x in left]
        b = [float(x) for x in right]
    except (TypeError, ValueError):
        return 0.0
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _accepts(fn: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    )


_REVIEW_PROMPT = """You maintain an assistant's memory of facts about people and things.
Below are entities, each with its stored attributes.

1. Within one entity, find pairs of attributes that are about the same thing: the same
   fact under two names (favorite_drink and preferred_drink), or an answer to the same
   question that changed (home_city and current_city). Different facts that merely look
   alike are not a pair.
2. Find entities that are the same person or thing under two names. An entity named
   "user" is the person the assistant talks to; if another entity is that person, it is
   the same entity.

Only list what you are sure of. Reply with JSON only, in this shape:
{"same_fact": [{"entity_id": "...", "attributes": ["<name>", "<name>"]}],
 "same_entity": [{"entity_ids": ["...", "..."]}]}
Reply {"same_fact": [], "same_entity": []} when there is nothing.

Entities:
"""


def known_facts(
    records: Any,
) -> Tuple[Dict[Tuple[str, str], Tuple[str, Mapping[str, Any]]], set, List[str]]:
    """What entity memory already holds, as the fact checks read it: the
    value per (entity, attribute), the values on their own, and prompt lines."""
    known: Dict[Tuple[str, str], Tuple[str, Mapping[str, Any]]] = {}
    for record in records or []:
        for attribute in record.get("attributes") or []:
            if isinstance(attribute, dict) and attribute.get("name"):
                known[
                    (
                        _normalized(record.get("name")),
                        str(attribute["name"]).casefold(),
                    )
                ] = (str(attribute.get("value") or ""), record)
    known_values = {
        _normalized(value)
        for value, _record in known.values()
        if len(_normalized(value)) >= 4
    } | {(name, _normalized(value)) for (_entity, name), (value, _r) in known.items()}
    known_lines = [
        f"- {entity}.{name} = {value[:120]}"
        for (entity, name), (value, _record) in list(known.items())[:80]
    ] or ["(nothing yet)"]
    return known, known_values, known_lines


_EXTRACT_PROMPT = """You maintain an assistant's long-term memory about the user.
Below are the messages the user wrote in one conversation, one per line.
List durable facts they state about themselves or about people, projects and things in
their life: names, roles, preferences, plans, relationships. Skip anything temporary,
guesses, and facts already known.

For each fact, "evidence" is a short phrase (at most 12 words) copied exactly from one
of the user's lines. Use "user" as the entity for facts about the user. Attribute names
are short snake_case. Reply with JSON only, in this shape:
{"facts": [{"entity": "user", "entity_type": "person", "attribute": "...", "value": "...", "evidence": "..."}]}
Reply {"facts": []} when there is nothing new.

Already known:
{known}

The user's messages:
{transcript}
"""


class MemoryConsolidator:
    """Plan, apply and undo consolidation of one memory and user."""

    def __init__(
        self,
        provider: Any,
        *,
        agent_id: Optional[str] = None,
        model: Any = None,
        config: Optional[ConsolidationConfig] = None,
        knowledge_base_ids: Iterable[str] = (),
    ) -> None:
        if provider is None:
            raise ValueError("Memory consolidation requires a memory provider")
        from ..long_term.semantic.entity_memory import EntityMemory

        self.provider = provider
        self.agent_id = str(agent_id or "").strip() or None
        self.model = model
        self.config = config or ConsolidationConfig()
        self.knowledge_base_ids = frozenset(str(item) for item in knowledge_base_ids)
        self.entities = EntityMemory(provider)

    # -- planning ------------------------------------------------------------

    def recent_threads(
        self, memory_id: str, user_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """The memory's conversations, most recently active first.

        Threads that ran sealed hold only sealed rows and are left out.
        """
        retrieve = getattr(
            self.provider, "retrieve_conversation_history_ordered_by_timestamp", None
        )
        if not callable(retrieve):
            return []
        kwargs: Dict[str, Any] = {"memory_id": memory_id}
        if _accepts(retrieve, "user_id"):
            kwargs["user_id"] = user_id
        if _accepts(retrieve, "include_embedding"):
            kwargs["include_embedding"] = False
        latest: Dict[str, float] = {}
        for row in retrieve(**kwargs) or []:
            if (
                not isinstance(row, dict)
                or row.get("user_id") != user_id
                or str(row.get("retention_state") or "").lower() == SEALED_STATE
            ):
                continue
            thread = _row_thread(row)
            if thread:
                when = _row_time(row, MemoryType.CONVERSATION_MEMORY)
                latest[thread] = max(latest.get(thread, 0.0), when)
        ordered = sorted(latest, key=lambda thread: latest[thread], reverse=True)
        return ordered[: limit or self.config.max_threads]

    def plan(
        self,
        *,
        memory_id: str,
        user_id: Optional[str] = None,
        thread_ids: Sequence[str] = (),
        since: Optional[float] = None,
        knowledge_base: bool = False,
        entities: bool = True,
    ) -> ConsolidationPlan:
        """Dry run over one memory and user; nothing is changed.

        ``thread_ids`` are the conversations mined for new facts (none are
        mined without them); ``since`` (epoch seconds) limits that to
        newer messages.
        ``knowledge_base`` also suppresses duplicate knowledge-base chunks,
        which every user of the agent shares. ``entities=False`` leaves
        entity memory alone (an agent that keeps entity facts per thread).
        """
        memory = str(memory_id or "").strip()
        if not memory:
            raise ValueError("memory_id is required for memory consolidation")
        ops: List[ConsolidationOp] = []
        warnings: List[str] = []
        supports_entities = getattr(self.provider, "supports_entity_memory", None)
        if entities and (not callable(supports_entities) or supports_entities()):
            ops.extend(self._entity_duplicates(memory, user_id, warnings))
        ops.extend(
            self._duplicate_rows(MemoryType.SUMMARIES, memory, user_id, warnings)
        )
        if knowledge_base:
            ops.extend(
                self._duplicate_rows(
                    MemoryType.KNOWLEDGE_BASE, memory, user_id, warnings
                )
            )
        generate = text_generator(self.model)
        if generate is not None and entities:
            if self.config.review_attributes:
                merged = {
                    entity_id
                    for op in ops
                    if op.kind == MERGE_ENTITIES
                    for entity_id in (op.target_id, *op.details["duplicate_entity_ids"])
                }
                ops.extend(
                    self._attribute_review(generate, memory, user_id, merged, warnings)
                )
            if self.config.extract_facts and thread_ids:
                ops.extend(
                    self._new_facts(
                        generate, memory, user_id, list(thread_ids), since, warnings
                    )
                )
        elif (
            generate is None
            and entities
            and (
                self.config.review_attributes
                or (self.config.extract_facts and thread_ids)
            )
        ):
            warnings.append("no model: attribute review and fact extraction skipped")
        return ConsolidationPlan(
            plan_id=ConsolidationPlan.make_id(memory, user_id, ops),
            memory_id=memory,
            user_id=user_id,
            ops=tuple(ops),
            model=_model_label(self.model) if generate is not None else None,
            warnings=tuple(warnings),
        )

    def _entity_duplicates(
        self, memory_id: str, user_id: Optional[str], warnings: List[str]
    ) -> List[ConsolidationOp]:
        try:
            report = self.entities.consolidate_duplicate_entities(
                memory_id=memory_id,
                user_id=user_id,
                apply=False,
                allow_legacy_scope=user_id is None,
            )
        except Exception as exc:
            warnings.append(f"entity duplicates skipped: {type(exc).__name__}")
            return []
        groups = report.get("groups") or []
        names: Dict[str, Any] = {}
        if groups:
            names = {
                str(record.get("entity_id")): record.get("name")
                for record in self.entities.list_entities(
                    memory_id=memory_id, user_id=user_id
                )
            }
        return [
            ConsolidationOp(
                kind=MERGE_ENTITIES,
                memory_type=MemoryType.ENTITY_MEMORY.value,
                target_id=group["canonical_entity_id"],
                reason=f"{group['record_count']} records for the same entity",
                details={
                    "duplicate_entity_ids": list(group["duplicate_entity_ids"]),
                    "names": [
                        names.get(entity_id)
                        for entity_id in (
                            group["canonical_entity_id"],
                            *group["duplicate_entity_ids"],
                        )
                    ],
                    "conflicting_attribute_names": list(
                        group.get("conflicting_attribute_names") or []
                    ),
                },
            )
            for group in groups
        ]

    def _rows(self, memory_type: MemoryType, user_id: Optional[str]) -> List[Dict]:
        """Rows of one user, or every row of a knowledge base (shared by all users)."""
        if memory_type == MemoryType.KNOWLEDGE_BASE:
            rows = self.provider.list_all(memory_type)
        else:
            try:
                rows = self.provider.list_all(memory_type, user_id=user_id)
            except TypeError:
                rows = self.provider.list_all(memory_type)
        return [row for row in rows or [] if isinstance(row, dict)]

    def _duplicate_rows(
        self,
        memory_type: MemoryType,
        memory_id: str,
        user_id: Optional[str],
        warnings: List[str],
    ) -> List[ConsolidationOp]:
        """Rows whose text repeats another row's in the same scope.

        The oldest row stays (other records may cite it). Summaries of one
        thread also count as duplicates at cosine similarity 0.97 or more.
        """
        try:
            rows = self._rows(memory_type, user_id)
        except Exception as exc:
            warnings.append(
                f"{memory_type.value} duplicates skipped: {type(exc).__name__}"
            )
            return []
        is_knowledge = memory_type == MemoryType.KNOWLEDGE_BASE
        groups: Dict[Tuple[str, ...], List[Dict]] = {}
        for row in rows:
            if is_suppressed(row) or not _row_id(row):
                continue
            if not is_knowledge and row.get("user_id") != user_id:
                continue
            if is_knowledge:
                base = str(row.get("knowledge_base_id") or "")
                if (
                    base not in self.knowledge_base_ids
                    and row.get("memory_id") != memory_id
                ):
                    continue
                key = (
                    base or str(row.get("memory_id") or ""),
                    str(row.get("namespace") or ""),
                )
            else:
                if row.get("memory_id") != memory_id:
                    continue
                if self.agent_id and row.get("agent_id") not in (None, self.agent_id):
                    continue
                key = (_row_thread(row),)
            if _row_text(row):
                groups.setdefault(key, []).append(row)
        ops: List[ConsolidationOp] = []
        for key, members in sorted(groups.items()):
            members.sort(key=lambda row: (_row_time(row, memory_type), _row_id(row)))
            kept: List[Dict] = []
            for row in members:
                text = _normalized(_row_text(row))
                original = next(
                    (
                        other
                        for other in kept
                        if _normalized(_row_text(other)) == text
                        or (
                            not is_knowledge
                            and _cosine(row.get("embedding"), other.get("embedding"))
                            >= 0.97
                        )
                    ),
                    None,
                )
                if original is None or _flag(row, "pinned") or _flag(row, "verified"):
                    kept.append(row)
                    continue
                exact = _normalized(_row_text(original)) == text
                ops.append(
                    ConsolidationOp(
                        kind=SUPPRESS_DUPLICATE,
                        memory_type=memory_type.value,
                        target_id=_row_id(row),
                        reason=(
                            "same text as an older row"
                            if exact
                            else "near-identical to an older summary of this thread"
                        ),
                        details={"duplicate_of": _row_id(original)},
                    )
                )
        return ops

    def _ask(self, generate: Callable[[str], str], prompt: str, warnings: List[str]):
        try:
            answer = _parse_json(generate(prompt))
        except Exception as exc:
            warnings.append(f"model call failed: {type(exc).__name__}")
            return None
        if answer is None:
            warnings.append("model reply was not JSON; skipped")
        return answer

    def _attribute_review(
        self,
        generate: Callable[[str], str],
        memory_id: str,
        user_id: Optional[str],
        merged: set,
        warnings: List[str],
    ) -> List[ConsolidationOp]:
        """The model names the pairs; the code decides what goes.

        Of two attributes holding one fact, the newer stays. Two entities
        merge only when they also share at least two facts word for word.
        """
        try:
            records = self.entities.list_entities(memory_id=memory_id, user_id=user_id)
        except Exception as exc:
            warnings.append(f"attribute review skipped: {type(exc).__name__}")
            return []
        candidates = [
            record
            for record in records
            if len([a for a in record.get("attributes") or [] if isinstance(a, dict)])
            >= 2
        ]
        candidates.sort(
            key=lambda record: str(record.get("updated_at") or ""), reverse=True
        )
        candidates = candidates[: self.config.max_entities]
        ops: List[ConsolidationOp] = []
        for start in range(0, len(candidates), 8):
            batch = {
                str(record.get("entity_id")): record
                for record in candidates[start : start + 8]
            }
            listing = [
                {
                    "entity_id": entity_id,
                    "name": record.get("name"),
                    "type": record.get("entity_type"),
                    "attributes": [
                        {
                            "name": attribute.get("name"),
                            "value": str(attribute.get("value") or "")[:300],
                            "updated_at": attribute.get("updated_at"),
                        }
                        for attribute in record.get("attributes") or []
                        if isinstance(attribute, dict)
                    ],
                }
                for entity_id, record in batch.items()
            ]
            answer = self._ask(
                generate,
                _REVIEW_PROMPT + json.dumps(listing, ensure_ascii=False),
                warnings,
            )
            if not answer:
                continue
            merges = self._same_entity_ops(answer.get("same_entity"), batch, merged)
            ops.extend(merges)
            ops.extend(
                self._same_fact_ops(answer.get("same_fact"), batch, merged, warnings)
            )
        return ops

    @staticmethod
    def _same_entity_ops(
        proposals: Any, batch: Mapping[str, Mapping[str, Any]], merged: set
    ) -> List[ConsolidationOp]:
        from ..long_term.semantic.entity_memory import EntityMemory

        ops: List[ConsolidationOp] = []
        for item in proposals if isinstance(proposals, list) else []:
            raw = item.get("entity_ids") if isinstance(item, Mapping) else None
            ids = list(dict.fromkeys(str(value) for value in raw or [] if value))
            if len(ids) < 2 or any(i not in batch or i in merged for i in ids):
                continue
            records = [batch[i] for i in ids]
            types = {
                str(record.get("entity_type")).casefold()
                for record in records
                if record.get("entity_type")
            }
            facts = [
                {
                    (str(a.get("name")).casefold(), _normalized(a.get("value")))
                    for a in record.get("attributes") or []
                    if isinstance(a, dict) and a.get("name")
                }
                for record in records
            ]
            shared = set.intersection(*facts)
            if len(types) > 1 or len(shared) < 2:
                continue
            canonical = EntityMemory._choose_canonical(records)
            merged.update(ids)
            ops.append(
                ConsolidationOp(
                    kind=MERGE_ENTITIES,
                    memory_type=MemoryType.ENTITY_MEMORY.value,
                    target_id=str(canonical["entity_id"]),
                    reason=(
                        f"one entity under {len(ids)} names, sharing "
                        f"{len(shared)} facts"
                    ),
                    details={
                        "duplicate_entity_ids": sorted(
                            i for i in ids if i != str(canonical["entity_id"])
                        ),
                        "names": [record.get("name") for record in records],
                        "conflicting_attribute_names": EntityMemory._attribute_conflicts(
                            records
                        ),
                    },
                )
            )
        return ops

    @staticmethod
    def _same_fact_ops(
        proposals: Any,
        batch: Mapping[str, Mapping[str, Any]],
        merged: set,
        warnings: List[str],
    ) -> List[ConsolidationOp]:
        ops: List[ConsolidationOp] = []
        retired: Dict[str, set] = {}
        kept: Dict[str, set] = {}
        for item in proposals if isinstance(proposals, list) else []:
            if not isinstance(item, Mapping) or not isinstance(
                item.get("attributes"), list
            ):
                continue
            entity_id = str(item.get("entity_id") or "")
            record = batch.get(entity_id)
            names = [str(name or "").casefold() for name in item["attributes"]]
            # An entity merged in this plan is reviewed again on the next.
            if record is None or entity_id in merged:
                continue
            attributes = {
                str(attribute.get("name") or "").casefold(): attribute
                for attribute in record.get("attributes") or []
                if isinstance(attribute, dict) and attribute.get("name")
            }
            if len(names) != 2 or names[0] == names[1]:
                continue
            if any(name not in attributes for name in names):
                continue
            first, second = attributes[names[0]], attributes[names[1]]
            same_value = _normalized(first.get("value")) == _normalized(
                second.get("value")
            )
            first_at = str(first.get("updated_at") or "")
            second_at = str(second.get("updated_at") or "")
            if first_at == second_at and not same_value:
                warnings.append(
                    f"kept {entity_id}.{first.get('name')} and "
                    f"{second.get('name')}: they disagree and neither is newer"
                )
                continue
            older, newer = (first, second) if first_at <= second_at else (second, first)
            name = str(older.get("name")).casefold()
            keep = str(newer.get("name")).casefold()
            gone = retired.setdefault(entity_id, set())
            stays = kept.setdefault(entity_id, set())
            if keep in gone or name in stays or name in gone:
                continue
            gone.add(name)
            stays.add(keep)
            ops.append(
                ConsolidationOp(
                    kind=RETIRE_ATTRIBUTE,
                    memory_type=MemoryType.ENTITY_MEMORY.value,
                    target_id=entity_id,
                    reason=(
                        f"same fact as {newer.get('name')}"
                        if same_value
                        else f"replaced by the newer {newer.get('name')}"
                    ),
                    details={
                        "entity": record.get("name"),
                        "attribute": older.get("name"),
                        "value": older.get("value"),
                        "kept": newer.get("name"),
                        "kept_value": newer.get("value"),
                        "why": "duplicate" if same_value else "outdated",
                    },
                )
            )
        return ops

    def _thread_messages(
        self,
        memory_id: str,
        user_id: Optional[str],
        thread_id: str,
        since: Optional[float],
    ) -> List[Dict]:
        retrieve = getattr(
            self.provider, "retrieve_conversation_history_ordered_by_timestamp", None
        )
        if not callable(retrieve):
            return []
        kwargs: Dict[str, Any] = {"memory_id": memory_id}
        if _accepts(retrieve, "user_id"):
            kwargs["user_id"] = user_id
        if _accepts(retrieve, "thread_id"):
            kwargs["thread_id"] = thread_id
        if _accepts(retrieve, "include_embedding"):
            kwargs["include_embedding"] = False
        rows = [
            row
            for row in retrieve(**kwargs) or []
            if isinstance(row, dict)
            and row.get("user_id") == user_id
            and _row_thread(row) == str(thread_id)
            and not is_suppressed(row)
            and str(row.get("retention_state") or "").lower() != SEALED_STATE
            and str(row.get("role") or "").lower() in {"user", "assistant"}
        ]
        if since:
            rows = [
                row
                for row in rows
                if _row_time(row, MemoryType.CONVERSATION_MEMORY) > since
            ]
        return rows[-self.config.max_messages_per_thread :]

    def _new_facts(
        self,
        generate: Callable[[str], str],
        memory_id: str,
        user_id: Optional[str],
        thread_ids: List[str],
        since: Optional[float],
        warnings: List[str],
    ) -> List[ConsolidationOp]:
        try:
            records = self.entities.list_entities(memory_id=memory_id, user_id=user_id)
        except Exception as exc:
            warnings.append(f"fact extraction skipped: {type(exc).__name__}")
            return []
        known, known_values, known_lines = known_facts(records)
        ops: List[ConsolidationOp] = []
        seen: set = set()
        for thread_id in thread_ids[: self.config.max_threads]:
            try:
                messages = self._thread_messages(memory_id, user_id, thread_id, since)
            except Exception as exc:
                warnings.append(f"thread {thread_id} skipped: {type(exc).__name__}")
                continue
            said = " ".join(
                _normalized(_row_text(row))
                for row in messages
                if str(row.get("role") or "").lower() == "user"
            )
            if not said:
                continue
            transcript = "\n".join(
                "- " + " ".join(_row_text(row).split())[:1200]
                for row in messages
                if str(row.get("role") or "").lower() == "user"
            )
            answer = self._ask(
                generate,
                _EXTRACT_PROMPT.replace("{known}", "\n".join(known_lines)).replace(
                    "{transcript}", transcript
                ),
                warnings,
            )
            for item in (answer or {}).get("facts") or []:
                op = self._valid_fact(item, said, known, known_values, thread_id)
                if op is None:
                    continue
                key = (
                    _normalized(op.details["entity"]),
                    op.details["attribute"].casefold(),
                )
                if key in seen:
                    continue
                seen.add(key)
                ops.append(op)
        return ops

    @staticmethod
    def _valid_fact(
        item: Any,
        said: str,
        known: Mapping[Tuple[str, str], Tuple[str, Mapping[str, Any]]],
        known_values: set,
        thread_id: str,
    ) -> Optional[ConsolidationOp]:
        if not isinstance(item, Mapping):
            return None
        entity = " ".join(str(item.get("entity") or "").split())[:80]
        attribute = re.sub(
            r"[^a-z0-9_]+", "_", str(item.get("attribute") or "").strip().lower()
        )
        attribute = attribute.strip("_")[:60]
        value = " ".join(str(item.get("value") or "").split())[:300]
        evidence = re.sub(
            r"^(?:-\s*)?(?:user\s*:\s*)?", "", _normalized(item.get("evidence"))
        ).strip(" \"'.")
        if not entity or not attribute or not value or len(evidence) < 3:
            return None
        # The fact has to rest on words the user actually wrote, and those
        # words have to say it.
        if evidence not in said:
            return None
        value_words = set(re.findall(r"[a-z0-9]{3,}", value.casefold()))
        if value_words and not value_words & set(re.findall(r"[a-z0-9]{3,}", evidence)):
            return None
        existing = known.get((_normalized(entity), attribute))
        if existing is not None and _normalized(existing[0]) == _normalized(value):
            return None
        # Already stored, under this entity or another name for it.
        if existing is None and (
            _normalized(value) in known_values
            or (attribute, _normalized(value)) in known_values
        ):
            return None
        record = existing[1] if existing is not None else None
        details: Dict[str, Any] = {
            "entity": entity,
            "entity_type": str(item.get("entity_type") or "").strip()[:40] or None,
            "attribute": attribute,
            "value": value,
            "evidence": str(item.get("evidence") or "")[:300],
            "thread_id": thread_id,
        }
        if existing is not None:
            details["replaces"] = existing[0]
        return ConsolidationOp(
            kind=ADD_FACT,
            memory_type=MemoryType.ENTITY_MEMORY.value,
            target_id=str((record or {}).get("entity_id") or ""),
            reason=(
                "the user said this; memory still had the old value"
                if existing is not None
                else "the user said this and it was not in memory"
            ),
            details=details,
        )

    # -- applying ------------------------------------------------------------

    def apply(
        self,
        plan: ConsolidationPlan,
        *,
        approved_by: str,
        reason: Optional[str] = None,
    ) -> ConsolidationResult:
        """Make a plan's changes. Every one can be undone with :meth:`restore`.

        Each change is journalled (when memory history is on) with the
        approver as writer, ``consolidation:<plan id>`` as source and a label
        that says what it did, such as "Merged duplicates of Ada Lovelace".
        """
        approver = str(approved_by or "").strip()
        if not approver:
            raise ValueError("approved_by is required to apply a consolidation plan")
        if plan.plan_id != ConsolidationPlan.make_id(
            plan.memory_id, plan.user_id, plan.ops
        ):
            raise ValueError("Consolidation plan content does not match its plan_id")
        from ..memory_history import memory_change_context

        applied = {
            MERGE_ENTITIES: 0,
            SUPPRESS_DUPLICATE: 0,
            RETIRE_ATTRIBUTE: 0,
            ADD_FACT: 0,
        }
        errors: List[str] = []
        source = f"consolidation:{plan.plan_id}"
        # Retirements are applied per entity, one write each.
        retirements: Dict[str, List[ConsolidationOp]] = {}
        with memory_change_context(
            actor=approver, source=source, agent_id=self.agent_id
        ):
            for op in plan.ops:
                if op.kind == RETIRE_ATTRIBUTE:
                    retirements.setdefault(op.target_id, []).append(op)
                    continue
                try:
                    with memory_change_context(label=_op_label(op)):
                        self._apply_one(op, plan, approver, reason, source)
                    applied[op.kind] += 1
                except Exception as exc:
                    errors.append(f"{op.kind} {op.target_id}: {exc}")
            for entity_id, ops in retirements.items():
                label = "Retired " + ", ".join(
                    f"{op.details.get('entity')}.{op.details['attribute']}"
                    for op in ops
                )
                try:
                    with memory_change_context(label=label):
                        applied[RETIRE_ATTRIBUTE] += self.entities.retire_attributes(
                            entity_id,
                            memory_id=plan.memory_id,
                            user_id=plan.user_id,
                            retirements=[
                                {
                                    "name": op.details["attribute"],
                                    "reason": op.reason,
                                    "kept": op.details.get("kept"),
                                }
                                for op in ops
                            ],
                            retired_by=plan.plan_id,
                        )
                except Exception as exc:
                    errors.append(f"{RETIRE_ATTRIBUTE} {entity_id}: {exc}")
        return ConsolidationResult(
            plan_id=plan.plan_id,
            approved_by=approver,
            applied=applied,
            errors=tuple(errors),
        )

    def _apply_one(
        self,
        op: ConsolidationOp,
        plan: ConsolidationPlan,
        approver: str,
        reason: Optional[str],
        source: str,
    ) -> None:
        if op.kind == MERGE_ENTITIES:
            self.entities.consolidate_duplicate_entities(
                memory_id=plan.memory_id,
                user_id=plan.user_id,
                apply=True,
                allow_legacy_scope=plan.user_id is None,
                entity_ids=[op.target_id, *op.details["duplicate_entity_ids"]],
                provenance=plan.plan_id,
            )
        elif op.kind == SUPPRESS_DUPLICATE:
            self._suppress(op, plan.plan_id, approver, reason)
        elif op.kind == ADD_FACT:
            self._add_fact(op, plan, source)
        else:
            raise ValueError(f"unknown operation {op.kind}")

    def _suppress(
        self,
        op: ConsolidationOp,
        plan_id: str,
        approver: str,
        reason: Optional[str],
    ) -> None:
        memory_type = MemoryType(op.memory_type)
        row = self.provider.retrieve_by_id(op.target_id, memory_type)
        if not isinstance(row, dict):
            raise ValueError("record no longer exists")
        if is_suppressed(row):
            raise ValueError("record is already suppressed")
        if _flag(row, "pinned") or _flag(row, "verified"):
            raise ValueError("pinned and verified records are kept")
        ok = self.provider.update_by_id(
            op.target_id,
            {
                "retention_state": "suppressed",
                "suppressed_by": plan_id,
                "suppressed_at": utc_now_iso(),
                "suppression_reason": op.reason,
                "suppression_approver": approver[:240],
                "suppression_note": str(reason or "")[:2000],
                "superseded_by": op.details.get("duplicate_of"),
                "suppressed_from": row.get("retention_state") or "active",
            },
            memory_type,
        )
        if ok is False:
            raise ValueError("provider refused the update")

    def _add_fact(
        self, op: ConsolidationOp, plan: ConsolidationPlan, source: str
    ) -> None:
        details = op.details
        if details.get("replaces") is not None and op.target_id:
            self.entities.retire_attributes(
                op.target_id,
                memory_id=plan.memory_id,
                user_id=plan.user_id,
                retirements=[
                    {
                        "name": details["attribute"],
                        "reason": "replaced by what the user said later",
                        "kept": details["attribute"],
                    }
                ],
                retired_by=plan.plan_id,
            )
        self.entities.upsert_entity(
            entity_id=op.target_id or None,
            name=details["entity"],
            entity_type=details.get("entity_type"),
            attributes=[
                {
                    "name": details["attribute"],
                    "value": details["value"],
                    "confidence": 0.7,
                    "source": source,
                }
            ],
            memory_id=plan.memory_id,
            user_id=plan.user_id,
        )

    # -- undoing -------------------------------------------------------------

    def restore(
        self,
        plan_id: str,
        *,
        memory_id: str,
        user_id: Optional[str] = None,
        approved_by: str,
    ) -> Dict[str, int]:
        """Undo one applied plan in this memory and user."""
        approver = str(approved_by or "").strip()
        if not approver:
            raise ValueError("approved_by is required")
        from ..memory_history import memory_change_context

        with memory_change_context(
            actor=approver,
            source=f"consolidation:{plan_id}:restore",
            agent_id=self.agent_id,
            label="Undid memory consolidation",
        ):
            counts = self.entities.restore_consolidation(
                plan_id, memory_id=memory_id, user_id=user_id
            )
            counts["rows"] = 0
            for memory_type in (MemoryType.SUMMARIES, MemoryType.KNOWLEDGE_BASE):
                for row in self._rows(memory_type, user_id):
                    if row.get("suppressed_by") != plan_id or not _row_id(row):
                        continue
                    ok = self.provider.update_by_id(
                        _row_id(row),
                        {
                            "retention_state": _state_before(row),
                            "unsuppressed_by": approver[:240],
                            "unsuppressed_at": utc_now_iso(),
                        },
                        memory_type,
                    )
                    if ok is not False:
                        counts["rows"] += 1
        return counts


__all__ = [
    "ADD_FACT",
    "AUTOMATIC_APPROVER",
    "MERGE_ENTITIES",
    "RETIRE_ATTRIBUTE",
    "SUPPRESS_DUPLICATE",
    "ConsolidationConfig",
    "ConsolidationOp",
    "ConsolidationPlan",
    "ConsolidationResult",
    "MemoryConsolidator",
    "model_for_agent",
    "text_generator",
]
