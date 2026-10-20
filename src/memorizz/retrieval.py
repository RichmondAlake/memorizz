# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Explicit policy for automatic pre-inference memory retrieval.

Memory partitions describe what an agent may store and expose through tools.
They no longer have to double as an implicit instruction to inject semantic
matches before every model call. ``RetrievalPolicy`` controls only that
automatic recall layer; exact current-thread conversation history is loaded
independently and is never disabled by this policy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

_CONVERSATION_SCOPES = {"memory", "thread", "disabled"}
_KNOWLEDGE_SCOPES = {"memory", "namespace", "disabled"}
_ENTITY_SCOPES = {"memory", "thread", "disabled"}


def _scope(value: Any, *, allowed: set[str], label: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{label} must be one of: {choices}")
    return normalized


def _namespaces(value: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    result = []
    seen = set()
    for item in value:
        namespace = str(item or "").strip()
        if namespace and namespace not in seen:
            result.append(namespace)
            seen.add(namespace)
    return tuple(result)


# ``retention_state`` of conversation rows and summaries a sealed thread
# wrote: recalled inside that thread only.
SEALED_STATE = "sealed"


def sealed_elsewhere(row: Any, thread_id: Optional[str]) -> bool:
    """True for a row a sealed thread wrote, read from any other thread."""
    if not isinstance(row, dict):
        return False
    content = row.get("content") if isinstance(row.get("content"), dict) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    state = (
        row.get("retention_state")
        or content.get("retention_state")
        or metadata.get("retention_state")
    )
    if str(state or "").strip().lower() != SEALED_STATE:
        return False
    row_thread = (
        row.get("thread_id")
        or row.get("conversation_id")
        or content.get("thread_id")
        or content.get("conversation_id")
    )
    return str(row_thread or "") != str(thread_id or "")


def thread_entity_memory_id(memory_id: str, thread_id: str) -> str:
    """The memory scope holding one thread's entity facts (``entity_scope="thread"``).

    Entity rows are scoped by memory and user only, so a thread gets its own
    memory scope. It stays under the 255 characters every provider allows.
    """
    scoped = f"{memory_id}:thread:{thread_id}"
    if len(scoped) <= 255:
        return scoped
    digest = hashlib.sha256(scoped.encode("utf-8")).hexdigest()[:32]
    return f"{str(memory_id)[:200]}:thread:{digest}"


@dataclass(frozen=True)
class RetrievalPolicy:
    """Configure automatic semantic recall without changing storage.

    The defaults preserve the pre-0.5.2 behaviour for compatibility. Safer
    conversational applications can use ``RetrievalPolicy.disabled()``,
    select ``conversation_scope="thread"`` and explicit KB namespaces, or seal
    each thread with ``RetrievalPolicy.sealed()``.

    ``entity_scope`` decides where entity facts live: ``"memory"`` shares
    them across the threads of a memory and user, ``"thread"`` keeps the
    facts a thread writes, and the facts it recalls, inside that thread, and
    ``"disabled"`` stops recalling them automatically (the entity tools still
    work).
    """

    conversation_scope: str = "memory"
    knowledge_base_scope: str = "memory"
    knowledge_base_namespaces: Tuple[str, ...] = ()
    entity_scope: str = "memory"
    candidate_limit: int = 5
    max_items: int = 4
    query_expansion: bool = False
    max_query_variants: int = 2
    dedupe_parent_sources: bool = False
    # Per-agent forgetting overrides (see RetrievalScoring / RetentionConfig);
    # None means "inherit the global MEMORIZZ_* settings".
    scoring: Optional[Dict[str, Any]] = None
    retention: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        for name in ("scoring", "retention"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be a mapping or None")
            object.__setattr__(self, name, dict(value) if value else None)
        object.__setattr__(
            self,
            "conversation_scope",
            _scope(
                self.conversation_scope,
                allowed=_CONVERSATION_SCOPES,
                label="conversation_scope",
            ),
        )
        object.__setattr__(
            self,
            "knowledge_base_scope",
            _scope(
                self.knowledge_base_scope,
                allowed=_KNOWLEDGE_SCOPES,
                label="knowledge_base_scope",
            ),
        )
        object.__setattr__(
            self,
            "knowledge_base_namespaces",
            _namespaces(self.knowledge_base_namespaces),
        )
        object.__setattr__(
            self,
            "entity_scope",
            _scope(self.entity_scope, allowed=_ENTITY_SCOPES, label="entity_scope"),
        )
        if isinstance(self.candidate_limit, bool) or int(self.candidate_limit) < 1:
            raise ValueError("candidate_limit must be a positive integer")
        if isinstance(self.max_items, bool) or int(self.max_items) < 1:
            raise ValueError("max_items must be a positive integer")
        object.__setattr__(self, "candidate_limit", int(self.candidate_limit))
        object.__setattr__(self, "max_items", int(self.max_items))
        object.__setattr__(self, "query_expansion", bool(self.query_expansion))
        if (
            isinstance(self.max_query_variants, bool)
            or int(self.max_query_variants) < 1
        ):
            raise ValueError("max_query_variants must be a positive integer")
        object.__setattr__(self, "max_query_variants", int(self.max_query_variants))
        object.__setattr__(
            self, "dedupe_parent_sources", bool(self.dedupe_parent_sources)
        )

    @classmethod
    def disabled(cls) -> "RetrievalPolicy":
        return cls(
            conversation_scope="disabled",
            knowledge_base_scope="disabled",
        )

    @classmethod
    def sealed(cls, **overrides: Any) -> "RetrievalPolicy":
        """Nothing a user said in one thread reaches another.

        Conversation recall and entity facts stay inside the current thread,
        the semantic cache only answers from the same thread, and the agent's
        persona is read-only. The knowledge base is shared reference material
        and stays on.
        """
        return cls(
            **{"conversation_scope": "thread", "entity_scope": "thread", **overrides}
        )

    @property
    def is_sealed(self) -> bool:
        return self.conversation_scope != "memory" and self.entity_scope != "memory"

    @classmethod
    def from_value(
        cls,
        value: Optional[Union["RetrievalPolicy", Dict[str, Any], str, bool]],
    ) -> "RetrievalPolicy":
        if value is None or value is True:
            return cls()
        if value is False:
            return cls.disabled()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "disabled":
                return cls.disabled()
            if normalized == "thread":
                # Narrows conversation recall only; the knowledge base is
                # shared reference material and stays on.
                return cls(conversation_scope="thread")
            if normalized == "memory":
                return cls()
            if normalized == "sealed":
                return cls.sealed()
            if normalized == "namespace":
                return cls(
                    conversation_scope="disabled",
                    knowledge_base_scope="namespace",
                )
            raise ValueError(
                "retrieval_policy string must be thread, sealed, memory, namespace, "
                "or disabled"
            )
        if isinstance(value, dict):
            data = dict(value)
            generic_scope = data.pop("scope", None)
            if generic_scope is not None:
                base = cls.from_value(str(generic_scope)).to_dict()
                base.update(data)
                data = base
            if "namespaces" in data and "knowledge_base_namespaces" not in data:
                data["knowledge_base_namespaces"] = data.pop("namespaces")
            return cls(**data)
        raise TypeError(
            "retrieval_policy must be RetrievalPolicy, dict, scope string, bool, or None"
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "conversation_scope": self.conversation_scope,
            "knowledge_base_scope": self.knowledge_base_scope,
            "knowledge_base_namespaces": list(self.knowledge_base_namespaces),
        }
        if self.entity_scope != "memory":
            result["entity_scope"] = self.entity_scope
        if self.candidate_limit != 5:
            result["candidate_limit"] = self.candidate_limit
        if self.max_items != 4:
            result["max_items"] = self.max_items
        if self.query_expansion:
            result["query_expansion"] = True
        if self.max_query_variants != 2:
            result["max_query_variants"] = self.max_query_variants
        if self.dedupe_parent_sources:
            result["dedupe_parent_sources"] = True
        if self.scoring:
            result["scoring"] = dict(self.scoring)
        if self.retention:
            result["retention"] = dict(self.retention)
        return result
