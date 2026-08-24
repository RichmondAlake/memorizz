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

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

_CONVERSATION_SCOPES = {"memory", "thread", "disabled"}
_KNOWLEDGE_SCOPES = {"memory", "namespace", "disabled"}


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


@dataclass(frozen=True)
class RetrievalPolicy:
    """Configure automatic semantic recall without changing storage.

    The defaults preserve the pre-0.5.2 behaviour for compatibility. Safer
    conversational applications can use ``RetrievalPolicy.disabled()`` or
    select ``conversation_scope="thread"`` and explicit KB namespaces.
    """

    conversation_scope: str = "memory"
    knowledge_base_scope: str = "memory"
    knowledge_base_namespaces: Tuple[str, ...] = ()
    candidate_limit: int = 5
    max_items: int = 4
    query_expansion: bool = False
    max_query_variants: int = 2
    dedupe_parent_sources: bool = False

    def __post_init__(self) -> None:
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
                return cls(
                    conversation_scope="thread",
                    knowledge_base_scope="disabled",
                )
            if normalized == "memory":
                return cls()
            if normalized == "namespace":
                return cls(
                    conversation_scope="disabled",
                    knowledge_base_scope="namespace",
                )
            raise ValueError(
                "retrieval_policy string must be thread, memory, namespace, or disabled"
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
        return result
