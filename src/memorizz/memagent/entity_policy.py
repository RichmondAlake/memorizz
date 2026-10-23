# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Where an agent files entity facts: one canonical record for its user, and
whether facts about anyone else are kept.

Without a policy, entity writes find their record by name, so "user", "me"
and the user's real name can become three records of one person, and a host
that wants only its user's profile has no way to say so. An
:class:`EntityPolicy` makes that host authority explicit for every write
path: the ``entity_memory_upsert`` tool, entity capture after a turn, memory
routing, :meth:`MemAgent.capture_entity_facts` and ``remember``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

SUBJECTS = ("all", "self")
#: How capture names the user, and other names that mean the user.
SELF_NAMES = frozenset({"user", "the user", "me", "myself", "self", "current user"})
#: Attributes of the user's record that hold a name the user goes by.
NAME_ATTRIBUTES = frozenset(
    {"name", "first_name", "full_name", "preferred_name", "nickname", "last_name"}
)
#: What a refused write about someone else tells the model.
SELF_ONLY_REFUSAL = (
    "This memory keeps facts about the user only. Store a fact about the user "
    'with name "user"; do not store facts about other people.'
)
_KEY_RE = re.compile(r"[^a-z0-9_.:-]+")


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


@dataclass(frozen=True)
class EntityPolicy:
    """How entity writes are bound. Defaults change nothing.

    ``self_identity_key``: facts about the user land on the one record with
    this identity key (for example ``"authenticated_user"``), whatever name a
    writer gives the user, and a stale or invented entity id never splits it.
    ``subjects``: ``"all"`` keeps facts about named people and things too;
    ``"self"`` keeps only facts about the user: a tool write about anyone else
    is refused (the model is told to store facts about the user as "user"),
    and capture drops facts about anyone else.
    """

    self_identity_key: Optional[str] = None
    subjects: str = "all"

    def __post_init__(self) -> None:
        key = self.self_identity_key
        if key is not None:
            key = _KEY_RE.sub("_", str(key).strip().lower()).strip("_") or None
            if key is not None and len(key) > 160:
                raise ValueError("self_identity_key must be 160 characters or fewer")
        object.__setattr__(self, "self_identity_key", key)
        subjects = str(self.subjects or "all").strip().lower()
        if subjects not in SUBJECTS:
            raise ValueError(f"subjects must be one of {', '.join(SUBJECTS)}")
        if subjects == "self" and key is None:
            raise ValueError('subjects="self" needs a self_identity_key')
        object.__setattr__(self, "subjects", subjects)

    @classmethod
    def from_value(cls, value: Any) -> "EntityPolicy":
        """``None`` is the default policy; a mapping of the fields or an
        :class:`EntityPolicy` sets one."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("entity_policy must be a mapping or EntityPolicy")
        return cls(
            self_identity_key=value.get("self_identity_key"),
            subjects=value.get("subjects") or "all",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"self_identity_key": self.self_identity_key, "subjects": self.subjects}

    @property
    def active(self) -> bool:
        return self.self_identity_key is not None

    @property
    def self_only(self) -> bool:
        return self.subjects == "self"

    # -- reading records ----------------------------------------------------

    def is_self_record(self, record: Mapping[str, Any]) -> bool:
        if not self.active or not isinstance(record, Mapping):
            return False
        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        return str(metadata.get("identity_key") or "") == self.self_identity_key

    def self_record(
        self, records: List[Mapping[str, Any]]
    ) -> Optional[Mapping[str, Any]]:
        return next((row for row in records or [] if self.is_self_record(row)), None)

    def present(self, records: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Records as fact extraction should read them: the user's record
        named ``user`` (the name extraction gives the user), and with
        ``subjects="self"`` nothing else."""
        out: List[Dict[str, Any]] = []
        for record in records or []:
            if self.is_self_record(record):
                out.append({**record, "name": "user"})
            elif not self.self_only:
                out.append(dict(record))
        return out

    # -- binding writes -----------------------------------------------------

    def is_self_name(
        self, name: Any, self_record: Optional[Mapping[str, Any]] = None
    ) -> bool:
        """``name`` is "user", "me" and the like, or a name the user's record
        goes by (its own name or a name attribute such as ``first_name``)."""
        normalized = _normalized(name)
        if not normalized:
            return False
        if normalized in SELF_NAMES:
            return True
        if not self_record:
            return False
        names = {_normalized(self_record.get("name"))}
        for attribute in self_record.get("attributes") or []:
            if (
                isinstance(attribute, Mapping)
                and str(attribute.get("name") or "").casefold() in NAME_ATTRIBUTES
            ):
                value = _normalized(attribute.get("value"))
                names.add(value)
                names.add(value.split(" ")[0] if value else value)
        names.discard("")
        return normalized in names

    def binds(
        self,
        *,
        name: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        self_record: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Whether a write naming ``name`` is about the user."""
        if not self.active:
            return False
        if isinstance(metadata, Mapping) and (
            metadata.get("is_self") is True or metadata.get("subject") == "current_user"
        ):
            return True
        return self.is_self_name(name, self_record)

    def refuses(
        self,
        *,
        name: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        self_record: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Whether a write is about someone other than the user under
        ``subjects="self"``: it is refused, never filed on the user."""
        return self.self_only and not self.binds(
            name=name, metadata=metadata, self_record=self_record
        )


__all__ = [
    "EntityPolicy",
    "NAME_ATTRIBUTES",
    "SELF_NAMES",
    "SELF_ONLY_REFUSAL",
    "SUBJECTS",
]
