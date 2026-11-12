# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Propagation: when a fact changes, find what still says the old thing.

Entity memory supersedes a value and keeps the old one in its history, so
the *record* is right. A skill distilled last month, or a note the agent
wrote, can still carry the old value in its text, and the model reads that
text as an instruction or as a fact. Nothing here rewrites anything. When a
value is superseded, the skills of the agent and the notes in the same
memory that still mention the old value are **flagged** for a person:

    agent.stale_references()                         # open flags
    agent.resolve_stale_reference(flag_id, by="dev") # the text was fixed
    agent.dismiss_stale_reference(flag_id, by="dev") # not about this fact
    agent.check_stale_references()                   # sweep what is stored

Matching is deliberately narrow. The old value has to appear as a whole
phrase. A name-like value ("Leeds", "Nadia Hussain", "Drop") is enough on
its own, at low confidence. Anything else, a number, an amount or ordinary
words, also needs the entity (its name or an alias) or the attribute named
in the same text. Values under four characters are never matched.
"""

from __future__ import annotations

import hashlib
import logging
import re
import weakref
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .._time import utc_now_iso
from ..enums.memory_type import MemoryType

logger = logging.getLogger(__name__)

STALE_REFERENCE_RECORD = "learning_stale_reference"
OPEN, RESOLVED, DISMISSED = "open", "resolved", "dismissed"
#: The shortest old value that is ever looked for.
MIN_VALUE_CHARS = 4
#: Rows of one store read for one supersession.
MAX_ROWS = 2000
#: Stores whose text a model is later shown as a note.
NOTE_STORES = (MemoryType.KNOWLEDGE_BASE, MemoryType.SUMMARIES)
_SKILL_SKIP = frozenset({"deprecated", "archived", "rejected"})


def _phrase(value: Any) -> Optional["re.Pattern[str]"]:
    """A pattern for ``value`` as a whole phrase: not inside a longer word,
    and not inside a longer number ("2,500" does not match "2,500,000")."""
    words = str(value or "").split()
    if not words:
        return None
    body = r"\s+".join(re.escape(word) for word in words)
    return re.compile(
        r"(?<![A-Za-z0-9])(?<!\d[,.])" + body + r"(?![A-Za-z0-9])(?![,.]\d)"
    )


def _name_like(value: str) -> bool:
    """True for a value that reads as a name: every word capitalised, such
    as "Leeds", "Nadia Hussain" or "Drop". Matched with its capitals."""
    words = value.split()
    return bool(words) and all(
        word[:1].isupper() and any(ch.isalpha() for ch in word) for word in words
    )


def mentions(
    text: str,
    old_value: str,
    *,
    entity_terms: Sequence[str] = (),
    attribute: str = "",
) -> Optional[Dict[str, str]]:
    """Whether ``text`` still states ``old_value``; None when it does not.

    Returns ``{"excerpt", "confidence", "matched_on"}``. ``matched_on`` is
    "entity" or "attribute" when the text also names what the value belongs
    to (high confidence), or "value" for a name-like value found on its own
    (low confidence).
    """
    value = " ".join(str(old_value or "").split())
    if len(value) < MIN_VALUE_CHARS or not re.search(r"[A-Za-z0-9]{3,}", value):
        return None
    name_like = _name_like(value)
    pattern = _phrase(value)
    if pattern is None:
        return None
    if not name_like:
        pattern = re.compile(pattern.pattern, re.IGNORECASE)
    found = pattern.search(text or "")
    if found is None:
        return None
    lowered = (text or "").casefold()
    matched_on = ""
    for term in entity_terms:
        term = " ".join(str(term or "").split())
        if len(term) < 3 or term.casefold() == value.casefold():
            continue
        named = _phrase(term)
        if named is not None and re.search(named.pattern, text or "", re.IGNORECASE):
            matched_on = "entity"
            break
    if not matched_on:
        spoken = " ".join(re.findall(r"[a-z0-9]+", str(attribute or "").casefold()))
        if len(spoken) >= 5 and spoken in " ".join(re.findall(r"[a-z0-9]+", lowered)):
            matched_on = "attribute"
    if not matched_on and not name_like:
        return None
    start, end = max(0, found.start() - 70), min(len(text), found.end() + 70)
    return {
        "excerpt": " ".join(text[start:end].split()),
        "confidence": "high" if matched_on else "low",
        "matched_on": matched_on or "value",
    }


def _text_of(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    parts: List[str] = []
    for key in fields:
        value = row.get(key)
        if isinstance(value, Mapping):
            value = value.get("content") or value.get("text")
        if isinstance(value, (list, tuple)):
            value = "\n".join(str(item) for item in value)
        if value is not None and str(value).strip():
            parts.append(str(value))
    return "\n".join(parts)


class SupersessionPropagator:
    """Flag the skills and notes of one agent that still state a value
    entity memory has superseded. See the module docstring."""

    def __init__(self, agent: Any):
        if agent.memory_provider is None:
            raise ValueError("Propagation requires a memory provider")
        if not getattr(agent, "agent_id", None):
            raise ValueError("Propagation requires an agent_id")
        self.provider = agent.memory_provider
        self.agent_id = str(agent.agent_id)

    # ------------------------------------------------------------------ store

    def _store(self) -> Any:
        from .store import LearningControlPlaneStore

        return LearningControlPlaneStore(self.provider, agent_id=self.agent_id)

    def list(
        self,
        status: Optional[str] = OPEN,
        *,
        memory_id: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Stored flags of this agent, newest first. ``status`` is "open"
        (the default), "resolved", "dismissed" or None for all."""
        rows = self._store().list_records(
            record_type=STALE_REFERENCE_RECORD, limit=5000
        )
        rows.sort(key=lambda row: str(row.get("flagged_at") or ""), reverse=True)
        return [
            row
            for row in rows
            if (status is None or row.get("status") == status)
            and (memory_id is None or row.get("memory_id") == memory_id)
            and (target_type is None or row.get("target_type") == target_type)
        ]

    def _close(
        self, flag_id: str, status: str, by: str, note: Optional[str]
    ) -> Dict[str, Any]:
        who = str(by or "").strip()
        if not who:
            raise ValueError("a name is required to close a stale-reference flag")
        record = self._store().get(flag_id)
        if not record or record.get("record_type") != STALE_REFERENCE_RECORD:
            raise ValueError(f"No stale-reference flag {flag_id!r}")
        record.update(
            status=status,
            closed_by=who[:240],
            closed_at=utc_now_iso(),
            note=str(note or "")[:2000] or None,
        )
        return self._store()._put(record, immutable=False)

    def resolve(
        self, flag_id: str, *, by: str, note: Optional[str] = None
    ) -> Dict[str, Any]:
        """Close a flag because the skill or note was brought up to date."""
        return self._close(flag_id, RESOLVED, by, note)

    def dismiss(
        self, flag_id: str, *, by: str, note: Optional[str] = None
    ) -> Dict[str, Any]:
        """Close a flag because the text was not about this fact. The same
        record is not flagged again for the same old value."""
        return self._close(flag_id, DISMISSED, by, note)

    # --------------------------------------------------------------- flagging

    def _targets(self, record: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """The skills of this agent and the notes in the entity's memory."""
        memory_id, user_id = record.get("memory_id"), record.get("user_id")
        targets: List[Dict[str, Any]] = []
        for row in self._rows(MemoryType.SKILLBOX):
            if str(row.get("agent_id") or "") != self.agent_id:
                continue
            if str(row.get("status") or "").casefold() in _SKILL_SKIP:
                continue
            if user_id and row.get("user_id") and row.get("user_id") != user_id:
                continue
            targets.append(
                {
                    "target_type": "skill",
                    "target_store": MemoryType.SKILLBOX.value,
                    "target_id": str(row.get("skill_id") or row.get("_id") or ""),
                    "target_name": str(row.get("name") or ""),
                    "text": _text_of(
                        row, ("description", "content", "preconditions", "queries")
                    ),
                }
            )
        if not memory_id:
            return targets
        for store in NOTE_STORES:
            for row in self._rows(store):
                if row.get("memory_id") != memory_id:
                    continue
                if user_id and row.get("user_id") and row.get("user_id") != user_id:
                    continue
                if str(row.get("retention_state") or "").casefold() == "suppressed":
                    continue
                text = _text_of(row, ("text", "summary_content", "content", "excerpt"))
                targets.append(
                    {
                        "target_type": "note",
                        "target_store": store.value,
                        "target_id": str(row.get("_id") or row.get("id") or ""),
                        "target_name": " ".join(text.split())[:80],
                        "text": text,
                    }
                )
        return targets

    def _rows(self, memory_type: MemoryType) -> List[Dict[str, Any]]:
        try:
            rows = self.provider.list_all(memory_type) or []
        except Exception as exc:
            logger.debug("Propagation could not read %s: %s", memory_type, exc)
            return []
        return [row for row in rows if isinstance(row, dict)][:MAX_ROWS]

    def flag(
        self,
        record: Mapping[str, Any],
        changes: Sequence[Mapping[str, Any]],
        *,
        preview: bool = False,
    ) -> List[Dict[str, Any]]:
        """Flag what still states the old value of each change to ``record``.

        ``changes`` is ``[{"attribute", "old_value", "new_value", "at"}]``.
        Returns the flags this call opened. A record already flagged for the
        same old value (open or closed) is left as it is. ``preview`` returns
        the flags it would open and stores nothing.
        """
        changes = [
            change
            for change in changes or []
            if len(" ".join(str(change.get("old_value") or "").split()))
            >= MIN_VALUE_CHARS
            # "Leeds" becoming "Leeds office" leaves text saying "Leeds" true.
            and _phrase(change.get("old_value")) is not None
            and not re.search(
                _phrase(change.get("old_value")).pattern,  # type: ignore[union-attr]
                str(change.get("new_value") or ""),
                re.IGNORECASE,
            )
        ]
        if not changes:
            return []
        from ..long_term.semantic.entity_memory.entity_memory import entity_aliases

        entity = str(record.get("name") or "")
        terms = [entity, *entity_aliases(record)]
        store = self._store()
        opened: List[Dict[str, Any]] = []
        targets = self._targets(record)
        for change in changes:
            for target in targets:
                if not target["target_id"]:
                    continue
                found = mentions(
                    target["text"],
                    str(change["old_value"]),
                    entity_terms=terms,
                    attribute=str(change.get("attribute") or ""),
                )
                if found is None:
                    continue
                key = "|".join(
                    [
                        self.agent_id,
                        target["target_store"],
                        target["target_id"],
                        str(record.get("entity_id") or ""),
                        str(change.get("attribute") or "").casefold(),
                        " ".join(str(change["old_value"]).casefold().split()),
                    ]
                )
                flag_id = "stale-" + hashlib.sha256(key.encode()).hexdigest()[:28]
                if store.lookup(flag_id):
                    continue
                flag = {
                    "record_type": STALE_REFERENCE_RECORD,
                    "record_id": flag_id,
                    "flag_id": flag_id,
                    "status": OPEN,
                    "memory_id": record.get("memory_id"),
                    "user_id": record.get("user_id"),
                    "target_type": target["target_type"],
                    "target_store": target["target_store"],
                    "target_id": target["target_id"],
                    "target_name": target["target_name"],
                    "entity_id": record.get("entity_id"),
                    "entity": entity,
                    "attribute": change.get("attribute"),
                    "old_value": change["old_value"],
                    "new_value": change.get("new_value"),
                    "superseded_at": change.get("at"),
                    "flagged_at": utc_now_iso(),
                    **found,
                }
                opened.append(flag if preview else store._put(flag, immutable=False))
        return opened

    def sweep(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        preview: bool = False,
    ) -> List[Dict[str, Any]]:
        """Check what is stored now: for every entity in a memory, flag what
        still states a value it has since replaced. Returns the new flags;
        ``preview`` returns them without storing any."""
        from ..long_term.semantic.entity_memory.entity_memory import (
            EntityMemory,
            entity_attribute_timeline,
        )

        opened: List[Dict[str, Any]] = []
        entities = EntityMemory(self.provider)
        for record in entities.list_entities(memory_id=memory_id, user_id=user_id):
            changes: List[Dict[str, Any]] = []
            for attribute in record.get("attributes") or []:
                if not isinstance(attribute, dict) or not attribute.get("name"):
                    continue
                current = str(attribute.get("value") or "")
                seen = {" ".join(current.casefold().split())}
                rows = entity_attribute_timeline(record, attribute["name"])
                for row in rows:
                    old = str(row.get("value") or "")
                    folded = " ".join(old.casefold().split())
                    if row.get("current") or not row.get("applied") or folded in seen:
                        continue
                    seen.add(folded)
                    changes.append(
                        {
                            "attribute": attribute["name"],
                            "old_value": old,
                            "new_value": current,
                            "at": row.get("superseded_at") or row.get("valid_to"),
                        }
                    )
            opened.extend(self.flag(record, changes, preview=preview))
        return opened

    def close_fixed(
        self, *, by: str, memory_id: Optional[str] = None, preview: bool = False
    ) -> List[Dict[str, Any]]:
        """Resolve the open flags whose skill or note no longer states the
        old value, or no longer exists: someone fixed the text. Returns the
        flags closed (with ``preview``, the ones that would be)."""
        open_flags = self.list(OPEN, memory_id=memory_id)
        if not open_flags:
            return []
        texts: Dict[tuple, str] = {}
        for store in (MemoryType.SKILLBOX, *NOTE_STORES):
            skill = store is MemoryType.SKILLBOX
            for row in self._rows(store):
                row_id = str(
                    (row.get("skill_id") if skill else None)
                    or row.get("_id")
                    or row.get("id")
                    or ""
                )
                live = (
                    str(row.get("status") or "").casefold() not in _SKILL_SKIP
                    if skill
                    else str(row.get("retention_state") or "").casefold()
                    != "suppressed"
                )
                if row_id and live:
                    texts[(store.value, row_id)] = _text_of(
                        row,
                        ("description", "content", "preconditions", "queries")
                        if skill
                        else ("text", "summary_content", "content", "excerpt"),
                    )
        closed: List[Dict[str, Any]] = []
        for flag in open_flags:
            text = texts.get(
                (str(flag.get("target_store")), str(flag.get("target_id")))
            )
            pattern = _phrase(flag.get("old_value"))
            if text is not None and pattern is not None:
                flags = 0 if _name_like(str(flag.get("old_value"))) else re.IGNORECASE
                if re.search(pattern.pattern, text, flags):
                    continue
            closed.append(
                flag
                if preview
                else self.resolve(
                    str(flag["flag_id"]),
                    by=by,
                    note="the text no longer states the old value",
                )
            )
        return closed


def watch_supersessions(agent: Any) -> None:
    """Have ``agent`` flag stale references whenever an entity in one of its
    memories has a value superseded, whoever made the write.

    Holds the agent weakly. Set ``agent.propagate_supersessions = False`` to
    switch it off.
    """
    provider = getattr(agent, "memory_provider", None)
    if provider is None or not getattr(agent, "agent_id", None):
        return
    from ..long_term.semantic.entity_memory.entity_memory import (
        add_supersession_listener,
    )

    reference = weakref.ref(agent)
    agent_id = str(agent.agent_id)
    marker = f"_memorizz_supersession_watch_{agent_id}"
    if getattr(provider, marker, None) is not None and getattr(provider, marker)():
        # Another live instance of this agent already listens here.
        return

    def listener(record: Dict[str, Any], changes: List[Dict[str, Any]]) -> Any:
        live = reference()
        if live is None:
            return False
        if getattr(live, "propagate_supersessions", True) is False:
            return None
        memory_ids = {str(item) for item in getattr(live, "memory_ids", None) or []}
        current = getattr(live, "memory_id", None)
        if current:
            memory_ids.add(str(current))
        if str(record.get("memory_id") or "") not in memory_ids:
            return None
        SupersessionPropagator(live).flag(record, changes)
        return None

    try:
        setattr(provider, marker, reference)
    except Exception:
        return
    add_supersession_listener(provider, listener)


__all__ = [
    "STALE_REFERENCE_RECORD",
    "SupersessionPropagator",
    "mentions",
    "watch_supersessions",
]
