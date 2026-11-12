# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Verification: does a stored note still rest on the conversation it cites?

Memory an agent wrote from a conversation is only as good as the
conversation. A message is edited or forgotten, a thread is deleted, a
capture misread a sentence: the note stays, and the model goes on reading it
as a fact. A :class:`NoteVerifier` checks three kinds of note against the
transcript each one names, and **flags** what it cannot support. It deletes
and rewrites nothing.

=================  ==========================================  ==============
Note               What it cites                               How it is checked
=================  ==========================================  ==============
A stated fact      The user's words, kept when consolidation   The quote is still in a
                   added it (``learning_fact_evidence``)       user message: code
A captured fact    The conversation an entity value came from  The value's words are in
                   (the attribute's ``thread_id``)             the conversation: code;
                                                               otherwise the background
                                                               model, a few per pass
A summary          The conversation it summarises              The conversation exists
=================  ==========================================  ==============

A flag is one of ``unsupported`` (the conversation is there and does not say
it), ``source_missing`` (the conversation is gone) and ``source_sealed``
(the conversation was sealed, so the note can no longer be checked). A
summary of a sealed conversation is what sealing is for and is not flagged.

    agent.unverified_notes()                          # open flags
    agent.resolve_unverified_note(flag_id, by="dev")  # the note was fixed
    agent.dismiss_unverified_note(flag_id, by="dev")  # the note is right

A note that was flagged once, open or closed, is not flagged again for the
same value. A captured fact the model confirmed is remembered, so it is asked
about once.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .._time import utc_now_iso
from ..enums.memory_type import MemoryType

logger = logging.getLogger(__name__)

UNVERIFIED_NOTE_RECORD = "learning_unverified_note"
FACT_EVIDENCE_RECORD = "learning_fact_evidence"
VERIFY_STATE_RECORD = "learning_verify_state"
OPEN, RESOLVED, DISMISSED = "open", "resolved", "dismissed"
UNSUPPORTED, SOURCE_MISSING, SOURCE_SEALED = (
    "unsupported",
    "source_missing",
    "source_sealed",
)
#: Captured facts the background model is asked about in one pass.
DEFAULT_MODEL_CHECKS = 3
#: Facts and summaries read in one pass, so a large memory stays bounded.
MAX_NOTES = 400
#: Confirmed facts remembered between passes.
MAX_REMEMBERED = 4000

_VERIFY_PROMPT = """Below is one conversation between a user and an assistant, one message per line.
A fact was recorded from that conversation:

  {entity} . {attribute} = {value}

Does the conversation say this? Reply with JSON only:
{{"supported": true or false, "quote": "<at most 15 words copied exactly from one line, or empty>"}}
Say true only when one of the lines states the fact, and copy the words that state it.

The conversation:
{transcript}
"""


def _quote(text: Any) -> str:
    """A quote as it is compared: case, spacing, a leading "- user:" and the
    quotation marks around it do not count."""
    folded = " ".join(str(text or "").split()).casefold()
    return re.sub(r"^(?:-\s*)?(?:user\s*:\s*)?", "", folded).strip(" \"'.“”‘’")


def _words(text: Any) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", str(text or "").casefold()))


def record_fact_evidence(
    provider: Any,
    agent_id: Optional[str],
    *,
    entity_id: str,
    entity: str,
    attribute: str,
    value: str,
    evidence: str,
    thread_id: Optional[str],
    memory_id: Optional[str],
    user_id: Optional[str],
    plan_id: Optional[str] = None,
) -> None:
    """Keep the user's words a stated fact rests on, so a later pass can
    check they are still in the conversation. Best effort: a fact is stored
    whether or not its evidence could be."""
    if provider is None or not agent_id or not entity_id or not _quote(evidence):
        return
    key = "|".join(
        [str(agent_id), str(entity_id), str(attribute).casefold(), _quote(value)]
    )
    try:
        from .store import LearningControlPlaneStore

        LearningControlPlaneStore(provider, agent_id=str(agent_id))._put(
            {
                "record_type": FACT_EVIDENCE_RECORD,
                "record_id": "evidence-"
                + hashlib.sha256(key.encode()).hexdigest()[:28],
                "entity_id": str(entity_id),
                "entity": str(entity)[:120],
                "attribute": str(attribute)[:80],
                "value": str(value)[:300],
                "evidence": str(evidence)[:300],
                "thread_id": thread_id,
                "memory_id": memory_id,
                "user_id": user_id,
                "plan_id": plan_id,
                "timestamp": utc_now_iso(),
            },
            immutable=False,
        )
    except Exception as exc:
        logger.debug("Fact evidence was not stored: %s", exc)


class NoteVerifier:
    """Check one agent's notes against the conversations they cite. See the
    module docstring."""

    def __init__(self, agent: Any):
        if agent.memory_provider is None:
            raise ValueError("Verification requires a memory provider")
        if not getattr(agent, "agent_id", None):
            raise ValueError("Verification requires an agent_id")
        self.agent = agent
        self.provider = agent.memory_provider
        self.agent_id = str(agent.agent_id)

    # ------------------------------------------------------------------ store

    def _store(self) -> Any:
        from .store import LearningControlPlaneStore

        return LearningControlPlaneStore(self.provider, agent_id=self.agent_id)

    def list(
        self, status: Optional[str] = OPEN, *, memory_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Stored flags of this agent, newest first. ``status`` is "open"
        (the default), "resolved", "dismissed" or None for all."""
        rows = self._store().list_records(
            record_type=UNVERIFIED_NOTE_RECORD, limit=5000
        )
        rows.sort(key=lambda row: str(row.get("flagged_at") or ""), reverse=True)
        return [
            row
            for row in rows
            if (status is None or row.get("status") == status)
            and (memory_id is None or row.get("memory_id") == memory_id)
        ]

    def _close(
        self, flag_id: str, status: str, by: str, note: Optional[str]
    ) -> Dict[str, Any]:
        who = str(by or "").strip()
        if not who:
            raise ValueError("a name is required to close an unverified-note flag")
        record = self._store().get(flag_id)
        if not record or record.get("record_type") != UNVERIFIED_NOTE_RECORD:
            raise ValueError(f"No unverified-note flag {flag_id!r}")
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
        """Close a flag because the note was corrected or removed."""
        return self._close(flag_id, RESOLVED, by, note)

    def dismiss(
        self, flag_id: str, *, by: str, note: Optional[str] = None
    ) -> Dict[str, Any]:
        """Close a flag because the note is right as it stands."""
        return self._close(flag_id, DISMISSED, by, note)

    def _state_id(self) -> str:
        return f"verify-state-{self.agent_id}"[:120]

    def _remembered(self) -> List[str]:
        record = self._store().lookup(self._state_id()) or {}
        return [str(item) for item in record.get("confirmed") or []]

    def _remember(self, keys: List[str]) -> None:
        if not keys:
            return
        kept = list(dict.fromkeys([*self._remembered(), *keys]))[-MAX_REMEMBERED:]
        self._store()._put(
            {
                "record_type": VERIFY_STATE_RECORD,
                "record_id": self._state_id(),
                "confirmed": kept,
                "timestamp": utc_now_iso(),
            },
            immutable=False,
        )

    # ------------------------------------------------------------ transcripts

    def _conversations(
        self, memory_id: str, user_id: Optional[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Each conversation of a memory: the messages that can still be
        read (``said`` is the user's, ``text`` everyone's) and whether the
        rest was sealed."""
        from ..memagent.utils.context_dedup import is_suppressed
        from ..retrieval import SEALED_STATE
        from .consolidation import _accepts, _row_text, _row_thread

        reader = getattr(
            self.provider, "retrieve_conversation_history_ordered_by_timestamp", None
        )
        if not callable(reader):
            raise NotImplementedError("the provider cannot list a conversation")
        kwargs: Dict[str, Any] = {"memory_id": memory_id}
        if _accepts(reader, "include_embedding"):
            kwargs["include_embedding"] = False
        threads: Dict[str, Dict[str, Any]] = {}
        for row in reader(**kwargs) or []:
            if not isinstance(row, dict):
                continue
            if user_id and row.get("user_id") not in (None, "", user_id):
                continue
            thread = threads.setdefault(
                _row_thread(row), {"lines": [], "all": [], "live": 0, "sealed": 0}
            )
            if str(row.get("retention_state") or "").lower() == SEALED_STATE:
                thread["sealed"] += 1
                continue
            if is_suppressed(row):
                continue
            thread["live"] += 1
            role = str(row.get("role") or "").lower()
            line = " ".join(_row_text(row).split())
            if role == "user":
                thread["lines"].append(line)
            if role in {"user", "assistant"}:
                thread["all"].append(f"{role}: {line}")
        for thread in threads.values():
            thread["said"] = " ".join(line.casefold() for line in thread["lines"])
            thread["text"] = " ".join(line.casefold() for line in thread["all"])
        return threads

    @staticmethod
    def _source_state(thread: Optional[Mapping[str, Any]]) -> Optional[str]:
        """Why a conversation cannot be read, or None when it can."""
        if not thread or not (thread["live"] or thread["sealed"]):
            return SOURCE_MISSING
        if not thread["live"]:
            return SOURCE_SEALED
        return None

    # ------------------------------------------------------------------ check

    def check(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
        *,
        store: bool = True,
        model_checks: int = DEFAULT_MODEL_CHECKS,
    ) -> Dict[str, Any]:
        """Check the notes of one memory. Returns ``{"checked", "supported",
        "model_checked", "left", "findings", "opened"}``: every note that
        failed (``findings``) and the flags this call stored (``opened``).
        With ``store=False`` nothing is written."""
        from ..long_term.semantic.entity_memory.entity_memory import (
            EntityMemory,
            source_level,
        )
        from .consolidation import _parse_json, _row_thread, text_generator

        threads = self._conversations(memory_id, user_id)
        entities = EntityMemory(self.provider)
        records = {
            str(record.get("entity_id")): record
            for record in entities.list_entities(memory_id=memory_id, user_id=user_id)
        }
        result: Dict[str, Any] = {
            "checked": 0,
            "supported": 0,
            "model_checked": 0,
            "left": 0,
            "findings": [],
            "opened": [],
        }
        findings: List[Dict[str, Any]] = result["findings"]

        def current(record: Mapping[str, Any], attribute: str) -> Optional[Dict]:
            for item in record.get("attributes") or []:
                if (
                    isinstance(item, dict)
                    and str(item.get("name") or "").casefold() == attribute.casefold()
                ):
                    return item
            return None

        def fail(kind: str, reason: str, **target: Any) -> None:
            findings.append(
                {
                    "kind": kind,
                    "reason": reason,
                    "memory_id": memory_id,
                    "user_id": user_id,
                    **target,
                }
            )

        # 1. Stated facts: the user's words were kept, so code can check them.
        quoted: set = set()
        evidence_rows = self._store().list_records(
            record_type=FACT_EVIDENCE_RECORD, memory_id=memory_id, limit=MAX_NOTES
        )
        for row in evidence_rows:
            record = records.get(str(row.get("entity_id")))
            held = current(record or {}, str(row.get("attribute") or ""))
            if held is None or _quote(held.get("value")) != _quote(row.get("value")):
                continue  # No longer what memory says: nothing to verify.
            quoted.add((str(row.get("entity_id")), str(row["attribute"]).casefold()))
            result["checked"] += 1
            target = {
                "target_type": "stated_fact",
                "target_store": MemoryType.ENTITY_MEMORY.value,
                "target_id": str(row.get("entity_id")),
                "entity": (record or {}).get("name") or row.get("entity"),
                "attribute": row.get("attribute"),
                "value": row.get("value"),
                "evidence": row.get("evidence"),
                "thread_id": row.get("thread_id"),
            }
            thread = threads.get(str(row.get("thread_id") or ""))
            state = self._source_state(thread)
            if state == SOURCE_MISSING:
                fail(state, "the conversation it was stated in is gone", **target)
            elif state == SOURCE_SEALED:
                fail(state, "the conversation it was stated in is sealed", **target)
            elif _quote(row.get("evidence")) in thread["said"]:  # type: ignore[index]
                result["supported"] += 1
            else:
                fail(
                    UNSUPPORTED,
                    "the quoted words are no longer in the conversation",
                    **target,
                )

        # 2. Captured facts: a conversation is named, the words are not kept.
        remembered = set(self._remembered())
        confirmed: List[str] = []
        budget = max(0, int(model_checks))
        generate = text_generator(getattr(self.agent, "background_model", None))
        for entity_id, record in list(records.items())[:MAX_NOTES]:
            for item in record.get("attributes") or []:
                if not isinstance(item, dict) or not item.get("thread_id"):
                    continue
                if (entity_id, str(item.get("name") or "").casefold()) in quoted:
                    continue
                if source_level(item) >= 3:
                    continue  # The host's own write is not a note to check.
                result["checked"] += 1
                thread_id = str(item["thread_id"])
                target = {
                    "target_type": "captured_fact",
                    "target_store": MemoryType.ENTITY_MEMORY.value,
                    "target_id": entity_id,
                    "entity": record.get("name"),
                    "attribute": item.get("name"),
                    "value": item.get("value"),
                    "thread_id": thread_id,
                }
                thread = threads.get(thread_id)
                state = self._source_state(thread)
                if state == SOURCE_MISSING:
                    fail(state, "the conversation it came from is gone", **target)
                    continue
                if state == SOURCE_SEALED:
                    fail(state, "the conversation it came from is sealed", **target)
                    continue
                wanted = _words(item.get("value"))
                if not wanted or wanted <= _words(thread["text"]):  # type: ignore[index]
                    result["supported"] += 1
                    continue
                key = hashlib.sha256(
                    "|".join(
                        [entity_id, str(item.get("name")), _quote(item.get("value"))]
                        + [thread_id]
                    ).encode()
                ).hexdigest()[:24]
                if key in remembered:
                    result["supported"] += 1
                    continue
                if generate is None or budget <= 0:
                    result["left"] += 1
                    continue
                budget -= 1
                result["model_checked"] += 1
                transcript = "\n".join(f"- {line[:1200]}" for line in thread["all"])  # type: ignore[index]
                try:
                    answer = _parse_json(
                        generate(
                            _VERIFY_PROMPT.format(
                                entity=record.get("name"),
                                attribute=item.get("name"),
                                value=item.get("value"),
                                transcript=transcript[-12000:],
                            )
                        )
                    )
                except Exception as exc:
                    logger.warning("Verification call failed: %s", exc)
                    result["left"] += 1
                    continue
                said = _quote((answer or {}).get("quote"))
                # The model's yes counts only with words from the conversation.
                if (
                    answer
                    and answer.get("supported") is True
                    and len(said) >= 3
                    and said in thread["text"]  # type: ignore[index]
                ):
                    result["supported"] += 1
                    confirmed.append(key)
                else:
                    fail(
                        UNSUPPORTED,
                        "nothing in that conversation says it",
                        **target,
                    )

        # 3. Summaries: the conversation they summarise has to exist.
        try:
            summaries = self.provider.list_all(MemoryType.SUMMARIES) or []
        except Exception:
            summaries = []
        from ..memagent.utils.context_dedup import is_suppressed

        for row in [r for r in summaries if isinstance(r, dict)][:MAX_NOTES]:
            if row.get("memory_id") != memory_id or is_suppressed(row):
                continue
            if user_id and row.get("user_id") not in (None, "", user_id):
                continue
            thread_id = _row_thread(row)
            if not thread_id:
                continue
            result["checked"] += 1
            if self._source_state(threads.get(thread_id)) == SOURCE_MISSING:
                fail(
                    SOURCE_MISSING,
                    "the conversation it summarises is gone",
                    target_type="summary",
                    target_store=MemoryType.SUMMARIES.value,
                    target_id=str(row.get("_id") or row.get("id") or ""),
                    value=" ".join(
                        str(
                            row.get("summary_content")
                            or row.get("text")
                            or row.get("content")
                            or ""
                        ).split()
                    )[:160],
                    thread_id=thread_id,
                )
            else:
                result["supported"] += 1

        if store:
            self._remember(confirmed)
            result["opened"] = self._open(findings)
        return result

    def _open(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        store = self._store()
        opened: List[Dict[str, Any]] = []
        for finding in findings:
            key = "|".join(
                [
                    self.agent_id,
                    str(finding.get("target_type")),
                    str(finding.get("target_id")),
                    str(finding.get("attribute") or "").casefold(),
                    _quote(finding.get("value")),
                ]
            )
            flag_id = "unverified-" + hashlib.sha256(key.encode()).hexdigest()[:28]
            if store.lookup(flag_id):
                continue
            opened.append(
                store._put(
                    {
                        "record_type": UNVERIFIED_NOTE_RECORD,
                        "record_id": flag_id,
                        "flag_id": flag_id,
                        "status": OPEN,
                        "flagged_at": utc_now_iso(),
                        **finding,
                    },
                    immutable=False,
                )
            )
        return opened


def describe_finding(finding: Mapping[str, Any]) -> Tuple[str, str]:
    """A finding as a person reads it: what, and why."""
    if finding.get("target_type") == "summary":
        what = f"A summary: “{finding.get('value')}”"
    else:
        what = (
            f"{finding.get('entity')}.{finding.get('attribute')} = "
            f"{finding.get('value')}"
        )
    why = str(finding.get("reason") or "")
    if finding.get("evidence"):
        why += f" (it quoted “{finding['evidence']}”)"
    return what, why[:1].upper() + why[1:]


__all__ = [
    "DEFAULT_MODEL_CHECKS",
    "FACT_EVIDENCE_RECORD",
    "NoteVerifier",
    "SOURCE_MISSING",
    "SOURCE_SEALED",
    "UNSUPPORTED",
    "UNVERIFIED_NOTE_RECORD",
    "describe_finding",
    "record_fact_evidence",
]
