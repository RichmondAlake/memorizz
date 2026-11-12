# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Compaction: keep the vectors of one memory in step with its text.

Retrieval finds a record by its vector. After a cycle has merged entities,
replaced values and retired facts, the text is right; this pass makes sure
the vectors are too, and changes no text at all.

What :meth:`Compactor.run` does, for one memory and user:

1. **Entities.** A live entity record is embedded again when it changed
   since ``since`` (the start of the cycle), or when its vector is missing,
   covers only part of it, or is the wrong size for the current embedder. An
   entity's vector is built from its name, type and *current* values;
   earlier and retired values are never part of it, so a search for the old
   value stops finding the record. (A write already embeds a record whose
   text changed. This pass makes that certain after merges and retirements,
   and repairs vectors that failed or predate a change of embedder.)
2. **Notes.** A live knowledge-base note or summary in the memory whose
   vector is missing or the wrong size is embedded again from its text.
3. **Suppressed rows** (only with ``drop_suppressed_vectors``). The vector of
   a suppressed note, or of an entity record a merge replaced, is removed, so
   nothing can match it. The row and its text stay. A row that is restored
   later has no vector until step 1 or 2 of the next pass gives it one.

Nothing is deleted and no text is rewritten, so the pass loses no content:
a vector can always be computed again from the text it came from. A
provider that builds its own vectors is left alone.

The length of an attribute's history is bounded where it is written (the
newest 20 earlier values, and one entry per value that was offered and not
applied), so there is nothing here to fold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..enums.memory_type import MemoryType

logger = logging.getLogger(__name__)

#: Stores whose text a model is later shown as a note.
NOTE_STORES = (MemoryType.KNOWLEDGE_BASE, MemoryType.SUMMARIES)
#: Records read from one store in one pass.
MAX_ROWS = 2000
#: Records embedded in one pass, so a first run on a large memory is bounded.
MAX_EMBEDS = 200


class Compactor:
    """Refresh the vectors of one agent's memory. See the module docstring."""

    def __init__(self, agent: Any):
        if agent.memory_provider is None:
            raise ValueError("Compaction requires a memory provider")
        self.provider = agent.memory_provider

    @staticmethod
    def _dimensions() -> Optional[int]:
        try:
            from ..embeddings import get_embedding_dimensions

            return int(get_embedding_dimensions())
        except Exception:
            return None

    @staticmethod
    def _epoch(value: Any) -> Optional[float]:
        if not value:
            return None
        try:
            from .._time import to_epoch

            return float(to_epoch(value))
        except Exception:
            return None

    def run(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
        *,
        since: Any = None,
        store: bool = True,
        drop_suppressed_vectors: bool = False,
    ) -> Dict[str, Any]:
        """Refresh one memory's vectors. Returns counts (``entities``,
        ``notes``, ``dropped``, ``left``) and one line per record in
        ``items``. With ``store=False`` nothing is embedded or written and
        the counts say what would be."""
        from ..long_term.semantic.entity_memory.entity_memory import EntityMemory
        from ..memagent.utils.context_dedup import is_suppressed
        from ..memory_provider.base import provider_manages_embeddings
        from .consolidation import _row_text

        result: Dict[str, Any] = {
            "entities": 0,
            "notes": 0,
            "dropped": 0,
            "left": 0,
            "items": [],
            "skipped": None,
        }
        if provider_manages_embeddings(self.provider):
            result["skipped"] = "The memory provider builds its own vectors."
            return result
        size = self._dimensions()
        cutoff = self._epoch(since)
        budget = MAX_EMBEDS
        entities = EntityMemory(self.provider)
        rows = entities.list_entities(
            memory_id=memory_id, user_id=user_id, include_superseded=True
        )[:MAX_ROWS]
        # A listing that carries no vectors at all cannot say which are missing.
        sees_vectors = any(row.get("embedding") for row in rows)

        def why(row: Dict[str, Any], changed: bool) -> Optional[str]:
            vector = row.get("embedding")
            if not vector:
                return "it had no vector" if sees_vectors else None
            if size and len(vector) != size:
                return f"its vector was {len(vector)} wide, the embedder is {size}"
            if changed:
                return "it changed in this cycle"
            return None

        for row in rows:
            name = str(row.get("name") or row.get("entity_id") or "")
            if entities._is_superseded(row):
                if drop_suppressed_vectors and row.get("embedding"):
                    result["dropped"] += 1
                    result["items"].append(
                        ("drop_vector", f"Removed the vector of {name}", "merged away")
                    )
                    if store:
                        payload = {k: v for k, v in row.items() if k != "score"}
                        payload["embedding"] = None
                        self.provider.store(
                            data=payload, memory_store_type=MemoryType.ENTITY_MEMORY
                        )
                continue
            stamp = self._epoch(row.get("updated_at"))
            changed = bool(cutoff and stamp and stamp >= cutoff)
            reason = why(row, changed)
            if reason is None and entities.embedding_report(row)["status"] != "full":
                reason = "its vector covered only part of it"
            if reason is None:
                continue
            if budget <= 0:
                result["left"] += 1
                continue
            budget -= 1
            result["entities"] += 1
            result["items"].append(("re_embed", f"Embedded {name} again", reason))
            if store:
                payload = {k: v for k, v in row.items() if k != "score"}
                try:
                    entities._embed_record(payload)
                    self.provider.store(
                        data=payload, memory_store_type=MemoryType.ENTITY_MEMORY
                    )
                except Exception as exc:
                    logger.warning("Entity %s was not embedded again: %s", name, exc)
                    result["entities"] -= 1
                    result["items"].pop()

        for note_store in NOTE_STORES:
            try:
                notes = [
                    row
                    for row in (self.provider.list_all(note_store) or [])
                    if isinstance(row, dict) and row.get("memory_id") == memory_id
                ][:MAX_ROWS]
            except Exception as exc:
                logger.debug("Compaction could not read %s: %s", note_store, exc)
                continue
            if user_id:
                notes = [
                    row for row in notes if row.get("user_id") in (None, "", user_id)
                ]
            notes_show_vectors = any(row.get("embedding") for row in notes)
            for row in notes:
                text = _row_text(row)
                label = " ".join(text.split())[:60]
                vector = row.get("embedding")
                if is_suppressed(row):
                    if drop_suppressed_vectors and vector:
                        result["dropped"] += 1
                        result["items"].append(
                            (
                                "drop_vector",
                                f"Removed the vector of “{label}”",
                                "hidden",
                            )
                        )
                        if store:
                            self.provider.store(
                                data={**row, "embedding": None},
                                memory_store_type=note_store,
                            )
                    continue
                if not text:
                    continue
                if vector and not (size and len(vector) != size):
                    continue
                if not vector and not notes_show_vectors:
                    continue
                if budget <= 0:
                    result["left"] += 1
                    continue
                budget -= 1
                result["notes"] += 1
                result["items"].append(
                    (
                        "re_embed",
                        f"Embedded “{label}” again",
                        "it had no vector"
                        if not vector
                        else "its vector was the wrong size",
                    )
                )
                if store:
                    try:
                        from ..embeddings import get_embedding

                        self.provider.store(
                            data={**row, "embedding": get_embedding(text)},
                            memory_store_type=note_store,
                        )
                    except Exception as exc:
                        logger.warning("A note was not embedded again: %s", exc)
                        result["notes"] -= 1
                        result["items"].pop()
        return result


__all__ = ["Compactor", "MAX_EMBEDS", "NOTE_STORES"]
