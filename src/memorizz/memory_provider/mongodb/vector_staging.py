"""Re-embed a MongoDB store beside its live vectors, then switch over.

Staged and previous vectors live in their own collection, never in the
memory documents, while search keeps reading each document's ``embedding``:

    memorizz_vector_staging: {slot: "next"|"prev", store, key, embedding, sig}

The switch writes each staged vector into its document and keeps the one it
replaces as ``prev``; swapping back turns ``prev`` into the staged slot and
only re-embeds what changed. An Atlas vector index fixes its vector size
(``numDimensions``), so a MongoDB store keeps its size.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...enums.memory_type import MemoryType

STAGING_COLLECTION = "memorizz_vector_staging"


def _id_filter(key: str) -> Dict[str, Any]:
    try:
        from bson import ObjectId

        if ObjectId.is_valid(key):
            return {"_id": {"$in": [key, ObjectId(key)]}}
    except ImportError:  # pragma: no cover - bson ships with pymongo
        pass
    return {"_id": key}


class MongoDBVectorStaging:
    """Staged re-embedding for ``MongoDBProvider``."""

    VECTOR_STAGING = {"fixed_size": True, "resize": False}

    def _staging(self):
        return self.db[STAGING_COLLECTION]

    def vector_stage_begin(
        self, dimensions: int, *, reuse_previous: bool = False
    ) -> None:
        staging = self._staging()
        staging.delete_many({"slot": "next"})
        if reuse_previous:
            if not staging.count_documents({"slot": "prev"}, limit=1):
                raise RuntimeError("No previous vectors are kept in this store")
            staging.update_many({"slot": "prev"}, {"$set": {"slot": "next"}})

    def vector_stage_keys(
        self, memory_type: MemoryType, candidates: Sequence[str]
    ) -> List[str]:
        return [str(value) for value in candidates]

    def vector_stage_signatures(self, memory_type: MemoryType) -> Dict[str, str]:
        return {
            row["key"]: row.get("sig") or ""
            for row in self._staging().find(
                {"slot": "next", "store": memory_type.value}, {"key": 1, "sig": 1}
            )
        }

    def vector_stage_put(
        self,
        memory_type: MemoryType,
        items: Sequence[Tuple[Sequence[str], Sequence[float], str]],
    ) -> List[bool]:
        collection = self._collection(memory_type)
        staging = self._staging()
        done = []
        for candidates, vector, sig in items:
            key = None
            for value in candidates:
                found = collection.find_one(_id_filter(str(value)), {"_id": 1})
                if found is not None:
                    key = str(found["_id"])
                    break
            if key is None:
                done.append(False)
                continue
            staging.replace_one(
                {"slot": "next", "store": memory_type.value, "key": key},
                {
                    "slot": "next",
                    "store": memory_type.value,
                    "key": key,
                    "embedding": [float(x) for x in vector],
                    "sig": sig,
                },
                upsert=True,
            )
            done.append(True)
        return done

    def vector_stage_switch(self, staged: Iterable[MemoryType]) -> Dict[str, Any]:
        staging = self._staging()
        if not staging.count_documents({"slot": "next"}, limit=1):
            raise RuntimeError("Nothing is staged in this store")
        staging.delete_many({"slot": "prev"})
        switched: Dict[str, int] = {}
        for row in staging.find({"slot": "next"}):
            collection = self._collection(MemoryType(row["store"]))
            current = collection.find_one(_id_filter(row["key"]), {"embedding": 1})
            if current is None:
                continue
            if current.get("embedding"):
                staging.insert_one(
                    {
                        "slot": "prev",
                        "store": row["store"],
                        "key": row["key"],
                        "embedding": current["embedding"],
                        "sig": row.get("sig"),
                    }
                )
            collection.update_one(
                {"_id": current["_id"]}, {"$set": {"embedding": row["embedding"]}}
            )
            switched[row["store"]] = switched.get(row["store"], 0) + 1
        staging.delete_many({"slot": "next"})
        return {"switched": switched, "carried": {}, "dropped": {}, "indexes": []}

    def vector_stage_drop(self) -> None:
        self._staging().delete_many({"slot": "next"})

    def vector_previous(self) -> Optional[Dict[str, Any]]:
        staging = self._staging()
        rows = staging.count_documents({"slot": "prev"})
        if not rows:
            return None
        sample = staging.find_one({"slot": "prev"}, {"embedding": 1}) or {}
        return {"dimensions": len(sample.get("embedding") or []) or None, "rows": rows}

    def vector_previous_drop(self) -> None:
        self._staging().delete_many({"slot": "prev"})


__all__ = ["MongoDBVectorStaging", "STAGING_COLLECTION"]
