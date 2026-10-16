"""Swap the embedding model with no window of mixed vectors.

``migrate`` re-embeds every stored memory into the store's staged slot while
search keeps using the live vectors, catches up with memories written or
edited meanwhile (each staged vector carries a signature of its text), then
switches over in one step and keeps the vectors it replaced. Swapping back
turns those kept vectors into the staged slot, so only memories written or
edited since the swap are embedded again.

Stores implement the slot (``vector_stage_*`` and ``vector_previous*``):
Oracle with shadow VECTOR columns renamed into place (which is also how its
fixed-size columns change size), the filesystem store with sidecar files,
MongoDB with a staging collection. Other stores re-embed in place
(``swap.reembed``) and keep their size.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..enums.memory_type import MemoryType
from .swap import EMBEDDED_STORES, _ids, embedding_text

logger = logging.getLogger(__name__)


class StagingError(RuntimeError):
    """A staged re-embed stopped before switching; the live vectors are untouched."""


def supported(provider) -> bool:
    return callable(getattr(provider, "vector_stage_begin", None))


def capabilities(provider) -> Dict[str, bool]:
    """``staged`` (re-embeds beside the live vectors), ``fixed_size`` (the
    store pins one vector size) and ``resize`` (it can change that size)."""
    info = dict(getattr(provider, "VECTOR_STAGING", None) or {})
    staged = supported(provider)
    return {
        "staged": staged,
        "fixed_size": bool(info.get("fixed_size", True)),
        "resize": staged and bool(info.get("resize")),
    }


def signature(text: str) -> str:
    """What a vector was made from: memories whose text changes get a new one."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _rows(provider, memory_type: MemoryType) -> List[Dict[str, Any]]:
    try:
        return [
            r for r in (provider.list_all(memory_type) or []) if isinstance(r, dict)
        ]
    except Exception as exc:
        logger.info("Skipping %s: %s", memory_type.value, exc)
        return []


def migrate(
    provider,
    manager,
    *,
    dimensions: int,
    reuse_previous: bool = False,
    stores: Iterable[MemoryType] = EMBEDDED_STORES,
    progress: Optional[Callable[[str], None]] = None,
    batch_size: int = 32,
    max_passes: int = 4,
) -> Dict[str, Any]:
    """Re-embed ``stores`` with ``manager`` at ``dimensions`` and switch.

    Pass 1 embeds every memory that lacks a staged vector for its current
    text; later passes catch up with memories written or edited while it
    ran, until a pass finds nothing to do. Nothing changes for readers until
    the switch; any failure before it discards the staged slot (all or
    nothing). Returns per-store counts and the store's switch report."""
    say = progress or (lambda message: None)
    stores = list(stores)
    dimensions = int(dimensions)
    provider.vector_stage_begin(dimensions, reuse_previous=reuse_previous)
    counts: Dict[str, Dict[str, int]] = {}
    failed: Dict[Tuple[str, Tuple[str, ...]], str] = {}
    try:
        for attempt in range(max_passes):
            pending = 0
            for memory_type in stores:
                rows = _rows(provider, memory_type)
                if not rows:
                    continue
                tally = counts.setdefault(
                    memory_type.value,
                    {"rows": 0, "embedded": 0, "kept": 0, "skipped": 0, "failed": 0},
                )
                tally["rows"] = len(rows)
                staged = provider.vector_stage_signatures(memory_type)
                work: List[Tuple[List[str], str, str]] = []
                kept = skipped = 0
                for row in rows:
                    try:
                        text = embedding_text(memory_type, row)
                    except Exception as exc:  # one odd row never stops a swap
                        logger.warning(
                            "Could not build a %s text: %s", memory_type.value, exc
                        )
                        text = ""
                    ids = _ids(memory_type, row)
                    if not text or not ids:
                        skipped += 1
                        continue
                    if (memory_type.value, tuple(ids)) in failed:
                        continue
                    sig = signature(text)
                    keys = provider.vector_stage_keys(memory_type, ids)
                    if any(staged.get(key) == sig for key in keys):
                        kept += 1
                        continue
                    work.append((ids, text, sig))
                if attempt == 0:
                    tally["kept"], tally["skipped"] = kept, skipped
                pending += len(work)
                for start in range(0, len(work), batch_size):
                    chunk = work[start : start + batch_size]
                    say(
                        f"{'Catching up' if attempt else 'Re-embedding'} {memory_type.value}: "
                        f"{start + len(chunk)}/{len(work)}"
                    )
                    vectors = manager.get_embeddings([text for _, text, _ in chunk])
                    for vector in vectors:
                        if len(vector) != dimensions:
                            raise StagingError(
                                f"The model made {len(vector)}-dimension vectors, not {dimensions}"
                            )
                    done = provider.vector_stage_put(
                        memory_type,
                        [
                            (ids, vector, sig)
                            for (ids, _, sig), vector in zip(chunk, vectors)
                        ],
                    )
                    for (ids, _, _), ok in zip(chunk, done):
                        if ok:
                            tally["embedded"] += 1
                        else:
                            failed[(memory_type.value, tuple(ids))] = "no matching row"
                            tally["failed"] += 1
            if not pending:
                break
        else:
            raise StagingError(
                "Memories kept changing while re-embedding; try again when the store is quieter"
            )
        if failed:
            raise StagingError(
                f"{len(failed)} memories could not be staged; nothing was switched"
            )
    except Exception:
        try:
            provider.vector_stage_drop()
        except Exception as cleanup:  # pragma: no cover - reported, not raised
            logger.warning("Could not discard the staged vectors: %s", cleanup)
        raise
    say("Switching to the new vectors…")
    try:
        switch = provider.vector_stage_switch(stores)
    except Exception as exc:
        # Some stores may have switched: keep what is there, say so.
        raise StagingError(
            f"The switch failed part-way ({exc}). Run the swap again to finish it, "
            "or swap back."
        ) from exc
    return {
        "stores": counts,
        "dimensions": dimensions,
        "reused": reuse_previous,
        **switch,
    }


__all__ = ["StagingError", "capabilities", "migrate", "signature", "supported"]
