"""Swap a fine-tuned embedding model into the memory layer, and back.

Stored memories carry vectors from the embedding model that wrote them, so
changing the model means re-embedding them: ``reembed`` rebuilds every
stored vector with the text each store embeds (a knowledge chunk's content,
an entity's name/type/attributes, a persona's name/role/goals/background, a
tool's name/description/signature/queries...). The new model is cut to the
size the store's vectors already have (Oracle VECTOR columns and MongoDB
vector indexes are fixed-size); EmbeddingGemma and Qwen3-Embedding are
trained to keep their quality at smaller sizes.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..enums.memory_type import MemoryType

logger = logging.getLogger(__name__)

EMBEDDING_ENV = (
    "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER",
    "MEMORIZZ_DEFAULT_EMBEDDING_MODEL",
    "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS",
    "MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL",
)

# Stores whose rows carry an ``embedding`` that search compares against.
EMBEDDED_STORES = (
    MemoryType.KNOWLEDGE_BASE,
    MemoryType.CONVERSATION_MEMORY,
    MemoryType.SUMMARIES,
    MemoryType.SHORT_TERM_MEMORY,
    MemoryType.ENTITY_MEMORY,
    MemoryType.PERSONAS,
    MemoryType.TOOLBOX,
    MemoryType.WORKFLOW_MEMORY,
    MemoryType.SKILLBOX,
    MemoryType.SEMANTIC_CACHE,
)

# The logical id each store's rows are updated by, after ``_id``.
_ID_FIELDS = {
    MemoryType.ENTITY_MEMORY: ("entity_id",),
    MemoryType.SUMMARIES: ("summary_id",),
    MemoryType.WORKFLOW_MEMORY: ("workflow_id",),
    MemoryType.TOOLBOX: ("tool_id",),
    MemoryType.SKILLBOX: ("skill_id",),
    MemoryType.PERSONAS: ("persona_id",),
    MemoryType.SEMANTIC_CACHE: ("cache_key",),
}


def _text(value: Any) -> str:
    if isinstance(value, dict):
        value = (
            value.get("content") or value.get("text") or json.dumps(value, default=str)
        )
    return str(value or "").strip()


def embedding_text(memory_type: MemoryType, row: Dict[str, Any]) -> str:
    """The text a store embeds for one of its rows (as when it was written)."""
    if memory_type == MemoryType.ENTITY_MEMORY:
        from ..long_term.semantic.entity_memory.entity_memory import EntityMemory

        return EntityMemory._build_embedding_text(_entity_shape(row))
    if memory_type == MemoryType.PERSONAS:
        return " ".join(
            str(row.get(k) or "") for k in ("name", "role", "goals", "background")
        ).strip()
    if memory_type == MemoryType.TOOLBOX:
        queries = row.get("queries") or []
        if isinstance(queries, list):
            queries = " ".join(str(q) for q in queries)
        signature = row.get("signature") or ""
        return " ".join(
            str(x)
            for x in (
                row.get("name") or "",
                row.get("description") or row.get("docstring") or "",
                signature,
                queries,
            )
            if x
        ).strip()
    if memory_type == MemoryType.WORKFLOW_MEMORY:
        from ..long_term.procedural.workflow.workflow import workflow_embedding_text

        return workflow_embedding_text(row)
    if memory_type == MemoryType.SKILLBOX:

        def joined(value):
            return (
                " ".join(str(v) for v in value)
                if isinstance(value, list)
                else str(value or "")
            )

        return " ".join(
            [
                str(row.get("name") or ""),
                str(row.get("description") or ""),
                joined(row.get("preconditions")),
                joined(row.get("queries")),
            ]
        ).strip()
    if memory_type == MemoryType.SEMANTIC_CACHE:
        return _text(row.get("query") or row.get("query_text"))
    for key in ("content", "text", "summary_content", "summary"):
        if row.get(key):
            return _text(row[key])
    return ""


def _entity_shape(row: Dict[str, Any]) -> Dict[str, Any]:
    """Entity rows as EntityMemory builds its text from them: attributes a
    list of {name, value} (some providers return a name -> value mapping or
    JSON text), relations a list of mappings, metadata a mapping."""

    def loaded(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value

    out = dict(row)
    attributes = loaded(row.get("attributes"))
    if isinstance(attributes, dict):
        attributes = [{"name": k, "value": v} for k, v in attributes.items()]
    out["attributes"] = (
        [a for a in attributes or [] if isinstance(a, dict)]
        if isinstance(attributes, list)
        else []
    )
    relations = loaded(row.get("relations"))
    out["relations"] = (
        [r for r in relations or [] if isinstance(r, dict)]
        if isinstance(relations, list)
        else []
    )
    metadata = loaded(row.get("metadata"))
    out["metadata"] = metadata if isinstance(metadata, dict) else {}
    return out


def _ids(memory_type: MemoryType, row: Dict[str, Any]) -> List[str]:
    out = []
    for key in _ID_FIELDS.get(memory_type, ()) + ("_id", "id", "memory_id"):
        value = row.get(key)
        if value not in (None, "") and str(value) not in out:
            out.append(str(value))
    return out


def store_dimensions(provider, manager) -> Optional[int]:
    """The vector size this store already holds: Oracle's declared VECTOR
    size, else the size of the embedding model that wrote it."""
    declared = getattr(provider, "get_vector_schema_dimensions", None)
    if callable(declared):
        try:
            sizes = set(declared().values())
        except Exception:
            sizes = set()
        if len(sizes) == 1:
            return sizes.pop()
    try:
        return int(manager.get_dimensions())
    except Exception:
        return None


def model_dimensions(model_dir: str) -> int:
    meta_file = Path(model_dir) / "memorizz_model.json"
    meta = (
        json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
    )
    if meta.get("dimensions"):
        return int(meta["dimensions"])
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device="cpu")
    getter = getattr(model, "get_embedding_dimension", None)
    return int(
        getter() if callable(getter) else model.get_sentence_embedding_dimension()
    )


def rebind(provider, manager) -> None:
    """Point a connected provider at the new embedding model. Providers that
    fixed an embedder at construction (Oracle, MongoDB, a FileSystem store
    given one) write new memories with it; the rest read the global one."""
    if getattr(provider, "_embedding_provider", None) is not None:
        provider._embedding_provider = manager
    if hasattr(provider, "_embedding_dimension_mismatch_detected"):
        provider._embedding_dimension_mismatch_detected = False
    cache = getattr(provider, "_vector_dimension_mismatch_warning_emitted_for", None)
    if isinstance(cache, set):
        cache.clear()


def reembed(
    provider,
    manager,
    *,
    stores: Iterable[MemoryType] = EMBEDDED_STORES,
    progress: Optional[Callable[[str], None]] = None,
    batch_size: int = 32,
) -> Dict[str, Dict[str, int]]:
    """Re-embed every row of ``stores`` that has text, with ``manager``.
    Returns per-store counts: rows, updated, skipped (no text), failed."""
    report: Dict[str, Dict[str, int]] = {}
    for memory_type in stores:
        try:
            rows = provider.list_all(memory_type) or []
        except Exception as exc:
            logger.info("Skipping %s: %s", memory_type.value, exc)
            continue
        counts = {"rows": len(rows), "updated": 0, "skipped": 0, "failed": 0}
        work = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                text = embedding_text(memory_type, row)
            except Exception as exc:  # one odd row never stops a re-embed
                logger.warning(
                    "Could not build the %s text of a row: %s", memory_type.value, exc
                )
                counts["failed"] += 1
                continue
            ids = _ids(memory_type, row)
            if not text or not ids:
                counts["skipped"] += 1
                continue
            work.append((ids, text))
        for start in range(0, len(work), batch_size):
            chunk = work[start : start + batch_size]
            if progress:
                progress(
                    f"Re-embedding {memory_type.value}: {start + len(chunk)}/{len(work)}"
                )
            try:
                vectors = manager.get_embeddings([text for _, text in chunk])
            except Exception as exc:
                logger.warning("Embedding %s failed: %s", memory_type.value, exc)
                counts["failed"] += len(chunk)
                continue
            for (ids, _), vector in zip(chunk, vectors):
                done = False
                for record_id in ids:
                    try:
                        if provider.update_by_id(
                            record_id, {"embedding": list(vector)}, memory_type
                        ):
                            done = True
                            break
                    except Exception as exc:
                        logger.debug(
                            "update %s %s failed: %s", memory_type.value, record_id, exc
                        )
                counts["updated" if done else "failed"] += 1
        mark = getattr(provider, "_mark_vector_index_dirty", None)
        if callable(mark):
            mark(memory_type)
        if rows:
            report[memory_type.value] = counts
    return report


def manager_for(values: Dict[str, str]):
    """An embedding model built from MEMORIZZ_DEFAULT_EMBEDDING_* values,
    without making it the one in use."""
    from ..embeddings import EmbeddingManager, embedding_settings_from

    name, config = embedding_settings_from(values)
    if not name:
        raise ValueError("no embedding provider is set")
    return EmbeddingManager(name, config)


def current_settings() -> Dict[str, str]:
    return {key: os.environ.get(key, "") for key in EMBEDDING_ENV}


def apply_settings(
    values: Dict[str, str], persist: Optional[Callable[[Dict[str, str]], None]] = None
) -> None:
    """Set the embedding settings in this process (and through ``persist``,
    e.g. the UI's .env writer), then rebuild the global embedding model."""
    from ..embeddings import reset_embedding_manager

    for key in EMBEDDING_ENV:
        value = values.get(key, "")
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    if persist is not None:
        persist({key: values.get(key, "") for key in EMBEDDING_ENV})
    reset_embedding_manager()


__all__ = [
    "EMBEDDED_STORES",
    "EMBEDDING_ENV",
    "apply_settings",
    "current_settings",
    "embedding_text",
    "manager_for",
    "model_dimensions",
    "rebind",
    "reembed",
    "store_dimensions",
]
