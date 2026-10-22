# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Does the store's vector space match the embedder this UI searches with?

Run once when the UI connects. It reads what the store declares:

- Oracle: the declared size of every ``EMBEDDING`` VECTOR column;
- MongoDB: ``numDimensions`` of each Atlas vector index, else the length of
  one stored embedding per collection;
- FileSystem: the length of one stored embedding per store type;
- Notion: the same checks on its separate vector store.

and the embedding settings saved with the store's agents (``embedding_provider``
and ``embedding_config``), when any are recorded. It compares them with the
embedder the UI is configured to use. When the store records one embedder,
its vectors have that size and the provider can be built here (its package
is installed and its key is set), the UI adopts it for this process.
Otherwise it returns a warning saying what differs and how to fix it.

Nothing is written and nothing is re-embedded. Reading dimensions never calls
an embedding API: configured sizes come from the embedder's settings, not
from a test embedding.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..enums.memory_type import MemoryType

logger = logging.getLogger(__name__)

# Embedders the UI may build by itself: cheap to construct, no model to load.
ADOPTABLE_PROVIDERS = frozenset({"openai", "ollama", "voyageai"})
_PROVIDER_PACKAGES = {"openai": "openai", "ollama": "ollama", "voyageai": "voyageai"}
_PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "voyageai": "VOYAGE_API_KEY"}
_AGENT_SCAN_LIMIT = 500
_MONGO_SAMPLE_MS = 2000
_MONGO_BUDGET_SECONDS = 4.0

FIX_SETTINGS = (
    "set MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER, MEMORIZZ_DEFAULT_EMBEDDING_MODEL "
    "and MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS to the model that wrote them"
)


# ---------------------------------------------------------------------------
# What the UI is configured to embed with
# ---------------------------------------------------------------------------


def unwrap(provider: Any) -> Any:
    """The provider under the UI's read-only proxy."""
    from .security import ReadOnlyProviderProxy

    if isinstance(provider, ReadOnlyProviderProxy):
        return provider.wrapped_provider
    return provider


def configured_embedder(provider: Any) -> Any:
    """The embedder searches on ``provider`` use: its own, else the global one."""
    own = getattr(provider, "_embedding_provider", None)
    if own is not None:
        return own
    try:
        from .. import embeddings

        return embeddings.get_embedding_manager()
    except Exception as exc:  # no key for the default provider, for example
        logger.debug("No global embedding manager: %s", exc)
        return None


def describe_embedder(embedder: Any) -> Dict[str, Any]:
    """``{"provider", "model", "dimensions"}`` without calling the embedder.

    Ollama's size is the configured or catalogued one, not a probe; an
    OpenAI-compatible server without a configured size is unknown.
    """
    if embedder is None:
        return {"provider": None, "model": None, "dimensions": None}
    if type(embedder).__name__ == "OracleInDatabaseEmbeddingProvider":
        return {
            "provider": "oracle in-database",
            "model": getattr(embedder, "model_name", None),
            "dimensions": _int(getattr(embedder, "dimensions", None)),
        }
    provider_type = getattr(embedder, "provider_type", None)
    inner = getattr(embedder, "_provider", None)
    if provider_type is not None and inner is not None:
        name = str(getattr(provider_type, "value", provider_type))
        model = getattr(inner, "model", None) or getattr(inner, "model_name", None)
        dimensions = _int(getattr(inner, "dimensions", None))
        if dimensions is None and name == "huggingface":
            dimensions = _safe_call(embedder, "get_dimensions")
        return {"provider": name, "model": model, "dimensions": dimensions}
    info = _safe_call(embedder, "get_provider_info")
    info = info if isinstance(info, dict) else {}
    return {
        "provider": info.get("provider") or type(embedder).__name__,
        "model": info.get("model"),
        "dimensions": _int(info.get("dimensions")),
    }


def embedder_label(description: Dict[str, Any]) -> str:
    """``ollama nomic-embed-text (768)``."""
    if not description.get("provider"):
        return "no embedding model"
    text = str(description["provider"])
    if description.get("model"):
        text += f" {description['model']}"
    if description.get("dimensions"):
        text += f" ({description['dimensions']})"
    return text


# ---------------------------------------------------------------------------
# What the store holds
# ---------------------------------------------------------------------------


def store_dimensions(provider: Any) -> Tuple[Dict[str, int], str]:
    """``({store: dimensions}, where they came from)`` for one provider."""
    if callable(getattr(provider, "get_vector_schema_dimensions", None)):
        declared = provider.get_vector_schema_dimensions()
        return (
            {
                str(name).split(".", 1)[0].lower(): int(size)
                for name, size in (declared or {}).items()
            },
            "Oracle VECTOR columns",
        )
    if getattr(provider, "db", None) is not None and hasattr(provider, "client"):
        return _mongodb_dimensions(provider)
    if hasattr(provider, "_indexes") and hasattr(provider, "_read_document"):
        return _filesystem_dimensions(provider), "stored embeddings"
    return {}, ""


def _mongodb_dimensions(provider: Any) -> Tuple[Dict[str, int], str]:
    db = provider.db
    try:
        names = set(db.list_collection_names())
    except Exception as exc:
        logger.debug("Could not list MongoDB collections: %s", exc)
        return {}, ""
    found: Dict[str, int] = {}
    sources = set()
    deadline = time.monotonic() + _MONGO_BUDGET_SECONDS
    for memory_type in MemoryType:
        if memory_type.value not in names:
            continue
        collection = db[memory_type.value]
        size = _mongodb_index_dimensions(collection)
        if size is not None:
            found[memory_type.value] = size
            sources.add("vector search indexes")
            continue
        if time.monotonic() > deadline:
            continue  # enough sampling for one connect
        size = _mongodb_sample_dimensions(collection)
        if size is not None:
            found[memory_type.value] = size
            sources.add("stored embeddings")
    return found, " and ".join(sorted(sources))


def _mongodb_index_dimensions(collection: Any) -> Optional[int]:
    try:
        indexes = list(collection.list_search_indexes())
    except Exception:
        return None  # not Atlas, or no permission to list search indexes
    for index in indexes:
        if index.get("type") not in (None, "vectorSearch"):
            continue
        definition = index.get("latestDefinition") or index.get("definition") or {}
        for field in definition.get("fields") or []:
            if field.get("type") == "vector" and field.get("numDimensions"):
                return _int(field["numDimensions"])
    return None


def _mongodb_sample_dimensions(collection: Any) -> Optional[int]:
    try:
        rows = list(
            collection.aggregate(
                [
                    {"$match": {"embedding.0": {"$exists": True}}},
                    {"$limit": 1},
                    {"$project": {"_id": 0, "size": {"$size": "$embedding"}}},
                ],
                maxTimeMS=_MONGO_SAMPLE_MS,
            )
        )
    except Exception as exc:
        logger.debug("Could not sample %s embeddings: %s", collection.name, exc)
        return None
    return _int(rows[0].get("size")) if rows else None


def _filesystem_dimensions(provider: Any) -> Dict[str, int]:
    found: Dict[str, int] = {}
    indexes = getattr(provider, "_indexes", {}) or {}
    for memory_type in MemoryType:
        items = indexes.get(memory_type) or {}
        for document_id, meta in items.items():
            if not (isinstance(meta, dict) and meta.get("has_embedding")):
                continue
            try:
                document = provider._read_document(memory_type, document_id)
            except Exception:
                continue
            vector = (document or {}).get("embedding")
            if isinstance(vector, (list, tuple)) and vector:
                found[memory_type.value] = len(vector)
            break
    return found


def recorded_embedders(provider: Any) -> List[Dict[str, Any]]:
    """Distinct embedding settings saved with the store's agents."""
    try:
        rows = provider.list_all(MemoryType.MEMAGENT) or []
    except Exception as exc:
        logger.debug("Could not read agents for embedding settings: %s", exc)
        return []
    counts: Counter = Counter()
    for row in list(rows)[:_AGENT_SCAN_LIMIT]:
        data = _as_dict(row)
        name = data.get("embedding_provider")
        if not (isinstance(name, str) and name.strip()):
            continue
        config = data.get("embedding_config")
        config = config if isinstance(config, dict) else {}
        size = _int(config.get("dimensions") or config.get("output_dimension"))
        model = config.get("model")
        base_url = config.get("base_url")
        counts[
            (
                name.strip().lower(),
                str(model) if model else None,
                size,
                str(base_url) if base_url else None,
            )
        ] += 1
    return [
        {
            "provider": name,
            "model": model,
            "dimensions": size,
            "base_url": base_url,
            "agents": count,
        }
        for (name, model, size, base_url), count in counts.most_common()
    ]


# ---------------------------------------------------------------------------
# Compare, adopt or warn
# ---------------------------------------------------------------------------


def check_store_embeddings(
    provider: Any, provider_type: Optional[str]
) -> Dict[str, Any]:
    """Compare the store's vectors with the configured embedder.

    Returns ``{"status": ...}``: ``match``, ``adopted`` (the UI now uses the
    store's recorded embedder), ``mismatch`` (with ``message``), ``empty``
    (no vectors to compare) or ``unknown``. Never raises.
    """
    try:
        return _check(unwrap(provider), provider_type)
    except Exception as exc:  # the check must never stop a connection
        logger.warning("Embedding check failed (%s)", type(exc).__name__)
        return {"status": "unknown", "message": None}


def _check(provider: Any, provider_type: Optional[str]) -> Dict[str, Any]:
    target = provider
    if provider_type == "notion":
        target = getattr(provider, "semantic_provider", None)
        if target is None:
            return {"status": "empty", "message": None}
    try:
        sizes, source = store_dimensions(target)
    except Exception as exc:
        logger.info("Could not read the store's vector dimensions: %s", exc)
        return {"status": "unknown", "message": None}
    if not sizes:
        return {"status": "empty", "message": None}

    store_size, _ = Counter(sizes.values()).most_common(1)[0]
    configured = describe_embedder(configured_embedder(target))
    # Notion lists its agents through the remote API: no scan at connect.
    recorded_all = recorded_embedders(target) if provider_type != "notion" else []
    recorded = _recorded_for(recorded_all, store_size)
    result: Dict[str, Any] = {
        "store_dimensions": store_size,
        "dimensions_by_store": dict(sorted(sizes.items())),
        "source": source,
        "configured": configured,
        "recorded": recorded,
    }

    if (
        configured.get("provider")
        and configured.get("dimensions") is None
        and recorded is None
    ):
        # An OpenAI-compatible server with no size set: its size is only
        # known after a test embedding, which this check never makes.
        return {**result, "status": "unknown", "message": None}
    same_size = configured.get("dimensions") == store_size and all(
        size == store_size for size in sizes.values()
    )
    same_model = recorded is None or _same_embedder(recorded, configured)
    if same_size and same_model:
        return {**result, "status": "match", "message": None}

    if recorded is not None:
        reason = _unavailable_reason(recorded)
        if reason is None:
            manager = _build(recorded, store_size)
            if manager is not None:
                _adopt(provider, target, manager)
                adopted = describe_embedder(manager)
                return {
                    **result,
                    "status": "adopted",
                    "adopted": adopted,
                    "message": (
                        f"This store holds {store_size}-dimension vectors from "
                        f"{embedder_label(adopted)}, so the UI searches with it "
                        f"instead of {embedder_label(configured)} from the "
                        "environment. Nothing was re-embedded."
                    ),
                }
            reason = f"{recorded['provider']} could not be set up here"
        result["unavailable"] = reason

    return {
        **result,
        "status": "mismatch",
        "message": mismatch_message(
            store_size,
            configured,
            recorded=recorded,
            unavailable=result.get("unavailable"),
            mixed=sorted(set(sizes.values())) if len(set(sizes.values())) > 1 else None,
        ),
    }


def mismatch_message(
    store_size: int,
    configured: Dict[str, Any],
    *,
    recorded: Optional[Dict[str, Any]] = None,
    unavailable: Optional[str] = None,
    mixed: Optional[List[int]] = None,
    connected: bool = True,
) -> str:
    """The warning shown on the dashboard and the connect page."""
    if mixed:
        holds = (
            "This store holds vectors of "
            + ", ".join(str(size) for size in mixed[:-1])
            + f" and {mixed[-1]} dimensions"
        )
    else:
        holds = f"This store holds {store_size}-dimension vectors"
    if recorded is not None:
        holds += f" from {embedder_label({**recorded, 'dimensions': None})}"
    lead = f"{holds}; the UI is configured for {embedder_label(configured)}."
    if not connected:
        lead += " The UI did not connect."
    fix = "Searches will not match until you "
    if recorded is not None and unavailable:
        fix += f"{unavailable}, then connect again."
    else:
        steps = []
        if str(configured.get("provider") or "").startswith("oracle"):
            steps.append("set MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING=false")
        if recorded is not None:
            steps.append(
                "set MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER="
                f"{recorded['provider']}"
                + (
                    f", MEMORIZZ_DEFAULT_EMBEDDING_MODEL={recorded['model']}"
                    if recorded.get("model")
                    else ""
                )
                + f" and MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS={store_size}"
            )
        else:
            steps.append(f"{FIX_SETTINGS} ({store_size} dimensions)")
        fix += " and ".join(steps) + ", then connect again."
    return f"{lead} {fix} Nothing was re-embedded."


_ORACLE_MISMATCH = re.compile(
    r"Oracle VECTOR dimension mismatch: the configured embedding provider "
    r"outputs (?P<configured>\d+) dimensions, but schema (?P<schema>\S+) "
    r"declares (?P<declared>.+?)\. ",
)


def oracle_mismatch_error(message: str) -> Optional[str]:
    """Rewrite the Oracle provider's dimension-mismatch failure as the same
    warning, for the connect page (the provider refuses to start then)."""
    match = _ORACLE_MISMATCH.search(str(message or ""))
    if not match:
        return None
    sizes = [int(size) for size in re.findall(r"=(\d+)", match.group("declared"))]
    if not sizes:
        return None
    store_size, _ = Counter(sizes).most_common(1)[0]
    try:
        from ..memory_provider.oracle.embedding import in_database_embedding_options

        model = in_database_embedding_options({})["model_name"]
    except Exception:
        model = None
    configured = {
        "provider": "oracle in-database",
        "model": model,
        "dimensions": int(match.group("configured")),
    }
    distinct = sorted(set(sizes))
    return mismatch_message(
        store_size,
        configured,
        mixed=distinct if len(distinct) > 1 else None,
        connected=False,
    )


def _recorded_for(
    recorded: Iterable[Dict[str, Any]], store_size: int
) -> Optional[Dict[str, Any]]:
    """The one recorded embedder that fits the store, if exactly one does."""
    fitting = [
        item
        for item in recorded
        if item.get("dimensions") in (None, store_size)
        and _default_size(item) in (None, store_size)
    ]
    distinct = {(item["provider"], item.get("model")) for item in fitting}
    if len(distinct) != 1:
        return None
    return fitting[0]


def _default_size(item: Dict[str, Any]) -> Optional[int]:
    """The size a recorded embedder without saved dimensions produces."""
    if item.get("dimensions"):
        return item["dimensions"]
    if item.get("provider") == "openai" and not item.get("base_url"):
        model = item.get("model") or "text-embedding-3-small"
        return 1536 if model == "text-embedding-ada-002" else 256
    return None


def _same_embedder(recorded: Dict[str, Any], configured: Dict[str, Any]) -> bool:
    if recorded.get("provider") != configured.get("provider"):
        return False
    model = recorded.get("model")
    return not model or model == configured.get("model")


def _unavailable_reason(recorded: Dict[str, Any]) -> Optional[str]:
    """Why the recorded embedder cannot be built here, or None when it can."""
    name = recorded["provider"]
    if name not in ADOPTABLE_PROVIDERS:
        return f"configure {name} embeddings for this UI"
    package = _PROVIDER_PACKAGES[name]
    if importlib.util.find_spec(package) is None:
        return f"install the {package} package"
    key = _PROVIDER_KEYS.get(name)
    if key and not (name == "openai" and recorded.get("base_url")):
        if not str(os.environ.get(key, "")).strip():
            return f"set {key}"
    return None


def _build(recorded: Dict[str, Any], store_size: int) -> Any:
    from ..embeddings import EmbeddingManager

    config: Dict[str, Any] = {}
    if recorded.get("model"):
        config["model"] = recorded["model"]
    size_key = (
        "output_dimension" if recorded["provider"] == "voyageai" else "dimensions"
    )
    config[size_key] = int(recorded.get("dimensions") or store_size)
    if recorded.get("base_url"):
        config["base_url"] = recorded["base_url"]
    try:
        return EmbeddingManager(recorded["provider"], config)
    except Exception as exc:
        logger.info("Could not build the store's embedder: %s", exc)
        return None


def _adopt(provider: Any, target: Any, manager: Any) -> None:
    """Search with ``manager`` in this process. Nothing is written."""
    from ..embeddings import set_global_embedding_manager
    from ..finetune.swap import rebind

    for item in {id(provider): provider, id(target): target}.values():
        rebind(item, manager)
    set_global_embedding_manager(manager)


def _as_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return row
    dump = getattr(row, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            return {}
    return dict(getattr(row, "__dict__", {}) or {})


def _int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _safe_call(target: Any, name: str) -> Any:
    method = getattr(target, name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


__all__ = [
    "check_store_embeddings",
    "configured_embedder",
    "describe_embedder",
    "embedder_label",
    "mismatch_message",
    "oracle_mismatch_error",
    "recorded_embedders",
    "store_dimensions",
]
