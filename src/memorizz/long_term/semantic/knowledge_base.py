# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ...embeddings import get_embedding
from ...enums.memory_type import MemoryType
from ...memory_provider import MemoryProvider
from . import extractors as _extractors
from .extractors import EmptyDocumentError, ExtractorError, FileSource

logger = logging.getLogger(__name__)


def _persist_agent(agent: Any) -> None:
    """Persist agent state after mutating ``knowledge_base_ids``.

    ``MemAgent`` exposes ``save()`` (which serializes to a ``MemAgentModel``
    and hands it to the memory provider). Older code called ``agent.update()``
    which doesn't exist — failing silently and leaving the attach unpersisted.
    Try ``save`` first, then fall back to provider-level updates for any
    custom agent-like object that callers might pass.
    """
    save = getattr(agent, "save", None)
    if callable(save):
        save()
        return
    # Best-effort fallback: if the caller gave us something with a memory_provider
    # and an agent_id, at least update the agent_knowledge_bases relation.
    provider = getattr(agent, "memory_provider", None)
    agent_id = getattr(agent, "agent_id", None)
    ids = getattr(agent, "knowledge_base_ids", None) or []
    updater = getattr(provider, "update_memagent_knowledge_base_ids", None)
    if callable(updater) and agent_id:
        updater(agent_id, list(ids))
        return
    logger.warning(
        "Could not persist agent (no save() and no update_memagent_knowledge_base_ids)"
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

ChunkingStrategy = str  # "fixed" | "sentence" | "paragraph" | "none"
# Callable form: (corpus, **kwargs) -> list of chunk strings
ChunkerFn = Callable[..., List[str]]

DEFAULT_CHUNK_SIZE = 1000  # characters per chunk for "fixed"
DEFAULT_CHUNK_OVERLAP = 100  # characters of overlap between adjacent chunks


def _chunk_fixed(
    corpus: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    **_: Any,
) -> List[str]:
    """Split by fixed character window with optional overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    text = corpus or ""
    if len(text) <= chunk_size:
        return [text] if text else []
    step = chunk_size - chunk_overlap
    chunks: List[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size]
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])")


def _chunk_sentence(
    corpus: str, chunk_size: int = DEFAULT_CHUNK_SIZE, **_: Any
) -> List[str]:
    """Split on sentence boundaries, packing sentences up to chunk_size."""
    text = (corpus or "").strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT.split(text)
    chunks: List[str] = []
    buffer = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if not buffer:
            buffer = sent
        elif len(buffer) + 1 + len(sent) <= chunk_size:
            buffer = f"{buffer} {sent}"
        else:
            chunks.append(buffer)
            buffer = sent
    if buffer:
        chunks.append(buffer)
    return chunks


def _chunk_paragraph(corpus: str, **_: Any) -> List[str]:
    """Split on blank lines (paragraph boundaries)."""
    text = corpus or ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paragraphs if p]


def _chunk_none(corpus: str, **_: Any) -> List[str]:
    """No chunking — return the corpus as a single item."""
    return [corpus] if corpus else []


DEFAULT_SEMANTIC_BREAKPOINT_PERCENTILE = 95.0


def _chunk_semantic(
    corpus: str,
    breakpoint_percentile: float = DEFAULT_SEMANTIC_BREAKPOINT_PERCENTILE,
    **_: Any,
) -> List[str]:
    """Split by semantic similarity of adjacent sentences.

    Each sentence is embedded, and cosine distance is computed between every
    adjacent pair. Sentences are grouped into chunks, breaking whenever the
    distance to the next sentence falls in the top ``(100 - breakpoint_percentile)``
    percent of distances (i.e. the largest topic shifts). A higher percentile
    produces fewer, larger chunks; a lower percentile produces more chunks.

    Requires embeddings, so this is slower and more expensive than the other
    strategies. For short corpora (< 2 sentences) it degrades gracefully to
    a single chunk.
    """
    import numpy as np

    text = (corpus or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 1:
        return sentences

    embeddings = np.array(
        [get_embedding(sentence) for sentence in sentences], dtype=float
    )
    norms = np.linalg.norm(embeddings, axis=1)
    # Guard against zero-vectors (unusual but possible for empty-ish sentences).
    norms[norms == 0] = 1.0
    unit = embeddings / norms[:, None]
    # Cosine distance between sentence i and i+1
    sims = np.sum(unit[:-1] * unit[1:], axis=1)
    distances = 1.0 - sims

    # Clamp the percentile to a sensible range to avoid degenerate cuts
    pct = min(max(float(breakpoint_percentile), 0.0), 100.0)
    threshold = float(np.percentile(distances, pct))
    breakpoint_indices = [i for i, d in enumerate(distances.tolist()) if d >= threshold]

    chunks: List[str] = []
    start = 0
    for bp in breakpoint_indices:
        # Include sentence `bp` in the current chunk, break after it
        segment = " ".join(sentences[start : bp + 1]).strip()
        if segment:
            chunks.append(segment)
        start = bp + 1
    tail = " ".join(sentences[start:]).strip()
    if tail:
        chunks.append(tail)
    # Percentile with all-equal distances collapses to a single break at the
    # last pair; ensure we don't drop sentences.
    return chunks or [text]


_BUILTIN_CHUNKERS: Dict[str, ChunkerFn] = {
    "fixed": _chunk_fixed,
    "sentence": _chunk_sentence,
    "paragraph": _chunk_paragraph,
    "semantic": _chunk_semantic,
    "none": _chunk_none,
}


def _resolve_chunker(strategy: Union[ChunkingStrategy, ChunkerFn]) -> ChunkerFn:
    if callable(strategy):
        return strategy  # custom chunker
    key = str(strategy).lower().strip()
    if key not in _BUILTIN_CHUNKERS:
        valid = ", ".join(sorted(_BUILTIN_CHUNKERS))
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Valid options: {valid}, "
            "or pass a callable (corpus, **kwargs) -> list[str]."
        )
    return _BUILTIN_CHUNKERS[key]


class KnowledgeBase:
    """
    KnowledgeBase class that implements a specialized form of long-term memory.

    Instead of writing documents to a separate "knowledge_base" collection,
    each ingestion is stored directly in the agent's `knowledge_base` store.
    Each ingest may be split into multiple chunks (one document per chunk,
    all sharing the same ``knowledge_base_id``) so that semantic retrieval
    can match at a finer granularity than the full source document.
    """

    def __init__(self, memory_provider: Optional[MemoryProvider] = None):
        """
        Initialize a new KnowledgeBase instance.

        Parameters:
        -----------
        memory_provider : Optional[MemoryProvider]
            The memory provider to use for storage and retrieval.
            If not provided, a default MemoryProvider will be used.
        """
        self.memory_provider = memory_provider or MemoryProvider()

    def ingest_knowledge(
        self,
        corpus: str,
        namespace: str,
        chunking_strategy: Union[ChunkingStrategy, ChunkerFn] = "fixed",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        breakpoint_percentile: float = DEFAULT_SEMANTIC_BREAKPOINT_PERCENTILE,
    ) -> str:
        """
        Embed and save text content to the memory provider under the given namespace.

        The corpus is split into chunks according to ``chunking_strategy``, and
        each chunk is embedded and stored as its own document sharing one
        ``knowledge_base_id``. This makes semantic retrieval land on the most
        relevant passage rather than always returning the full source text.

        Parameters:
        -----------
        corpus : str
            The text content to be ingested and stored.
        namespace : str
            A namespace to organize and categorize the knowledge.
        chunking_strategy : str | callable, default "fixed"
            One of:
              - "fixed":     split into ``chunk_size``-character windows with ``chunk_overlap``
              - "sentence":  pack sentences up to ``chunk_size`` characters
              - "paragraph": split on blank lines
              - "semantic":  embed sentences and break at the largest topic shifts
                             (tuned by ``breakpoint_percentile``)
              - "none":      store the full corpus as a single document
            Or a callable ``(corpus, **kwargs) -> list[str]`` for custom logic.
        chunk_size : int, default 1000
            Target chunk size in characters (used by "fixed" and "sentence").
        chunk_overlap : int, default 100
            Characters of overlap between adjacent "fixed" chunks. Must be
            ``>= 0`` and ``< chunk_size``.
        breakpoint_percentile : float, default 95.0
            Only used by "semantic". Cut between sentences whose cosine
            distance is in the top ``(100 - breakpoint_percentile)`` percent.
            Lower values produce more chunks; higher values produce fewer.

        Returns:
        --------
        str
            A unique knowledge_base_id that can be attached to an agent to scope its knowledge base.
        """
        knowledge_base_id = str(uuid.uuid4())

        chunker = _resolve_chunker(chunking_strategy)
        chunks = chunker(
            corpus,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            breakpoint_percentile=breakpoint_percentile,
        )
        if not chunks:
            # Nothing to store; still return the id so callers can detect
            # empty ingests and decide whether to treat as error.
            logger.warning(
                "ingest_knowledge: corpus for namespace=%r produced no chunks",
                namespace,
            )
            return knowledge_base_id

        strategy_label = (
            chunking_strategy
            if isinstance(chunking_strategy, str)
            else getattr(chunking_strategy, "__name__", "custom")
        )
        now = datetime.now().isoformat()

        for index, chunk_text in enumerate(chunks):
            embedding = get_embedding(chunk_text)
            entry = {
                "content": chunk_text,
                "embedding": embedding,
                "namespace": namespace,
                "knowledge_base_id": knowledge_base_id,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "chunking_strategy": strategy_label,
                "created_at": now,
                "updated_at": now,
            }
            self.memory_provider.store(
                entry, memory_store_type=MemoryType.KNOWLEDGE_BASE
            )

        return knowledge_base_id

    def retrieve_knowledge(self, knowledge_base_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all knowledge entries associated with a given knowledge_base_id.

        When the source was chunked at ingest time, this returns every chunk
        sorted by ``chunk_index`` so callers can reassemble the original text.

        Parameters:
        -----------
        knowledge_base_id : str
            The unique ID to retrieve knowledge for.

        Returns:
        --------
        List[Dict[str, Any]]
            A list of knowledge documents, each containing the original content,
            its embedding, and the memory ID.
        """
        all_entries = self.memory_provider.list_all(
            memory_store_type=MemoryType.KNOWLEDGE_BASE
        )

        knowledge_entries = [
            entry
            for entry in all_entries
            if entry.get("knowledge_base_id") == knowledge_base_id
        ]
        knowledge_entries.sort(key=lambda e: e.get("chunk_index", 0))
        return knowledge_entries

    def retrieve_knowledge_by_query(
        self, query: str, namespace: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge entries that are semantically similar to the query.

        Parameters:
        -----------
        query : str
            The query to retrieve relevant knowledge for.
        namespace : Optional[str]
            If provided, limit the search to knowledge within this namespace.
        limit : int
            Maximum number of entries to return.

        Returns:
        --------
        List[Dict[str, Any]]
            A list of knowledge documents that are semantically similar to the query.
        """
        # Generate embedding for the query
        query_embedding = get_embedding(query)

        # Create a query object for semantic search
        query_obj = {"embedding": query_embedding, "limit": limit}

        # If namespace is provided, add it to the query
        if namespace:
            query_obj["namespace"] = namespace

        # Use the retrieve_by_query method for semantics search
        results = self.memory_provider.retrieve_by_query(
            query_obj, memory_store_type=MemoryType.KNOWLEDGE_BASE, limit=limit
        )

        # If results is a single dict, wrap it in a list
        if results and isinstance(results, dict):
            results = [results]

        # If no results, return empty list
        return results or []

    def delete_knowledge(self, knowledge_base_id: str) -> bool:
        """
        Delete all knowledge entries (all chunks) associated with a given knowledge_base_id.

        Parameters:
        -----------
        knowledge_base_id : str
            The unique ID of the knowledge to delete.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        # Get all entries with this memory ID
        entries = self.retrieve_knowledge(knowledge_base_id)

        # Delete each entry
        success = True
        for entry in entries:
            entry_id = entry.get("_id")
            if entry_id:
                if not self.memory_provider.delete_by_id(
                    entry_id, memory_store_type=MemoryType.KNOWLEDGE_BASE
                ):
                    success = False

        return success

    def update_knowledge(
        self,
        knowledge_base_id: str,
        corpus: str,
        chunking_strategy: Union[ChunkingStrategy, ChunkerFn] = "fixed",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        breakpoint_percentile: float = DEFAULT_SEMANTIC_BREAKPOINT_PERCENTILE,
    ) -> bool:
        """
        Replace a knowledge entry's content, re-chunking and re-embedding.

        Implemented as delete-then-reinsert so that the chunk count after
        update reflects the new corpus and chosen strategy, not the old one.

        Parameters:
        -----------
        knowledge_base_id : str
            The unique ID of the knowledge to update.
        corpus : str
            The new text content.
        chunking_strategy, chunk_size, chunk_overlap:
            Chunking parameters. See ``ingest_knowledge``.

        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        existing = self.retrieve_knowledge(knowledge_base_id)
        if not existing:
            return False
        namespace = existing[0].get("namespace", "")

        if not self.delete_knowledge(knowledge_base_id):
            return False

        # Re-ingest under the SAME knowledge_base_id so any agent attachments
        # to that id continue to point at the new content.
        chunker = _resolve_chunker(chunking_strategy)
        chunks = chunker(
            corpus,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            breakpoint_percentile=breakpoint_percentile,
        )
        if not chunks:
            return True  # nothing to store; deletion already succeeded

        strategy_label = (
            chunking_strategy
            if isinstance(chunking_strategy, str)
            else getattr(chunking_strategy, "__name__", "custom")
        )
        now = datetime.now().isoformat()
        for index, chunk_text in enumerate(chunks):
            entry = {
                "content": chunk_text,
                "embedding": get_embedding(chunk_text),
                "namespace": namespace,
                "knowledge_base_id": knowledge_base_id,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "chunking_strategy": strategy_label,
                "created_at": now,
                "updated_at": now,
            }
            self.memory_provider.store(
                entry, memory_store_type=MemoryType.KNOWLEDGE_BASE
            )
        return True

    # ------------------------------------------------------------------
    # File / directory ingestion (centralized via extractors module)
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        source: FileSource,
        namespace: Optional[str] = None,
        *,
        filename: Optional[str] = None,
        chunking_strategy: Union[ChunkingStrategy, ChunkerFn] = "fixed",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        breakpoint_percentile: float = DEFAULT_SEMANTIC_BREAKPOINT_PERCENTILE,
    ) -> str:
        """Ingest a single file from disk, raw bytes, or a file-like object.

        Text extraction is dispatched by extension through
        :mod:`memorizz.long_term.semantic.extractors` — the same module the
        UI uses, so SDK and UI behave identically. PDFs go through ``pypdf``
        when ``memorizz[ingest-pdf]`` is installed; all text-like formats
        (``.txt``, ``.md``, ``.json``, ``.csv``, ``.py``, ``.html``, …) work
        out of the box. Register custom handlers via
        ``extractors.register_extractor``.

        Parameters
        ----------
        source:
            Path (``str``/``Path``), raw ``bytes``, or binary file-like.
        namespace:
            Namespace stored with the chunks. Defaults to the file name.
        filename:
            Required only when ``source`` is raw bytes or a file-like object
            without a usable ``.name``. Provides the extension used to pick
            an extractor.
        chunking_strategy, chunk_size, chunk_overlap, breakpoint_percentile:
            Forwarded to :meth:`ingest_knowledge`.

        Returns
        -------
        str
            The ``knowledge_base_id`` grouping every chunk from this file.

        Raises
        ------
        UnsupportedFileType, MissingExtractorDependency, EmptyDocumentError,
        ExtractionError:
            Propagated from the extractor; see :mod:`extractors` for details.
        """
        corpus = _extractors.extract_text(source, filename=filename)

        # Infer a sensible default namespace from the source.
        if namespace is None:
            if isinstance(source, (str, os.PathLike)) and not isinstance(
                source, (bytes, bytearray)
            ):
                namespace = Path(source).name
            else:
                namespace = filename or "upload"

        return self.ingest_knowledge(
            corpus=corpus,
            namespace=namespace,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            breakpoint_percentile=breakpoint_percentile,
        )

    def ingest_directory(
        self,
        root: Union[str, os.PathLike],
        namespace: Optional[str] = None,
        *,
        recursive: bool = True,
        pattern: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        chunking_strategy: Union[ChunkingStrategy, ChunkerFn] = "fixed",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        breakpoint_percentile: float = DEFAULT_SEMANTIC_BREAKPOINT_PERCENTILE,
    ) -> Dict[str, Any]:
        """Ingest every supported file under ``root``.

        One :meth:`ingest_file` call per file, collecting per-file results
        rather than failing fast — so a single corrupt PDF won't abort a
        big batch.

        Parameters
        ----------
        root:
            Directory to scan.
        namespace:
            Namespace applied to every ingested file. If None, each file
            uses its own filename as its namespace (so you can filter by
            source document later).
        recursive:
            If True, walk the tree; if False, scan only immediate children.
        pattern:
            Optional glob (``"*.pdf"``) applied before the extension filter.
        extensions:
            Whitelist of extensions to include. Defaults to every registered
            extractor — pass e.g. ``[".pdf"]`` to restrict the scan.
        chunking_strategy, chunk_size, chunk_overlap, breakpoint_percentile:
            Forwarded to :meth:`ingest_knowledge` for every file.

        Returns
        -------
        dict
            ``{"total": int, "ingested": int, "results": [
                {"path": str, "ok": bool, "knowledge_base_id": str,
                 "chunk_count": int, "error": str | None}, ...
            ]}``
        """
        paths = list(
            _extractors.iter_ingestable_files(
                root,
                recursive=recursive,
                pattern=pattern,
                extensions=extensions,
            )
        )
        results: List[Dict[str, Any]] = []
        for path in paths:
            ns = namespace or path.name
            entry: Dict[str, Any] = {
                "path": str(path),
                "ok": False,
                "knowledge_base_id": None,
                "chunk_count": 0,
                "error": None,
            }
            try:
                kb_id = self.ingest_file(
                    path,
                    namespace=ns,
                    chunking_strategy=chunking_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    breakpoint_percentile=breakpoint_percentile,
                )
            except ExtractorError as exc:
                entry["error"] = str(exc)
            except (
                EmptyDocumentError
            ) as exc:  # pragma: no cover - subtype of ExtractorError
                entry["error"] = str(exc)
            except Exception as exc:
                logger.exception("ingest_directory: failed on %s", path)
                entry["error"] = f"{type(exc).__name__}: {exc}"
            else:
                entry["ok"] = True
                entry["knowledge_base_id"] = kb_id
                try:
                    entry["chunk_count"] = len(self.retrieve_knowledge(kb_id))
                except Exception:
                    pass
            results.append(entry)

        return {
            "total": len(results),
            "ingested": sum(1 for r in results if r["ok"]),
            "results": results,
        }

    def attach_to_agent(self, agent, knowledge_base_id: str) -> bool:
        """
        Attach the knowledge base entry to a MemAgent.

        This updates the agent's configuration to include the knowledge base ID.

        Parameters:
        -----------
        agent : MemAgent
            The agent to attach the knowledge to.
        knowledge_base_id : str
            The unique ID of the knowledge to attach.

        Returns:
        --------
        bool
            True if the attachment was successful, False otherwise.
        """
        try:
            # Verify that the knowledge exists
            entries = self.retrieve_knowledge(knowledge_base_id)
            if not entries:
                return False

            # Store the knowledge_base_id in the agent's attributes if it doesn't exist
            if not hasattr(agent, "knowledge_base_ids"):
                agent.knowledge_base_ids = []

            # Add the ID if it's not already there
            if knowledge_base_id not in agent.knowledge_base_ids:
                agent.knowledge_base_ids.append(knowledge_base_id)
                _persist_agent(agent)

            return True
        except Exception as e:
            logger.error("Error attaching knowledge to agent: %s", e)
            return False

    def detach_from_agent(self, agent, knowledge_base_id: str) -> bool:
        """
        Remove a knowledge_base_id from an agent's ``knowledge_base_ids`` list.

        This does NOT delete the underlying knowledge entries — it only
        detaches them from this agent. Use :meth:`delete_knowledge` afterward
        if you also want to purge the data.

        Parameters
        ----------
        agent : MemAgent
        knowledge_base_id : str

        Returns
        -------
        bool
            True if the id was found and removed; False if the agent wasn't
            attached to this id in the first place, or if persistence failed.
        """
        try:
            ids = list(getattr(agent, "knowledge_base_ids", None) or [])
            if knowledge_base_id not in ids:
                return False
            ids.remove(knowledge_base_id)
            agent.knowledge_base_ids = ids
            _persist_agent(agent)
            return True
        except Exception as e:
            logger.error("Error detaching knowledge from agent: %s", e)
            return False
