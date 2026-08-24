"""LongMemEval-V2 memory backend built on MemoRizz primitives.

The official benchmark is intentionally optional.  Call
``register_longmemeval_v2_backend`` after adding an official checkout to
``sys.path``; this module then registers a concrete implementation against the
benchmark's own ``Memory`` interface without vendoring or modifying its code.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import types
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def load_longmemeval_v2_memory_registry(official_root: str | Path) -> Any:
    """Load the official base registry without importing unrelated backends.

    LongMemEval-V2's ``memory_modules.memory`` appends eager imports for every
    bundled backend. Some of those backends have mutually incompatible optional
    dependencies even when MemoRizz is the selected backend. The benchmark's
    base interface and registry are self-contained above that import block, so
    load exactly that public contract and leave the official harness untouched.
    """
    existing = sys.modules.get("memory_modules.memory")
    if existing is not None and all(
        hasattr(existing, name)
        for name in ("Memory", "register_memory", "build_memory")
    ):
        return existing

    package_root = Path(official_root).expanduser().resolve() / "memory_modules"
    source_path = package_root / "memory.py"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing official LongMemEval-V2 memory registry: {source_path}"
        )
    source = source_path.read_text(encoding="utf-8")
    marker = "\nfrom .no_retrieval import"
    base_source, separator, _ = source.partition(marker)
    if not separator:
        raise RuntimeError(
            "Unsupported LongMemEval-V2 registry layout: optional backend import "
            "boundary was not found"
        )

    package = sys.modules.get("memory_modules")
    if package is None:
        package = types.ModuleType("memory_modules")
        package.__path__ = [str(package_root)]
        package.__package__ = "memory_modules"
        sys.modules["memory_modules"] = package

    module = types.ModuleType("memory_modules.memory")
    module.__file__ = str(source_path)
    module.__package__ = "memory_modules"
    sys.modules["memory_modules.memory"] = module
    try:
        exec(compile(base_source, str(source_path), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop("memory_modules.memory", None)
        raise
    setattr(package, "memory", module)
    return module


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_accessibility_tree(value: Any, *, max_chars: int = 16_000) -> str:
    """Keep UI facts and affordances while removing structural tree noise."""
    lines: List[str] = []
    signals = (
        "statictext ",
        "button ",
        "link ",
        "option ",
        "textbox ",
        "searchbox ",
        "combobox ",
        "heading ",
        "checkbox ",
        "menuitem ",
        "tab ",
    )
    for raw_line in str(value or "").splitlines():
        normalized = " ".join(raw_line.strip().split())
        lowered = normalized.lower()
        if not normalized or not any(signal in lowered for signal in signals):
            continue
        normalized = re.sub(r"^\[[^\]]+\]\s*", "", normalized)
        normalized = re.sub(
            r",\s*(?:clickable|visible|focused|required)(?:=(?:True|False))?",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        lines.append(normalized)
    cleaned = "\n".join(lines)
    if len(cleaned) <= max_chars:
        return cleaned
    # Menus, dialogs, and late-rendered options frequently occur at the end of
    # an accessibility tree. A prefix-only cap silently erases exactly those
    # facts, so retain both boundaries within the deterministic size budget.
    marker = "\n[... accessibility tree middle omitted ...]\n"
    remaining = max(0, max_chars - len(marker))
    head = remaining // 2
    tail = remaining - head
    return cleaned[:head] + marker + cleaned[-tail:]


_LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "when",
    "which",
    "with",
}


def _lexical_terms(value: Any) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]+", str(value or "").lower())
        if token not in _LEXICAL_STOPWORDS
    ]


def _retrieval_query(value: Any) -> str:
    """Remove evaluator output-format boilerplate from the retrieval query."""
    query = str(value or "").strip()
    return re.sub(
        r"\n+\s*Mark your final answer\b.*$",
        "",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def rank_lexical_documents(
    query: str,
    documents: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    max_per_trajectory: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return bounded BM25-like matches over already tenant-scoped documents.

    ``max_per_trajectory`` prevents a single long trajectory from consuming the
    entire lexical lane.  Excluded rows are retained as a deterministic
    backfill, so sparse corpora can still satisfy ``limit``.
    """
    rows = [dict(item) for item in documents]
    query_counts = Counter(_lexical_terms(query))
    if not rows or not query_counts or limit <= 0:
        return []
    tokenized = [Counter(_lexical_terms(row.get("content"))) for row in rows]
    document_frequency = {
        term: sum(1 for counts in tokenized if counts.get(term, 0) > 0)
        for term in query_counts
    }
    total = len(rows)
    scored: List[tuple[float, Dict[str, Any]]] = []
    for row, counts in zip(rows, tokenized):
        score = 0.0
        for term, query_frequency in query_counts.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1.0
                + (total - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            score += (
                inverse_frequency
                * (1.0 + math.log(frequency))
                * (1.0 + math.log(query_frequency))
            )
        if score > 0:
            row["lexical_score"] = score
            scored.append((score, row))
    scored.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("trajectory_id") or ""),
            int(item[1].get("chunk_index") or 0),
        )
    )
    requested = max(0, int(limit))
    if not max_per_trajectory:
        return [row for _, row in scored[:requested]]

    cap = max(1, int(max_per_trajectory))
    selected: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for _, row in scored:
        trajectory_id = str(row.get("trajectory_id") or row.get("namespace") or "")
        if counts[trajectory_id] >= cap:
            deferred.append(row)
            continue
        selected.append(row)
        counts[trajectory_id] += 1
        if len(selected) >= requested:
            return selected
    if len(selected) < requested:
        selected.extend(deferred[: requested - len(selected)])
    return selected


def _state_text(trajectory_id: str, state: Mapping[str, Any]) -> str:
    fields = [
        f"trajectory_id: {trajectory_id}",
        f"state_index: {state.get('state_index')}",
        f"step: {state.get('step')}",
    ]
    for key in ("url", "action", "thought", "observation", "error"):
        value = state.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        fields.append(f"{key}: {value}")
    tree = _clean_accessibility_tree(
        state.get("accessibility_tree") or state.get("axtree") or ""
    )
    if tree:
        fields.append("visible_ui:\n" + tree)
    return "\n".join(fields)


def trajectory_chunks(
    trajectory: Mapping[str, Any], *, max_chars: int = 12_000
) -> List[str]:
    """Create stable, state-aligned text chunks suitable for embedding."""
    trajectory_id = str(trajectory.get("id") or uuid.uuid4())
    header = (
        f"trajectory_id: {trajectory_id}\n"
        f"domain: {trajectory.get('domain')}\n"
        f"environment: {trajectory.get('environment')}\n"
        f"outcome: {trajectory.get('outcome')}\n"
        f"goal: {trajectory.get('goal')}\n"
    )
    segments = [header]
    states = trajectory.get("states")
    if isinstance(states, list):
        segments.extend(
            _state_text(trajectory_id, state)
            for state in states
            if isinstance(state, Mapping)
        )
    chunks: List[str] = []
    current = ""
    for segment in segments:
        segment = str(segment).strip()
        if not segment:
            continue
        pieces = [
            segment[offset : offset + max_chars]
            for offset in range(0, len(segment), max_chars)
        ]
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current}\n\n{piece}".strip() if current else piece
    if current:
        chunks.append(current)
    return chunks


def trajectory_digest(trajectory: Mapping[str, Any], *, max_chars: int = 5_000) -> str:
    """Produce a compact episodic record used by MemoRizz summarization."""
    trajectory_id = str(trajectory.get("id") or "unknown")
    lines = [
        f"Trajectory {trajectory_id}",
        f"Goal: {trajectory.get('goal')}",
        f"Outcome: {trajectory.get('outcome')}",
    ]
    states = trajectory.get("states")
    if isinstance(states, list):
        for state in states:
            if not isinstance(state, Mapping):
                continue
            action = state.get("action")
            thought = state.get("thought")
            if action:
                lines.append(f"Step {state.get('step')} action: {action}")
            if thought:
                lines.append(f"Step {state.get('step')} lesson: {thought}")
            if sum(len(line) + 1 for line in lines) >= max_chars:
                break
    return "\n".join(lines)[:max_chars]


def register_longmemeval_v2_backend():
    """Register and return MemoRizz's backend with the official LME-V2 registry."""
    try:
        from memory_modules.memory import Memory, register_memory
    except ImportError as exc:  # pragma: no cover - depends on optional checkout
        raise ImportError(
            "The official LongMemEval-V2 repository must be on sys.path before "
            "registering the MemoRizz backend."
        ) from exc

    from memorizz import ContextPolicy, MemAgent
    from memorizz.benchmarks.common import (
        create_benchmark_memory_provider,
        normalize_memory_backend,
    )
    from memorizz.benchmarks.pricing import estimate_openai_text_cost
    from memorizz.embeddings import EmbeddingManager, set_global_embedding_manager
    from memorizz.enums import MemoryType
    from memorizz.llms.openai import OpenAI

    class _TrackedMemoryModel(OpenAI):
        """Account for MemoRizz-owned synthesis calls in benchmark reports."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.call_count = 0
            self.total_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
            }
            self.estimated_cost_usd = 0.0

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            response = super().generate(*args, **kwargs)
            usage = dict(self.get_last_usage() or {})
            for key in self.total_usage:
                self.total_usage[key] += max(0, int(usage.get(key) or 0))
            self.estimated_cost_usd += estimate_openai_text_cost(self.model, usage)
            self.call_count += 1
            return response

    @register_memory
    class MemoRizzLongMemEvalV2Memory(Memory):
        memory_type = "memorizz"

        def __init__(self, memory_params: Dict[str, object]) -> None:
            super().__init__(memory_params)
            root_value = memory_params.get("workspace_dir") or memory_params.get(
                "root_path"
            )
            if not root_value:
                raise ValueError(
                    "MemoRizz LongMemEval-V2 memory requires workspace_dir"
                )
            self.root_path = Path(str(root_value)).expanduser().resolve()
            self.root_path.mkdir(parents=True, exist_ok=True)
            self.memory_id = str(
                memory_params.get("memory_id") or f"longmemeval-v2:{uuid.uuid4().hex}"
            )
            self.user_id = str(memory_params.get("user_id") or "longmemeval-v2")
            self.ingest_thread_id = "longmemeval-v2-ingest"
            self.query_thread_id = "longmemeval-v2-query"
            self.top_k = max(1, int(memory_params.get("top_k") or 12))
            self.lexical_ratio = max(
                0.0, min(float(memory_params.get("lexical_ratio", 0.5)), 1.0)
            )
            self.summary_top_k = max(0, int(memory_params.get("summary_top_k") or 3))
            self.chunk_chars = max(
                2_000, min(int(memory_params.get("chunk_chars") or 12_000), 28_000)
            )
            self.embedding_batch_size = max(
                1, min(int(memory_params.get("embedding_batch_size") or 64), 128)
            )
            self.summary_batch_size = max(
                2, int(memory_params.get("summary_batch_size") or 20)
            )
            self.enable_summaries = bool(memory_params.get("enable_summaries", True))
            self.cache_probe = bool(memory_params.get("cache_probe", True))
            self.memory_backend = normalize_memory_backend(
                str(memory_params.get("memory_backend") or "filesystem")
            )
            self.learning_control_plane = bool(
                memory_params.get("learning_control_plane", True)
            )
            self.evidence_token_budget = max(
                256, int(memory_params.get("evidence_token_budget") or 6_000)
            )
            self.model_name = str(memory_params.get("model") or "gpt-5-mini")
            self.embedding_model = str(
                memory_params.get("embedding_model") or "text-embedding-3-small"
            )
            self.embedding_dimensions = max(
                64, int(memory_params.get("embedding_dimensions") or 256)
            )
            api_key_env = str(memory_params.get("api_key_env") or "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"{api_key_env} is required for this backend")

            self.embedding_manager = EmbeddingManager(
                "openai",
                {
                    "api_key": api_key,
                    "model": self.embedding_model,
                    "dimensions": self.embedding_dimensions,
                },
            )
            set_global_embedding_manager(self.embedding_manager)
            self.provider = create_benchmark_memory_provider(
                self.memory_backend,
                filesystem_root=self.root_path / "provider",
                embedding_provider=self.embedding_manager,
            )
            self.model = _TrackedMemoryModel(
                api_key=api_key,
                model=self.model_name,
                api_mode="responses",
                reasoning_effort="low",
                max_tokens=4_096,
            )
            self.agent: Optional[MemAgent] = None
            self._pending_chunks: List[Dict[str, Any]] = []
            self._lexical_documents: List[Dict[str, Any]] = []
            self._pending_digests: List[Dict[str, Any]] = []
            self._trajectory_count = 0
            self._chunk_count = 0
            self._summary_ids: List[str] = []
            self._compaction_duration_seconds = 0.0
            self._last_query: Optional[str] = None
            self._last_request_context: Optional[Dict[str, Any]] = None
            self._last_memory_context: List[Dict[str, str]] = []
            self._last_query_stats: Dict[str, Any] = {}
            self._ingest_started = time.perf_counter()

        def _flush_chunks(self, *, force: bool = False) -> None:
            while len(self._pending_chunks) >= self.embedding_batch_size or (
                force and self._pending_chunks
            ):
                count = min(self.embedding_batch_size, len(self._pending_chunks))
                batch = self._pending_chunks[:count]
                del self._pending_chunks[:count]
                embeddings = self.embedding_manager.get_embeddings(
                    [str(item["content"]) for item in batch]
                )
                rows = []
                for item, embedding in zip(batch, embeddings):
                    self._lexical_documents.append(dict(item))
                    rows.append(
                        {
                            **item,
                            "embedding": embedding,
                            "memory_id": self.memory_id,
                            "user_id": self.user_id,
                            "agent_id": "longmemeval-v2-memory",
                        }
                    )
                self.provider.store_many(rows, MemoryType.KNOWLEDGE_BASE)
                self._chunk_count += len(rows)

        def _flush_digests(self, *, force: bool = False) -> None:
            while len(self._pending_digests) >= self.embedding_batch_size or (
                force and self._pending_digests
            ):
                count = min(self.embedding_batch_size, len(self._pending_digests))
                batch = self._pending_digests[:count]
                del self._pending_digests[:count]
                embeddings = self.embedding_manager.get_embeddings(
                    [str(item["content"]) for item in batch]
                )
                self.provider.store_many(
                    [
                        {**item, "embedding": embedding}
                        for item, embedding in zip(batch, embeddings)
                    ],
                    MemoryType.CONVERSATION_MEMORY,
                )

        def insert(self, trajectory: Dict[str, object]) -> None:
            trajectory_id = str(trajectory.get("id") or uuid.uuid4().hex)
            chunks = trajectory_chunks(trajectory, max_chars=self.chunk_chars)
            for chunk_index, content in enumerate(chunks):
                self._pending_chunks.append(
                    {
                        "content": content,
                        "namespace": trajectory_id,
                        "trajectory_id": trajectory_id,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                        "source_type": "longmemeval_v2_trajectory",
                        "timestamp": _utc_now(),
                    }
                )
            self._flush_chunks()
            digest = trajectory_digest(trajectory)
            self._pending_digests.append(
                {
                    "role": "tool",
                    "content": digest,
                    "thread_id": self.ingest_thread_id,
                    "memory_id": self.memory_id,
                    "user_id": self.user_id,
                    "agent_id": "longmemeval-v2-memory",
                    "trajectory_id": trajectory_id,
                    "timestamp": _utc_now(),
                }
            )
            self._flush_digests()
            self._trajectory_count += 1

        def _ensure_agent_and_compaction(self) -> None:
            self._flush_chunks(force=True)
            self._flush_digests(force=True)
            if self.agent is None:
                self.agent = MemAgent(
                    model=self.model,
                    llm_config={
                        "provider": "openai",
                        "model": self.model_name,
                        "api_mode": "responses",
                    },
                    instruction=(
                        "You are the MemoRizz memory synthesizer for LongMemEval-V2. "
                        "Use only retrieved memory evidence. Produce a compact factual "
                        "briefing for the downstream reader, preserve exact UI labels, "
                        "ordered workflow steps, outcomes, exceptions, and uncertainty. "
                        "Never invent benchmark facts and never mention a gold answer."
                    ),
                    memory_provider=self.provider,
                    memory_ids=[self.memory_id],
                    memory_types=[
                        MemoryType.CONVERSATION_MEMORY,
                        MemoryType.KNOWLEDGE_BASE,
                        MemoryType.SUMMARIES,
                    ],
                    max_steps=4,
                    semantic_cache=True,
                    semantic_cache_config={
                        "similarity_threshold": 0.995,
                        "scope": "session",
                        "ttl_hours": 24.0,
                        "admission_policy": "read_only_deterministic",
                        "freshness_by_domain": {"longmemeval_v2": 86_400.0},
                    },
                    automations_enabled=False,
                    context_policy=ContextPolicy(
                        progressive_tool_disclosure=True,
                        tool_top_k=3,
                        max_tool_invocations_per_turn=4,
                    ),
                    name=f"MemoRizz LongMemEval-V2 {self.memory_id[-8:]}",
                    auto_register=True,
                    learning_control_plane=(
                        {
                            "enabled": True,
                            "evidence_sources": ["knowledge_base", "summaries"],
                            "evidence_token_budget": self.evidence_token_budget,
                            "evidence_max_items": min(self.top_k, 10),
                            "evidence_candidates_per_source": self.top_k,
                            "evidence_max_per_source": min(self.top_k, 10),
                        }
                        if self.learning_control_plane
                        else False
                    ),
                )
            if self.enable_summaries and not self._summary_ids:
                started = time.perf_counter()
                self._summary_ids = self.agent.generate_summaries(
                    days_back=1,
                    max_memories_per_summary=self.summary_batch_size,
                    memory_id=self.memory_id,
                    user_id=self.user_id,
                    thread_id=self.ingest_thread_id,
                )
                self._compaction_duration_seconds = time.perf_counter() - started

        @staticmethod
        def _content(row: Mapping[str, Any]) -> str:
            value = row.get("content")
            if isinstance(value, Mapping):
                value = value.get("content") or value.get("text")
            return str(value or "").strip()

        def query(
            self, query: str, query_image: Optional[str] = None
        ) -> List[Dict[str, str]]:
            started = time.perf_counter()
            self._ensure_agent_and_compaction()
            assert self.agent is not None
            retrieval_query = _retrieval_query(query)
            semantic_rows = (
                self.provider.retrieve_by_query(
                    retrieval_query,
                    MemoryType.KNOWLEDGE_BASE,
                    limit=self.top_k,
                    memory_id=self.memory_id,
                    user_id=self.user_id,
                )
                or []
            )
            lexical_limit = min(
                self.top_k, max(0, round(self.top_k * self.lexical_ratio))
            )
            lexical_rows = rank_lexical_documents(
                retrieval_query,
                self._lexical_documents,
                limit=lexical_limit,
                max_per_trajectory=max(1, math.ceil(lexical_limit / 3)),
            )
            # Reserve a lexical lane for exact UI labels and code symbols,
            # then fill the remaining fixed context budget with semantic hits.
            raw_rows: List[Dict[str, Any]] = []
            seen_sources = set()
            for row in [*lexical_rows, *semantic_rows]:
                source = (
                    row.get("trajectory_id"),
                    row.get("chunk_index"),
                    row.get("_id") or row.get("id"),
                )
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                raw_rows.append(dict(row))
                if len(raw_rows) >= self.top_k:
                    break
            summary_rows = (
                self.provider.retrieve_by_query(
                    query,
                    MemoryType.SUMMARIES,
                    limit=self.summary_top_k,
                    memory_id=self.memory_id,
                    user_id=self.user_id,
                )
                or []
            )
            raw_evidence = [self._content(row) for row in raw_rows]
            raw_evidence = [item for item in raw_evidence if item]
            summaries = [self._content(row) for row in summary_rows]
            summaries = [item for item in summaries if item]
            request_context = {
                "benchmark": "longmemeval-v2",
                "cache_domain": "longmemeval_v2",
                "cache_data_version": self.memory_id,
                "query_image_present": bool(query_image),
            }
            if self.learning_control_plane:
                # Keep a small lexical lane for exact UI labels. Semantic and
                # summary selection is owned by the bounded EvidencePack.
                request_context["lexical_trajectory_evidence"] = [
                    self._content(row) for row in lexical_rows[:3]
                ]
            else:
                request_context["retrieved_trajectory_chunks"] = raw_evidence
                request_context["compacted_trajectory_summaries"] = summaries
            briefing = self.agent.run(
                query,
                memory_id=self.memory_id,
                thread_id=self.query_thread_id,
                user_id=self.user_id,
                context=request_context,
            )
            context_items: List[Dict[str, str]] = []
            if summaries and not self.learning_control_plane:
                context_items.append(
                    {
                        "type": "text",
                        "value": "MemoRizz compacted experience:\n\n"
                        + "\n\n---\n\n".join(summaries),
                    }
                )
            if raw_evidence and not self.learning_control_plane:
                context_items.append(
                    {
                        "type": "text",
                        "value": "MemoRizz retrieved trajectory evidence:\n\n"
                        + "\n\n---\n\n".join(raw_evidence),
                    }
                )
            context_items.append(
                {"type": "text", "value": "MemoRizz memory briefing:\n" + briefing}
            )
            self._last_query = query
            self._last_request_context = request_context
            self._last_memory_context = context_items
            self._last_query_stats = {
                "duration_seconds": time.perf_counter() - started,
                "retrieval_query_chars": len(retrieval_query),
                "raw_match_count": len(raw_evidence),
                "lexical_match_count": len(lexical_rows),
                "lexical_ratio": self.lexical_ratio,
                "summary_match_count": len(summaries),
                "memory_context_chars": sum(
                    len(str(item.get("value") or "")) for item in context_items
                ),
                "learning_control_plane": self.learning_control_plane,
            }
            return context_items

        def post_query_hook(
            self,
            *,
            query: str,
            query_image: Optional[str],
            memory_context: List[Dict[str, str]],
        ) -> Dict[str, object]:
            assert self.agent is not None
            before = self.agent.semantic_cache_stats()
            cache_probe_match = None
            if self.cache_probe and self._last_request_context is not None:
                probe = self.agent.run(
                    query,
                    memory_id=self.memory_id,
                    thread_id=self.query_thread_id,
                    user_id=self.user_id,
                    context=self._last_request_context,
                )
                expected = memory_context[-1]["value"].split(
                    "MemoRizz memory briefing:\n", 1
                )[-1]
                cache_probe_match = probe == expected
            after = self.agent.semantic_cache_stats()
            try:
                observability = self.agent.observability_summary(
                    self.memory_id, self.user_id
                )
            except Exception as exc:
                observability = {"error": f"{type(exc).__name__}: {exc}"}
            learning_report = self.agent.learning_report(
                memory_id=self.memory_id, user_id=self.user_id
            )
            return {
                "backend": "memorizz",
                "memory_backend": self.memory_backend,
                "memory_id": self.memory_id,
                "trajectory_count": self._trajectory_count,
                "indexed_chunk_count": self._chunk_count,
                "summary_count": len(self._summary_ids),
                "compaction_duration_seconds": self._compaction_duration_seconds,
                "ingest_elapsed_seconds": time.perf_counter() - self._ingest_started,
                "query": dict(self._last_query_stats),
                "semantic_cache_before_probe": before,
                "semantic_cache_after_probe": after,
                "semantic_cache_probe_match": cache_probe_match,
                "observability": observability,
                "learning_control_plane": learning_report,
                "evidence_decision": self.agent.explain_memory_decision(),
                "memory_model_usage": dict(self.model.total_usage),
                "memory_model_call_count": self.model.call_count,
                "memory_model_estimated_cost_usd": self.model.estimated_cost_usd,
                "features": {
                    "semantic_vector_retrieval": True,
                    "hybrid_lexical_retrieval": self.lexical_ratio > 0,
                    "source_diversified_retrieval": self.lexical_ratio > 0,
                    "semantic_cache": True,
                    "summarization": self.enable_summaries,
                    "conversation_compaction": self.enable_summaries,
                    "tenant_scope": self.user_id,
                    "bounded_evidence_pack": self.learning_control_plane,
                    "immutable_learning_events": self.learning_control_plane,
                },
            }

        def _save_backend(self, output_dir: Path) -> None:
            compiler_report = None
            learning_report = None
            if self.agent is not None and self.learning_control_plane:
                compiler_report = self.agent.compile_memory(
                    memory_id=self.memory_id,
                    user_id=self.user_id,
                    thread_id=self.query_thread_id,
                )
                learning_report = self.agent.learning_report(
                    memory_id=self.memory_id, user_id=self.user_id
                )
            (output_dir / "memorizz_backend.json").write_text(
                json.dumps(
                    {
                        "memory_id": self.memory_id,
                        "trajectory_count": self._trajectory_count,
                        "indexed_chunk_count": self._chunk_count,
                        "summary_ids": self._summary_ids,
                        "memory_backend": self.memory_backend,
                        "learning_control_plane": learning_report,
                        "memory_compiler": compiler_report,
                        "memory_model_usage": dict(self.model.total_usage),
                        "memory_model_call_count": self.model.call_count,
                        "memory_model_estimated_cost_usd": (
                            self.model.estimated_cost_usd
                        ),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if self.agent is not None:
                self.agent.close(close_memory_provider=True)

    return MemoRizzLongMemEvalV2Memory


__all__ = [
    "load_longmemeval_v2_memory_registry",
    "rank_lexical_documents",
    "register_longmemeval_v2_backend",
    "trajectory_chunks",
    "trajectory_digest",
]
