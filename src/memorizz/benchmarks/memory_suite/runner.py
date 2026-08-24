"""Local-only MemoRizz runner shared by the five memory benchmarks."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ...embeddings import EmbeddingManager
from ...enums import MemoryType
from ...llms.ollama import OllamaLLM
from ..common import create_benchmark_memory_provider
from ..longmemeval_v2 import rank_lexical_documents
from .catalog import get_benchmark_spec
from .corpus_cache import CorpusEmbeddingCache
from .datasets import verify_dataset
from .loaders import load_benchmark_cases
from .models import MemoryBenchmarkCase, MemoryDocument
from .protocols import (
    assess_comparability,
    get_evaluation_profile,
    get_protocol_manifest,
    resolve_profile_limit,
)
from .retrieval import FusionConfig, QueryExpander, build_query_variants, fuse_rankings
from .scoring import (
    build_judge_prompt,
    citation_metrics,
    parse_judge_response,
    parse_reader_response,
    reader_response_needs_repair,
    retrieval_metrics,
    score_answer,
)
from .semantic_memory import derive_semantic_memories

ProgressCallback = Callable[[str], None]


class _AgentTemplateProvider:
    """Read-only adapter used to hydrate one isolated evaluation agent."""

    def __init__(self, template: Any) -> None:
        self.template = template

    def retrieve_memagent(self, agent_id: str) -> Any:
        template_id = str(getattr(self.template, "agent_id", None) or "")
        return (
            self.template if not template_id or template_id == str(agent_id) else None
        )


_OPENAI_TOKEN_PRICING_USD_PER_MILLION: Dict[str, Dict[str, float]] = {
    "gpt-5.5": {"input": 5.0, "cached_input": 0.5, "output": 30.0},
}


def _mean(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, samples: int = 1_000
) -> Dict[str, float | int]:
    """Return a deterministic percentile interval for one mean."""

    present = [float(value) for value in values]
    if not present:
        return {"samples": 0, "mean": 0.0, "lower": 0.0, "upper": 0.0}
    observed = statistics.fmean(present)
    if len(present) == 1:
        return {
            "samples": 1,
            "mean": observed,
            "lower": observed,
            "upper": observed,
        }
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choice(present) for _ in present)
        for _ in range(max(100, int(samples)))
    )
    lower_index = int(0.025 * (len(means) - 1))
    upper_index = int(0.975 * (len(means) - 1))
    return {
        "samples": len(present),
        "mean": observed,
        "lower": means[lower_index],
        "upper": means[upper_index],
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    present = sorted(float(value) for value in values)
    if not present:
        return None
    index = max(0, min(len(present) - 1, round((len(present) - 1) * percentile)))
    return present[index]


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _cases_fingerprint(cases: Sequence[MemoryBenchmarkCase]) -> str:
    """Hash the exact selected questions and de-duplicated memory corpora."""
    digest = hashlib.sha256()
    seen_corpora = set()
    for case in cases:
        for value in (case.case_id, case.corpus_id, case.question, *case.answers):
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        if case.corpus_id in seen_corpora:
            continue
        seen_corpora.add(case.corpus_id)
        for document in case.documents:
            digest.update(document.source_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document.content.encode("utf-8"))
            digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _file_fingerprint(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _dependency_lock_fingerprint() -> str:
    repository = Path(__file__).resolve().parents[4]
    for filename in ("uv.lock", "requirements.lock", "pyproject.toml"):
        fingerprint = _file_fingerprint(repository / filename)
        if fingerprint:
            return fingerprint
    return f"python:{platform.python_version()}"


class MemorySuiteRunner:
    """Evaluate normalized cases with MemoRizz storage and selectable readers.

    ``model`` and ``embedding_manager`` are injectable to keep unit tests fast;
    production calls always default to the local Ollama providers below.
    """

    def __init__(
        self,
        *,
        workspace: str | Path,
        model_provider: str = "ollama",
        model_name: str = "qwen2.5:3b",
        judge_model_name: str | None = None,
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
        top_k: int = 8,
        lexical_ratio: float = 0.35,
        candidate_pool_size: int | None = None,
        query_expander: QueryExpander | None = None,
        query_expansion: bool = True,
        temporal_weight: float = 0.05,
        entity_weight: float = 0.05,
        rerank_weight: float = 0.15,
        diversity_lambda: float = 0.9,
        max_evidence_chars: int = 24_000,
        reasoning_effort: str | None = "low",
        max_output_tokens: int = 512,
        oracle_reader: bool = True,
        profile: str = "smoke",
        seed: int = 0,
        corpus_cache_dir: str | Path | None = None,
        corpus_cache_enabled: bool = True,
        embedding_model_digest: str | None = None,
        dataset_readiness: Mapping[str, Any] | None = None,
        memory_backend: str = "filesystem",
        semantic_memory: bool = True,
        reader_repair: bool = True,
        evaluation_mode: str = "retrieval",
        agent_template: Any = None,
        agent: Any = None,
        model: Any = None,
        judge_model: Any = None,
        embedding_manager: Any = None,
        provider: Any = None,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model_provider = str(model_provider or "ollama").strip().lower()
        if self.model_provider not in {"ollama", "openai"}:
            raise ValueError("model_provider must be 'ollama' or 'openai'")
        self.model_name = str(model_name)
        self.judge_model_name = str(judge_model_name or model_name)
        self.embedding_model = str(embedding_model)
        self.ollama_host = str(ollama_host)
        self.top_k = max(1, int(top_k))
        self.lexical_ratio = max(0.0, min(float(lexical_ratio), 1.0))
        self.candidate_pool_size = max(
            self.top_k,
            int(candidate_pool_size or max(256, self.top_k * 16)),
        )
        self.query_expander = query_expander
        self.query_expansion = bool(query_expansion)
        self.fusion_config = FusionConfig(
            top_k=self.top_k,
            candidate_pool_size=self.candidate_pool_size,
            semantic_weight=1.0,
            lexical_weight=self.lexical_ratio,
            temporal_weight=max(0.0, float(temporal_weight)),
            entity_weight=max(0.0, float(entity_weight)),
            rerank_weight=max(0.0, float(rerank_weight)),
            diversity_lambda=float(diversity_lambda),
        )
        self.max_evidence_chars = max(512, int(max_evidence_chars))
        self.reasoning_effort = (
            str(reasoning_effort).strip().lower() if reasoning_effort else None
        )
        self.max_output_tokens = max(64, int(max_output_tokens))
        self.oracle_reader = bool(oracle_reader)
        self.profile = get_evaluation_profile(profile).name
        self.seed = int(seed)
        self.embedding_model_digest = str(embedding_model_digest or "").strip() or None
        self.dataset_readiness = dict(dataset_readiness or {})
        self.memory_backend = str(memory_backend or "filesystem").strip().lower()
        self.semantic_memory = bool(semantic_memory)
        self.reader_repair = bool(reader_repair)
        self.evaluation_mode = str(evaluation_mode or "retrieval").strip().lower()
        if self.evaluation_mode not in {"retrieval", "memagent"}:
            raise ValueError("evaluation_mode must be 'retrieval' or 'memagent'")
        self.progress = progress or (lambda _line: None)
        self._run_nonce = uuid.uuid4().hex[:12]
        self.memory_root = self.workspace / "runs" / self._run_nonce / "memory"
        self.corpus_cache = CorpusEmbeddingCache(
            corpus_cache_dir or self.workspace.parent / "corpus-cache",
            enabled=corpus_cache_enabled,
        )
        self.embedding_manager = embedding_manager or EmbeddingManager(
            "ollama",
            {
                "model": self.embedding_model,
                "base_url": self.ollama_host,
                "timeout": 120,
            },
        )
        self.model = model or self._create_model(self.model_name, judge=False)
        if judge_model is not None:
            self.judge_model = judge_model
        elif judge_model_name and judge_model_name != model_name:
            self.judge_model = self._create_model(self.judge_model_name, judge=True)
        else:
            self.judge_model = self.model
        self.provider = provider or create_benchmark_memory_provider(
            self.memory_backend,
            filesystem_root=self.memory_root,
            embedding_provider=self.embedding_manager,
        )
        self.agent = agent
        self.agent_template_id: str | None = None
        self.agent_template_name: str | None = None
        self.agent_evaluation_overrides: Dict[str, Any] = {}
        if self.evaluation_mode == "memagent" and self.agent is None:
            if agent_template is None:
                raise ValueError("memagent evaluation requires an agent template")
            self.agent = self._create_evaluation_agent(agent_template)
        if self.evaluation_mode == "memagent" and self.agent is not None:
            self.agent_template_id = (
                str(getattr(self.agent, "agent_id", "") or "") or None
            )
            self.agent_template_name = (
                str(getattr(self.agent, "name", "") or "") or None
            )
        self._corpora: Dict[str, Dict[str, Any]] = {}
        self._judge_seconds: List[float] = []
        self._usage = {
            "generation_prompt_tokens": 0,
            "generation_completion_tokens": 0,
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
            "generation_cached_tokens": 0,
            "generation_reasoning_tokens": 0,
            "judge_cached_tokens": 0,
            "judge_reasoning_tokens": 0,
            "oracle_prompt_tokens": 0,
            "oracle_completion_tokens": 0,
            "oracle_cached_tokens": 0,
            "oracle_reasoning_tokens": 0,
        }

    def _create_evaluation_agent(self, template: Any) -> Any:
        """Hydrate one agent template onto the isolated benchmark provider."""

        from ...memagent import MemAgent
        from ...memagent.models import MemAgentModel
        from ...retrieval import RetrievalPolicy

        if isinstance(template, Mapping):
            template = MemAgentModel.model_validate(dict(template))
        if not isinstance(template, MemAgentModel):
            raise TypeError("agent_template must be a MemAgentModel or mapping")

        template_id = str(template.agent_id or f"memory-suite-{self._run_nonce}")
        memory_types = list(template.memory_types or [])
        if MemoryType.KNOWLEDGE_BASE.value not in {
            str(getattr(item, "value", item)) for item in memory_types
        }:
            memory_types.append(MemoryType.KNOWLEDGE_BASE.value)

        base_policy = RetrievalPolicy.from_value(template.retrieval_policy)
        retrieval_policy = {
            **base_policy.to_dict(),
            "knowledge_base_scope": "memory",
            "candidate_limit": self.candidate_pool_size,
            "max_items": self.top_k,
            "query_expansion": self.query_expansion,
            "max_query_variants": self.fusion_config.max_query_variants,
            "dedupe_parent_sources": True,
        }
        instruction = str(template.instruction or "").rstrip()
        instruction += (
            "\n\nEvaluation output contract: answer from the automatically retrieved "
            "memory context without calling tools. Return JSON only with "
            'the shape {"answer":"...","source_ids":["..."],'
            '"abstained":false}. Cite only source IDs present in memory context. '
            "For temporal questions, resolve relative dates from the supplied "
            "event timestamp and return an absolute date with a named month. "
            "When the input describes a situation instead of asking an explicit "
            "question, respond with the applicable prior constraint, preference, "
            "lesson, or advice."
        )
        isolated_template = template.model_copy(
            update={
                "agent_id": template_id,
                "delegates": [],
                "tools": [],
                "internet_access_provider": None,
                "internet_access_config": None,
                "mcp_servers": [],
                "sandbox_provider": None,
                "browser_control": None,
                "meta_harness": False,
                "skill_retrieval": False,
                "continual_learning": False,
                "learning_control_plane": False,
                "learning_control_plane_config": None,
                "self_aware": False,
                "automations_enabled": False,
                "memory_ids": [],
                "knowledge_base_ids": [],
                "memory_types": memory_types or template.memory_types,
            }
        )
        agent = MemAgent.load(
            template_id,
            memory_provider=_AgentTemplateProvider(isolated_template),
            runtime_memory_provider=self.provider,
            model=self.model,
            instruction=instruction,
            max_steps=1,
            tools=[],
            delegates=[],
            mcp_servers=[],
            internet_access_provider=None,
            sandbox_provider=None,
            browser_control=None,
            meta_harness=False,
            skill_retrieval=False,
            continual_learning=False,
            learning_control_plane=False,
            self_aware=False,
            automations_enabled=False,
            memory_ids=[],
            knowledge_base_ids=[],
            memory_types=memory_types or template.memory_types,
            retrieval_policy=retrieval_policy,
            semantic_cache=bool(template.semantic_cache),
            semantic_cache_config=template.semantic_cache_config,
            auto_register=False,
            streaming=False,
        )
        self.agent_evaluation_overrides = {
            "isolated_runtime_memory": True,
            "side_effect_tools_disabled": True,
            "max_steps": 1,
            "knowledge_base_scope": "memory",
            "retrieval_candidate_limit": self.candidate_pool_size,
            "retrieval_max_items": self.top_k,
            "query_expansion": self.query_expansion,
            "dedupe_parent_sources": True,
            "reader_model_override": self.model_name,
        }
        return agent

    def _create_model(self, model_name: str, *, judge: bool) -> Any:
        if self.model_provider == "ollama":
            return OllamaLLM(
                model=model_name,
                host=self.ollama_host,
                temperature=0,
                seed=0,
                num_predict=256 if judge else self.max_output_tokens,
                think=False,
                timeout=180,
            )

        from ...llms.openai import OpenAI

        return OpenAI(
            model=model_name,
            api_mode="responses",
            reasoning_effort=self.reasoning_effort,
            max_completion_tokens=(
                min(self.max_output_tokens, 256) if judge else self.max_output_tokens
            ),
        )

    def _capture_usage(self, model: Any, *, lane: str) -> None:
        getter = getattr(model, "get_last_usage", None)
        usage = getter() if callable(getter) else None
        if not isinstance(usage, Mapping):
            return
        self._usage[f"{lane}_prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self._usage[f"{lane}_completion_tokens"] += int(
            usage.get("completion_tokens") or 0
        )
        self._usage[f"{lane}_cached_tokens"] += int(usage.get("cached_tokens") or 0)
        self._usage[f"{lane}_reasoning_tokens"] += int(
            usage.get("reasoning_tokens") or 0
        )

    @staticmethod
    def _pricing_for_model(model_name: str) -> Dict[str, float] | None:
        normalized = str(model_name or "").lower()
        for prefix, pricing in _OPENAI_TOKEN_PRICING_USD_PER_MILLION.items():
            if normalized == prefix or normalized.startswith(f"{prefix}-"):
                return pricing
        return None

    def _lane_cost(self, lane: str, model_name: str) -> float | None:
        if self.model_provider != "openai":
            return 0.0
        pricing = self._pricing_for_model(model_name)
        if pricing is None:
            return None
        prompt_tokens = int(self._usage[f"{lane}_prompt_tokens"])
        cached_tokens = min(prompt_tokens, int(self._usage[f"{lane}_cached_tokens"]))
        uncached_tokens = max(0, prompt_tokens - cached_tokens)
        completion_tokens = int(self._usage[f"{lane}_completion_tokens"])
        return (
            uncached_tokens * pricing["input"]
            + cached_tokens * pricing["cached_input"]
            + completion_tokens * pricing["output"]
        ) / 1_000_000

    def _estimated_cost(self) -> float | None:
        generation = self._lane_cost("generation", self.model_name)
        judge = self._lane_cost("judge", self.judge_model_name)
        oracle = self._lane_cost("oracle", self.model_name)
        if generation is None or judge is None or oracle is None:
            return None
        return generation + judge + oracle

    def _embedding_identity(self) -> Dict[str, Any]:
        provider = "custom"
        dimensions = None
        getter = getattr(self.embedding_manager, "get_provider_info", None)
        if callable(getter):
            try:
                info = getter()
            except Exception:
                info = {}
            if isinstance(info, Mapping):
                provider = str(info.get("provider") or provider)
                dimensions = info.get("dimensions")
        return {
            "provider": provider,
            "model": self.embedding_model,
            "model_digest": self.embedding_model_digest or "unresolved",
            "dimensions": dimensions,
            "chunking": "loader-stable-v1",
        }

    def _memory_id(self, corpus_id: str) -> str:
        stable = uuid.uuid5(uuid.NAMESPACE_URL, corpus_id).hex[:16]
        return f"memory-suite:{self._run_nonce}:{stable}"

    def _ingest_corpus(
        self, corpus_id: str, documents: Sequence[MemoryDocument]
    ) -> Dict[str, Any]:
        cached = self._corpora.get(corpus_id)
        if cached is not None:
            return cached
        started = time.perf_counter()
        memory_id = self._memory_id(corpus_id)
        original_documents = list(documents)
        derived_documents = (
            derive_semantic_memories(original_documents) if self.semantic_memory else []
        )
        indexed_documents = [*original_documents, *derived_documents]
        contents = [document.content for document in indexed_documents]
        embedding_identity = self._embedding_identity()
        embeddings, cache_state = self.corpus_cache.load(
            indexed_documents, embedding_identity
        )
        embedding_started = time.perf_counter()
        if embeddings is None:
            embeddings = self.embedding_manager.get_embeddings(contents)
            write_state = self.corpus_cache.store(
                indexed_documents, embedding_identity, embeddings
            )
            cache_state["write"] = write_state
            cache_state["bytes"] = int(write_state.get("bytes") or 0)
        embedding_seconds = time.perf_counter() - embedding_started
        if len(embeddings) != len(indexed_documents):
            raise RuntimeError("Embedding count did not match corpus document count")
        lexical_rows: List[Dict[str, Any]] = []
        for document, embedding in zip(indexed_documents, embeddings):
            row = {
                "_id": uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self._run_nonce}:{corpus_id}:{document.source_id}",
                ).hex,
                "content": document.content,
                "source_id": document.source_id,
                "parent_source_id": document.parent_source_id or document.source_id,
                "linked_source_ids": list(document.linked_source_ids),
                "memory_id": memory_id,
                "knowledge_base_id": memory_id,
                "user_id": "memory-suite",
                "namespace": corpus_id,
                "embedding": embedding,
                "metadata": dict(document.metadata),
            }
            lexical_rows.append(dict(row))
        self.provider.store_many(lexical_rows, MemoryType.KNOWLEDGE_BASE)
        state = {
            "memory_id": memory_id,
            "knowledge_base_id": memory_id,
            "documents": lexical_rows,
            "document_count": len(indexed_documents),
            "original_document_count": len(original_documents),
            "derived_semantic_memory_count": len(derived_documents),
            "ingestion_seconds": time.perf_counter() - started,
            "embedding_seconds": embedding_seconds,
            "embedding_cache": cache_state,
        }
        self._corpora[corpus_id] = state
        self.progress(
            f"Ingested corpus {corpus_id} once ({len(original_documents)} source + "
            f"{len(derived_documents)} semantic records, "
            f"{state['ingestion_seconds']:.2f}s, embedding cache "
            f"{'hit' if cache_state.get('hit') else 'miss'})."
        )
        return state

    def _retrieve(
        self, case: MemoryBenchmarkCase, corpus: Mapping[str, Any]
    ) -> tuple[List[Dict[str, Any]], Dict[str, float], List[str]]:
        rankings = []
        variants = build_query_variants(
            case.question,
            expander=self.query_expander,
            max_variants=self.fusion_config.max_query_variants,
            expand_concepts=self.query_expansion,
        )
        # Semantic query variants form one lane group in fusion. Each variant
        # retains the full semantic weight, while grouped RRF takes the best
        # rank per source so appearing in several paraphrases cannot double-count.
        variant_weight = self.fusion_config.semantic_weight
        semantic_started = time.perf_counter()
        for index, query in enumerate(variants):
            semantic = self.provider.search_memory(
                query,
                MemoryType.KNOWLEDGE_BASE,
                limit=self.candidate_pool_size,
                memory_id=str(corpus["memory_id"]),
                user_id="memory-suite",
                namespace=case.corpus_id,
            )
            rankings.append((f"semantic:{index}", semantic, variant_weight))
        semantic_seconds = time.perf_counter() - semantic_started
        lexical_started = time.perf_counter()
        lexical = rank_lexical_documents(
            case.question,
            corpus["documents"],
            limit=self.candidate_pool_size,
        )
        lexical_seconds = time.perf_counter() - lexical_started
        rankings.append(("lexical", lexical, self.fusion_config.lexical_weight))
        fusion_started = time.perf_counter()
        selected = fuse_rankings(
            rankings,
            question=case.question,
            config=self.fusion_config,
            query_time=case.metadata.get("query_time"),
            query_variants=variants[1:],
        )
        return (
            selected,
            {
                "semantic_seconds": semantic_seconds,
                "lexical_seconds": lexical_seconds,
                "fusion_seconds": time.perf_counter() - fusion_started,
            },
            variants,
        )

    def _evidence_text(self, rows: Sequence[Mapping[str, Any]]) -> str:
        evidence: List[str] = []
        length = 0
        for index, row in enumerate(rows, start=1):
            source_ids = [
                str(item) for item in row.get("linked_source_ids") or [] if item
            ]
            if not source_ids:
                source_ids = [
                    str(
                        row.get("parent_source_id") or row.get("source_id") or "unknown"
                    )
                ]
            block = (
                f"[Memory {index}; source_ids="
                f"{json.dumps(source_ids, ensure_ascii=False)}]\n"
                f"{row.get('content') or ''}"
            )
            remaining = self.max_evidence_chars - length
            if remaining <= 0:
                break
            evidence.append(block[:remaining])
            length += min(len(block), remaining)
        return "\n\n".join(evidence) or "No relevant memory was retrieved."

    def _evidence_prompt(self, case: MemoryBenchmarkCase, evidence_text: str) -> str:
        return (
            f"Retrieved memory:\n{evidence_text}\n\n"
            f"Question:\n{case.question}\n\n"
            "Answer directly and concisely using only retrieved memory. Cite the "
            "exact source IDs shown in the evidence, with each ID as its own "
            "source_ids array element; never join multiple IDs into one string. "
            "If memory does not support an answer, abstain. For temporal questions, "
            "resolve relative dates from "
            "the supplied event timestamp and use an absolute date with a named "
            "month. When the input describes a situation rather than asking an "
            "explicit question, respond with the applicable prior constraint, "
            "preference, lesson, or advice. Return JSON only with this shape: "
            '{"answer":"...","source_ids":["..."],"abstained":false}.'
        )

    def _generate_reader(
        self,
        case: MemoryBenchmarkCase,
        rows: Sequence[Mapping[str, Any]],
        *,
        lane: str,
    ) -> Dict[str, Any]:
        evidence_text = self._evidence_text(rows)
        started = time.perf_counter()
        raw = self.model.generate_text(
            self._evidence_prompt(case, evidence_text),
            instructions=(
                "You are the grounded answer reader in a long-term-memory "
                "evaluation. Never invent facts and return the requested JSON only."
            ),
        ).strip()
        elapsed = time.perf_counter() - started
        self._capture_usage(self.model, lane=lane)
        parsed = self._parse_reader_with_repair(case, raw, lane=lane)
        parsed["evidence_text"] = evidence_text
        parsed["seconds"] = elapsed
        return parsed

    def _parse_reader_with_repair(
        self,
        case: MemoryBenchmarkCase,
        raw: str,
        *,
        lane: str,
    ) -> Dict[str, Any]:
        parsed = parse_reader_response(raw)
        parsed["repair_attempted"] = False
        parsed["repair_succeeded"] = False
        if not self.reader_repair or not reader_response_needs_repair(raw):
            return parsed
        parsed["repair_attempted"] = True
        repaired_raw = self.model.generate_text(
            (
                "Reformat the response below as valid JSON without adding, removing, "
                "or changing factual claims. Use exactly this shape: "
                '{"answer":"...","source_ids":["..."],"abstained":false}.\n\n'
                f"Question:\n{case.question}\n\nResponse to repair:\n{raw}"
            ),
            instructions="Repair formatting only. Return one JSON object and nothing else.",
        ).strip()
        self._capture_usage(self.model, lane=lane)
        repaired = parse_reader_response(repaired_raw)
        if repaired["structured"]:
            repaired["repair_attempted"] = True
            repaired["repair_succeeded"] = True
            repaired["original_raw"] = raw
            return repaired
        parsed["repair_raw"] = repaired_raw
        return parsed

    def _generate_memagent_reader(
        self,
        case: MemoryBenchmarkCase,
        corpus: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Run the selected MemAgent against one isolated benchmark scope."""

        if self.agent is None:
            raise RuntimeError("MemAgent evaluation was not initialized")
        self.agent.knowledge_base_ids = [str(corpus["knowledge_base_id"])]
        started = time.perf_counter()
        raw = str(
            self.agent.run(
                case.question,
                memory_id=str(corpus["memory_id"]),
                thread_id=f"memory-suite:{case.case_id}",
                user_id="memory-suite",
                context={
                    "evaluation": {
                        "benchmark": case.benchmark_id,
                        "case_id": case.case_id,
                        "requires_grounded_memory": True,
                    }
                },
                observability_context={
                    "request_id": f"memory-suite:{case.case_id}",
                    "grounding_status": "evaluation",
                    "grounding_source": "automatic-memory-retrieval",
                },
            )
        ).strip()
        elapsed = time.perf_counter() - started
        self._capture_usage(self.model, lane="generation")
        parsed = self._parse_reader_with_repair(case, raw, lane="generation")
        evidence_getter = getattr(self.agent, "last_retrieval_evidence", None)
        evidence = evidence_getter() if callable(evidence_getter) else {}
        evidence_rows = list(evidence.get("items") or [])
        parsed["evidence_rows"] = evidence_rows
        parsed["evidence_text"] = self._evidence_text(
            [
                {
                    "content": row.get("text") or "",
                    "source_id": row.get("source_id") or row.get("id"),
                    "parent_source_id": row.get("parent_source_id"),
                    "linked_source_ids": row.get("linked_source_ids") or [],
                }
                for row in evidence_rows
            ]
        )
        retrieval_seconds = float(evidence.get("duration_ms") or 0.0) / 1000
        parsed["seconds"] = max(0.0, elapsed - retrieval_seconds)
        parsed["total_seconds"] = elapsed
        parsed["retrieval_seconds"] = retrieval_seconds
        parsed["retrieval_candidate_count"] = int(evidence.get("candidate_count") or 0)
        parsed["query_variants"] = list(
            evidence.get("query_variants") or [case.question]
        )
        return parsed

    @staticmethod
    def _gold_evidence(
        case: MemoryBenchmarkCase, corpus: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        relevant = {str(item) for item in case.relevant_source_ids}
        if not relevant:
            return []
        return [
            dict(row)
            for row in corpus.get("documents") or []
            if relevant.intersection(
                {
                    str(row.get("source_id") or ""),
                    str(row.get("parent_source_id") or ""),
                }
            )
        ]

    def _judge(self, case: MemoryBenchmarkCase, prediction: str) -> Dict[str, Any]:
        started = time.perf_counter()
        raw = self.judge_model.generate_text(
            build_judge_prompt(case, prediction),
            instructions="You are a strict benchmark grader. Return JSON only.",
        )
        self._judge_seconds.append(time.perf_counter() - started)
        self._capture_usage(self.judge_model, lane="judge")
        return parse_judge_response(raw)

    def _faithfulness(
        self,
        case: MemoryBenchmarkCase,
        prediction: str,
        evidence_text: str,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        raw = self.judge_model.generate_text(
            (
                "Score how fully the candidate's factual claims are supported by "
                "the retrieved context. Return 1 when every claim is supported, a "
                "number between 0 and 1 for partial support, and 0 when unsupported "
                "or contradicted. Do not use outside knowledge.\n\n"
                f"Question:\n{case.question}\n\n"
                f"Retrieved context:\n{evidence_text}\n\n"
                f"Candidate response:\n{prediction}\n\n"
                'Return JSON only: {"score": 0.0, "reason": "brief explanation"}'
            ),
            instructions=(
                "You are a strict AgentMemBench faithfulness grader. "
                "Return JSON only."
            ),
        )
        self._judge_seconds.append(time.perf_counter() - started)
        self._capture_usage(self.judge_model, lane="judge")
        return parse_judge_response(raw)

    def _run_case(
        self, case: MemoryBenchmarkCase, index: int, total: int
    ) -> Dict[str, Any]:
        corpus = self._ingest_corpus(case.corpus_id, case.documents)
        if self.evaluation_mode == "memagent":
            reader = self._generate_memagent_reader(case, corpus)
            retrieved = [
                {
                    **dict(row),
                    "content": row.get("text") or "",
                    "source_id": row.get("source_id") or row.get("id"),
                }
                for row in reader.get("evidence_rows") or []
            ]
            retrieval_seconds = float(reader.get("retrieval_seconds") or 0.0)
            retrieval_timing = {
                "semantic_seconds": retrieval_seconds,
                "lexical_seconds": 0.0,
                "fusion_seconds": 0.0,
            }
            query_variants = list(reader.get("query_variants") or [case.question])
        else:
            retrieval_started = time.perf_counter()
            retrieved, retrieval_timing, query_variants = self._retrieve(case, corpus)
            retrieval_seconds = time.perf_counter() - retrieval_started
            reader = self._generate_reader(case, retrieved, lane="generation")
        prediction = str(reader["answer"])
        evidence_text = str(reader["evidence_text"])
        generation_seconds = float(reader["seconds"])
        answer_result = score_answer(
            case,
            prediction,
            judge=self._judge if case.scorer == "llm_judge" else None,
        )
        faithfulness = (
            self._faithfulness(case, prediction, evidence_text)
            if case.benchmark_id == "agentmembench"
            else None
        )
        retrieval_result = retrieval_metrics(retrieved, case.relevant_source_ids)
        citations = citation_metrics(
            reader["source_ids"], retrieved, case.relevant_source_ids
        )
        gold_rows = self._gold_evidence(case, corpus)
        oracle_result = None
        if self.oracle_reader and gold_rows:
            oracle_reader = self._generate_reader(case, gold_rows, lane="oracle")
            oracle_answer = score_answer(
                case,
                str(oracle_reader["answer"]),
                judge=self._judge if case.scorer == "llm_judge" else None,
            )
            oracle_result = {
                "prediction": oracle_reader["answer"],
                "source_ids": oracle_reader["source_ids"],
                "structured": oracle_reader["structured"],
                "score": oracle_answer["score"],
                "correct": oracle_answer["correct"],
                "metric": oracle_answer["metric"],
                "judge": oracle_answer.get("judge"),
                "generation_seconds": oracle_reader["seconds"],
                "evidence_source_ids": [
                    source_id
                    for source_id in dict.fromkeys(
                        row.get("parent_source_id") or row.get("source_id")
                        for row in gold_rows
                    )
                    if source_id
                ],
                "memory_context_chars": len(str(oracle_reader["evidence_text"])),
            }
        retrieval_hit = (
            retrieval_result.get("recall_at_k") is not None
            and float(retrieval_result["recall_at_k"] or 0.0) > 0.0
        )
        warnings = []
        if not retrieval_hit and float(answer_result["score"]) > 0:
            warnings.append(
                "answer received credit although no gold source was retrieved"
            )
        if citations["grounding_status"] != "grounded":
            warnings.append(
                "reader response was not fully grounded with valid retrieved source IDs"
            )
        self.progress(
            f"[{index}/{total}] {case.case_id} ({case.category}): "
            f"score={answer_result['score']:.3f}"
        )
        return {
            "case_id": case.case_id,
            "corpus_id": case.corpus_id,
            "category": case.category,
            "question": case.question,
            "answers": list(case.answers),
            "prediction": prediction,
            "reader_output": {
                "source_ids": reader["source_ids"],
                "abstained": reader["abstained"],
                "structured": reader["structured"],
                "repair_attempted": reader.get("repair_attempted", False),
                "repair_succeeded": reader.get("repair_succeeded", False),
            },
            "evaluation_mode": self.evaluation_mode,
            "scorer": case.scorer,
            "metric": answer_result["metric"],
            "score": answer_result["score"],
            "correct": answer_result["correct"],
            "judge": answer_result.get("judge"),
            "faithfulness": faithfulness,
            "faithfulness_score": (
                float(faithfulness["score"]) if faithfulness is not None else None
            ),
            "retrieved_source_ids": [
                source_id
                for source_id in dict.fromkeys(
                    source_id
                    for row in retrieved
                    for source_id in (
                        list(row.get("linked_source_ids") or [])
                        or [row.get("parent_source_id") or row.get("source_id")]
                    )
                    if source_id
                )
            ],
            "relevant_source_ids": list(case.relevant_source_ids),
            "retrieval": retrieval_result,
            "retrieval_hit": retrieval_hit,
            "citations": citations,
            "grounding_status": citations["grounding_status"],
            "oracle_reader": oracle_result,
            "warnings": warnings,
            "retrieval_seconds": retrieval_seconds,
            "retrieval_timing": retrieval_timing,
            "query_variants": query_variants,
            "generation_seconds": generation_seconds,
            "memory_context_chars": len(evidence_text),
            "memory_context_token_estimate": (len(evidence_text) + 3) // 4,
            "metadata": dict(case.metadata),
            "retrieval_trace": [
                {
                    "source_id": row.get("source_id"),
                    "parent_source_id": row.get("parent_source_id"),
                    "linked_source_ids": list(row.get("linked_source_ids") or []),
                    **dict(row.get("_retrieval") or {}),
                }
                for row in retrieved
            ],
        }

    def run(
        self,
        cases: Sequence[MemoryBenchmarkCase],
        *,
        variant: str,
        dataset_path: str | Path,
    ) -> Dict[str, Any]:
        if not cases:
            raise ValueError("At least one benchmark case is required")
        benchmark_ids = {case.benchmark_id for case in cases}
        if len(benchmark_ids) != 1:
            raise ValueError("One runner invocation may contain only one benchmark")
        benchmark_id = next(iter(benchmark_ids))
        spec = get_benchmark_spec(benchmark_id)
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        provider_note = (
            "external API cost is fixed at $0"
            if self.model_provider == "ollama"
            else "OpenAI token usage and estimated cost will be recorded"
        )
        self.progress(
            f"Starting {spec.name} {variant} with "
            f"{self.model_provider}:{self.model_name}; mode={self.evaluation_mode}; "
            f"{provider_note}."
        )
        results = [
            self._run_case(case, index, len(cases))
            for index, case in enumerate(cases, start=1)
        ]
        elapsed = time.perf_counter() - started
        category_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for result in results:
            category_rows[str(result["category"])].append(result)
        category_results: Dict[str, Dict[str, Any]] = {}
        for category_index, (category, rows) in enumerate(category_rows.items()):
            scores = [float(row["score"]) for row in rows]
            category_results[category] = {
                "num_samples": len(rows),
                "accuracy": statistics.fmean(float(row["correct"]) for row in rows),
                "average_score": statistics.fmean(scores),
                "score_confidence_interval_95": _bootstrap_mean_ci(
                    scores, seed=self.seed + category_index
                ),
                "retrieval_recall_at_k": _mean(
                    [row["retrieval"]["recall_at_k"] for row in rows]
                ),
                "mrr": _mean([row["retrieval"]["mrr"] for row in rows]),
                "ndcg_at_k": _mean([row["retrieval"]["ndcg_at_k"] for row in rows]),
                "faithfulness": _mean([row.get("faithfulness_score") for row in rows]),
                "oracle_reader_score": _mean(
                    [(row.get("oracle_reader") or {}).get("score") for row in rows]
                ),
                "grounded_rate": statistics.fmean(
                    float(row.get("grounding_status") == "grounded") for row in rows
                ),
            }
        overall_score = statistics.fmean(float(row["score"]) for row in results)
        overall_accuracy = statistics.fmean(float(row["correct"]) for row in results)
        oracle_score = _mean(
            [(row.get("oracle_reader") or {}).get("score") for row in results]
        )
        retrieved_hit_rows = [row for row in results if row.get("retrieval_hit")]
        estimated_cost = self._estimated_cost()
        primary_token_total = sum(
            int(self._usage[key])
            for key in (
                "generation_prompt_tokens",
                "generation_completion_tokens",
                "judge_prompt_tokens",
                "judge_completion_tokens",
                "oracle_prompt_tokens",
                "oracle_completion_tokens",
            )
        )
        readiness = self.dataset_readiness or verify_dataset(
            benchmark_id,
            data_path=dataset_path,
            variant=variant,
        )
        subset_fingerprint = _cases_fingerprint(cases)
        protocol_evidence = {
            "official_runner": False,
            "official_scorer": False,
            "dataset_verified": bool(readiness.get("ready")),
            "dataset_revision": (readiness.get("source") or {}).get("revision"),
            "dataset_fingerprint": readiness.get("dataset_fingerprint")
            or subset_fingerprint,
            "upstream_revision": (readiness.get("source") or {}).get("revision"),
            "num_samples": len(results),
            "reader_model": self.model_name,
            "embedding_model": self.embedding_model,
            "judge_model": self.judge_model_name,
            "top_k": self.top_k,
            "decoding": {
                "strategy": "greedy",
                "temperature": 0,
                "seed": self.seed,
                "quantization": None,
            },
            "prompt_hash": "sha256:"
            + hashlib.sha256(
                f"memory-suite-grounded-reader-v4:{self.evaluation_mode}".encode()
            ).hexdigest(),
            "scorer_hash": "sha256:"
            + hashlib.sha256(b"memory-suite-diagnostic-scoring-v2").hexdigest(),
            "dependency_lock_hash": _dependency_lock_fingerprint(),
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
            },
            "seed": self.seed,
        }
        protocol = assess_comparability(
            get_protocol_manifest(benchmark_id),
            protocol_evidence,
            profile=self.profile,
        )
        run_warnings = [
            {"case_id": row["case_id"], "warning": warning}
            for row in results
            for warning in row.get("warnings") or []
        ]
        report = {
            "schema_version": "memorizz.memory-suite.v3",
            "benchmark": benchmark_id,
            "benchmark_name": spec.name,
            "paper_url": spec.paper_url,
            "repository_url": spec.repository_url,
            "paper_comparable": protocol["paper_comparable"],
            "comparison_label": protocol["comparison_label"],
            "non_comparability_reasons": protocol["non_comparability_reasons"],
            "protocol": protocol,
            "dataset_readiness": readiness,
            "overall_accuracy": overall_accuracy,
            "overall_score": overall_score,
            "score_confidence_interval_95": _bootstrap_mean_ci(
                [float(row["score"]) for row in results], seed=self.seed
            ),
            "category_results": category_results,
            "answer_quality": {
                "task_score": overall_score,
                "retrieved_evidence_score": (
                    overall_score if self.evaluation_mode == "retrieval" else None
                ),
                "memagent_end_to_end_score": (
                    overall_score if self.evaluation_mode == "memagent" else None
                ),
                "gold_evidence_oracle_score": oracle_score,
                "reader_gap_to_oracle": (
                    oracle_score - overall_score if oracle_score is not None else None
                ),
                "score_conditioned_on_retrieval_hit": _mean(
                    [float(row["score"]) for row in retrieved_hit_rows]
                ),
                "faithfulness": _mean(
                    [row.get("faithfulness_score") for row in results]
                ),
                "grounded_rate": statistics.fmean(
                    float(row.get("grounding_status") == "grounded") for row in results
                ),
            },
            "retrieval": {
                "recall_at_k": _mean(
                    [row["retrieval"]["recall_at_k"] for row in results]
                ),
                "mrr": _mean([row["retrieval"]["mrr"] for row in results]),
                "ndcg_at_k": _mean([row["retrieval"]["ndcg_at_k"] for row in results]),
                "top_k": self.top_k,
                "candidate_pool_size": self.candidate_pool_size,
                "basis": (
                    "memagent_automatic_retrieval"
                    if self.evaluation_mode == "memagent"
                    else "diagnostic_fusion"
                ),
                "strategy": (
                    "automatic_provider_search+query_expansion+context_dedup_mmr"
                    if self.evaluation_mode == "memagent"
                    else "grouped_weighted_rrf+concept+entity+temporal+local_rerank+mmr"
                ),
                "fusion": (
                    None
                    if self.evaluation_mode == "memagent"
                    else "grouped_weighted_rrf"
                ),
                "semantic_weight": (
                    None
                    if self.evaluation_mode == "memagent"
                    else self.fusion_config.semantic_weight
                ),
                "lexical_weight": (
                    None
                    if self.evaluation_mode == "memagent"
                    else self.fusion_config.lexical_weight
                ),
                "entity_weight": (
                    None
                    if self.evaluation_mode == "memagent"
                    else self.fusion_config.entity_weight
                ),
                "temporal_weight": (
                    None
                    if self.evaluation_mode == "memagent"
                    else self.fusion_config.temporal_weight
                ),
                "rerank_weight": (
                    None
                    if self.evaluation_mode == "memagent"
                    else self.fusion_config.rerank_weight
                ),
                "concept_weight": (
                    None
                    if self.evaluation_mode == "memagent"
                    else self.fusion_config.concept_weight
                ),
                "diversity_lambda": (
                    None
                    if self.evaluation_mode == "memagent"
                    else self.fusion_config.diversity_lambda
                ),
                "query_expansion": self.query_expansion,
                "parent_source_deduplication": True,
            },
            "efficiency": {
                "corpus_count": len(self._corpora),
                "document_count": sum(
                    int(corpus["document_count"]) for corpus in self._corpora.values()
                ),
                "original_document_count": sum(
                    int(corpus["original_document_count"])
                    for corpus in self._corpora.values()
                ),
                "derived_semantic_memory_count": sum(
                    int(corpus["derived_semantic_memory_count"])
                    for corpus in self._corpora.values()
                ),
                "ingestion_seconds": sum(
                    float(corpus["ingestion_seconds"])
                    for corpus in self._corpora.values()
                ),
                "cold_ingestion_seconds": sum(
                    float(corpus["ingestion_seconds"])
                    for corpus in self._corpora.values()
                    if not corpus["embedding_cache"].get("hit")
                ),
                "warm_ingestion_seconds": sum(
                    float(corpus["ingestion_seconds"])
                    for corpus in self._corpora.values()
                    if corpus["embedding_cache"].get("hit")
                ),
                "embedding_seconds": sum(
                    float(corpus["embedding_seconds"])
                    for corpus in self._corpora.values()
                ),
                "embedding_cache_hits": sum(
                    int(bool(corpus["embedding_cache"].get("hit")))
                    for corpus in self._corpora.values()
                ),
                "embedding_cache_misses": sum(
                    int(not bool(corpus["embedding_cache"].get("hit")))
                    for corpus in self._corpora.values()
                ),
                "embedding_cache_bytes": sum(
                    int(corpus["embedding_cache"].get("bytes") or 0)
                    for corpus in self._corpora.values()
                ),
                "reader_repair_attempts": sum(
                    int(bool(row["reader_output"].get("repair_attempted")))
                    for row in results
                ),
                "reader_repair_successes": sum(
                    int(bool(row["reader_output"].get("repair_succeeded")))
                    for row in results
                ),
                "average_retrieval_seconds": statistics.fmean(
                    float(row["retrieval_seconds"]) for row in results
                ),
                "retrieval_p50_seconds": _percentile(
                    [float(row["retrieval_seconds"]) for row in results], 0.5
                ),
                "retrieval_p95_seconds": _percentile(
                    [float(row["retrieval_seconds"]) for row in results], 0.95
                ),
                "average_semantic_search_seconds": statistics.fmean(
                    float(row["retrieval_timing"]["semantic_seconds"])
                    for row in results
                ),
                "average_lexical_search_seconds": statistics.fmean(
                    float(row["retrieval_timing"]["lexical_seconds"]) for row in results
                ),
                "average_reranking_seconds": statistics.fmean(
                    float(row["retrieval_timing"]["fusion_seconds"]) for row in results
                ),
                "average_generation_seconds": statistics.fmean(
                    float(row["generation_seconds"]) for row in results
                ),
                "generation_p50_seconds": _percentile(
                    [float(row["generation_seconds"]) for row in results], 0.5
                ),
                "generation_p95_seconds": _percentile(
                    [float(row["generation_seconds"]) for row in results], 0.95
                ),
                "judge_seconds": sum(self._judge_seconds),
                "judge_p95_seconds": _percentile(self._judge_seconds, 0.95),
                "average_oracle_generation_seconds": _mean(
                    [
                        (row.get("oracle_reader") or {}).get("generation_seconds")
                        for row in results
                    ]
                ),
                "average_answer_latency_seconds": statistics.fmean(
                    float(row["retrieval_seconds"]) + float(row["generation_seconds"])
                    for row in results
                ),
                "average_memory_context_chars": statistics.fmean(
                    float(row["memory_context_chars"]) for row in results
                ),
                "average_memory_context_token_estimate": statistics.fmean(
                    float(row["memory_context_token_estimate"]) for row in results
                ),
                "memory_footprint_bytes": _directory_size(self.memory_root),
                "stored_memory_token_estimate": sum(
                    (len(str(row.get("content") or "")) + 3) // 4
                    for corpus in self._corpora.values()
                    for row in corpus["documents"]
                ),
                "total_processing_time": elapsed,
            },
            "usage": {
                **self._usage,
                "total_tokens": primary_token_total,
                "cost_usd": estimated_cost,
                "cost_is_estimate": self.model_provider == "openai",
                "pricing_usd_per_million_tokens": (
                    self._pricing_for_model(self.model_name)
                    if self.model_provider == "openai"
                    else None
                ),
                "lane_cost_usd": {
                    "retrieved_reader": self._lane_cost("generation", self.model_name),
                    "oracle_reader": self._lane_cost("oracle", self.model_name),
                    "judge": self._lane_cost("judge", self.judge_model_name),
                },
            },
            "metadata": {
                "dataset_variant": variant,
                "dataset_path": str(Path(dataset_path).expanduser().resolve()),
                "dataset_subset_fingerprint": _cases_fingerprint(cases),
                "evaluation_profile": self.profile,
                "num_samples": len(results),
                "timestamp": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "total_processing_time": elapsed,
                "application_mode": (
                    str(
                        getattr(
                            getattr(self.agent, "application_mode", None),
                            "value",
                            getattr(self.agent, "application_mode", "memagent"),
                        )
                    )
                    if self.evaluation_mode == "memagent"
                    else "memory-suite"
                ),
                "evaluation_mode": self.evaluation_mode,
                "agent_template_id": self.agent_template_id,
                "agent_template_name": self.agent_template_name,
                "agent_evaluation_overrides": self.agent_evaluation_overrides,
                "model_provider": self.model_provider,
                "reader_model": self.model_name,
                "judge_model": self.judge_model_name,
                "reasoning_effort": (
                    self.reasoning_effort if self.model_provider == "openai" else None
                ),
                "embedding_provider": "ollama",
                "embedding_model": self.embedding_model,
                "embedding_model_digest": self.embedding_model_digest,
                "provider_capabilities": self.provider.memory_capabilities().to_dict(),
                "semantic_memory_extraction": self.semantic_memory,
                "semantic_cache_enabled": bool(
                    getattr(
                        getattr(self.agent, "cache_manager", None), "enabled", False
                    )
                ),
                "features_exercised": {
                    "provider_storage": True,
                    "automatic_memory_retrieval": self.evaluation_mode == "memagent",
                    "memagent_context_assembly": self.evaluation_mode == "memagent",
                    "semantic_cache_decision": (
                        self.evaluation_mode == "memagent"
                        and bool(
                            getattr(
                                getattr(self.agent, "cache_manager", None),
                                "enabled",
                                False,
                            )
                        )
                    ),
                    "summarization_compaction": False,
                    "note": (
                        "Summarization requires an incremental multi-turn profile; "
                        "this static QA runner does not claim to exercise it."
                    ),
                },
                "external_api_cost_usd": estimated_cost,
            },
            "scorer_provenance": {
                "official": False,
                "mode": "memorizz_diagnostic",
                "note": get_protocol_manifest(benchmark_id).official_scorer_note,
            },
            "warnings": run_warnings,
            "cases": results,
        }
        cost_label = "unpriced" if estimated_cost is None else f"${estimated_cost:.6f}"
        self.progress(
            f"Completed {spec.name}: score={overall_score:.3f}, "
            f"accuracy={overall_accuracy:.3f}, cost={cost_label}."
        )
        return report

    def close(self) -> None:
        if self.agent is not None:
            close_agent = getattr(self.agent, "close", None)
            if callable(close_agent):
                close_agent(
                    close_memory_provider=False,
                    close_model_provider=False,
                )
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


def run_memory_suite(
    benchmark_id: str,
    data_path: str | Path,
    *,
    variant: str | None = None,
    limit: int | None = None,
    profile: str = "smoke",
    output_path: str | Path | None = None,
    workspace: str | Path,
    model_provider: str = "ollama",
    model_name: str = "qwen2.5:3b",
    judge_model_name: str | None = None,
    embedding_model: str = "nomic-embed-text",
    ollama_host: str = "http://localhost:11434",
    top_k: int = 8,
    lexical_ratio: float = 0.35,
    candidate_pool_size: int | None = None,
    query_expansion: bool = True,
    temporal_weight: float = 0.05,
    entity_weight: float = 0.05,
    rerank_weight: float = 0.15,
    diversity_lambda: float = 0.9,
    max_document_chars: int = 3_500,
    max_evidence_chars: int = 24_000,
    reasoning_effort: str | None = "low",
    max_output_tokens: int = 512,
    oracle_reader: bool = True,
    seed: int = 0,
    corpus_cache_dir: str | Path | None = None,
    corpus_cache_enabled: bool = True,
    embedding_model_digest: str | None = None,
    memory_backend: str = "filesystem",
    semantic_memory: bool = True,
    reader_repair: bool = True,
    evaluation_mode: str = "retrieval",
    agent_template: Any = None,
    agent: Any = None,
    strict_paper: bool = False,
    source_path: str | Path | None = None,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Load, run, and optionally write one protocol-labelled evaluation."""
    spec = get_benchmark_spec(benchmark_id)
    selected_variant = variant or spec.default_variant
    selected_profile = get_evaluation_profile(profile)
    one_per_category = selected_profile.name == "smoke" and limit is None
    selected_limit = resolve_profile_limit(profile, limit)
    readiness = verify_dataset(
        spec.benchmark_id,
        data_path=data_path,
        source_path=source_path,
        variant=selected_variant,
    )
    cases = load_benchmark_cases(
        spec.benchmark_id,
        data_path,
        variant=selected_variant,
        limit=selected_limit,
        max_document_chars=max_document_chars,
        one_per_category=one_per_category,
    )
    runner = MemorySuiteRunner(
        workspace=workspace,
        model_provider=model_provider,
        model_name=model_name,
        judge_model_name=judge_model_name,
        embedding_model=embedding_model,
        ollama_host=ollama_host,
        top_k=top_k,
        lexical_ratio=lexical_ratio,
        candidate_pool_size=candidate_pool_size,
        query_expansion=query_expansion,
        temporal_weight=temporal_weight,
        entity_weight=entity_weight,
        rerank_weight=rerank_weight,
        diversity_lambda=diversity_lambda,
        max_evidence_chars=max_evidence_chars,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        oracle_reader=oracle_reader,
        profile=profile,
        seed=seed,
        corpus_cache_dir=corpus_cache_dir,
        corpus_cache_enabled=corpus_cache_enabled,
        embedding_model_digest=embedding_model_digest,
        dataset_readiness=readiness,
        memory_backend=memory_backend,
        semantic_memory=semantic_memory,
        reader_repair=reader_repair,
        evaluation_mode=evaluation_mode,
        agent_template=agent_template,
        agent=agent,
        progress=progress,
    )
    try:
        report = runner.run(
            cases,
            variant=selected_variant,
            dataset_path=data_path,
        )
    finally:
        runner.close()
    if strict_paper and not report["paper_comparable"]:
        reasons = "; ".join(report["non_comparability_reasons"])
        raise RuntimeError(f"Strict paper reproduction rejected this run: {reasons}")
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


__all__ = ["MemorySuiteRunner", "run_memory_suite"]
