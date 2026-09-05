# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import importlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ...memagent import MemAgentModel

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

from ...enums.memory_type import MemoryType
from ..base import MemoryProvider, MemoryProviderCapabilities, filter_tool_log_rows

logger = logging.getLogger(__name__)


class _FsUserIdUnset:
    """Sentinel for "user_id filter not supplied" — distinct from explicit None."""

    def __repr__(self) -> str:  # pragma: no cover
        return "<user_id unset>"


_FS_UNSET = _FsUserIdUnset()


_FS_USER_SCOPED_TYPES = frozenset(
    {
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.KNOWLEDGE_BASE,
        MemoryType.SHORT_TERM_MEMORY,
        MemoryType.WORKFLOW_MEMORY,
        MemoryType.SUMMARIES,
        MemoryType.SEMANTIC_CACHE,
        MemoryType.ENTITY_MEMORY,
        MemoryType.TOOL_LOG,
    }
)


@dataclass
class FileSystemConfig:
    """Configuration for the filesystem provider."""

    root_path: Union[str, Path]
    lazy_vector_indexes: bool = False
    use_faiss: bool = True
    embedding_provider: Optional[Any] = None
    embedding_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.root_path = Path(self.root_path).expanduser().resolve()
        self.embedding_config = self.embedding_config or {}


class FileSystemProvider(MemoryProvider):
    """
    Filesystem-backed implementation of the MemoryProvider interface.

    Data is stored as JSON documents arranged by MemoryType. Vector search is
    handled via FAISS when available, falling back to brute-force cosine scoring
    when the dependency is missing.
    """

    INDEX_VERSION = 1

    def memory_capabilities(self) -> MemoryProviderCapabilities:
        return MemoryProviderCapabilities(
            provider=type(self).__name__,
            batch_store=True,
            transactional_batch=False,
            scoped_search=True,
            result_scores=True,
            provenance=True,
            native_vector_search=bool(self.config.use_faiss),
            native_hybrid_search=False,
        )

    def __init__(self, config: FileSystemConfig):
        self.config = config
        self.root_path = config.root_path
        self.root_path.mkdir(parents=True, exist_ok=True)

        self._embedding_provider = self._setup_embedding_provider(config)
        self._faiss = self._load_faiss() if config.use_faiss else None
        self._store_paths: Dict[MemoryType, Path] = {}
        self._indexes: Dict[MemoryType, Dict[str, Dict[str, Any]]] = {}
        self._locks: Dict[MemoryType, threading.RLock] = {
            memory_type: threading.RLock() for memory_type in MemoryType
        }
        self._vector_state: Dict[
            MemoryType, Dict[str, Any]
        ] = {}  # {"index": faiss.Index, "doc_ids": [...], "dirty": bool}

        self._initialize_storage()

    @staticmethod
    def _load_faiss() -> Optional[Any]:
        try:
            return importlib.import_module("faiss")
        except ImportError:  # pragma: no cover - optional dependency
            return None

    # ---------------------------------------------------------------------
    # Public API - Required by MemoryProvider
    # ---------------------------------------------------------------------
    def store(
        self,
        data: Dict[str, Any] = None,
        memory_store_type: Union[str, MemoryType, None] = None,
        memory_id: Optional[str] = None,
        memory_unit: Any = None,
    ) -> str:
        if memory_unit is not None:
            data = self._convert_memory_unit(memory_unit)
            if memory_id:
                data["memory_id"] = memory_id
            if getattr(memory_unit, "memory_type", None):
                memory_store_type = memory_unit.memory_type
            elif data.get("memory_type"):
                memory_store_type = data["memory_type"]
            else:
                memory_store_type = MemoryType.CONVERSATION_MEMORY

        if data is None or memory_store_type is None:
            raise ValueError(
                "Either (data, memory_store_type) or (memory_unit) must be provided"
            )

        memory_type = self._normalize_memory_type(memory_store_type)
        if memory_type == MemoryType.MEMAGENT:
            from ...memagent import MemAgentModel

            memagent = (
                data if isinstance(data, MemAgentModel) else MemAgentModel(**data)
            )
            return self.store_memagent(memagent)

        document = self._prepare_document(data)
        if memory_id:
            document.setdefault("memory_id", memory_id)

        record_id = str(document.get("_id") or document.get("id") or uuid.uuid4())
        document["_id"] = record_id
        document["id"] = record_id

        self._write_document(memory_type, record_id, document)
        return record_id

    def store_many(
        self,
        rows: List[Dict[str, Any]],
        memory_store_type: Union[str, MemoryType],
        *,
        memory_id: Optional[str] = None,
    ) -> List[str]:
        """Store a batch while persisting the shared index only once.

        Benchmark ingestion can contain thousands of independently retrievable
        chunks. Rewriting the complete JSON index after every chunk makes that
        path quadratic in corpus size, so this optional provider extension
        retains ordinary ``store`` semantics with one final index checkpoint.
        """
        memory_type = self._normalize_memory_type(memory_store_type)
        if memory_type == MemoryType.MEMAGENT:
            return [
                self.store(row, memory_store_type=memory_type, memory_id=memory_id)
                for row in rows
            ]
        prepared: List[Tuple[str, Dict[str, Any]]] = []
        for row in rows:
            document = self._prepare_document(row)
            if memory_id:
                document.setdefault("memory_id", memory_id)
            record_id = str(document.get("_id") or document.get("id") or uuid.uuid4())
            document["_id"] = record_id
            document["id"] = record_id
            prepared.append((record_id, document))
        if not prepared:
            return []

        written = False
        with self._locks[memory_type]:
            try:
                for record_id, document in prepared:
                    self._write_document(
                        memory_type,
                        record_id,
                        document,
                        persist_index=False,
                    )
                    written = True
            finally:
                if written:
                    self._save_index(memory_type)
                    self._mark_vector_index_dirty(memory_type)
        return [record_id for record_id, _ in prepared]

    def retrieve_by_query(
        self,
        query: Union[Dict[str, Any], str],
        memory_store_type: Union[str, MemoryType, None] = None,
        limit: int = 1,
        memory_id: Optional[str] = None,
        memory_type: Union[str, MemoryType, None] = None,
        **kwargs,
    ) -> Optional[List[Dict[str, Any]]]:
        resolved_type = self._normalize_memory_type(memory_type or memory_store_type)
        user_id_scope = kwargs.get("user_id", _FS_UNSET)
        thread_id = kwargs.get("thread_id")
        namespace = kwargs.get("namespace")

        if isinstance(query, dict):
            return self._filter_documents(
                resolved_type,
                query,
                limit,
                memory_id=memory_id,
                user_id=user_id_scope,
                thread_id=thread_id,
                namespace=namespace,
            )
        elif isinstance(query, str):
            return self._semantic_search(
                resolved_type,
                query,
                limit,
                memory_id=memory_id,
                user_id=user_id_scope,
                thread_id=thread_id,
                namespace=namespace,
            )
        else:
            raise ValueError("query must be either a dict filter or a string")

    def retrieve_skillbox_candidates(
        self,
        query: str,
        *,
        limit: int,
        statuses: List[str],
        agent_id: Optional[str],
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Score only lifecycle/tenant-eligible skill documents.

        The generic FAISS index is global to the store. Filtering after its
        top-k can let ACTIVE or another tenant's skills crowd out a valid
        SHADOW candidate, so this dedicated background path narrows the
        documents before final ranking.
        """
        wanted = {str(status) for status in statuses}
        candidates: List[Dict[str, Any]] = []
        with self._locks[MemoryType.SKILLBOX]:
            for doc_id in self._indexes[MemoryType.SKILLBOX]:
                document = self._read_document(MemoryType.SKILLBOX, doc_id)
                if not document:
                    continue
                if document.get("status") not in wanted:
                    continue
                if document.get("agent_id") != agent_id:
                    continue
                if document.get("user_id") != user_id:
                    continue
                candidates.append(document)

        embedding_provider = self._get_embedding_provider()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        if embedding_provider is not None:
            query_embedding = embedding_provider.get_embedding(query)
            for document in candidates:
                target = document.get("embedding")
                if not target:
                    continue
                similarity = self._cosine_similarity(query_embedding, target)
                if similarity is None:
                    continue
                document["score"] = float(similarity)
                scored.append((float(similarity), document))
        else:
            query_terms = set(str(query).lower().split())
            for document in candidates:
                applicability = " ".join(
                    str(document.get(field) or "")
                    for field in (
                        "name",
                        "description",
                        "preconditions",
                        "queries",
                    )
                )
                terms = set(applicability.lower().split())
                similarity = len(query_terms.intersection(terms)) / max(
                    len(query_terms.union(terms)), 1
                )
                document["score"] = float(similarity)
                scored.append((float(similarity), document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[: max(int(limit), 1)]]

    def retrieve_by_id(
        self, id: str, memory_store_type: Union[str, MemoryType, None] = None
    ) -> Optional[Dict[str, Any]]:
        if memory_store_type:
            memory_types = [self._normalize_memory_type(memory_store_type)]
        else:
            memory_types = list(MemoryType)

        for memory_type in memory_types:
            with self._locks[memory_type]:
                metadata = self._indexes.get(memory_type, {})
                if id in metadata:
                    return self._read_document(memory_type, id)
        return None

    def retrieve_by_name(
        self,
        name: str,
        memory_store_type: Union[str, MemoryType, None] = None,
        include_embedding: bool = False,
    ) -> Optional[Dict[str, Any]]:
        memory_types = (
            [self._normalize_memory_type(memory_store_type)]
            if memory_store_type
            else list(MemoryType)
        )
        for memory_type in memory_types:
            with self._locks[memory_type]:
                for doc_id, meta in self._indexes[memory_type].items():
                    if meta.get("name") == name:
                        document = self._read_document(memory_type, doc_id)
                        if not include_embedding and document:
                            document.pop("embedding", None)
                        return document
        return None

    def delete_observability_bundle(self, record_id, fingerprint):
        from ...observability.index import digest
        from ...observability.normalization import read_payload

        with self._locks[MemoryType.SHARED_MEMORY]:
            row = self.retrieve_by_id(record_id, MemoryType.SHARED_MEMORY)
            payload = read_payload(row) if row else None
            if (
                not payload
                or payload.get("record_type") != "observability_trace_bundle"
                or digest(payload) != fingerprint
            ):
                return False
            return self.delete_by_id(record_id, MemoryType.SHARED_MEMORY)

    def delete_by_id(
        self, id: str, memory_store_type: Union[str, MemoryType, None]
    ) -> bool:
        memory_type = self._normalize_memory_type(memory_store_type)
        with self._locks[memory_type]:
            metadata = self._indexes[memory_type]
            if id not in metadata:
                return False
            file_path = self._document_path(memory_type, id)
            if file_path.exists():
                file_path.unlink()
            metadata.pop(id, None)
            self._save_index(memory_type)
            self._mark_vector_index_dirty(memory_type)
            return True

    def delete_by_name(
        self, name: str, memory_store_type: Union[str, MemoryType, None]
    ) -> bool:
        memory_type = self._normalize_memory_type(memory_store_type)
        with self._locks[memory_type]:
            metadata = self._indexes[memory_type]
            for doc_id, meta in list(metadata.items()):
                if meta.get("name") == name:
                    return self.delete_by_id(doc_id, memory_type)
        return False

    def delete_all(self, memory_store_type: Union[str, MemoryType, None]) -> bool:
        memory_type = self._normalize_memory_type(memory_store_type)
        with self._locks[memory_type]:
            store_path = self._store_paths[memory_type]
            for path in store_path.glob("*.json"):
                path.unlink()
            self._indexes[memory_type] = {}
            self._save_index(memory_type)
            self._mark_vector_index_dirty(memory_type)
            return True

    def list_all(
        self,
        memory_store_type: Union[str, MemoryType, None],
        user_id: Any = _FS_UNSET,
    ) -> List[Dict[str, Any]]:
        memory_type = self._normalize_memory_type(memory_store_type)
        documents: List[Dict[str, Any]] = []
        apply_user_filter = (
            user_id is not _FS_UNSET and memory_type in _FS_USER_SCOPED_TYPES
        )
        with self._locks[memory_type]:
            for doc_id in self._indexes[memory_type].keys():
                document = self._read_document(memory_type, doc_id)
                if not document:
                    continue
                if apply_user_filter and document.get("user_id") != user_id:
                    continue
                documents.append(document)
        return documents

    def list_tool_logs(
        self,
        memory_id: Optional[str] = None,
        user_id: Any = _FS_UNSET,
        limit: int = 20,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Recent tool-log rows for a memory/thread (most-recent first).

        Native counterpart to the MongoDB provider's indexed query: reads every
        tool_log document and narrows in Python via the shared
        :func:`filter_tool_log_rows`, so the ``thread_id`` scoping that keeps the
        digest to one conversation behaves identically across providers.
        """
        rows = self.list_all(MemoryType.TOOL_LOG)
        return filter_tool_log_rows(
            rows,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            limit=limit,
        )

    def retrieve_conversation_history_ordered_by_timestamp(
        self,
        memory_id: str,
        memory_type: Union[str, MemoryType, None] = None,
        limit: Optional[int] = None,
        user_id: Any = _FS_UNSET,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        resolved_type = self._normalize_memory_type(
            memory_type or MemoryType.CONVERSATION_MEMORY
        )
        apply_user_filter = user_id is not _FS_UNSET
        with self._locks[resolved_type]:
            documents: List[Tuple[float, Dict[str, Any]]] = []
            for doc_id in self._indexes[resolved_type]:
                document = self._read_document(resolved_type, doc_id)
                if not document:
                    continue
                if document.get("memory_id") != memory_id:
                    continue
                if apply_user_filter and document.get("user_id") != user_id:
                    continue
                if thread_id is not None and str(
                    document.get("thread_id") or document.get("conversation_id") or ""
                ) != str(thread_id):
                    continue
                timestamp = self._coerce_timestamp(
                    document.get("timestamp") or document.get("created_at")
                )
                documents.append((timestamp, document))

        sorted_docs = sorted(documents, key=lambda item: item[0])
        final_docs = [doc for _, doc in sorted_docs]
        if limit:
            return final_docs[-limit:]
        return final_docs

    def update_by_id(
        self,
        id: str,
        data: Dict[str, Any],
        memory_store_type: Union[str, MemoryType, None],
    ) -> bool:
        memory_type = self._normalize_memory_type(memory_store_type)
        with self._locks[memory_type]:
            document = self._read_document(memory_type, id)
            if not document:
                return False
            document.update(self._prepare_document(data))
            document["_id"] = id
            document["id"] = id
            document["updated_at"] = datetime.utcnow().isoformat()
            self._write_document(memory_type, id, document)
            return True

    def get_observability_index(self):
        from ...observability.sql_index import SQLiteSpanIndex

        if not hasattr(self, "_observability_index"):
            self._observability_index = SQLiteSpanIndex(self.root_path)
        return self._observability_index

    def close(self) -> None:
        """No persistent connections to close; provided for API parity."""
        return

    def store_memagent(self, memagent: "MemAgentModel") -> str:  # noqa: F821
        memagent_dict = memagent.model_dump()
        agent_id = memagent_dict.get("agent_id") or str(uuid.uuid4())
        memagent_dict["agent_id"] = agent_id
        memagent_dict["_id"] = agent_id
        memagent_dict["id"] = agent_id

        if memagent.persona:
            memagent_dict["persona"] = memagent.persona.to_dict()

        tools = memagent_dict.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if (
                    isinstance(tool, dict)
                    and "function" in tool
                    and callable(tool["function"])
                ):
                    tool.pop("function")

        self._write_document(MemoryType.MEMAGENT, agent_id, memagent_dict)
        self._sync_agent_tools_to_toolbox(agent_id, tools)
        return agent_id

    def _sync_agent_tools_to_toolbox(
        self, agent_id: str, tools: Optional[List[Dict[str, Any]]]
    ) -> None:
        """Mirror an agent's tool list into the TOOLBOX store.

        Deletes any existing TOOLBOX rows for this ``agent_id`` before
        re-inserting the current set so tools removed from the agent
        don't linger in the playground's toolbox-memory pane.
        ``tools=None`` is treated as "caller didn't include tools in this
        save" and is a no-op — only an explicit empty list clears rows.
        """
        if not agent_id or tools is None:
            return

        try:
            for doc in list(self.list_all(MemoryType.TOOLBOX) or []):
                if not isinstance(doc, dict):
                    continue
                if doc.get("agent_id") != agent_id:
                    continue
                doc_id = doc.get("_id") or doc.get("id")
                if doc_id:
                    self.delete_by_id(str(doc_id), MemoryType.TOOLBOX)
        except Exception as exc:
            logger.warning(
                "Failed to clear toolbox rows for agent %s: %s", agent_id, exc
            )
            return

        if not tools:
            return

        for tool_meta in tools:
            if not isinstance(tool_meta, dict):
                continue
            raw_id = tool_meta.get("_id") or tool_meta.get("name")
            if not raw_id:
                continue
            tool_doc = {
                "_id": f"{agent_id}:{raw_id}",
                "tool_id": f"{agent_id}:{raw_id}",
                "name": tool_meta.get("name"),
                "description": tool_meta.get("description", ""),
                "signature": tool_meta.get("signature", ""),
                "docstring": tool_meta.get(
                    "docstring", tool_meta.get("description", "")
                ),
                "tool_type": tool_meta.get("type", "function"),
                "parameters": tool_meta.get("parameters", {}),
                "agent_id": agent_id,
            }
            try:
                self.store(tool_doc, memory_store_type=MemoryType.TOOLBOX)
            except Exception as exc:
                logger.warning(
                    "Failed to sync tool %s for agent %s to TOOLBOX: %s",
                    tool_doc.get("name"),
                    agent_id,
                    exc,
                )

    def delete_memagent(self, agent_id: str, cascade: bool = False) -> bool:
        if cascade:
            memagent = self.retrieve_memagent(agent_id)
            if memagent and memagent.memory_ids:
                for memory_id in memagent.memory_ids:
                    for memory_type in MemoryType:
                        if memory_type == MemoryType.MEMAGENT:
                            continue
                        self._delete_memory_units_by_memory_id(memory_id, memory_type)
        return self.delete_by_id(agent_id, MemoryType.MEMAGENT)

    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        return self.update_by_id(
            agent_id, {"memory_ids": memory_ids}, MemoryType.MEMAGENT
        )

    def delete_memagent_memory_ids(self, agent_id: str) -> bool:
        return self.update_by_id(agent_id, {"memory_ids": []}, MemoryType.MEMAGENT)

    def list_memagents(self) -> List["MemAgentModel"]:
        documents = self.list_all(MemoryType.MEMAGENT)
        agents: List["MemAgentModel"] = []
        if not documents:
            return agents

        from ...long_term.semantic.persona.persona import Persona
        from ...memagent import MemAgentModel

        for doc in documents:
            agent = MemAgentModel(
                name=doc.get("name"),
                application_id=doc.get("application_id"),
                instruction=doc.get("instruction"),
                application_mode=doc.get("application_mode", "assistant"),
                memory_types=doc.get("memory_types"),
                max_steps=doc.get("max_steps"),
                memory_ids=doc.get("memory_ids") or [],
                agent_id=doc.get("agent_id") or doc.get("_id"),
                is_favorite=bool(doc.get("is_favorite", False)),
                tools=doc.get("tools"),
                tool_access=doc.get("tool_access"),
                knowledge_base_ids=doc.get("knowledge_base_ids"),
                delegates=doc.get("delegates"),
                llm_config=doc.get("llm_config"),
                embedding_config=doc.get("embedding_config"),
                semantic_cache=bool(doc.get("semantic_cache", False)),
                semantic_cache_config=doc.get("semantic_cache_config"),
                tool_result_policy=doc.get("tool_result_policy"),
                context_policy=doc.get("context_policy"),
                retrieval_policy=doc.get("retrieval_policy"),
                delegation_config=doc.get("delegation_config"),
                skill_retrieval=bool(doc.get("skill_retrieval", False)),
                skill_retrieval_config=doc.get("skill_retrieval_config"),
                semantic_layer_config=doc.get("semantic_layer_config"),
                context_window_tokens=doc.get("context_window_tokens"),
                sandbox_provider=doc.get("sandbox_provider"),
                browser_control=doc.get("browser_control"),
                meta_harness=bool(doc.get("meta_harness", False)),
                meta_harness_mode=doc.get("meta_harness_mode"),
                default_harness=doc.get("default_harness", "auto"),
                harness_config=doc.get("harness_config"),
                internet_access_provider=doc.get("internet_access_provider"),
                internet_access_config=doc.get("internet_access_config"),
                skills_marketplace_provider=doc.get("skills_marketplace_provider"),
                skills_marketplace_config=doc.get("skills_marketplace_config"),
                skill_paths=doc.get("skill_paths"),
                mcp_servers=doc.get("mcp_servers"),
                self_aware=bool(doc.get("self_aware", False)),
                continual_learning=bool(doc.get("continual_learning", False)),
                continual_learning_config=doc.get("continual_learning_config"),
                learning_control_plane=bool(doc.get("learning_control_plane", False)),
                learning_control_plane_config=doc.get("learning_control_plane_config"),
                self_aware_config=doc.get("self_aware_config"),
                automations_enabled=bool(doc.get("automations_enabled", True)),
                default_timezone=doc.get("default_timezone"),
                whatsapp_enabled=bool(doc.get("whatsapp_enabled", False)),
                whatsapp_config=doc.get("whatsapp_config"),
                memory_provider=self,
            )

            persona_data = doc.get("persona")
            if persona_data:
                agent.persona = Persona.from_dict(persona_data)
            agents.append(agent)
        return agents

    def retrieve_memagent(self, agent_id: str) -> Optional["MemAgentModel"]:
        document = self.retrieve_by_id(agent_id, MemoryType.MEMAGENT)
        if not document:
            return None

        from ...long_term.semantic.persona.persona import Persona
        from ...memagent import MemAgentModel

        memagent = MemAgentModel(
            name=document.get("name"),
            application_id=document.get("application_id"),
            instruction=document.get("instruction"),
            application_mode=document.get("application_mode", "assistant"),
            memory_types=document.get("memory_types"),
            max_steps=document.get("max_steps"),
            memory_ids=document.get("memory_ids") or [],
            agent_id=document.get("agent_id") or document.get("_id"),
            is_favorite=bool(document.get("is_favorite", False)),
            tools=document.get("tools"),
            tool_access=document.get("tool_access"),
            knowledge_base_ids=document.get("knowledge_base_ids"),
            delegates=document.get("delegates"),
            llm_config=document.get("llm_config"),
            embedding_config=document.get("embedding_config"),
            semantic_cache=bool(document.get("semantic_cache", False)),
            semantic_cache_config=document.get("semantic_cache_config"),
            tool_result_policy=document.get("tool_result_policy"),
            context_policy=document.get("context_policy"),
            retrieval_policy=document.get("retrieval_policy"),
            delegation_config=document.get("delegation_config"),
            skill_retrieval=bool(document.get("skill_retrieval", False)),
            skill_retrieval_config=document.get("skill_retrieval_config"),
            semantic_layer_config=document.get("semantic_layer_config"),
            context_window_tokens=document.get("context_window_tokens"),
            sandbox_provider=document.get("sandbox_provider"),
            browser_control=document.get("browser_control"),
            meta_harness=bool(document.get("meta_harness", False)),
            meta_harness_mode=document.get("meta_harness_mode"),
            default_harness=document.get("default_harness", "auto"),
            harness_config=document.get("harness_config"),
            internet_access_provider=document.get("internet_access_provider"),
            internet_access_config=document.get("internet_access_config"),
            skills_marketplace_provider=document.get("skills_marketplace_provider"),
            skills_marketplace_config=document.get("skills_marketplace_config"),
            skill_paths=document.get("skill_paths"),
            mcp_servers=document.get("mcp_servers"),
            self_aware=bool(document.get("self_aware", False)),
            continual_learning=bool(document.get("continual_learning", False)),
            continual_learning_config=document.get("continual_learning_config"),
            learning_control_plane=bool(document.get("learning_control_plane", False)),
            learning_control_plane_config=document.get("learning_control_plane_config"),
            self_aware_config=document.get("self_aware_config"),
            automations_enabled=bool(document.get("automations_enabled", True)),
            default_timezone=document.get("default_timezone"),
            whatsapp_enabled=bool(document.get("whatsapp_enabled", False)),
            whatsapp_config=document.get("whatsapp_config"),
            memory_provider=self,
        )

        persona_data = document.get("persona")
        if persona_data:
            memagent.persona = Persona.from_dict(persona_data)
        return memagent

    def supports_entity_memory(self) -> bool:
        return True

    def migrate_entity_memory_user_scope(self, *, memory_id: str, user_id: str) -> int:
        """Adopt anonymous entity rows under the filesystem provider write lock."""
        memory_type = MemoryType.ENTITY_MEMORY
        migrated = 0
        with self._locks[memory_type]:
            for document_id in list(self._indexes[memory_type]):
                document = self._read_document(memory_type, document_id)
                if not document:
                    continue
                if document.get("memory_id") != memory_id:
                    continue
                if document.get("user_id") is not None:
                    continue
                document["user_id"] = user_id
                self._write_document(memory_type, document_id, document)
                migrated += 1
        return migrated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_storage(self) -> None:
        for memory_type in MemoryType:
            store_path = self.root_path / memory_type.value
            store_path.mkdir(parents=True, exist_ok=True)
            self._store_paths[memory_type] = store_path
            index = self._read_index_file(store_path)
            self._indexes[memory_type] = index
            self._vector_state[memory_type] = {
                "index": None,
                "doc_ids": [],
                "dirty": True,
            }

    def _setup_embedding_provider(self, config: FileSystemConfig):
        if config.embedding_provider is None:
            return None
        if isinstance(config.embedding_provider, str):
            from ...embeddings import EmbeddingManager

            provider = EmbeddingManager(
                config.embedding_provider, config.embedding_config
            )
            logger.info(
                "Filesystem provider using explicit embedding provider: %s",
                provider.get_provider_info(),
            )
            return provider
        return config.embedding_provider

    def _get_embedding_provider(self):
        if self._embedding_provider is not None:
            return self._embedding_provider
        try:
            from ...embeddings import get_embedding_manager

            return get_embedding_manager()
        except Exception as exc:  # pragma: no cover - depends on env config
            logger.debug("Global embedding provider is not configured: %s", exc)
            return None

    def _normalize_memory_type(
        self, memory_type: Union[str, MemoryType, None]
    ) -> MemoryType:
        if memory_type is None:
            raise ValueError("memory_store_type or memory_type must be provided")
        if isinstance(memory_type, MemoryType):
            return memory_type
        return MemoryType(memory_type)

    def _convert_memory_unit(self, memory_unit: Any) -> Dict[str, Any]:
        if isinstance(memory_unit, dict):
            return dict(memory_unit)
        if hasattr(memory_unit, "model_dump"):
            return memory_unit.model_dump()
        if hasattr(memory_unit, "dict"):
            return memory_unit.dict()
        if hasattr(memory_unit, "__dict__"):
            return dict(memory_unit.__dict__)
        return dict(memory_unit)

    def _prepare_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        document = dict(data)
        if document.get("timestamp") is None:
            document["timestamp"] = datetime.utcnow().isoformat()

        tools = document.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if (
                    isinstance(tool, dict)
                    and "function" in tool
                    and callable(tool["function"])
                ):
                    tool.pop("function")

        persona = document.get("persona")
        if persona and hasattr(persona, "to_dict"):
            document["persona"] = persona.to_dict()

        # Ensure JSON-serializable payload
        document = json.loads(json.dumps(document, default=self._json_default_handler))
        return document

    def _json_default_handler(self, value: Any):
        if isinstance(value, (datetime,)):
            return value.isoformat()
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        if hasattr(value, "value"):
            return value.value
        if callable(value):
            return None
        return str(value)

    def _write_document(
        self,
        memory_type: MemoryType,
        document_id: str,
        document: Dict[str, Any],
        *,
        persist_index: bool = True,
    ) -> None:
        # Serialize writers per store: ``store()`` runs on the caller's
        # thread while the conversation-embedding backfill worker calls
        # ``update_by_id`` concurrently; without the (re-entrant) lock the
        # two racers clobber each other's index tmp-file rename.
        with self._locks[memory_type]:
            file_path = self._document_path(memory_type, document_id)
            tmp_path = file_path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False)
            if (
                document.get("immutable_trace") is True
                and document.get("record_type") == "observability_trace_bundle"
            ):
                try:
                    # Atomic no-replace publication, including across processes.
                    os.link(tmp_path, file_path)
                except FileExistsError:
                    document = self._read_document(memory_type, document_id)
                    if not document:
                        raise RuntimeError("Existing immutable trace cannot be read")
                finally:
                    tmp_path.unlink(missing_ok=True)
            else:
                os.replace(tmp_path, file_path)

            metadata = self._indexes[memory_type]
            metadata[document_id] = self._build_metadata(document)
            if persist_index:
                self._save_index(memory_type)
                self._mark_vector_index_dirty(memory_type)

    def _document_path(self, memory_type: MemoryType, document_id: str) -> Path:
        return self._store_paths[memory_type] / f"{document_id}.json"

    def _read_document(
        self, memory_type: MemoryType, document_id: str
    ) -> Optional[Dict[str, Any]]:
        file_path = self._document_path(memory_type, document_id)
        if not file_path.exists():
            return None
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                doc = json.load(handle)
            # Backward compat: migrate old conversation_id → thread_id on read
            if "conversation_id" in doc and "thread_id" not in doc:
                doc["thread_id"] = doc.pop("conversation_id")
            if (
                "associated_conversation_ids" in doc
                and "associated_thread_ids" not in doc
            ):
                doc["associated_thread_ids"] = doc.pop("associated_conversation_ids")
            return doc
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return None

    def _build_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = document.get("timestamp") or document.get("created_at")
        return {
            "id": document.get("id"),
            "name": document.get("name") or document.get("title"),
            "memory_id": document.get("memory_id"),
            "user_id": document.get("user_id"),
            "timestamp": timestamp,
            "has_embedding": bool(document.get("embedding")),
            "thread_id": document.get("thread_id") or document.get("conversation_id"),
        }

    def _read_index_file(self, store_path: Path) -> Dict[str, Dict[str, Any]]:
        index_path = store_path / "index.json"
        if not index_path.exists():
            return {}
        try:
            with index_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            items = payload.get("items")
            if isinstance(items, dict):
                return items
        except Exception as exc:
            logger.warning("Failed to load index from %s: %s", index_path, exc)
        return {}

    def _save_index(self, memory_type: MemoryType) -> None:
        store_path = self._store_paths[memory_type]
        index_path = store_path / "index.json"
        # Unique tmp name: two threads saving the same index must never race
        # on one shared tmp path (os.replace of a missing file raises ENOENT).
        tmp_path = index_path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        payload = {
            "version": self.INDEX_VERSION,
            "items": self._indexes[memory_type],
        }
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp_path, index_path)

    @staticmethod
    def _user_id_scope_matches(document: Dict[str, Any], user_id: Any) -> bool:
        """Return True when ``document`` belongs to the given tenant scope.

        ``_FS_UNSET`` disables filtering entirely. Any other value enforces
        strict equality — ``None`` only matches rows where ``user_id`` is
        missing or literally ``None``.
        """
        if user_id is _FS_UNSET:
            return True
        return document.get("user_id") == user_id

    @staticmethod
    def _retrieval_scope_matches(
        document: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> bool:
        """Apply exact episodic/knowledge boundaries before ranking."""
        if thread_id is not None:
            document_thread_id = document.get("thread_id") or document.get(
                "conversation_id"
            )
            if str(document_thread_id or "") != str(thread_id):
                return False
        if namespace is not None:
            content = document.get("content")
            nested = content if isinstance(content, dict) else {}
            document_namespace = document.get("namespace") or nested.get("namespace")
            if str(document_namespace or "") != str(namespace):
                return False
        return True

    def _filter_documents(
        self,
        memory_type: MemoryType,
        filters: Dict[str, Any],
        limit: int,
        memory_id: Optional[str],
        user_id: Any = _FS_UNSET,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        with self._locks[memory_type]:
            for doc_id in self._indexes[memory_type]:
                document = self._read_document(memory_type, doc_id)
                if not document:
                    continue
                if memory_id and document.get("memory_id") != memory_id:
                    continue
                if not self._user_id_scope_matches(document, user_id):
                    continue
                if not self._retrieval_scope_matches(
                    document, thread_id=thread_id, namespace=namespace
                ):
                    continue
                if all(document.get(k) == v for k, v in filters.items()):
                    matches.append(document)
                    if limit and len(matches) >= limit:
                        break
        return matches

    def _semantic_search(
        self,
        memory_type: MemoryType,
        query: str,
        limit: int,
        memory_id: Optional[str],
        user_id: Any = _FS_UNSET,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        embedding_provider = self._get_embedding_provider()
        if embedding_provider is None:
            logger.debug(
                "Embedding provider not configured; falling back to keyword search"
            )
            return self._keyword_search(
                memory_type,
                query,
                limit,
                memory_id,
                user_id=user_id,
                thread_id=thread_id,
                namespace=namespace,
            )

        # Documents created while embeddings were disabled have no vector to
        # rank. A later global embedder must not turn exact scoped recall into
        # a false miss or spend tokens embedding a query that cannot match.
        with self._locks[memory_type]:
            has_scoped_embedding = any(
                bool(meta.get("has_embedding"))
                and (not memory_id or meta.get("memory_id") == memory_id)
                and (user_id is _FS_UNSET or meta.get("user_id") == user_id)
                and (
                    thread_id is None
                    or str(meta.get("thread_id") or "") == str(thread_id)
                )
                for meta in self._indexes[memory_type].values()
            )
        if not has_scoped_embedding:
            return self._keyword_search(
                memory_type,
                query,
                limit,
                memory_id,
                user_id=user_id,
                thread_id=thread_id,
                namespace=namespace,
            )

        if self._faiss is None or np is None:
            logger.debug(
                "FAISS/numpy unavailable; falling back to brute-force cosine search"
            )
            matches = self._brute_force_search(
                memory_type,
                query,
                limit,
                memory_id,
                embedding_provider,
                user_id=user_id,
                thread_id=thread_id,
                namespace=namespace,
            )
            return matches or self._keyword_search(
                memory_type,
                query,
                limit,
                memory_id,
                user_id=user_id,
                thread_id=thread_id,
                namespace=namespace,
            )

        # A global FAISS top-k followed by filtering can return a false miss
        # when another thread/namespace crowds the requested scope. Rank only
        # eligible documents for explicitly scoped recall.
        if thread_id is not None or namespace is not None:
            matches = self._brute_force_search(
                memory_type,
                query,
                limit,
                memory_id,
                embedding_provider,
                user_id=user_id,
                thread_id=thread_id,
                namespace=namespace,
            )
            return matches or self._keyword_search(
                memory_type,
                query,
                limit,
                memory_id,
                user_id=user_id,
                thread_id=thread_id,
                namespace=namespace,
            )

        query_embedding = embedding_provider.get_embedding(query)
        query_vector = self._normalize_vector(
            np.array(query_embedding, dtype="float32")
        )

        index, doc_ids = self._ensure_vector_index(memory_type)
        if index is None or not doc_ids:
            return self._keyword_search(
                memory_type,
                query,
                limit,
                memory_id,
                user_id=user_id,
                thread_id=thread_id,
                namespace=namespace,
            )

        top_k = max(limit or 1, 1)
        # Over-fetch so the user_id filter still returns enough matches.
        distances, indices = index.search(query_vector.reshape(1, -1), top_k * 4)

        matches: List[Dict[str, Any]] = []
        for position, score in zip(indices[0], distances[0]):
            if position < 0 or position >= len(doc_ids):
                continue
            doc_id = doc_ids[position]
            document = self._read_document(memory_type, doc_id)
            if not document:
                continue
            if memory_id and document.get("memory_id") != memory_id:
                continue
            if not self._user_id_scope_matches(document, user_id):
                continue
            if not self._retrieval_scope_matches(
                document, thread_id=thread_id, namespace=namespace
            ):
                continue
            document["score"] = float(score)
            matches.append(document)
            if len(matches) >= top_k:
                break
        return matches or self._keyword_search(
            memory_type,
            query,
            limit,
            memory_id,
            user_id=user_id,
            thread_id=thread_id,
            namespace=namespace,
        )

    def _brute_force_search(
        self,
        memory_type: MemoryType,
        query: str,
        limit: int,
        memory_id: Optional[str],
        embedding_provider=None,
        user_id: Any = _FS_UNSET,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        embedding_provider = embedding_provider or self._get_embedding_provider()
        if embedding_provider is None:
            return self._keyword_search(
                memory_type,
                query,
                limit,
                memory_id,
                user_id=user_id,
                thread_id=thread_id,
                namespace=namespace,
            )

        query_embedding = embedding_provider.get_embedding(query)
        query_vector = (
            np.array(query_embedding, dtype="float32") if np else query_embedding
        )

        scored: List[Tuple[float, Dict[str, Any]]] = []
        with self._locks[memory_type]:
            for doc_id in self._indexes[memory_type]:
                document = self._read_document(memory_type, doc_id)
                if not document or "embedding" not in document:
                    continue
                if memory_id and document.get("memory_id") != memory_id:
                    continue
                if not self._user_id_scope_matches(document, user_id):
                    continue
                if not self._retrieval_scope_matches(
                    document, thread_id=thread_id, namespace=namespace
                ):
                    continue
                target = document["embedding"]
                similarity = self._cosine_similarity(query_vector, target)
                if similarity is None:
                    continue
                document["score"] = similarity
                scored.append((similarity, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[: limit or 1]]

    def _keyword_search(
        self,
        memory_type: MemoryType,
        query: str,
        limit: int,
        memory_id: Optional[str],
        user_id: Any = _FS_UNSET,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        needle = query.lower().strip()
        query_terms = set(re.findall(r"[a-z0-9_]+", needle))
        scored: List[Tuple[float, Dict[str, Any]]] = []
        with self._locks[memory_type]:
            for doc_id in self._indexes[memory_type]:
                document = self._read_document(memory_type, doc_id)
                if not document:
                    continue
                if memory_id and document.get("memory_id") != memory_id:
                    continue
                if not self._user_id_scope_matches(document, user_id):
                    continue
                if not self._retrieval_scope_matches(
                    document, thread_id=thread_id, namespace=namespace
                ):
                    continue
                haystacks = [
                    str(document.get("content", "")),
                    str(document.get("name", "")),
                    str(document.get("title", "")),
                ]
                searchable = "\n".join(haystacks).lower()
                if needle in searchable:
                    score = 2.0
                else:
                    document_terms = set(re.findall(r"[a-z0-9_]+", searchable))
                    overlap = len(query_terms.intersection(document_terms))
                    minimum_overlap = (
                        1
                        if len(query_terms) <= 2
                        else max(2, (len(query_terms) + 4) // 5)
                    )
                    if overlap < minimum_overlap:
                        continue
                    coverage = overlap / max(len(query_terms), 1)
                    specificity = overlap / max(len(document_terms), 1)
                    score = coverage + (0.2 * specificity)
                ranked = dict(document)
                ranked["score"] = float(score)
                scored.append((float(score), ranked))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[: max(int(limit or 1), 1)]]

    def _ensure_vector_index(
        self, memory_type: MemoryType
    ) -> Tuple[Optional[Any], List[str]]:
        if np is None or self._faiss is None:
            return None, []
        state = self._vector_state[memory_type]
        if state["index"] is not None and not state.get("dirty"):
            return state["index"], state["doc_ids"]

        doc_ids: List[str] = []
        vectors: List[np.ndarray] = []
        with self._locks[memory_type]:
            for doc_id in self._indexes[memory_type]:
                document = self._read_document(memory_type, doc_id)
                if not document:
                    continue
                embedding = document.get("embedding")
                if not embedding:
                    continue
                vector = self._normalize_vector(np.array(embedding, dtype="float32"))
                doc_ids.append(doc_id)
                vectors.append(vector)

        if not vectors:
            state["index"] = None
            state["doc_ids"] = []
            state["dirty"] = False
            return None, []

        dimension = vectors[0].shape[0]
        index = self._faiss.IndexFlatIP(dimension)
        stacked = np.stack(vectors, axis=0)
        index.add(stacked)

        state["index"] = index
        state["doc_ids"] = doc_ids
        state["dirty"] = False
        return index, doc_ids

    def _mark_vector_index_dirty(self, memory_type: MemoryType) -> None:
        if memory_type not in self._vector_state:
            return
        self._vector_state[memory_type]["dirty"] = True

    def _cosine_similarity(
        self, vector_a: Union[List[float], Any], vector_b: Union[List[float], Any]
    ) -> Optional[float]:
        if np is None:
            # Pure Python fallback
            try:
                dot = sum(a * b for a, b in zip(vector_a, vector_b))
                norm_a = sum(a * a for a in vector_a) ** 0.5
                norm_b = sum(b * b for b in vector_b) ** 0.5
                if norm_a == 0 or norm_b == 0:
                    return None
                return dot / (norm_a * norm_b)
            except Exception:
                return None

        vec_a = np.asarray(vector_a, dtype="float32").ravel()
        vec_b = np.asarray(vector_b, dtype="float32").ravel()
        # Guard against malformed/mismatched embeddings (e.g. a document embedded
        # with a different model/dimension, or a nested [[...]] shape). Flatten to
        # 1-D and bail on shape mismatch so np.dot stays scalar — otherwise
        # float(np.dot(...)) raises "only length-1 arrays can be converted to
        # Python scalars" and aborts brute-force retrieval.
        if vec_a.size == 0 or vec_a.shape != vec_b.shape:
            return None
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return None
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        if np is None:
            return vector
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _coerce_timestamp(self, value: Any) -> float:
        if value is None:
            return time.time()
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                try:
                    return datetime.fromisoformat(value).timestamp()
                except ValueError:
                    return time.time()
        return time.time()

    def _delete_memory_units_by_memory_id(
        self, memory_id: str, memory_type: MemoryType
    ) -> None:
        with self._locks[memory_type]:
            metadata = self._indexes[memory_type]
            for doc_id, meta in list(metadata.items()):
                if meta.get("memory_id") == memory_id:
                    file_path = self._document_path(memory_type, doc_id)
                    if file_path.exists():
                        file_path.unlink()
                    metadata.pop(doc_id, None)
            self._save_index(memory_type)
            self._mark_vector_index_dirty(memory_type)
