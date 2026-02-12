import json
import logging
import os
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

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

from ...enums.memory_type import MemoryType
from ..base import MemoryProvider

logger = logging.getLogger(__name__)


@dataclass
class FileSystemConfig:
    """Configuration for the filesystem provider."""

    root_path: Union[str, Path]
    lazy_vector_indexes: bool = False
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
    VECTOR_ENABLED_TYPES = {
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.LONG_TERM_MEMORY,
        MemoryType.SHORT_TERM_MEMORY,
        MemoryType.WORKFLOW_MEMORY,
        MemoryType.SUMMARIES,
        MemoryType.SEMANTIC_CACHE,
        MemoryType.ENTITY_MEMORY,
    }

    def __init__(self, config: FileSystemConfig):
        self.config = config
        self.root_path = config.root_path
        self.root_path.mkdir(parents=True, exist_ok=True)

        self._embedding_provider = self._setup_embedding_provider(config)
        self._store_paths: Dict[MemoryType, Path] = {}
        self._indexes: Dict[MemoryType, Dict[str, Dict[str, Any]]] = {}
        self._locks: Dict[MemoryType, threading.RLock] = {
            memory_type: threading.RLock() for memory_type in MemoryType
        }
        self._vector_state: Dict[
            MemoryType, Dict[str, Any]
        ] = {}  # {"index": faiss.Index, "doc_ids": [...], "dirty": bool}

        self._initialize_storage()

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

        if isinstance(query, dict):
            return self._filter_documents(
                resolved_type, query, limit, memory_id=memory_id
            )
        elif isinstance(query, str):
            return self._semantic_search(
                resolved_type, query, limit, memory_id=memory_id
            )
        else:
            raise ValueError("query must be either a dict filter or a string")

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
        self, memory_store_type: Union[str, MemoryType, None]
    ) -> List[Dict[str, Any]]:
        memory_type = self._normalize_memory_type(memory_store_type)
        documents: List[Dict[str, Any]] = []
        with self._locks[memory_type]:
            for doc_id in self._indexes[memory_type].keys():
                document = self._read_document(memory_type, doc_id)
                if document:
                    documents.append(document)
        return documents

    def retrieve_conversation_history_ordered_by_timestamp(
        self,
        memory_id: str,
        memory_type: Union[str, MemoryType, None] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        resolved_type = self._normalize_memory_type(
            memory_type or MemoryType.CONVERSATION_MEMORY
        )
        with self._locks[resolved_type]:
            documents: List[Tuple[float, Dict[str, Any]]] = []
            for doc_id in self._indexes[resolved_type]:
                document = self._read_document(resolved_type, doc_id)
                if not document:
                    continue
                if document.get("memory_id") != memory_id:
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
        return agent_id

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

        from ...long_term_memory.semantic.persona.persona import Persona
        from ...long_term_memory.semantic.persona.role_type import RoleType
        from ...memagent import MemAgentModel

        for doc in documents:
            agent = MemAgentModel(
                name=doc.get("name"),
                instruction=doc.get("instruction"),
                application_mode=doc.get("application_mode", "assistant"),
                memory_types=doc.get("memory_types"),
                max_steps=doc.get("max_steps"),
                memory_ids=doc.get("memory_ids") or [],
                agent_id=doc.get("agent_id") or doc.get("_id"),
                is_favorite=bool(doc.get("is_favorite", False)),
                tools=doc.get("tools"),
                long_term_memory_ids=doc.get("long_term_memory_ids"),
                sandbox_provider=doc.get("sandbox_provider"),
                internet_access_provider=doc.get("internet_access_provider"),
                internet_access_config=doc.get("internet_access_config"),
                skills_marketplace_provider=doc.get("skills_marketplace_provider"),
                skills_marketplace_config=doc.get("skills_marketplace_config"),
                skill_paths=doc.get("skill_paths"),
                mcp_servers=doc.get("mcp_servers"),
                memory_provider=self,
            )

            persona_data = doc.get("persona")
            if persona_data:
                role_value = persona_data.get("role")
                role = None
                for role_type in RoleType:
                    if role_type.value == role_value:
                        role = role_type
                        break
                role = role or RoleType.GENERAL
                agent.persona = Persona(
                    name=persona_data.get("name"),
                    role=role,
                    goals=persona_data.get("goals"),
                    background=persona_data.get("background"),
                    persona_id=persona_data.get("persona_id"),
                )
            agents.append(agent)
        return agents

    def retrieve_memagent(self, agent_id: str) -> Optional["MemAgentModel"]:
        document = self.retrieve_by_id(agent_id, MemoryType.MEMAGENT)
        if not document:
            return None

        from ...long_term_memory.semantic.persona.persona import Persona
        from ...long_term_memory.semantic.persona.role_type import RoleType
        from ...memagent import MemAgentModel

        memagent = MemAgentModel(
            name=document.get("name"),
            instruction=document.get("instruction"),
            application_mode=document.get("application_mode", "assistant"),
            memory_types=document.get("memory_types"),
            max_steps=document.get("max_steps"),
            memory_ids=document.get("memory_ids") or [],
            agent_id=document.get("agent_id") or document.get("_id"),
            is_favorite=bool(document.get("is_favorite", False)),
            tools=document.get("tools"),
            long_term_memory_ids=document.get("long_term_memory_ids"),
            sandbox_provider=document.get("sandbox_provider"),
            internet_access_provider=document.get("internet_access_provider"),
            internet_access_config=document.get("internet_access_config"),
            skills_marketplace_provider=document.get("skills_marketplace_provider"),
            skills_marketplace_config=document.get("skills_marketplace_config"),
            skill_paths=document.get("skill_paths"),
            mcp_servers=document.get("mcp_servers"),
            memory_provider=self,
        )

        persona_data = document.get("persona")
        if persona_data:
            role_value = persona_data.get("role")
            role = None
            for role_type in RoleType:
                if role_type.value == role_value:
                    role = role_type
                    break
            role = role or RoleType.GENERAL
            memagent.persona = Persona(
                name=persona_data.get("name"),
                role=role,
                goals=persona_data.get("goals"),
                background=persona_data.get("background"),
                persona_id=persona_data.get("persona_id"),
            )
        return memagent

    def supports_entity_memory(self) -> bool:
        return True

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
        self, memory_type: MemoryType, document_id: str, document: Dict[str, Any]
    ) -> None:
        file_path = self._document_path(memory_type, document_id)
        tmp_path = file_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False)
        os.replace(tmp_path, file_path)

        metadata = self._indexes[memory_type]
        metadata[document_id] = self._build_metadata(document)
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
                return json.load(handle)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return None

    def _build_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = document.get("timestamp") or document.get("created_at")
        return {
            "id": document.get("id"),
            "name": document.get("name") or document.get("title"),
            "memory_id": document.get("memory_id"),
            "timestamp": timestamp,
            "has_embedding": bool(document.get("embedding")),
            "conversation_id": document.get("conversation_id"),
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
        tmp_path = index_path.with_suffix(".tmp")
        payload = {
            "version": self.INDEX_VERSION,
            "items": self._indexes[memory_type],
        }
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp_path, index_path)

    def _filter_documents(
        self,
        memory_type: MemoryType,
        filters: Dict[str, Any],
        limit: int,
        memory_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        with self._locks[memory_type]:
            for doc_id in self._indexes[memory_type]:
                document = self._read_document(memory_type, doc_id)
                if not document:
                    continue
                if memory_id and document.get("memory_id") != memory_id:
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
    ) -> List[Dict[str, Any]]:
        embedding_provider = self._get_embedding_provider()
        if embedding_provider is None:
            logger.debug(
                "Embedding provider not configured; falling back to keyword search"
            )
            return self._keyword_search(memory_type, query, limit, memory_id)

        if faiss is None or np is None:
            logger.debug(
                "FAISS/numpy unavailable; falling back to brute-force cosine search"
            )
            return self._brute_force_search(
                memory_type, query, limit, memory_id, embedding_provider
            )

        query_embedding = embedding_provider.get_embedding(query)
        query_vector = self._normalize_vector(
            np.array(query_embedding, dtype="float32")
        )

        index, doc_ids = self._ensure_vector_index(memory_type)
        if index is None or not doc_ids:
            return []

        top_k = max(limit or 1, 1)
        distances, indices = index.search(query_vector.reshape(1, -1), top_k * 2)

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
            document["score"] = float(score)
            matches.append(document)
            if len(matches) >= top_k:
                break
        return matches

    def _brute_force_search(
        self,
        memory_type: MemoryType,
        query: str,
        limit: int,
        memory_id: Optional[str],
        embedding_provider=None,
    ) -> List[Dict[str, Any]]:
        embedding_provider = embedding_provider or self._get_embedding_provider()
        if embedding_provider is None:
            return self._keyword_search(memory_type, query, limit, memory_id)

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
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        needle = query.lower()
        matches: List[Dict[str, Any]] = []
        with self._locks[memory_type]:
            for doc_id in self._indexes[memory_type]:
                document = self._read_document(memory_type, doc_id)
                if not document:
                    continue
                if memory_id and document.get("memory_id") != memory_id:
                    continue
                haystacks = [
                    str(document.get("content", "")),
                    str(document.get("name", "")),
                    str(document.get("title", "")),
                ]
                if any(needle in hay.lower() for hay in haystacks if hay):
                    matches.append(document)
                    if limit and len(matches) >= limit:
                        break
        return matches

    def _ensure_vector_index(
        self, memory_type: MemoryType
    ) -> Tuple[Optional["faiss.Index"], List[str]]:
        if np is None or faiss is None:
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
        index = faiss.IndexFlatIP(dimension)
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

        vec_a = np.array(vector_a, dtype="float32")
        vec_b = np.array(vector_b, dtype="float32")
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
