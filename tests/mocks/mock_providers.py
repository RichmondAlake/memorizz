"""Mock providers for comprehensive testing."""
import heapq
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock

from memorizz.enums import MemoryType


class MockLLMProvider:
    """Comprehensive mock LLM provider for testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        provider: str = "mock",
        model: str = "mock-model",
    ):
        """Initialize with optional predefined responses."""
        self.responses = responses or [
            "This is a mock response.",
            "I understand your question.",
            "Let me help you with that.",
            "Here's the information you requested.",
            "I've completed the task successfully.",
        ]
        self.provider = provider
        self.model = model
        self.response_index = 0
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None
        self.last_tool_choice = None
        self.last_usage = None
        self.context_window_tokens = None

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> str:
        """Mock generate method."""
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools
        self.last_tool_choice = tool_choice

        # Return next response in cycle
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1

        return response

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        """Mock generate_text method."""
        messages = [{"role": "user", "content": prompt}]
        return self.generate(messages)

    def get_config(self) -> Dict[str, Any]:
        """Return mock configuration."""
        return {"provider": self.provider, "model": self.model}

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        """Return mock usage stats."""
        return self.last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        """Return mock context window size."""
        return self.context_window_tokens

    def reset(self):
        """Reset mock state."""
        self.response_index = 0
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None
        self.last_tool_choice = None
        self.last_usage = None


class MockMemoryProvider:
    """Comprehensive mock memory provider for testing."""

    def __init__(self):
        """Initialize mock memory provider."""
        self.storage = {}
        # Secondary index for faster retrievals in tests/perf suites:
        # memory_id -> {MemoryType: [MockMemoryUnit, ...]}
        self._storage_by_type: Dict[str, Dict[MemoryType, List["MockMemoryUnit"]]] = {}
        self.agents = {}
        self.call_history = []
        self.semantic_retrieval_map = {}
        self.episodic_retrieval_map = {}
        self.procedural_retrieval_map = {}
        self.capacity_limit = None

    def _normalize_memory_type_key(self, value: Any) -> Optional[MemoryType]:
        if value is None:
            return None
        if isinstance(value, MemoryType):
            return value
        if isinstance(value, str):
            try:
                return MemoryType(value)
            except ValueError:
                return None
        return None

    def _reindex_memory_id(self, memory_id: str) -> None:
        """Rebuild the per-type index for a bucket (used after capacity trimming)."""
        by_type: Dict[MemoryType, List["MockMemoryUnit"]] = {}
        for unit in self.storage.get(memory_id, []) or []:
            key = self._normalize_memory_type_key(getattr(unit, "memory_type", None))
            if key is None:
                continue
            by_type.setdefault(key, []).append(unit)
        self._storage_by_type[memory_id] = by_type

    def _normalize_timestamp(self, value: Any) -> datetime:
        if value is None:
            return datetime.now()
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.now()
        return datetime.now()

    def _ensure_memory_unit(
        self,
        memory_unit: Any,
        memory_id: str,
        default_memory_type: Optional[Any] = None,
    ):
        if isinstance(memory_unit, MockMemoryUnit):
            if memory_unit.memory_id is None:
                memory_unit.memory_id = memory_id
            return memory_unit

        if hasattr(memory_unit, "role") and hasattr(memory_unit, "conversation_id"):
            content = {
                "role": getattr(memory_unit, "role"),
                "content": getattr(memory_unit, "content"),
                "conversation_id": getattr(memory_unit, "conversation_id"),
                "timestamp": getattr(memory_unit, "timestamp", None),
            }
            return MockMemoryUnit(
                memory_type=MemoryType.CONVERSATION_MEMORY,
                content=content,
                timestamp=self._normalize_timestamp(
                    getattr(memory_unit, "timestamp", None)
                ),
                memory_id=memory_id,
            )

        if hasattr(memory_unit, "model_dump"):
            payload = memory_unit.model_dump()
        elif hasattr(memory_unit, "dict"):
            payload = memory_unit.dict()
        elif isinstance(memory_unit, dict):
            payload = memory_unit
        else:
            payload = {"content": memory_unit}

        memory_type = payload.get("memory_type") or payload.get("type")
        if isinstance(memory_type, str):
            try:
                memory_type = MemoryType(memory_type)
            except ValueError:
                memory_type = MemoryType.CONVERSATION_MEMORY
        if memory_type is None and default_memory_type is not None:
            if isinstance(default_memory_type, str):
                try:
                    memory_type = MemoryType(default_memory_type)
                except ValueError:
                    memory_type = MemoryType.CONVERSATION_MEMORY
            else:
                memory_type = default_memory_type
        if memory_type is None:
            memory_type = MemoryType.CONVERSATION_MEMORY

        timestamp = self._normalize_timestamp(payload.get("timestamp"))
        content = payload.get("content", payload)

        return MockMemoryUnit(
            memory_type=memory_type,
            content=content,
            timestamp=timestamp,
            memory_id=memory_id,
        )

    def _trim_to_capacity(self, memory_id: str) -> None:
        if self.capacity_limit is None:
            return
        if memory_id not in self.storage:
            return
        units = self.storage[memory_id]
        if len(units) <= self.capacity_limit:
            return
        units.sort(key=lambda unit: unit.timestamp)
        self.storage[memory_id] = units[-self.capacity_limit :]
        self._reindex_memory_id(memory_id)

    def store(
        self,
        data: Optional[Dict[str, Any]] = None,
        memory_store_type: Optional[Any] = None,
        memory_id: Optional[str] = None,
        memory_unit: Any = None,
        **kwargs,
    ) -> str:
        """Mock store method supporting legacy and new signatures."""
        # Support positional legacy signature: store(memory_id, memory_unit)
        if (
            memory_unit is None
            and memory_id is None
            and data is not None
            and memory_store_type is not None
            and not isinstance(data, dict)
        ):
            memory_id = str(data)
            memory_unit = memory_store_type
            memory_store_type = None
            data = None

        if memory_unit is None:
            memory_unit = data

        # For semantic cache entries, avoid using the entry's scoped memory_id as
        # the storage bucket key. Tests expect conversation history buckets to
        # contain only conversation messages.
        if memory_id is None and isinstance(memory_unit, dict):
            if (
                isinstance(memory_store_type, MemoryType)
                and memory_store_type == MemoryType.SEMANTIC_CACHE
            ):
                memory_id = None
            else:
                memory_id = memory_unit.get("memory_id")

        if memory_id is None:
            if isinstance(memory_store_type, MemoryType):
                memory_id = memory_store_type.value
            elif memory_store_type:
                memory_id = str(memory_store_type)
            else:
                memory_id = "default"

        self.call_history.append(("store", memory_id, memory_unit, memory_store_type))

        if memory_id not in self.storage:
            self.storage[memory_id] = []

        unit = self._ensure_memory_unit(
            memory_unit, memory_id, default_memory_type=memory_store_type
        )
        if unit.id is None:
            unit.id = str(uuid.uuid4())
        self.storage[memory_id].append(unit)
        key = self._normalize_memory_type_key(getattr(unit, "memory_type", None))
        if key is not None:
            self._storage_by_type.setdefault(memory_id, {}).setdefault(key, []).append(
                unit
            )
        self._trim_to_capacity(memory_id)
        return unit.id

    def retrieve_by_id(
        self, unit_id: str, memory_store_type: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Mock retrieve by ID method."""
        self.call_history.append(("retrieve_by_id", unit_id, memory_store_type))

        for memory_id, units in self.storage.items():
            for unit in units:
                if unit.id == unit_id:
                    return unit.to_dict()
        return None

    def retrieve_by_query(
        self,
        query: str,
        memory_id: Optional[str] = None,
        memory_type: Optional[Any] = None,
        limit: int = 5,
        memory_store_type: Optional[Any] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Mock retrieve by query method."""
        if limit is not None and isinstance(limit, int) and limit <= 0:
            self.call_history.append(
                (
                    "retrieve_by_query",
                    query,
                    memory_id,
                    memory_type or memory_store_type,
                    limit,
                )
            )
            return []
        if memory_type is None:
            memory_type = memory_store_type
        self.call_history.append(
            ("retrieve_by_query", query, memory_id, memory_type, limit)
        )

        type_key = self._normalize_memory_type_key(memory_type)

        # Resolve which bucket(s) to search. Most callsites pass memory_id, but
        # semantic cache lookups rely on memory_store_type only.
        if memory_id:
            if type_key is not None:
                units = self._storage_by_type.get(memory_id, {}).get(type_key, [])
            else:
                units = self.storage.get(memory_id, [])
        elif (
            isinstance(memory_type, MemoryType)
            and memory_type == MemoryType.SEMANTIC_CACHE
        ):
            cache_bucket = MemoryType.SEMANTIC_CACHE.value
            if type_key is not None:
                units = self._storage_by_type.get(cache_bucket, {}).get(type_key, [])
            else:
                units = self.storage.get(cache_bucket, [])
        else:
            units = []

        # Support dict-based filter queries (e.g. SemanticCache loading).
        if isinstance(query, dict):
            relevant_units: List[Dict[str, Any]] = []
            for unit in units:
                unit_dict = unit.to_dict()
                content = unit.content if isinstance(unit.content, dict) else {}
                matches = True
                for key, value in query.items():
                    # Prefer matching against the stored payload content when present.
                    if key in content:
                        if content.get(key) != value:
                            matches = False
                            break
                    else:
                        if unit_dict.get(key) != value:
                            matches = False
                            break
                if matches:
                    relevant_units.append(unit_dict)
                    if limit and len(relevant_units) >= limit:
                        break
            return relevant_units[:limit]

        query_text = str(query).lower()

        effective_type_key = ""
        if type_key is not None:
            effective_type_key = type_key.name.lower()
            if type_key == MemoryType.LONG_TERM_MEMORY:
                effective_type_key = "semantic_memory"
            elif type_key == MemoryType.CONVERSATION_MEMORY:
                effective_type_key = "episodic_memory"
            elif type_key in (MemoryType.TOOLBOX, MemoryType.WORKFLOW_MEMORY):
                effective_type_key = "procedural_memory"

        if (
            isinstance(query, str)
            and "semantic" in effective_type_key
            and query in self.semantic_retrieval_map
        ):
            return [unit.to_dict() for unit in self.semantic_retrieval_map[query]][
                :limit
            ]
        if (
            isinstance(query, str)
            and "episodic" in effective_type_key
            and query in self.episodic_retrieval_map
        ):
            return [unit.to_dict() for unit in self.episodic_retrieval_map[query]][
                :limit
            ]
        if (
            isinstance(query, str)
            and "procedural" in effective_type_key
            and query in self.procedural_retrieval_map
        ):
            return [unit.to_dict() for unit in self.procedural_retrieval_map[query]][
                :limit
            ]

        query_words = query_text.split()
        relevant_units = []

        for unit in units:
            unit_text = str(unit.content).lower()
            if any(word in unit_text for word in query_words):
                relevant_units.append(unit.to_dict())
                if limit and len(relevant_units) >= limit:
                    break

        return relevant_units[:limit]

    def retrieve_conversation_history_ordered_by_timestamp(
        self, memory_id: str, memory_type: Any, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Mock conversation history retrieval."""
        self.call_history.append(
            ("retrieve_conversation_history", memory_id, memory_type, limit)
        )

        type_key = self._normalize_memory_type_key(memory_type)
        if type_key is not None:
            units = self._storage_by_type.get(memory_id, {}).get(type_key, [])
        else:
            units = self.storage.get(memory_id, [])

        if not units:
            return []

        if not limit:
            sorted_units = sorted(units, key=lambda unit: unit.timestamp)
            return [unit.to_dict() for unit in sorted_units]

        # Fast path: select most recent N without sorting the full list.
        recent = heapq.nlargest(int(limit), units, key=lambda unit: unit.timestamp)
        recent.sort(key=lambda unit: unit.timestamp)
        return [unit.to_dict() for unit in recent]

    def store_memagent(self, agent_data: Dict[str, Any]) -> str:
        """Mock store agent method."""
        agent_id = agent_data.get("agent_id", str(uuid.uuid4()))
        self.agents[agent_id] = agent_data
        self.call_history.append(("store_memagent", agent_id))
        return agent_id

    def retrieve_memagent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Mock retrieve agent method."""
        self.call_history.append(("retrieve_memagent", agent_id))
        return self.agents.get(agent_id)

    def list_memagents(self) -> List[Dict[str, Any]]:
        """Mock list agents method."""
        self.call_history.append(("list_memagents",))
        return list(self.agents.values())

    def delete_by_id(self, memory_id: str) -> bool:
        """Mock delete by ID method."""
        self.call_history.append(("delete_by_id", memory_id))
        if memory_id in self.storage:
            del self.storage[memory_id]
            self._storage_by_type.pop(memory_id, None)
            return True
        return False

    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """Mock update agent memory IDs method."""
        self.call_history.append(("update_memagent_memory_ids", agent_id, memory_ids))
        if agent_id in self.agents:
            self.agents[agent_id]["memory_ids"] = memory_ids
            return True
        return False

    def get_call_history(self) -> List[tuple]:
        """Get history of all method calls."""
        return self.call_history.copy()

    def reset(self):
        """Reset mock state."""
        self.storage.clear()
        self._storage_by_type.clear()
        self.agents.clear()
        self.call_history.clear()
        self.semantic_retrieval_map = {}
        self.episodic_retrieval_map = {}
        self.procedural_retrieval_map = {}

    def configure_semantic_retrieval(self, retrieval_map: Dict[str, List[Any]]) -> None:
        self.semantic_retrieval_map = retrieval_map

    def configure_episodic_retrieval(self, retrieval_map: Dict[str, List[Any]]) -> None:
        self.episodic_retrieval_map = retrieval_map

    def configure_procedural_retrieval(
        self, retrieval_map: Dict[str, List[Any]]
    ) -> None:
        self.procedural_retrieval_map = retrieval_map

    def set_capacity_limit(self, limit: int) -> None:
        self.capacity_limit = limit


class MockMemoryUnit:
    """Mock memory unit for testing."""

    def __init__(
        self,
        memory_type: Any,
        content: Any,
        timestamp: Optional[datetime] = None,
        memory_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        unit_id: Optional[str] = None,
    ):
        self.id = unit_id or str(uuid.uuid4())
        self.memory_type = memory_type
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.memory_id = memory_id
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
            if hasattr(self.timestamp, "isoformat")
            else self.timestamp,
            "memory_id": self.memory_id,
            "embedding": self.embedding,
        }


class MockToolbox:
    """Mock toolbox for testing tool functionality."""

    def __init__(self, tools: Optional[Dict[str, Any]] = None):
        """Initialize with optional tools."""
        self.tools = tools or {
            "calculator": {
                "metadata": {
                    "name": "calculator",
                    "description": "Simple calculator",
                    "parameters": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                        "operation": {
                            "type": "string",
                            "description": "Operation (+, -, *, /)",
                        },
                    },
                    "required": ["a", "b", "operation"],
                },
                "function": lambda a, b, operation: self._calculator(a, b, operation),
            },
            "text_transformer": {
                "metadata": {
                    "name": "text_transformer",
                    "description": "Transform text",
                    "parameters": {
                        "text": {"type": "string", "description": "Text to transform"},
                        "transform": {
                            "type": "string",
                            "description": "Type of transformation",
                        },
                    },
                    "required": ["text"],
                },
                "function": lambda text, transform="upper": text.upper()
                if transform == "upper"
                else text.lower(),
            },
        }
        self.call_history = []

    def _calculator(self, a: float, b: float, operation: str) -> float:
        """Mock calculator function."""
        self.call_history.append(("calculator", a, b, operation))
        if operation == "+":
            return a + b
        elif operation == "-":
            return a - b
        elif operation == "*":
            return a * b
        elif operation == "/":
            return a / b if b != 0 else float("inf")
        else:
            raise ValueError(f"Unknown operation: {operation}")


class MockPersona:
    """Mock persona for testing."""

    def __init__(
        self,
        name: str = "TestBot",
        role: str = "Assistant",
        traits: List[str] = None,
        expertise: List[str] = None,
    ):
        """Initialize mock persona."""
        self.name = name
        self.role = role
        self.personality_traits = traits or ["helpful", "friendly"]
        self.expertise = expertise or ["testing", "mocking"]
        self.background = "I am a mock persona for testing purposes."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "personality_traits": self.personality_traits,
            "expertise": self.expertise,
            "background": self.background,
        }


class MockSemanticCache:
    """Mock semantic cache for testing."""

    def __init__(self, enabled: bool = True):
        """Initialize mock semantic cache."""
        self.enabled = enabled
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, query: str, session_id: Optional[str] = None) -> Optional[str]:
        """Mock get method."""
        cache_key = f"{query}_{session_id or 'default'}"
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        else:
            self.misses += 1
            return None

    def set(self, query: str, response: str, session_id: Optional[str] = None):
        """Mock set method."""
        cache_key = f"{query}_{session_id or 'default'}"
        self.cache[cache_key] = response

    def clear(self):
        """Mock clear method."""
        self.cache.clear()

    def clear_session(self, session_id: str):
        """Mock clear session method."""
        to_remove = [key for key in self.cache.keys() if key.endswith(f"_{session_id}")]
        for key in to_remove:
            del self.cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
            "hit_rate": self.hits / (self.hits + self.misses)
            if (self.hits + self.misses) > 0
            else 0,
        }
