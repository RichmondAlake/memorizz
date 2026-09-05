"""Pytest configuration and fixtures for MemAgent tests."""
import os
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock

import pytest

# Ensure third-party clients that expect OpenAI credentials during tests do not fail.
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")
# Bare MemAgent instances use filesystem memory by default. Keep that durable
# behavior isolated from a developer's real ~/.memorizz store during tests.
os.environ.setdefault("MEMORIZZ_HOME", tempfile.mkdtemp(prefix="memorizz-tests-"))
os.environ.setdefault(
    "MEMORIZZ_UI_AUDIT_LOG",
    os.path.join(os.environ["MEMORIZZ_HOME"], "trace-audit.jsonl"),
)

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


# Provide a deterministic embedding implementation for tests to avoid network calls.
from memorizz import embeddings as _embeddings  # noqa: E402
from memorizz.short_term_memory import semantic_cache as _semantic_cache  # noqa: E402

# Skip provider tests whose modules import an optional DB driver at import time,
# so a base-only checkout still collects cleanly. The dev extra installs both
# drivers, so the canonical local/CI suite exercises these tests.
collect_ignore = []
try:
    import pymongo  # noqa: F401
except ImportError:
    collect_ignore.append("unit/test_mongodb_provider_self_aware.py")

try:
    import oracledb  # noqa: F401
except ImportError:
    collect_ignore.extend(
        [
            "unit/test_oracle_in_database_embeddings.py",
            "unit/test_oracle_provider_close.py",
            "unit/test_oracle_provider_ids.py",
            "unit/test_oracle_workflow_record_ids.py",
        ]
    )


def _test_embedding(_text: str, **kwargs):
    """Return a stub embedding vector for test isolation."""
    return [0.0]


class _TestEmbeddingManager:
    def get_embedding(self, text: str, **kwargs):
        return _test_embedding(text, **kwargs)

    def get_dimensions(self) -> int:
        return 1

    def get_default_model(self) -> str:
        return "test-stub"

    def get_provider_info(self):
        return {"provider": "test", "model": "test-stub", "dimensions": 1}


_embeddings.get_embedding = _test_embedding
_embeddings.get_embedding_manager = lambda: _TestEmbeddingManager()  # type: ignore
_semantic_cache.get_embedding_manager = lambda: _TestEmbeddingManager()  # type: ignore


# =============================================================================
# Core Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    mock_llm = Mock()
    mock_llm.generate.return_value = "This is a test response from the LLM."
    mock_llm.model_name = "test-model"
    mock_llm.provider = "test-provider"
    return mock_llm


@pytest.fixture
def mock_memory_provider():
    """Mock memory provider for testing."""
    mock_memory = Mock()
    mock_memory.supports_entity_memory.return_value = True
    mock_memory.entity_memory_collection = True

    # Mock memory storage
    mock_memory._storage = {}

    def mock_store(memory_id: str, memory_unit: Any):
        if memory_id not in mock_memory._storage:
            mock_memory._storage[memory_id] = []
        mock_memory._storage[memory_id].append(memory_unit)
        return f"unit_{len(mock_memory._storage[memory_id])}"

    def mock_retrieve_by_id(memory_id: str):
        return mock_memory._storage.get(memory_id, [])

    def mock_retrieve_by_query(
        query: str, memory_id: str, memory_type: Any, limit: int = 5
    ):
        # Simple mock that returns some relevant memories
        memories = mock_memory._storage.get(memory_id, [])
        return memories[:limit]

    def mock_retrieve_conversation_history(
        memory_id: str,
        memory_type: Any,
        limit: int = 10,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        memories = mock_memory._storage.get(memory_id, [])
        rows = []
        for mem in memories:
            content = mem if isinstance(mem, dict) else {}
            if content.get("user_id") != user_id:
                continue
            if thread_id is not None and str(content.get("thread_id") or "") != str(
                thread_id
            ):
                continue
            rows.append({"content": mem, "timestamp": datetime.now().isoformat()})
        return rows[:limit] if limit else rows

    mock_memory.store.side_effect = mock_store
    mock_memory.retrieve_by_id.side_effect = mock_retrieve_by_id
    mock_memory.retrieve_by_query.side_effect = mock_retrieve_by_query
    mock_memory.retrieve_conversation_history_ordered_by_timestamp.side_effect = (
        mock_retrieve_conversation_history
    )

    return mock_memory


@pytest.fixture
def sample_tools():
    """Sample tools for testing."""

    def calculator_add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def text_processor(text: str, operation: str = "upper") -> str:
        """Process text with various operations."""
        if operation == "upper":
            return text.upper()
        elif operation == "lower":
            return text.lower()
        elif operation == "reverse":
            return text[::-1]
        return text

    return [calculator_add, text_processor]


@pytest.fixture
def sample_persona():
    """Sample persona for testing (a real Persona, built without embeddings)."""
    from memorizz.long_term.semantic.persona import Persona

    persona = Persona.from_dict(
        {
            "name": "TestBot",
            "role": "Assistant",
            "goals": "Help with various tasks.",
            "background": "I am a test assistant designed to help with various tasks.",
        }
    )
    persona.personality_traits = ["helpful", "friendly", "knowledgeable"]
    persona.expertise = ["python", "testing", "ai"]
    return persona


# =============================================================================
# MemAgent Fixtures
# =============================================================================


@pytest.fixture
def basic_memagent_config():
    """Basic MemAgent configuration for testing."""
    from memorizz.memagent.models import MemAgentConfig

    return MemAgentConfig(
        instruction="You are a test assistant.",
        max_steps=10,
        tool_access="private",
        semantic_cache=False,
    )


@pytest.fixture
def advanced_memagent_config():
    """Advanced MemAgent configuration for testing."""
    from memorizz.memagent.models import MemAgentConfig

    return MemAgentConfig(
        instruction="You are an advanced test assistant with multiple capabilities.",
        max_steps=25,
        tool_access="private",
        semantic_cache=True,
        application_mode="agent",
    )


@pytest.fixture
def memagent_with_mocks(
    mock_llm_provider, mock_memory_provider, sample_tools, sample_persona
):
    """MemAgent instance with all mocks configured."""
    from memorizz.memagent import MemAgent

    agent = MemAgent(
        model=mock_llm_provider,
        memory_provider=mock_memory_provider,
        tools=sample_tools,
        persona=sample_persona,
        instruction="You are a fully mocked test assistant.",
        max_steps=15,
        semantic_cache=False,
    )

    # Wrap load_conversation_history so tests can assert call counts.
    if agent.memory_manager:
        original = agent.memory_manager.load_conversation_history
        agent.memory_manager.load_conversation_history = MagicMock(wraps=original)

    return agent


@pytest.fixture
def multi_agent_setup(mock_llm_provider, mock_memory_provider):
    """Setup for multi-agent testing."""
    from memorizz.memagent import MemAgent

    # Create multiple agents with different specializations
    agent1 = MemAgent(
        model=mock_llm_provider,
        memory_provider=mock_memory_provider,
        instruction="You are Agent 1, specialized in data analysis.",
        agent_id="agent_1",
        max_steps=10,
    )

    agent2 = MemAgent(
        model=mock_llm_provider,
        memory_provider=mock_memory_provider,
        instruction="You are Agent 2, specialized in text processing.",
        agent_id="agent_2",
        max_steps=10,
    )

    agent3 = MemAgent(
        model=mock_llm_provider,
        memory_provider=mock_memory_provider,
        instruction="You are Agent 3, the coordinator.",
        agent_id="coordinator",
        delegates=[agent1, agent2],
        max_steps=20,
    )

    return {
        "agent1": agent1,
        "agent2": agent2,
        "coordinator": agent3,
        "all_agents": [agent1, agent2, agent3],
    }


# =============================================================================
# Memory Type Fixtures
# =============================================================================


@pytest.fixture
def memory_types():
    """All memory types for testing."""
    from memorizz.enums import MemoryType

    return [
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.KNOWLEDGE_BASE,
        MemoryType.TOOLBOX,
        MemoryType.WORKFLOW_MEMORY,
    ]


@pytest.fixture
def conversation_memory_setup(mock_memory_provider):
    """Setup for conversation memory testing."""
    from memorizz.enums import Role
    from memorizz.memagent.managers import MemoryManager

    memory_manager = MemoryManager(mock_memory_provider)

    # Pre-populate with some conversation history
    thread_id = "test_conv_123"
    memory_id = "test_memory_456"

    # Add some sample conversation entries
    user_memory = memory_manager.create_conversation_memory_unit(
        role=Role.USER,
        content="Hello, how are you?",
        thread_id=thread_id,
        memory_id=memory_id,
    )

    assistant_memory = memory_manager.create_conversation_memory_unit(
        role=Role.ASSISTANT,
        content="I'm doing well, thank you! How can I help you today?",
        thread_id=thread_id,
        memory_id=memory_id,
    )

    memory_manager.save_memory_unit(user_memory, memory_id)
    memory_manager.save_memory_unit(assistant_memory, memory_id)

    return {
        "memory_manager": memory_manager,
        "thread_id": thread_id,
        "memory_id": memory_id,
        "sample_memories": [user_memory, assistant_memory],
    }


@pytest.fixture
def semantic_cache_setup():
    """Setup for semantic cache testing."""
    from memorizz.memagent.managers import CacheManager

    cache_manager = CacheManager(
        enabled=True,
        config={"similarity_threshold": 0.8, "scope": "local"},
        agent_id="test_agent",
        memory_id="test_memory",
    )

    return cache_manager


# =============================================================================
# Builder Fixtures
# =============================================================================


@pytest.fixture
def agent_builder():
    """MemAgent builder for testing."""
    from memorizz.memagent.builders import MemAgentBuilder

    return MemAgentBuilder()


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_queries():
    """Sample queries for testing different scenarios."""
    return [
        "What is the capital of France?",
        "How do I sort a list in Python?",
        "Can you help me write a function to calculate fibonacci numbers?",
        "What's the weather like today?",
        "Explain machine learning in simple terms.",
        "Write a poem about testing software.",
        "Calculate 2 + 2",
        "What are the benefits of unit testing?",
        "How does semantic caching work?",
        "Create a TODO list for learning pytest.",
    ]


@pytest.fixture
def sample_llm_responses():
    """Sample LLM responses for testing."""
    return [
        "Paris is the capital of France.",
        "You can sort a list in Python using the sorted() function or the .sort() method.",
        "Here's a simple fibonacci function: def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        "I don't have access to real-time weather data, but I can help you find weather services.",
        "Machine learning is a way for computers to learn patterns from data without being explicitly programmed.",
        "Testing code is like checking your work,\nMaking sure each function doesn't shirk...",
        "2 + 2 equals 4.",
        "Unit testing helps ensure code reliability, makes debugging easier, and improves maintainability.",
        "Semantic caching stores responses based on meaning similarity, not exact matches.",
        "Here's a pytest learning TODO: 1. Learn fixtures, 2. Write unit tests, 3. Mock dependencies...",
    ]


# =============================================================================
# Performance Testing Fixtures
# =============================================================================


@pytest.fixture
def performance_config():
    """Configuration for performance testing."""
    return {
        "max_response_time": 5.0,  # seconds
        "max_memory_usage": 100,  # MB
        "concurrent_requests": 10,
        "stress_test_duration": 30,  # seconds
    }


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture
def integration_test_setup():
    """Setup for integration tests."""
    return {
        "test_agent_ids": ["agent_1", "agent_2", "coordinator"],
        "test_thread_ids": ["conv_1", "conv_2", "conv_3"],
        "test_memory_ids": ["mem_1", "mem_2", "mem_3"],
        "test_scenarios": [
            "single_turn_conversation",
            "multi_turn_conversation",
            "tool_execution",
            "memory_retrieval",
            "cache_hit_miss",
            "multi_agent_coordination",
        ],
    }


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Add any cleanup logic here if needed
    pass


@pytest.fixture(scope="session")
def test_session_setup():
    """Setup for the entire test session."""
    print("\\n🧪 Starting MemAgent test suite...")
    yield
    print("\\n✅ MemAgent test suite completed!")


# =============================================================================
# Utility Functions for Tests
# =============================================================================


def assert_agent_response_valid(response: str):
    """Assert that an agent response is valid."""
    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be a string"
    assert len(response.strip()) > 0, "Response should not be empty"
    assert len(response) < 10000, "Response should not be excessively long"


def assert_memory_unit_valid(memory_unit: Dict[str, Any]):
    """Assert that a memory unit is valid."""
    assert memory_unit is not None, "Memory unit should not be None"
    assert isinstance(memory_unit, dict), "Memory unit should be a dictionary"
    assert "content" in memory_unit, "Memory unit should have content"
    assert "memory_type" in memory_unit, "Memory unit should have memory_type"


def assert_agent_state_valid(agent):
    """Assert that an agent is in a valid state."""
    assert agent is not None, "Agent should not be None"
    assert hasattr(agent, "run"), "Agent should have run method"
    assert hasattr(agent, "agent_id"), "Agent should have agent_id"
    assert hasattr(agent, "instruction"), "Agent should have instruction"
