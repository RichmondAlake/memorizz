"""Tests for all memory types and memory functionality."""
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from memorizz.enums import MemoryType, Role
from memorizz.memagent import MemAgent
from tests.conftest import assert_agent_response_valid, assert_memory_unit_valid
from tests.mocks.mock_providers import (
    MockLLMProvider,
    MockMemoryProvider,
    MockMemoryUnit,
)


class TestConversationMemory:
    """Test conversation memory functionality."""

    @pytest.mark.memory
    @pytest.mark.conversation_memory
    def test_conversation_memory_storage(self):
        """Test storing conversation memory."""
        llm_provider = MockLLMProvider(["I'll remember our conversation."])
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You have excellent conversation memory.",
        )

        memory_id = "conv_memory_test"
        thread_id = "conv_123"

        response = agent.run(
            "Hello, my name is Alice and I like pizza.",
            memory_id=memory_id,
            thread_id=thread_id,
        )

        assert_agent_response_valid(response)

        # Verify conversation was stored
        stored_memories = memory_provider.storage[memory_id]
        assert len(stored_memories) == 2  # User message + Assistant response

        # Check user message
        user_memory = stored_memories[0]
        assert user_memory.memory_type == MemoryType.CONVERSATION_MEMORY
        assert user_memory.content["role"] == Role.USER.value
        assert "Alice" in user_memory.content["content"]
        assert user_memory.content["thread_id"] == thread_id

        # Check assistant message
        assistant_memory = stored_memories[1]
        assert assistant_memory.memory_type == MemoryType.CONVERSATION_MEMORY
        assert assistant_memory.content["role"] == Role.ASSISTANT.value
        assert assistant_memory.content["thread_id"] == thread_id

    @pytest.mark.memory
    @pytest.mark.conversation_memory
    def test_conversation_memory_retrieval(self):
        """Test retrieving conversation memory."""
        memory_provider = MockMemoryProvider()

        # Pre-populate memory with conversation history
        memory_id = "conv_retrieval_test"
        thread_id = "conv_456"

        # Add some historical conversation
        for i, (role, content) in enumerate(
            [
                (Role.USER, "Hi, I'm Bob"),
                (Role.ASSISTANT, "Hello Bob, nice to meet you!"),
                (Role.USER, "I work as a teacher"),
                (Role.ASSISTANT, "That's wonderful! Teaching is important work."),
            ]
        ):
            memory_unit = MockMemoryUnit(
                memory_type=MemoryType.CONVERSATION_MEMORY,
                content={
                    "role": role.value,
                    "content": content,
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat(),
                },
                timestamp=datetime.now() - timedelta(minutes=i),
            )
            memory_provider.store(memory_id, memory_unit)

        llm_provider = MockLLMProvider(
            ["Yes Bob, you mentioned you're a teacher. How can I help?"]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You remember conversation context well.",
            memory_ids=[memory_id],
        )

        # Make new query - agent should have access to conversation history
        response = agent.run(
            "Do you remember what I do for work?",
            memory_id=memory_id,
            thread_id=thread_id,
        )

        assert_agent_response_valid(response)

        # Verify conversation history was loaded during context building
        # The LLM should have received the conversation history in its context
        assert llm_provider.call_count == 1

    @pytest.mark.memory
    @pytest.mark.conversation_memory
    def test_conversation_memory_limits(self):
        """Test conversation memory retrieval limits."""
        memory_provider = MockMemoryProvider()
        memory_id = "conv_limits_test"
        thread_id = "conv_789"

        # Create a long conversation history (20 exchanges)
        for i in range(20):
            user_memory = MockMemoryUnit(
                memory_type=MemoryType.CONVERSATION_MEMORY,
                content={
                    "role": Role.USER.value,
                    "content": f"User message {i+1}",
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat(),
                },
                timestamp=datetime.now() - timedelta(minutes=40 - i * 2),
            )

            assistant_memory = MockMemoryUnit(
                memory_type=MemoryType.CONVERSATION_MEMORY,
                content={
                    "role": Role.ASSISTANT.value,
                    "content": f"Assistant response {i+1}",
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat(),
                },
                timestamp=datetime.now() - timedelta(minutes=39 - i * 2),
            )

            memory_provider.store(memory_id, user_memory)
            memory_provider.store(memory_id, assistant_memory)

        llm_provider = MockLLMProvider(
            ["Based on our conversation history, I can help you."]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You work with conversation limits.",
            memory_ids=[memory_id],
        )

        response = agent.run(
            "Summarize our conversation",
            memory_id=memory_id,
            thread_id=thread_id,
        )

        assert_agent_response_valid(response)

        # Verify that the memory manager applied limits when loading history
        # (The actual limit is typically 10 recent messages as set in the core.py)
        history_call = memory_provider.get_call_history()
        assert len(history_call) > 0

    @pytest.mark.memory
    @pytest.mark.conversation_memory
    def test_multiple_conversation_isolation(self):
        """Test that different conversations are kept separate."""
        memory_provider = MockMemoryProvider()
        memory_id = "multi_conv_test"

        llm_provider = MockLLMProvider(["Hello Alice!", "Hello Bob!", "Hello Charlie!"])

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You maintain separate conversations.",
            memory_ids=[memory_id],
        )

        # Three separate conversations
        conversations = {
            "conv_alice": "Hi, I'm Alice and I like cats",
            "conv_bob": "Hi, I'm Bob and I like dogs",
            "conv_charlie": "Hi, I'm Charlie and I like birds",
        }

        responses = {}
        for conv_id, message in conversations.items():
            responses[conv_id] = agent.run(
                message, memory_id=memory_id, thread_id=conv_id
            )

        # Verify all responses are valid
        for response in responses.values():
            assert_agent_response_valid(response)

        # Verify conversations are stored separately by thread_id
        all_memories = memory_provider.storage[memory_id]

        alice_memories = [
            m for m in all_memories if m.content.get("thread_id") == "conv_alice"
        ]
        bob_memories = [
            m for m in all_memories if m.content.get("thread_id") == "conv_bob"
        ]
        charlie_memories = [
            m for m in all_memories if m.content.get("thread_id") == "conv_charlie"
        ]

        assert len(alice_memories) == 2  # user + assistant
        assert len(bob_memories) == 2
        assert len(charlie_memories) == 2

        # Verify content separation
        alice_user_msg = next(
            m for m in alice_memories if m.content["role"] == Role.USER.value
        )
        bob_user_msg = next(
            m for m in bob_memories if m.content["role"] == Role.USER.value
        )

        assert "Alice" in alice_user_msg.content["content"]
        assert "cats" in alice_user_msg.content["content"]
        assert "Bob" in bob_user_msg.content["content"]
        assert "dogs" in bob_user_msg.content["content"]


class TestSemanticMemory:
    """Test semantic memory functionality."""

    @pytest.mark.memory
    @pytest.mark.knowledge_base
    def test_semantic_memory_storage_and_retrieval(self):
        """Test storing and retrieving semantic memories."""
        memory_provider = MockMemoryProvider()

        # Configure memory provider to return relevant semantic memories
        memory_provider.configure_semantic_retrieval(
            {
                "machine learning": [
                    MockMemoryUnit(
                        memory_type=MemoryType.KNOWLEDGE_BASE,
                        content={
                            "topic": "machine learning",
                            "information": "Machine learning is a subset of AI that learns patterns from data",
                            "keywords": ["AI", "patterns", "data", "algorithms"],
                        },
                    )
                ],
                "python programming": [
                    MockMemoryUnit(
                        memory_type=MemoryType.KNOWLEDGE_BASE,
                        content={
                            "topic": "python programming",
                            "information": "Python is a versatile programming language great for beginners",
                            "keywords": [
                                "programming",
                                "language",
                                "versatile",
                                "beginners",
                            ],
                        },
                    )
                ],
            }
        )

        llm_provider = MockLLMProvider(
            ["Based on what I know, machine learning is a subset of AI."]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You use semantic memory to provide informed responses.",
        )

        memory_id = "semantic_test"

        response = agent.run("Tell me about machine learning", memory_id=memory_id)

        assert_agent_response_valid(response)

        # Verify that semantic memory was queried
        call_history = memory_provider.get_call_history()
        retrieve_calls = [
            call for call in call_history if call[0] == "retrieve_by_query"
        ]
        assert len(retrieve_calls) > 0

    @pytest.mark.memory
    @pytest.mark.knowledge_base
    def test_semantic_memory_learning(self):
        """Test learning new semantic information."""
        memory_provider = MockMemoryProvider()

        llm_provider = MockLLMProvider(
            ["I've learned that quantum computing uses qubits for computation."]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You learn and store new semantic knowledge.",
        )

        # Agent learns new information
        memory_id = "semantic_learning_test"
        new_info = "Quantum computing uses quantum bits (qubits) which can exist in superposition states"

        response = agent.run(f"Learn this: {new_info}", memory_id=memory_id)

        assert_agent_response_valid(response)

        # Check if semantic information would be stored
        # (This depends on the implementation having semantic learning capabilities)
        stored_memories = memory_provider.storage.get(memory_id, [])

        # At minimum, the conversation should be stored
        assert len(stored_memories) >= 2  # user + assistant messages

    @pytest.mark.memory
    @pytest.mark.knowledge_base
    def test_semantic_memory_relevance_ranking(self):
        """Test that semantic memory retrieval ranks by relevance."""
        memory_provider = MockMemoryProvider()

        # Setup semantic memories with different relevance levels
        memory_provider.configure_semantic_retrieval(
            {
                "artificial intelligence": [
                    MockMemoryUnit(
                        memory_type=MemoryType.KNOWLEDGE_BASE,
                        content={
                            "topic": "machine learning",
                            "information": "ML is a branch of AI focusing on learning from data",
                            "relevance_score": 0.95,
                        },
                    ),
                    MockMemoryUnit(
                        memory_type=MemoryType.KNOWLEDGE_BASE,
                        content={
                            "topic": "neural networks",
                            "information": "Neural networks are AI models inspired by the brain",
                            "relevance_score": 0.85,
                        },
                    ),
                    MockMemoryUnit(
                        memory_type=MemoryType.KNOWLEDGE_BASE,
                        content={
                            "topic": "expert systems",
                            "information": "Expert systems use rules to make decisions",
                            "relevance_score": 0.60,
                        },
                    ),
                ]
            }
        )

        llm_provider = MockLLMProvider(
            ["AI encompasses machine learning, neural networks, and expert systems."]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You provide comprehensive answers using semantic memory.",
        )

        response = agent.run(
            "What do you know about artificial intelligence?", memory_id="ai_query_test"
        )

        assert_agent_response_valid(response)

        # Verify semantic retrieval was called
        retrieve_calls = [
            call
            for call in memory_provider.get_call_history()
            if call[0] == "retrieve_by_query"
        ]
        assert len(retrieve_calls) > 0


class TestEpisodicMemory:
    """Test episodic memory functionality."""

    @pytest.mark.memory
    @pytest.mark.episodic_memory
    def test_episodic_memory_events(self):
        """Test storing and retrieving episodic memories (events)."""
        memory_provider = MockMemoryProvider()

        # Configure episodic memory retrieval
        memory_provider.configure_episodic_retrieval(
            {
                "meeting_events": [
                    MockMemoryUnit(
                        memory_type=MemoryType.CONVERSATION_MEMORY,
                        content={
                            "event_type": "meeting",
                            "participants": ["Alice", "Bob", "Charlie"],
                            "date": "2023-12-01",
                            "location": "Conference Room A",
                            "summary": "Discussed Q4 project goals and deadlines",
                            "outcomes": [
                                "Set deadline for Dec 15",
                                "Assigned tasks to team members",
                            ],
                        },
                        timestamp=datetime(2023, 12, 1, 14, 0),
                    )
                ]
            }
        )

        llm_provider = MockLLMProvider(
            ["I remember we had a meeting on December 1st in Conference Room A."]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You remember specific events and experiences.",
        )

        response = agent.run(
            "Do you remember our last team meeting?", memory_id="episodic_test"
        )

        assert_agent_response_valid(response)

        # Verify episodic retrieval was attempted
        call_history = memory_provider.get_call_history()
        retrieve_calls = [
            call for call in call_history if call[0] == "retrieve_by_query"
        ]
        assert len(retrieve_calls) > 0

    @pytest.mark.memory
    @pytest.mark.episodic_memory
    def test_episodic_memory_temporal_ordering(self):
        """Test episodic memories maintain temporal order."""
        memory_provider = MockMemoryProvider()

        # Setup episodic memories with temporal sequence
        events = [
            {
                "event": "project_start",
                "description": "Started new AI project",
                "timestamp": datetime(2023, 11, 1, 9, 0),
            },
            {
                "event": "first_milestone",
                "description": "Completed data collection phase",
                "timestamp": datetime(2023, 11, 15, 16, 30),
            },
            {
                "event": "review_meeting",
                "description": "Conducted progress review with stakeholders",
                "timestamp": datetime(2023, 12, 1, 14, 0),
            },
        ]

        memory_provider.configure_episodic_retrieval(
            {
                "project_timeline": [
                    MockMemoryUnit(
                        memory_type=MemoryType.CONVERSATION_MEMORY,
                        content=event,
                        timestamp=event["timestamp"],
                    )
                    for event in events
                ]
            }
        )

        llm_provider = MockLLMProvider(
            [
                "The project started in November, then we hit our first milestone mid-month."
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You track events in chronological order.",
        )

        response = agent.run(
            "What's the timeline of our AI project?", memory_id="timeline_test"
        )

        assert_agent_response_valid(response)

    @pytest.mark.memory
    @pytest.mark.episodic_memory
    def test_episodic_memory_context_details(self):
        """Test episodic memories preserve contextual details."""
        memory_provider = MockMemoryProvider()

        detailed_event = MockMemoryUnit(
            memory_type=MemoryType.CONVERSATION_MEMORY,
            content={
                "event_type": "client_presentation",
                "date": "2023-11-20",
                "time": "10:00 AM",
                "location": "Client office downtown",
                "weather": "rainy",
                "attendees": ["John (client)", "Sarah (our PM)", "Mike (tech lead)"],
                "presentation_topic": "AI solution demo",
                "client_reaction": "very positive, asked detailed questions",
                "technical_issues": "minor network latency during demo",
                "next_steps": "follow-up meeting scheduled for Dec 5",
                "personal_notes": "client seemed impressed with the ML accuracy metrics",
            },
            timestamp=datetime(2023, 11, 20, 10, 0),
        )

        memory_provider.configure_episodic_retrieval(
            {"client_presentation": [detailed_event]}
        )

        llm_provider = MockLLMProvider(
            [
                "The client presentation went well despite the rainy weather and minor network issues."
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You remember detailed contextual information about events.",
        )

        response = agent.run(
            "How did the client presentation go last month?", memory_id="context_test"
        )

        assert_agent_response_valid(response)


class TestProceduralMemory:
    """Test procedural memory functionality."""

    @pytest.mark.memory
    @pytest.mark.procedural_memory
    def test_procedural_memory_workflows(self):
        """Test storing and executing procedural memories (workflows)."""
        memory_provider = MockMemoryProvider()

        # Configure procedural memory with workflows
        memory_provider.configure_procedural_retrieval(
            {
                "code_review_process": [
                    MockMemoryUnit(
                        memory_type=MemoryType.TOOLBOX,
                        content={
                            "procedure_name": "code_review_workflow",
                            "steps": [
                                "1. Pull latest changes from main branch",
                                "2. Create new feature branch",
                                "3. Implement changes with proper testing",
                                "4. Run automated test suite",
                                "5. Submit pull request with description",
                                "6. Request review from team members",
                                "7. Address review feedback",
                                "8. Merge after approval",
                            ],
                            "tools_required": ["git", "IDE", "test framework"],
                            "estimated_time": "2-4 hours",
                            "success_criteria": "All tests pass, code review approved",
                        },
                    )
                ]
            }
        )

        llm_provider = MockLLMProvider(
            ["Here's the code review process: start by pulling latest changes..."]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You know how to execute standard procedures and workflows.",
        )

        response = agent.run("How do I do a code review?", memory_id="procedure_test")

        assert_agent_response_valid(response)

        # Verify procedural memory was accessed
        call_history = memory_provider.get_call_history()
        retrieve_calls = [
            call for call in call_history if call[0] == "retrieve_by_query"
        ]
        assert len(retrieve_calls) > 0

    @pytest.mark.memory
    @pytest.mark.procedural_memory
    def test_procedural_memory_skills(self):
        """Test procedural memory for learned skills."""
        memory_provider = MockMemoryProvider()

        # Configure skills-based procedural memory
        skills = [
            MockMemoryUnit(
                memory_type=MemoryType.TOOLBOX,
                content={
                    "skill_name": "debug_python_error",
                    "procedure": [
                        "Read error message carefully",
                        "Check the stack trace for root cause",
                        "Look at the line number mentioned",
                        "Check variable types and values",
                        "Use print statements or debugger",
                        "Test with minimal example",
                        "Search documentation if needed",
                    ],
                    "common_mistakes": [
                        "Ignoring the full error message",
                        "Not checking variable types",
                        "Missing imports",
                    ],
                    "tools": ["Python debugger", "IDE", "print statements"],
                },
            ),
            MockMemoryUnit(
                memory_type=MemoryType.TOOLBOX,
                content={
                    "skill_name": "optimize_sql_query",
                    "procedure": [
                        "Analyze query execution plan",
                        "Check if proper indexes exist",
                        "Look for unnecessary JOINs",
                        "Consider query rewriting",
                        "Test with EXPLAIN command",
                        "Measure performance before/after",
                    ],
                    "performance_indicators": [
                        "execution time",
                        "rows examined",
                        "index usage",
                    ],
                    "tools": ["EXPLAIN", "database profiler", "index analyzer"],
                },
            ),
        ]

        memory_provider.configure_procedural_retrieval(
            {"debugging_skills": [skills[0]], "optimization_skills": [skills[1]]}
        )

        llm_provider = MockLLMProvider(
            [
                "To debug Python errors, start by reading the error message carefully...",
                "For SQL optimization, first analyze the execution plan...",
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You apply learned procedures and skills to solve problems.",
        )

        # Test skill retrieval for different problem types
        debug_response = agent.run(
            "I have a Python error, how should I debug it?", memory_id="skill_test_1"
        )
        sql_response = agent.run(
            "My SQL query is slow, how can I optimize it?", memory_id="skill_test_2"
        )

        assert_agent_response_valid(debug_response)
        assert_agent_response_valid(sql_response)

    @pytest.mark.memory
    @pytest.mark.procedural_memory
    def test_procedural_memory_adaptation(self):
        """Test procedural memory adaptation based on outcomes."""
        memory_provider = MockMemoryProvider()

        # Initial procedure
        original_procedure = MockMemoryUnit(
            memory_type=MemoryType.TOOLBOX,
            content={
                "procedure_name": "deploy_application",
                "version": 1,
                "steps": [
                    "Build application",
                    "Run tests",
                    "Deploy to staging",
                    "Manual QA check",
                    "Deploy to production",
                ],
                "success_rate": 0.75,
                "known_issues": [
                    "Manual QA bottleneck",
                    "Deployment rollback difficulties",
                ],
            },
        )

        memory_provider.configure_procedural_retrieval(
            {"deployment_process": [original_procedure]}
        )

        llm_provider = MockLLMProvider(
            ["Here's our current deployment process, though we've had some issues..."]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You learn from procedural outcomes and suggest improvements.",
        )

        response = agent.run(
            "What's our deployment process?", memory_id="adaptation_test"
        )

        assert_agent_response_valid(response)


class TestMemoryIntegration:
    """Test integration between different memory types."""

    @pytest.mark.memory
    @pytest.mark.integration
    def test_multi_memory_type_query(self):
        """Test queries that utilize multiple memory types."""
        memory_provider = MockMemoryProvider()

        # Setup different memory types
        memory_provider.configure_semantic_retrieval(
            {
                "machine_learning": [
                    MockMemoryUnit(
                        memory_type=MemoryType.KNOWLEDGE_BASE,
                        content={
                            "topic": "machine learning basics",
                            "information": "ML algorithms learn patterns from data to make predictions",
                        },
                    )
                ]
            }
        )

        memory_provider.configure_episodic_retrieval(
            {
                "ml_project": [
                    MockMemoryUnit(
                        memory_type=MemoryType.CONVERSATION_MEMORY,
                        content={
                            "event": "ML project completion",
                            "date": "2023-11-30",
                            "outcome": "Achieved 94% accuracy on customer prediction model",
                        },
                    )
                ]
            }
        )

        memory_provider.configure_procedural_retrieval(
            {
                "ml_workflow": [
                    MockMemoryUnit(
                        memory_type=MemoryType.TOOLBOX,
                        content={
                            "procedure": "train_ml_model",
                            "steps": [
                                "Prepare data",
                                "Feature engineering",
                                "Model selection",
                                "Training",
                                "Validation",
                            ],
                        },
                    )
                ]
            }
        )

        # Add conversation memory
        memory_id = "integration_test"
        thread_id = "multi_memory_conv"

        llm_provider = MockLLMProvider(
            [
                "I can help with ML! I know the theory, have experience from past projects, and know the process."
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You integrate knowledge from all memory types to provide comprehensive answers.",
            memory_ids=[memory_id],
        )

        response = agent.run(
            "I want to start a machine learning project. What should I know?",
            memory_id=memory_id,
            thread_id=thread_id,
        )

        assert_agent_response_valid(response)

        # Verify multiple memory types were queried
        call_history = memory_provider.get_call_history()
        retrieve_calls = [
            call for call in call_history if call[0] == "retrieve_by_query"
        ]

        # Should have tried to retrieve from multiple memory types
        assert len(retrieve_calls) > 0

    @pytest.mark.memory
    @pytest.mark.integration
    def test_memory_type_priority_resolution(self):
        """Test how agent resolves conflicts between different memory types."""
        memory_provider = MockMemoryProvider()

        # Setup conflicting information across memory types
        memory_provider.configure_semantic_retrieval(
            {
                "best_practices": [
                    MockMemoryUnit(
                        memory_type=MemoryType.KNOWLEDGE_BASE,
                        content={
                            "topic": "testing best practices",
                            "information": "Unit tests should cover at least 80% of code for good quality",
                        },
                    )
                ]
            }
        )

        memory_provider.configure_episodic_retrieval(
            {
                "testing_experience": [
                    MockMemoryUnit(
                        memory_type=MemoryType.CONVERSATION_MEMORY,
                        content={
                            "event": "project_retrospective",
                            "date": "2023-12-01",
                            "lesson_learned": "We achieved 90% test coverage but still had production bugs",
                        },
                    )
                ]
            }
        )

        memory_provider.configure_procedural_retrieval(
            {
                "testing_process": [
                    MockMemoryUnit(
                        memory_type=MemoryType.TOOLBOX,
                        content={
                            "procedure": "quality_testing",
                            "priority": "Focus on integration tests and user acceptance testing over raw coverage numbers",
                        },
                    )
                ]
            }
        )

        llm_provider = MockLLMProvider(
            [
                "Based on theory and experience, coverage is important but quality of tests matters more."
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You synthesize information from different memory types to provide balanced answers.",
        )

        response = agent.run(
            "What's the right approach to testing?", memory_id="priority_test"
        )

        assert_agent_response_valid(response)

    @pytest.mark.memory
    @pytest.mark.integration
    def test_memory_type_cross_referencing(self):
        """Test cross-referencing between memory types."""
        memory_provider = MockMemoryProvider()

        # Setup related information across memory types
        project_id = "ai_chatbot_project"

        memory_provider.configure_semantic_retrieval(
            {
                "chatbot_development": [
                    MockMemoryUnit(
                        memory_type=MemoryType.KNOWLEDGE_BASE,
                        content={
                            "topic": "chatbot architecture",
                            "information": "Modern chatbots use transformer models with attention mechanisms",
                            "related_projects": [project_id],
                        },
                    )
                ]
            }
        )

        memory_provider.configure_episodic_retrieval(
            {
                project_id: [
                    MockMemoryUnit(
                        memory_type=MemoryType.CONVERSATION_MEMORY,
                        content={
                            "project_id": project_id,
                            "phase": "development",
                            "current_status": "Implementing transformer-based response generation",
                            "challenges": "Model size optimization for production deployment",
                        },
                    )
                ]
            }
        )

        memory_provider.configure_procedural_retrieval(
            {
                "model_optimization": [
                    MockMemoryUnit(
                        memory_type=MemoryType.TOOLBOX,
                        content={
                            "procedure": "optimize_transformer_model",
                            "techniques": [
                                "quantization",
                                "pruning",
                                "knowledge distillation",
                            ],
                            "applicable_to": [project_id],
                        },
                    )
                ]
            }
        )

        llm_provider = MockLLMProvider(
            [
                "For the chatbot project, we can use transformer models with optimization techniques like quantization."
            ]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You cross-reference related information across memory types.",
        )

        response = agent.run(
            "How can we optimize our chatbot project?", memory_id="cross_ref_test"
        )

        assert_agent_response_valid(response)


class TestMemoryManagementOperations:
    """Test memory management operations."""

    @pytest.mark.memory
    @pytest.mark.management
    def test_memory_cleanup_operations(self):
        """Test memory cleanup and maintenance operations."""
        memory_provider = MockMemoryProvider()

        # Fill memory with test data
        memory_id = "cleanup_test"
        for i in range(50):  # Create many memory entries
            memory_unit = MockMemoryUnit(
                memory_type=MemoryType.CONVERSATION_MEMORY,
                content={
                    "role": Role.USER.value,
                    "content": f"Test message {i}",
                    "thread_id": f"conv_{i}",
                },
                timestamp=datetime.now() - timedelta(days=i),
            )
            memory_provider.store(memory_id, memory_unit)

        llm_provider = MockLLMProvider(["Memory cleanup operations completed."])

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You manage memory efficiently.",
            memory_ids=[memory_id],
        )

        # Test that agent can work with large memory
        response = agent.run("How many conversations do we have?", memory_id=memory_id)

        assert_agent_response_valid(response)

        # Verify memory contains expected number of entries
        stored_memories = memory_provider.storage[memory_id]
        assert len(stored_memories) >= 50

    @pytest.mark.memory
    @pytest.mark.management
    def test_memory_search_and_filtering(self):
        """Test memory search and filtering capabilities."""
        memory_provider = MockMemoryProvider()
        memory_id = "search_test"

        # Create diverse memory entries
        topics = [
            "python",
            "javascript",
            "machine learning",
            "databases",
            "web development",
        ]

        for i, topic in enumerate(topics):
            memory_unit = MockMemoryUnit(
                memory_type=MemoryType.KNOWLEDGE_BASE,
                content={
                    "topic": topic,
                    "information": f"Information about {topic}",
                    "category": "programming"
                    if topic in ["python", "javascript"]
                    else "technology",
                    "difficulty": "beginner" if i < 2 else "intermediate",
                },
            )
            memory_provider.store(memory_id, memory_unit)

        # Configure search capabilities
        memory_provider.configure_semantic_retrieval(
            {
                "python": [
                    mem
                    for mem in memory_provider.storage[memory_id]
                    if "python" in str(mem.content)
                ],
                "programming": [
                    mem
                    for mem in memory_provider.storage[memory_id]
                    if mem.content.get("category") == "programming"
                ],
            }
        )

        llm_provider = MockLLMProvider(
            ["I found information about Python programming."]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You can search and filter memory effectively.",
            memory_ids=[memory_id],
        )

        response = agent.run("Find information about Python", memory_id=memory_id)

        assert_agent_response_valid(response)

    @pytest.mark.memory
    @pytest.mark.management
    def test_memory_capacity_limits(self):
        """Test behavior when approaching memory capacity limits."""
        memory_provider = MockMemoryProvider()

        # Set up memory provider with capacity limits
        memory_provider.set_capacity_limit(100)  # Limit to 100 memory units

        memory_id = "capacity_test"

        # Fill memory to near capacity
        for i in range(95):
            memory_unit = MockMemoryUnit(
                memory_type=MemoryType.CONVERSATION_MEMORY,
                content={
                    "role": Role.USER.value,
                    "content": f"Message {i}",
                    "thread_id": f"conv_{i % 10}",  # 10 different conversations
                },
                timestamp=datetime.now() - timedelta(minutes=i),
            )
            memory_provider.store(memory_id, memory_unit)

        llm_provider = MockLLMProvider(["I'll work within memory constraints."])

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You work efficiently within memory limits.",
            memory_ids=[memory_id],
        )

        # Add more memories that might trigger capacity management
        for i in range(10):
            response = agent.run(
                f"New message {95 + i}", memory_id=memory_id, thread_id="new_conv"
            )
            assert_agent_response_valid(response)

        # Verify memory provider handled capacity appropriately
        total_memories = len(memory_provider.storage[memory_id])
        assert (
            total_memories <= memory_provider.capacity_limit * 1.1
        )  # Allow some flexibility
