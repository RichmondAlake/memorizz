"""Performance tests for MemAgent components."""
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from memorizz.enums import MemoryType, Role
from memorizz.memagent import MemAgent
from tests.conftest import assert_agent_response_valid
from tests.mocks.mock_providers import (
    MockLLMProvider,
    MockMemoryProvider,
    MockMemoryUnit,
)


class TestPerformanceBenchmarks:
    """Benchmark tests for agent performance."""

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_agent_initialization_performance(self):
        """Test agent initialization performance."""
        initialization_times = []

        # Measure multiple agent initializations
        for i in range(10):
            start_time = time.time()

            agent = MemAgent(
                model=MockLLMProvider(["Test response"]),
                memory_provider=MockMemoryProvider(),
                instruction=f"Test agent {i}",
                semantic_cache=True,
                agent_id=f"perf_agent_{i}",
            )

            end_time = time.time()
            initialization_times.append(end_time - start_time)

        # Performance assertions
        avg_init_time = statistics.mean(initialization_times)
        max_init_time = max(initialization_times)

        assert (
            avg_init_time < 0.1
        ), f"Average initialization time too slow: {avg_init_time:.4f}s"
        assert (
            max_init_time < 0.2
        ), f"Max initialization time too slow: {max_init_time:.4f}s"

        print(
            f"Agent initialization - Avg: {avg_init_time:.4f}s, Max: {max_init_time:.4f}s"
        )

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_single_query_response_time(self):
        """Test single query response time performance."""
        llm_provider = MockLLMProvider(["Quick response" for _ in range(50)])
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You respond quickly and efficiently.",
            agent_id="response_time_agent",
        )

        response_times = []

        # Measure response times for multiple queries
        for i in range(50):
            start_time = time.time()

            response = agent.run(f"Test query {i}", memory_id=f"test_{i}")

            end_time = time.time()
            response_times.append(end_time - start_time)

            assert_agent_response_valid(response)

        # Performance analysis
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[
            18
        ]  # 95th percentile
        max_response_time = max(response_times)

        assert (
            avg_response_time < 0.05
        ), f"Average response time too slow: {avg_response_time:.4f}s"
        assert (
            p95_response_time < 0.1
        ), f"95th percentile response time too slow: {p95_response_time:.4f}s"

        print(
            f"Query response - Avg: {avg_response_time:.4f}s, P95: {p95_response_time:.4f}s, Max: {max_response_time:.4f}s"
        )

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_memory_operation_performance(self):
        """Test memory operation performance."""
        memory_provider = MockMemoryProvider()

        # Pre-populate memory with test data
        memory_id = "perf_memory_test"
        for i in range(1000):
            memory_unit = MockMemoryUnit(
                memory_type=MemoryType.CONVERSATION_MEMORY,
                content={
                    "role": Role.USER.value,
                    "content": f"Test message {i}",
                    "thread_id": f"conv_{i % 100}",
                },
                timestamp=datetime.now() - timedelta(seconds=i),
            )
            memory_provider.store(memory_id, memory_unit)

        llm_provider = MockLLMProvider(["Response based on memory" for _ in range(20)])

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You work efficiently with large memory sets.",
            memory_ids=[memory_id],
            agent_id="memory_perf_agent",
        )

        memory_operation_times = []

        # Test memory-heavy operations
        for i in range(20):
            start_time = time.time()

            response = agent.run(
                f"Recall information about test {i * 50}", memory_id=memory_id
            )

            end_time = time.time()
            memory_operation_times.append(end_time - start_time)

            assert_agent_response_valid(response)

        avg_memory_time = statistics.mean(memory_operation_times)
        max_memory_time = max(memory_operation_times)

        assert (
            avg_memory_time < 0.2
        ), f"Average memory operation time too slow: {avg_memory_time:.4f}s"
        assert (
            max_memory_time < 0.5
        ), f"Max memory operation time too slow: {max_memory_time:.4f}s"

        print(
            f"Memory operations - Avg: {avg_memory_time:.4f}s, Max: {max_memory_time:.4f}s"
        )

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_tool_execution_performance(self):
        """Test tool execution performance."""

        def fast_tool(x: int) -> int:
            """Fast computational tool."""
            return x * 2

        def medium_tool(text: str) -> dict:
            """Medium complexity tool."""
            words = text.split()
            return {
                "word_count": len(words),
                "char_count": len(text),
                "unique_words": len(set(words)),
            }

        def slow_tool(n: int) -> list:
            """Simulate slower computation."""
            time.sleep(0.001)  # Simulate some processing time
            return [i**2 for i in range(min(n, 100))]

        llm_provider = MockLLMProvider(
            ["Tool executed successfully" for _ in range(30)]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=MockMemoryProvider(),
            tools=[fast_tool, medium_tool, slow_tool],
            instruction="You execute tools efficiently.",
            agent_id="tool_perf_agent",
        )

        # Test different tool performance
        tool_times = {"fast_tool": [], "medium_tool": [], "slow_tool": []}

        for i in range(10):
            # Fast tool
            start_time = time.time()
            result, _ = agent.tool_manager.execute_tool("fast_tool", {"x": i})
            tool_times["fast_tool"].append(time.time() - start_time)

            # Medium tool
            start_time = time.time()
            result, _ = agent.tool_manager.execute_tool(
                "medium_tool", {"text": f"test message {i} with some words"}
            )
            tool_times["medium_tool"].append(time.time() - start_time)

            # Slow tool
            start_time = time.time()
            result, _ = agent.tool_manager.execute_tool("slow_tool", {"n": 10})
            tool_times["slow_tool"].append(time.time() - start_time)

        # Performance assertions
        avg_fast = statistics.mean(tool_times["fast_tool"])
        avg_medium = statistics.mean(tool_times["medium_tool"])
        avg_slow = statistics.mean(tool_times["slow_tool"])

        assert avg_fast < 0.001, f"Fast tool too slow: {avg_fast:.6f}s"
        assert avg_medium < 0.01, f"Medium tool too slow: {avg_medium:.6f}s"
        assert avg_slow < 0.1, f"Slow tool too slow: {avg_slow:.6f}s"

        print(
            f"Tool execution - Fast: {avg_fast:.6f}s, Medium: {avg_medium:.6f}s, Slow: {avg_slow:.6f}s"
        )


class TestStressTests:
    """Stress tests for agent under load."""

    @pytest.mark.performance
    @pytest.mark.stress
    def test_concurrent_agent_requests(self):
        """Test agent handling concurrent requests."""
        llm_provider = MockLLMProvider([f"Concurrent response {i}" for i in range(100)])
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You handle concurrent requests efficiently.",
            agent_id="concurrent_test_agent",
        )

        def make_request(request_id):
            """Make a single request to the agent."""
            start_time = time.time()
            try:
                response = agent.run(
                    f"Concurrent request {request_id}",
                    memory_id=f"concurrent_{request_id}",
                )
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "success": True,
                    "response": response,
                    "duration": end_time - start_time,
                    "error": None,
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "success": False,
                    "response": None,
                    "duration": end_time - start_time,
                    "error": str(e),
                }

        # Execute concurrent requests
        concurrent_requests = 20
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_request, i) for i in range(concurrent_requests)
            ]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.time()
        total_duration = end_time - start_time

        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]

        success_rate = len(successful_requests) / len(results)
        avg_request_duration = statistics.mean(
            [r["duration"] for r in successful_requests]
        )

        # Performance assertions
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert (
            total_duration < 5.0
        ), f"Total execution time too long: {total_duration:.2f}s"
        assert (
            avg_request_duration < 0.1
        ), f"Average request duration too long: {avg_request_duration:.4f}s"

        print(
            f"Concurrent requests - Success: {success_rate:.2%}, Avg duration: {avg_request_duration:.4f}s"
        )

        if failed_requests:
            print(f"Failed requests: {len(failed_requests)}")
            for req in failed_requests[:3]:  # Show first 3 failures
                print(f"  Request {req['request_id']}: {req['error']}")

    @pytest.mark.performance
    @pytest.mark.stress
    def test_memory_stress_load(self):
        """Test agent with heavy memory load."""
        memory_provider = MockMemoryProvider()

        # Create agent with large memory dataset
        memory_id = "stress_memory"
        conversations = 50
        messages_per_conversation = 100

        print(
            f"Creating {conversations * messages_per_conversation * 2} memory entries..."
        )

        # Pre-populate with large conversation history
        for conv_i in range(conversations):
            thread_id = f"stress_conv_{conv_i}"
            for msg_i in range(messages_per_conversation):
                # User message
                user_memory = MockMemoryUnit(
                    memory_type=MemoryType.CONVERSATION_MEMORY,
                    content={
                        "role": Role.USER.value,
                        "content": f"User message {msg_i} in conversation {conv_i}",
                        "thread_id": thread_id,
                    },
                    timestamp=datetime.now()
                    - timedelta(minutes=(conv_i * 100) + (msg_i * 2)),
                )
                memory_provider.store(memory_id, user_memory)

                # Assistant response
                assistant_memory = MockMemoryUnit(
                    memory_type=MemoryType.CONVERSATION_MEMORY,
                    content={
                        "role": Role.ASSISTANT.value,
                        "content": f"Assistant response {msg_i} in conversation {conv_i}",
                        "thread_id": thread_id,
                    },
                    timestamp=datetime.now()
                    - timedelta(minutes=(conv_i * 100) + (msg_i * 2) - 1),
                )
                memory_provider.store(memory_id, assistant_memory)

        llm_provider = MockLLMProvider(
            [f"Response with heavy memory {i}" for i in range(20)]
        )

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You work efficiently with very large memory datasets.",
            memory_ids=[memory_id],
            agent_id="memory_stress_agent",
        )

        # Test performance with heavy memory load
        stress_test_times = []

        for i in range(20):
            start_time = time.time()

            response = agent.run(
                f"Query against heavy memory dataset {i}",
                memory_id=memory_id,
                thread_id=f"new_stress_conv_{i}",
            )

            end_time = time.time()
            stress_test_times.append(end_time - start_time)

            assert_agent_response_valid(response)

        avg_stress_time = statistics.mean(stress_test_times)
        max_stress_time = max(stress_test_times)
        p95_stress_time = statistics.quantiles(stress_test_times, n=20)[18]

        # Memory stress should not significantly degrade performance
        assert (
            avg_stress_time < 1.0
        ), f"Average stress test time too slow: {avg_stress_time:.4f}s"
        assert (
            max_stress_time < 2.0
        ), f"Max stress test time too slow: {max_stress_time:.4f}s"

        total_memory_entries = len(memory_provider.storage[memory_id])
        print(
            f"Memory stress test - {total_memory_entries} entries, Avg: {avg_stress_time:.4f}s, P95: {p95_stress_time:.4f}s"
        )

    @pytest.mark.performance
    @pytest.mark.stress
    def test_multi_agent_concurrent_stress(self):
        """Test multiple agents operating concurrently under stress."""
        shared_memory = MockMemoryProvider()

        def create_stress_agent(agent_id: str, response_prefix: str):
            """Create an agent for stress testing."""
            return MemAgent(
                model=MockLLMProvider(
                    [f"{response_prefix} response {i}" for i in range(50)]
                ),
                memory_provider=shared_memory,
                instruction=f"You are stress test agent {agent_id}.",
                agent_id=agent_id,
            )

        # Create multiple agents
        agents = [
            create_stress_agent(f"stress_agent_{i}", f"Agent-{i}") for i in range(5)
        ]

        def agent_stress_worker(agent, queries_per_agent):
            """Worker function for each agent."""
            results = []
            for i in range(queries_per_agent):
                start_time = time.time()
                try:
                    response = agent.run(
                        f"Stress query {i} from {agent.agent_id}",
                        memory_id=f"stress_{agent.agent_id}",
                        thread_id=f"stress_conv_{i}",
                    )
                    end_time = time.time()
                    results.append(
                        {
                            "agent_id": agent.agent_id,
                            "query_id": i,
                            "success": True,
                            "duration": end_time - start_time,
                            "error": None,
                        }
                    )
                except Exception as e:
                    end_time = time.time()
                    results.append(
                        {
                            "agent_id": agent.agent_id,
                            "query_id": i,
                            "success": False,
                            "duration": end_time - start_time,
                            "error": str(e),
                        }
                    )
            return results

        queries_per_agent = 10
        start_time = time.time()

        # Run all agents concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(agent_stress_worker, agent, queries_per_agent)
                for agent in agents
            ]
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())

        end_time = time.time()
        total_stress_duration = end_time - start_time

        # Analyze multi-agent stress results
        successful_queries = [r for r in all_results if r["success"]]
        failed_queries = [r for r in all_results if not r["success"]]

        success_rate = len(successful_queries) / len(all_results)
        avg_query_duration = statistics.mean(
            [r["duration"] for r in successful_queries]
        )
        total_queries = len(all_results)

        # Multi-agent stress assertions
        assert (
            success_rate >= 0.9
        ), f"Multi-agent success rate too low: {success_rate:.2%}"
        assert (
            avg_query_duration < 0.2
        ), f"Average query duration too slow: {avg_query_duration:.4f}s"
        assert (
            total_stress_duration < 10.0
        ), f"Total stress test duration too long: {total_stress_duration:.2f}s"

        queries_per_second = total_queries / total_stress_duration

        print(
            f"Multi-agent stress - {total_queries} queries, {queries_per_second:.1f} QPS, Success: {success_rate:.2%}"
        )

        if failed_queries:
            print(f"Failed queries: {len(failed_queries)}")
            for agent_id in set(r["agent_id"] for r in failed_queries):
                agent_failures = [
                    r for r in failed_queries if r["agent_id"] == agent_id
                ]
                print(f"  Agent {agent_id}: {len(agent_failures)} failures")


class TestMemoryPerformance:
    """Performance tests specifically for memory operations."""

    @pytest.mark.performance
    @pytest.mark.memory_perf
    def test_conversation_history_loading_performance(self):
        """Test performance of loading conversation history."""
        memory_provider = MockMemoryProvider()
        memory_id = "history_perf_test"

        # Create large conversation history
        conversation_sizes = [10, 50, 100, 500, 1000]

        for size in conversation_sizes:
            # Clear and repopulate memory
            memory_provider.storage[memory_id] = []

            for i in range(size):
                memory_unit = MockMemoryUnit(
                    memory_type=MemoryType.CONVERSATION_MEMORY,
                    content={
                        "role": Role.USER.value if i % 2 == 0 else Role.ASSISTANT.value,
                        "content": f"Message {i} of {size}",
                        "thread_id": "perf_conversation",
                    },
                    timestamp=datetime.now() - timedelta(minutes=size - i),
                )
                memory_provider.store(memory_id, memory_unit)

            agent = MemAgent(
                model=MockLLMProvider(["Response with history"]),
                memory_provider=memory_provider,
                instruction="You work with conversation history.",
                memory_ids=[memory_id],
                agent_id=f"history_perf_{size}",
            )

            # Measure history loading performance
            start_time = time.time()
            history = agent.load_conversation_history(memory_id)
            end_time = time.time()

            loading_time = end_time - start_time

            # Performance should scale reasonably
            max_allowed_time = size * 0.001  # Allow 1ms per history item
            assert (
                loading_time < max_allowed_time
            ), f"History loading too slow for size {size}: {loading_time:.4f}s"

            print(f"History loading - Size: {size}, Time: {loading_time:.4f}s")

    @pytest.mark.performance
    @pytest.mark.memory_perf
    def test_memory_retrieval_performance(self):
        """Test performance of memory retrieval operations."""
        memory_provider = MockMemoryProvider()
        memory_id = "retrieval_perf_test"

        # Create diverse memory content for retrieval
        topics = [
            "python",
            "javascript",
            "machine learning",
            "databases",
            "web development",
            "AI",
            "algorithms",
            "data structures",
        ]

        for i in range(1000):
            topic = topics[i % len(topics)]
            memory_unit = MockMemoryUnit(
                memory_type=MemoryType.KNOWLEDGE_BASE,
                content={
                    "topic": topic,
                    "information": f"Information about {topic} number {i}",
                    "keywords": [topic, f"keyword_{i}", f"tag_{i % 10}"],
                    "relevance_score": (i % 100) / 100.0,
                },
            )
            memory_provider.store(memory_id, memory_unit)

        # Configure retrieval for testing
        memory_provider.configure_semantic_retrieval(
            {
                query: [
                    mem
                    for mem in memory_provider.storage[memory_id]
                    if query in str(mem.content).lower()
                ][
                    :10
                ]  # Limit to top 10
                for query in topics
            }
        )

        agent = MemAgent(
            model=MockLLMProvider(["Retrieved relevant information"]),
            memory_provider=memory_provider,
            instruction="You retrieve information efficiently.",
            memory_ids=[memory_id],
            agent_id="retrieval_perf_agent",
        )

        retrieval_times = []

        # Test retrieval performance for different queries
        for topic in topics:
            start_time = time.time()

            response = agent.run(f"Tell me about {topic}", memory_id=memory_id)

            end_time = time.time()
            retrieval_times.append(end_time - start_time)

            assert_agent_response_valid(response)

        avg_retrieval_time = statistics.mean(retrieval_times)
        max_retrieval_time = max(retrieval_times)

        assert (
            avg_retrieval_time < 0.1
        ), f"Average retrieval time too slow: {avg_retrieval_time:.4f}s"
        assert (
            max_retrieval_time < 0.2
        ), f"Max retrieval time too slow: {max_retrieval_time:.4f}s"

        print(
            f"Memory retrieval - Avg: {avg_retrieval_time:.4f}s, Max: {max_retrieval_time:.4f}s"
        )


class TestCachingPerformance:
    """Performance tests for semantic caching."""

    @pytest.mark.performance
    @pytest.mark.caching_perf
    def test_cache_hit_performance(self):
        """Test performance of cache hits vs misses."""
        llm_provider = MockLLMProvider([f"LLM response {i}" for i in range(100)])
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=llm_provider,
            memory_provider=memory_provider,
            instruction="You use caching effectively.",
            semantic_cache=True,
            semantic_cache_config={"similarity_threshold": 0.9},
            agent_id="cache_perf_agent",
        )

        thread_id = "cache_perf_session"
        memory_id = "cache_perf_memory"

        # First, populate cache with some queries
        cache_population_queries = [
            "What is Python?",
            "How does machine learning work?",
            "What are data structures?",
            "Explain algorithms",
            "What is web development?",
        ]

        for query in cache_population_queries:
            agent.run(query, memory_id=memory_id, thread_id=thread_id)

        initial_llm_calls = llm_provider.call_count

        # Test cache hit performance (similar queries)
        cache_hit_times = []
        similar_queries = [
            "What is the Python programming language?",  # Similar to "What is Python?"
            "How do machine learning algorithms work?",  # Similar to "How does machine learning work?"
            "What are data structures in programming?",  # Similar to "What are data structures?"
        ]

        for query in similar_queries:
            start_time = time.time()
            response = agent.run(query, memory_id=memory_id, thread_id=thread_id)
            end_time = time.time()

            cache_hit_times.append(end_time - start_time)
            assert_agent_response_valid(response)

        # Test cache miss performance (new queries)
        cache_miss_times = []
        new_queries = [
            "What is quantum computing?",
            "How do neural networks function?",
            "What is blockchain technology?",
        ]

        for query in new_queries:
            start_time = time.time()
            response = agent.run(query, memory_id=memory_id, thread_id=thread_id)
            end_time = time.time()

            cache_miss_times.append(end_time - start_time)
            assert_agent_response_valid(response)

        # Performance analysis
        avg_hit_time = statistics.mean(cache_hit_times) if cache_hit_times else 0
        avg_miss_time = statistics.mean(cache_miss_times)

        # Cache hits should be faster than misses (if caching is working)
        # But we allow for implementation variations
        print(
            f"Cache performance - Hit avg: {avg_hit_time:.4f}s, Miss avg: {avg_miss_time:.4f}s"
        )

        # Basic performance requirements
        assert avg_hit_time < 0.1, f"Cache hit time too slow: {avg_hit_time:.4f}s"
        assert avg_miss_time < 0.2, f"Cache miss time too slow: {avg_miss_time:.4f}s"

    @pytest.mark.performance
    @pytest.mark.caching_perf
    def test_cache_scaling_performance(self):
        """Test cache performance as cache size grows."""
        memory_provider = MockMemoryProvider()

        agent = MemAgent(
            model=MockLLMProvider([f"Response {i}" for i in range(200)]),
            memory_provider=memory_provider,
            instruction="You work with growing cache sizes.",
            semantic_cache=True,
            agent_id="cache_scaling_agent",
        )

        thread_id = "scaling_session"
        memory_id = "scaling_memory"

        cache_sizes = [10, 25, 50, 100]
        scaling_results = {}

        for cache_size in cache_sizes:
            # Populate cache to target size
            for i in range(cache_size):
                agent.run(
                    f"Unique query number {i} for cache size {cache_size}",
                    memory_id=memory_id,
                    thread_id=f"{thread_id}_{cache_size}",
                )

            # Measure performance with current cache size
            query_times = []
            for i in range(5):
                start_time = time.time()
                response = agent.run(
                    f"Test query {i} with cache size {cache_size}",
                    memory_id=memory_id,
                    thread_id=f"{thread_id}_{cache_size}",
                )
                end_time = time.time()
                query_times.append(end_time - start_time)
                assert_agent_response_valid(response)

            avg_query_time = statistics.mean(query_times)
            scaling_results[cache_size] = avg_query_time

            print(f"Cache size {cache_size} - Avg query time: {avg_query_time:.4f}s")

        # Performance should not degrade significantly with cache size
        max_allowed_degradation = 2.0  # Allow 2x slowdown at most
        baseline_time = scaling_results[cache_sizes[0]]

        for cache_size in cache_sizes[1:]:
            # Avoid ratio checks when timings are too small to be stable.
            if baseline_time < 0.005:
                assert (
                    scaling_results[cache_size] < 0.05
                ), f"Cache query time too slow at size {cache_size}: {scaling_results[cache_size]:.4f}s"
                continue

            degradation_ratio = scaling_results[cache_size] / baseline_time
            assert (
                degradation_ratio < max_allowed_degradation
            ), f"Cache performance degraded too much at size {cache_size}: {degradation_ratio:.2f}x"
