# MemAgent Test Suite

This comprehensive test suite covers all aspects of the refactored MemAgent architecture, ensuring reliability, performance, and backward compatibility.

## Test Structure

### Unit Tests (`tests/unit/`)
- **`test_core.py`** - Core MemAgent class functionality, initialization, and run method
- **`test_managers.py`** - Individual manager components (Memory, Tool, Cache, Persona, Workflow)
- **`test_single_agent.py`** - Single agent functionality, tools, personas, caching, error handling
- **`test_multi_agent.py`** - Multi-agent coordination, communication patterns, delegation
- **`test_memory_types.py`** - All memory types: conversation, semantic, episodic, procedural
- **`test_backward_compatibility.py`** - Ensures 100% API compatibility with original implementation

### Integration Tests (`tests/integration/`)
- **`test_end_to_end.py`** - Complete workflows, component integration, error handling across systems

### Performance Tests (`tests/performance/`)
- **`test_performance.py`** - Benchmarks, stress tests, concurrent operations, scaling behavior

### Test Infrastructure
- **`conftest.py`** - Fixtures, utilities, and shared test configuration
- **`mocks/mock_providers.py`** - Mock implementations for LLM, Memory, and other providers
- **`pytest.ini`** - Test configuration with custom markers

## Test Categories & Markers

### Unit Test Markers
- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.memory` - Memory-related functionality

### Functional Test Markers
- `@pytest.mark.single_agent` - Single agent functionality
- `@pytest.mark.multi_agent` - Multi-agent coordination
- `@pytest.mark.conversation_memory` - Conversation memory tests
- `@pytest.mark.knowledge_base` - Knowledge base (semantic long-term memory) tests
- `@pytest.mark.episodic_memory` - Episodic memory tests
- `@pytest.mark.procedural_memory` - Procedural memory tests

### Integration & Performance Markers
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.stress` - Stress testing
- `@pytest.mark.compatibility` - Backward compatibility tests

## Running Tests

### Synthetic browser acceptance

The CI and release browser gate covers trace navigation, usage charts, JSON
exports, Evalground and host-adapter Persona Evolution workflows on desktop and
mobile. Fixtures use temporary storage and synthetic data, not live databases,
provider credentials or paid LLM calls.

Install `memorizz[dev,ui]` from this checkout and Node 24, then run from the
repository root:

```bash
python -m pip install -e '.[dev,ui]'
npm install --prefix /tmp/memorizz-browser-tools --no-save --ignore-scripts playwright@1.60.0
/tmp/memorizz-browser-tools/node_modules/.bin/playwright install chromium
MEMORIZZ_PLAYWRIGHT_MODULE=/tmp/memorizz-browser-tools/node_modules/playwright \
  python tests/browser/run_suite.py
```

The runner prints the temporary screenshot/log directory and stops its fixture
servers on success or failure. Use `--output-dir /path/to/results` to retain a
chosen output location, or `--node /path/to/node` to select a different Node
installation. Linux CI uses Playwright's `install --with-deps chromium` to install
the required system libraries. These checks do not replace live-provider or
consuming-application acceptance tests.

### Run All Tests
```bash
pytest
```

### Run Specific Categories
```bash
# Unit tests only
pytest -m unit

# Single agent functionality
pytest -m single_agent

# Multi-agent functionality
pytest -m multi_agent

# Memory type tests
pytest -m "conversation_memory or knowledge_base or episodic_memory or procedural_memory"

# Integration tests
pytest -m integration

# Performance tests
pytest -m performance

# Backward compatibility
pytest -m compatibility
```

### Run Specific Test Files
```bash
pytest tests/unit/test_core.py
pytest tests/unit/test_memory_types.py
pytest tests/integration/test_end_to_end.py
pytest tests/performance/test_performance.py
```

### Verbose Output
```bash
pytest -v                    # Verbose test names
pytest -s                    # Show print statements
pytest -v -s                 # Both verbose and show output
```

## Test Coverage

The test suite covers:

### Core Functionality
- ✅ Agent initialization with all parameter combinations
- ✅ Run method with memory and conversation IDs
- ✅ Error handling and graceful degradation
- ✅ Component manager integration

### Memory Systems
- ✅ Conversation memory storage and retrieval
- ✅ Semantic memory for knowledge storage
- ✅ Episodic memory for event tracking
- ✅ Procedural memory for workflow storage
- ✅ Memory integration and cross-referencing

### Agent Capabilities
- ✅ Tool management and execution
- ✅ Persona integration and prompting
- ✅ Semantic caching for performance
- ✅ Workflow orchestration

### Multi-Agent Systems
- ✅ Agent coordination patterns
- ✅ Shared memory communication
- ✅ Delegation and specialization
- ✅ Concurrent operations

### Performance & Reliability
- ✅ Response time benchmarks
- ✅ Memory operation performance
- ✅ Concurrent request handling
- ✅ Stress testing under load
- ✅ Error isolation and recovery

### Backward Compatibility
- ✅ Original API preservation
- ✅ Method signature compatibility
- ✅ Behavior consistency
- ✅ Model class compatibility

## Mock Infrastructure

The test suite includes comprehensive mocks:

- **MockLLMProvider** - Simulates LLM responses with configurable behaviors
- **MockMemoryProvider** - In-memory storage with realistic retrieval patterns
- **MockPersona** - Test personas with configurable traits
- **MockMemoryUnit** - Memory units for testing different memory types

## Test Utilities

- **Assertion Helpers** - `assert_agent_response_valid()`, `assert_agent_state_valid()`, etc.
- **Fixtures** - Reusable test components and configurations
- **Performance Helpers** - Timing and benchmarking utilities

## Best Practices

1. **Isolation** - Each test is independent and can run in any order
2. **Mocking** - External dependencies are mocked for reliability
3. **Assertions** - Clear, specific assertions with helpful error messages
4. **Documentation** - Each test class and method includes clear docstrings
5. **Performance** - Tests run quickly while still being comprehensive

## Continuous Integration

The test suite is designed for CI/CD integration:
- Fast unit tests for quick feedback
- Comprehensive integration tests for release validation
- Performance benchmarks to catch regressions
- Backward compatibility checks to ensure API stability

## Architecture Validation

These tests validate that the refactored architecture:
- ✅ Maintains 100% backward compatibility
- ✅ Improves modularity and testability
- ✅ Handles error cases gracefully
- ✅ Scales to complex multi-agent scenarios
- ✅ Preserves performance characteristics
- ✅ Supports all original functionality

The test suite provides confidence that the 3,216-line monolithic implementation has been successfully refactored into a clean, modular architecture without breaking existing functionality.
