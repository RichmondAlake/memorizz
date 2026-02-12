# Sandbox Code Execution

MemoRizz gives MemAgent the ability to **generate code, execute it in an isolated sandbox, and use the output** to answer questions, perform data analysis, or solve computational problems.

## Why Does an AI Agent Need a Sandbox?

Large Language Models are powerful at reasoning about code, but they cannot execute it natively. When an agent needs to:

- **Verify a computation** — "What's the 50th Fibonacci number?" requires actually running the math.
- **Analyze data** — Processing CSV files, calculating statistics, or building visualizations.
- **Test algorithms** — Writing and running code to validate correctness before presenting an answer.
- **Iterate on errors** — If code fails, the agent can see the traceback, fix it, and retry.

A sandbox provides the secure, isolated environment where this execution happens — without risking your local machine, databases, or credentials.

## Architecture

When you add a sandbox provider to MemAgent, three tools are automatically registered that the LLM can call:

| Tool | Description |
|------|-------------|
| `execute_code(code, language)` | Execute code and return stdout, stderr, and errors |
| `sandbox_write_file(path, content)` | Write a text file inside the sandbox |
| `sandbox_read_file(path)` | Read a text file from the sandbox |

The execution flow works like this:

1. The user asks a question (e.g., "What's the median of these numbers?")
2. The LLM decides it needs to run code and calls the `execute_code` tool
3. MemAgent routes the code to the configured sandbox provider
4. The sandbox executes the code in isolation and returns the output
5. The LLM reads the output and formulates a response
6. If the code had an error, the LLM can fix it and try again (up to `max_steps`)

Each `execute_code` call is **stateless** — a fresh sandbox is created and destroyed. Variables do not persist between calls.

## Supported Providers

MemoRizz supports three sandbox providers, covering both cloud and local execution:

| Provider | Type | Cold Start | Session Limit | GPU | Cost |
|----------|------|------------|---------------|-----|------|
| [**E2B**](e2b.md) (default) | Cloud | ~150ms | 24 hours | No | ~$0.08/hr |
| [**Daytona**](daytona.md) | Cloud | ~90ms | Unlimited | Yes | ~$0.08/hr |
| [**GraalPy**](graalpy.md) | Local | N/A | Unlimited | N/A | Free |

### Choosing a Provider

- **Getting started or prototyping?** Use **E2B** — simplest setup, purpose-built for AI agents.
- **Need GPU or unlimited runtime?** Use **Daytona** — full dev environments in the cloud.
- **Offline or cost-sensitive?** Use **GraalPy** — runs locally, no cloud dependency.

## Quick Start

### 1. Install

=== "E2B (default)"

    ```bash
    pip install "memorizz[sandbox-e2b]"
    export E2B_API_KEY="your-api-key"
    ```

=== "Daytona"

    ```bash
    pip install "memorizz[sandbox-daytona]"
    export DAYTONA_API_KEY="your-api-key"
    ```

=== "GraalPy"

    ```bash
    # Install GraalPy runtime from:
    # https://github.com/oracle/graalpython/releases
    #
    # Optional explicit path for MemAgent:
    export GRAALPY_PATH="/absolute/path/to/graalpy"
    graalpy --version
    ```

### 2. Create a MemAgent with Sandbox

```python
from memorizz.memagent.core import MemAgent

agent = MemAgent(
    model=my_llm,
    memory_provider=my_memory,
    sandbox_provider="e2b",  # or "daytona" or "graalpy"
    instruction="You are a data analyst. Use code execution to answer questions.",
)
```

For GraalPy, provider creation validates executable availability up front. If
`graalpy` is not resolvable (PATH, `GRAALPY_PATH`, or `graalpy_path`), MemAgent
raises a `ValueError` with an actionable setup message.

### 3. Ask a Question

```python
response = agent.run("What are the first 20 prime numbers?")
print(response)
# The agent writes code, executes it in the sandbox, and returns the answer.
```

## Runtime Provider Management

You can swap providers at runtime without creating a new agent:

```python
# Start with E2B
agent = MemAgent(model=llm, sandbox_provider="e2b")

# Switch to GraalPy for offline work
agent.with_sandbox_provider("graalpy")

# Switch to Daytona for GPU tasks
agent.with_sandbox_provider("daytona")

# Disable sandbox entirely
agent.with_sandbox_provider(None)
```

## Direct Code Execution

You can also execute code programmatically without going through the LLM:

```python
import json

result_json = agent.execute_code("print(2 ** 100)")
result = json.loads(result_json)
print(result["stdout"])  # ["1267650600228229401496703205376"]
```

## Custom Providers

You can implement your own sandbox provider by subclassing `SandboxProvider`:

```python
from memorizz.sandbox.base import SandboxProvider, register_provider
from memorizz.sandbox.models import ExecutionResult

class MySandboxProvider(SandboxProvider):
    provider_name = "my_sandbox"

    def execute_code(self, code, language="python", timeout=30, envs=None):
        # Your execution logic
        return ExecutionResult(stdout=["output"], exit_code=0)

    def write_file(self, path, content):
        return True

    def read_file(self, path):
        return "file contents"

register_provider("my_sandbox", MySandboxProvider)

# Now use it
agent = MemAgent(model=llm, sandbox_provider="my_sandbox")
```

## Installation Reference

| Extra | Command | What It Installs |
|-------|---------|-----------------|
| `sandbox` | `pip install "memorizz[sandbox]"` | E2B (default) |
| `sandbox-e2b` | `pip install "memorizz[sandbox-e2b]"` | E2B Code Interpreter SDK |
| `sandbox-daytona` | `pip install "memorizz[sandbox-daytona]"` | Daytona Python SDK |
| (none) | Install GraalPy runtime + optionally set `GRAALPY_PATH` | GraalPy runtime |
