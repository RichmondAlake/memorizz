# Sandbox — Code Execution for MemAgent

The `sandbox` module gives MemAgent the ability to **generate code and execute it in an isolated environment**, then use the output to continue reasoning. This is useful for data analysis, computation, file generation, and any task where the agent needs to run code safely.

## Supported Providers

| Provider | Type | Cold Start | Session Limit | GPU | Dependency |
|----------|------|------------|---------------|-----|------------|
| **E2B** (default) | Cloud | ~150ms | 24 hours | No | `E2B_API_KEY` |
| **Daytona** | Cloud | ~90ms | Unlimited | Yes | `DAYTONA_API_KEY` |
| **GraalPy** | Local | N/A | Unlimited | N/A | `graalpy` on PATH or `GRAALPY_PATH` |

## Quick Start

### Using E2B (Default)

```bash
pip install "memorizz[sandbox-e2b]"
export E2B_API_KEY="your-api-key"
```

```python
from memorizz.memagent import MemAgent

agent = MemAgent(
    model=my_llm,
    memory_provider=my_memory,
    sandbox_provider="e2b",
    instruction="You are a data analyst. Use code execution to answer questions.",
)

response = agent.run("What is the 50th Fibonacci number?")
```

### Using Daytona

```bash
pip install "memorizz[sandbox-daytona]"
export DAYTONA_API_KEY="your-api-key"
```

```python
agent = MemAgent(
    model=my_llm,
    memory_provider=my_memory,
    sandbox_provider="daytona",
)
```

### Using GraalPy (Local — No Cloud Required)

```bash
# 1) Install GraalPy runtime (pick your OS/CPU archive):
#    https://github.com/oracle/graalpython/releases
#
# 2) Extract and expose binary (example layout):
#    $HOME/.local/graalpy-community-<version>-<platform>/bin/graalpy
#    ln -sf "$HOME/.local/graalpy-community-<version>-<platform>/bin/graalpy" "$HOME/.local/bin/graalpy"
#
# 3) Ensure executable is discoverable:
export PATH="$HOME/.local/bin:$PATH"
export GRAALPY_PATH="$HOME/.local/bin/graalpy"  # optional explicit path

# 4) Verify:
graalpy --version
```

```python
# Simple subprocess mode (default).
# MemAgent now validates GraalPy at provider creation time and raises ValueError
# immediately if the executable cannot be found.
agent = MemAgent(
    model=my_llm,
    memory_provider=my_memory,
    sandbox_provider="graalpy",
)

# Explicit binary path (recommended in production)
agent = MemAgent(
    model=my_llm,
    memory_provider=my_memory,
    sandbox_provider={
        "provider": "graalpy",
        "graalpy_path": "/absolute/path/to/graalpy",
    },
)

# Java wrapper mode (full sandboxing via GraalVM SandboxPolicy)
agent = MemAgent(
    model=my_llm,
    memory_provider=my_memory,
    sandbox_provider={
        "provider": "graalpy",
        "mode": "java_wrapper",
        "java_wrapper_jar": "/path/to/graalpy-sandbox.jar",
        "sandbox_policy": "UNTRUSTED",
    },
)
```

## How It Works

1. The agent receives a user query.
2. The LLM decides to call the `execute_code` tool.
3. MemAgent routes the code to the configured sandbox provider.
4. The sandbox executes the code in isolation and returns stdout/stderr/errors.
5. The LLM inspects the output and either responds or iterates.

Each `execute_code` call is **stateless** — a fresh sandbox is created and destroyed per execution. Variables do not persist between calls.

## Registered Tools

When a sandbox provider is configured, three tools are automatically registered:

| Tool | Description |
|------|-------------|
| `execute_code(code, language)` | Execute code and return stdout/stderr/errors |
| `sandbox_write_file(path, content)` | Write a text file inside the sandbox |
| `sandbox_read_file(path)` | Read a text file from the sandbox |

## Advanced Configuration

### Passing a Config Dict

```python
agent = MemAgent(
    model=my_llm,
    memory_provider=my_memory,
    sandbox_provider={
        "provider": "e2b",
        "api_key": "sk-...",
        "template": "code-interpreter",
    },
)
```

### Runtime Provider Swap

```python
# Start with E2B
agent = MemAgent(model=my_llm, memory_provider=my_memory, sandbox_provider="e2b")

# Switch to GraalPy at runtime
agent.with_sandbox_provider("graalpy")

# Disable sandbox
agent.with_sandbox_provider(None)
```

### Direct Execution (Without LLM)

```python
result = agent.execute_code("print(2 ** 100)")
# Returns JSON: {"stdout": ["1267650600228229401496703205376"], ...}
```

## Provider Details

### E2B

- **Isolation**: Firecracker microVMs
- **Languages**: Python (default), JavaScript, R, and more
- **Filesystem**: Full read/write inside sandbox
- **Internet**: Available from inside sandbox
- **Pricing**: ~$0.08/hour, $100 free credits
- **Docs**: https://e2b.dev/docs

### Daytona

- **Isolation**: Docker containers with configurable resources
- **Languages**: Python, JavaScript, TypeScript
- **Filesystem**: Full read/write with Git support
- **GPU**: Supported (configurable CPU/memory/GPU/disk)
- **Pricing**: ~$0.08/hour, $200 free credits
- **Docs**: https://daytona.io/docs

### GraalPy

- **Isolation**: OS process boundaries (subprocess mode) or JVM sandbox (java_wrapper mode)
- **Languages**: Python only
- **Filesystem**: Configurable (restricted in java_wrapper mode)
- **Requirements**: GraalPy on PATH, `GRAALPY_PATH`, or `graalpy_path`; JVM for java_wrapper mode
- **Pricing**: Free (runs locally)
- **Docs**: https://www.graalvm.org/python/

#### GraalPy Modes

| Mode | Security | Setup | Use Case |
|------|----------|-------|----------|
| `subprocess` | OS-level process isolation | Just install GraalPy | Development, testing |
| `java_wrapper` | JVM SandboxPolicy (UNTRUSTED) | JVM + wrapper JAR | Production, multi-tenant |

## Writing a Custom Provider

Implement the `SandboxProvider` ABC and register it:

```python
from memorizz.sandbox.base import SandboxProvider, register_provider
from memorizz.sandbox.models import ExecutionResult

class MySandboxProvider(SandboxProvider):
    provider_name = "my_sandbox"

    def execute_code(self, code, language="python", timeout=30, envs=None):
        # Your execution logic here
        return ExecutionResult(stdout=["hello"], exit_code=0)

    def write_file(self, path, content):
        return True

    def read_file(self, path):
        return "file contents"

# Register so it can be used by name
register_provider("my_sandbox", MySandboxProvider)

# Now use it
agent = MemAgent(model=my_llm, memory_provider=my_memory, sandbox_provider="my_sandbox")
```
