# E2B Provider

E2B is the **default sandbox provider** for MemoRizz. It is purpose-built for AI agent code execution, using **Firecracker microVMs** for secure isolation with fast cold starts.

## Overview

| Feature | Details |
|---------|---------|
| **Type** | Cloud-hosted |
| **Isolation** | Firecracker microVMs |
| **Cold Start** | ~150ms |
| **Session Limit** | 24 hours |
| **Languages** | Python (default), JavaScript, R, and more |
| **GPU** | Not supported |
| **Filesystem** | Full read/write inside sandbox |
| **Internet** | Available from inside sandbox |
| **Pricing** | ~$0.08/hour, $100 free credits |
| **Website** | [e2b.dev](https://e2b.dev) |

## Setup

### 1. Install the SDK

```bash
pip install "memorizz[sandbox-e2b]"
```

### 2. Get an API Key

1. Sign up at [e2b.dev](https://e2b.dev)
2. Navigate to your dashboard
3. Copy your API key

### 3. Set the Environment Variable

```bash
export E2B_API_KEY="your-api-key"
```

## Usage

### Basic

```python
from memorizz.memagent.core import MemAgent

agent = MemAgent(
    model=my_llm,
    sandbox_provider="e2b",
    instruction="You are a data analyst. Execute code to answer questions.",
)

response = agent.run("Calculate the standard deviation of [4, 8, 15, 16, 23, 42]")
```

### With Explicit Configuration

```python
agent = MemAgent(
    model=my_llm,
    sandbox_provider={
        "provider": "e2b",
        "api_key": "your-api-key",
        "template": "code-interpreter",
    },
    instruction="You are a helpful coding assistant.",
)
```

### Direct Provider Usage

You can also use the E2B provider directly outside of MemAgent:

```python
from memorizz.sandbox.providers.e2b_provider import E2BSandboxProvider

provider = E2BSandboxProvider(api_key="your-api-key")
result = provider.execute_code("print('hello from E2B!')")
print(result.output)     # "hello from E2B!"
print(result.success)    # True
```

## How It Works

Each `execute_code` call follows this lifecycle:

1. **Create** — A new E2B sandbox is created using the Firecracker microVM backend (~150ms)
2. **Execute** — The code is sent to the sandbox's Jupyter kernel for execution
3. **Capture** — stdout, stderr, execution results, and any errors are collected
4. **Destroy** — The sandbox is torn down immediately after execution

This ensures every call is stateless and isolated.

## Execution Result

The `execute_code` tool returns a JSON string with:

```json
{
    "stdout": ["line1", "line2"],
    "stderr": [],
    "error": null,
    "exit_code": 0,
    "results": ["output representations"],
    "success": true
}
```

If the code produces an error:

```json
{
    "stdout": [],
    "stderr": ["Traceback (most recent call last):"],
    "error": "NameError: name 'undefined_var' is not defined\n...",
    "exit_code": 1,
    "results": [],
    "success": false
}
```

## File Operations

E2B supports reading and writing files inside the sandbox:

```python
# Through the agent (as tool calls)
# The LLM can call sandbox_write_file and sandbox_read_file automatically

# Through direct execution
result = agent.execute_code("""
with open('/home/user/data.txt', 'w') as f:
    f.write('hello from sandbox!')

with open('/home/user/data.txt', 'r') as f:
    print(f.read())
""")
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `E2B_API_KEY` env var | E2B API key |
| `template` | `str` | `"code-interpreter"` | Sandbox template to use |

## Limitations

- **24-hour session limit** — Sandboxes are automatically destroyed after 24 hours
- **No GPU support** — Use Daytona if you need GPU access
- **Cloud-only** — Requires internet connectivity and an API key
- **Stateless** — Variables do not persist between `execute_code` calls

## Troubleshooting

### "e2b-code-interpreter is not installed"

```bash
pip install e2b-code-interpreter
```

### "E2B API key not provided"

Set the environment variable:
```bash
export E2B_API_KEY="your-api-key"
```

Or pass it explicitly:
```python
agent = MemAgent(
    model=llm,
    sandbox_provider={"provider": "e2b", "api_key": "your-key"},
)
```
