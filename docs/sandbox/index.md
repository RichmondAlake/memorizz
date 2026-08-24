# Sandbox code execution

MemoRizz exposes code execution through one provider-neutral interface. The
security boundary and lifecycle depend on the provider; the word *sandbox* does
not make every implementation suitable for hostile code.

## Model-facing tools

Attaching a provider registers three tools:

| Tool | Purpose |
|---|---|
| `execute_code(code, language)` | Execute code and return a normalized `ExecutionResult`. |
| `sandbox_write_file(path, content)` | Write a text file in the provider workspace. |
| `sandbox_read_file(path)` | Read a text file from the provider workspace. |

`ExecutionResult.to_dict()` always includes `stdout`, `stderr`, `error`,
`exit_code`, `results`, `success`, and provider `metadata`.

## Provider choices

| Provider | Runtime | Lifecycle | Security statement |
|---|---|---|---|
| [E2B](e2b.md) | Remote microVM service | One bounded session per provider instance | Remote isolation; egress is denied by default by MemoRizz. |
| [Daytona](daytona.md) | Remote development environment | Fresh environment per operation in the current adapter | Remote container boundary; file operations are not session-coherent. |
| [GraalPy](graalpy.md) | Local GraalPy/JVM | Private provider directory with bounded processes | Subprocess mode is trusted-code execution only; validated Java `UNTRUSTED` mode is the hostile-code boundary. |

## Configure an agent

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_sandbox(
        {
            "provider": "e2b",
            "session_timeout": 300,
            "max_execution_timeout": 60,
            "allow_internet_access": False,
        }
    )
    .build()
)
```

The equivalent constructor form, when `my_llm` is an already configured
`LLMProvider`, is:

```python
from memorizz import MemAgent

agent = MemAgent(model=my_llm, sandbox_provider={"provider": "e2b"})
```

Swap or detach at runtime:

```python
agent.with_sandbox_provider("graalpy")
agent.with_sandbox_provider(None)
```

Direct host execution does not require an LLM:

```python
import json

result = json.loads(agent.execute_code("print(6 * 7)"))
assert result["success"] is True
assert result["stdout"] == ["42"]
```

## Installation

```bash
pip install "memorizz[sandbox-e2b]"
pip install "memorizz[sandbox-daytona]"
```

GraalPy is a separate system runtime. See the provider guide for installation
and the Java wrapper build requirements.

## Production rules

- Keep provider credentials in environment variables or a secret manager.
- Set explicit wall-clock, CPU, memory, process, output, and egress policies
  where the provider supports them.
- Close providers (or the parent agent) so remote sessions are terminated.
- Treat model-authored code as hostile. Do not use GraalPy subprocess mode as
  an isolation boundary.
- Test the exact provider SDK versions and deployment template used in
  production. MemoRizz validates its supported E2B compatibility range at
  runtime.
