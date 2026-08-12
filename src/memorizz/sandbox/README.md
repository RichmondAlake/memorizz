# MemoRizz sandbox providers

This package implements the provider-neutral code-execution interface used by
`MemAgent`.

| Provider | Lifecycle | Boundary |
|---|---|---|
| `e2b` | One bounded remote session per provider instance | Remote E2B isolation; network denied by default. |
| `daytona` | Fresh remote environment per operation | Remote container environment; file operations are not coherent across calls. |
| `graalpy` | Private local directory and bounded processes | Subprocess mode is trusted-code execution only; Java `UNTRUSTED` mode is the hardened boundary. |

Configure through the builder:

```python
agent = (
    MemAgentBuilder()
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

The agent registers `execute_code`, `sandbox_write_file`, and
`sandbox_read_file`. Every execution returns the normalized `ExecutionResult`
contract, including provider audit metadata.

See the public documentation:

- `docs/sandbox/index.md`
- `docs/sandbox/e2b.md`
- `docs/sandbox/daytona.md`
- `docs/sandbox/graalpy.md`
