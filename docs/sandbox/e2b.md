# E2B provider

The E2B adapter owns one bounded Code Interpreter session for the lifetime of
the provider. A write, execution, and subsequent read therefore share one
remote filesystem.

## Install and configure

MemoRizz 0.5 pins a tested SDK pair because newer `e2b` transport releases are
not compatible with the current `e2b-code-interpreter` 2.9 client:

```bash
pip install "memorizz[sandbox-e2b]"
export E2B_API_KEY="<secret>"
```

Supported bounds are:

```text
e2b>=2.26.0,<2.38.0
e2b-code-interpreter>=2.9.0,<2.10.0
```

Construction fails immediately when `E2B_API_KEY` is missing or the installed
SDK pair is outside that range.

## Agent configuration

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_sandbox(
        {
            "provider": "e2b",
            "template": "my-bounded-template",
            "session_timeout": 300,
            "max_execution_timeout": 60,
            "allow_internet_access": False,
            "cpu_count": 2,
            "memory_mb": 1024,
        }
    )
    .build()
)
```

When all configuration is environment-driven, use the fluent preset:

```python
agent = (
    MemAgentBuilder()
    .with_e2b_from_env(
        template="my-bounded-template",
        session_timeout=300,
        max_execution_timeout=60,
        allow_internet_access=False,
        cpu_count=2,
        memory_mb=1024,
    )
    .build()
)
```

The preset validates `E2B_API_KEY` and the supported SDK pair immediately and
retains only secret-free provider metadata in `agent.environment_reports`.

CPU and memory are properties of the selected E2B template. MemoRizz requires
an explicit template whenever those policy declarations are supplied; deployers
must build and validate that template with the matching resources. The session
timeout is clamped to 30–3600 seconds, and every execution is capped by
`max_execution_timeout`.

Outbound internet access defaults to `False` and is sent to the E2B create API.

## Coherent session lifecycle

```python
from memorizz.sandbox.providers.e2b_provider import E2BSandboxProvider

with E2BSandboxProvider(api_key="<secret>") as sandbox:
    assert sandbox.write_file("/workspace/input.txt", "41")
    result = sandbox.execute_code(
        "print(int(open('/workspace/input.txt').read()) + 1)"
    )
    assert result.stdout == ["42"]
    assert sandbox.read_file("/workspace/input.txt") == "41"
```

The session is created lazily through `Sandbox.create(...)`, reused for file
and code operations, and killed by `close()` or context-manager exit. Older SDK
constructor/result shapes are normalized only through the compatibility path.

## Result and audit metadata

```json
{
  "stdout": ["42"],
  "stderr": [],
  "error": null,
  "exit_code": 0,
  "results": [],
  "success": true,
  "metadata": {
    "provider": "e2b",
    "stateful_session": true,
    "execution_timeout": 30,
    "egress_allowed": false,
    "resource_policy_enforcement": "e2b_template"
  }
}
```

MemoRizz normalizes dictionary and object variants of stdout, stderr, results,
and errors. It never includes the E2B API key in persisted provider config or
execution metadata.

## Configuration reference

| Setting | Default | Meaning |
|---|---|---|
| `api_key` | `E2B_API_KEY` | Credential; required and never persisted. |
| `template` | E2B Code Interpreter default | Explicit template name/ID. |
| `session_timeout` | `300` | Bounded provider-session lifetime in seconds. |
| `max_execution_timeout` | `120` | Maximum time for one execution. |
| `allow_internet_access` | `False` | Outbound-network policy passed at sandbox creation. |
| `cpu_count` | unset | Declared CPU policy; requires an explicit template. |
| `memory_mb` | unset | Declared memory policy; requires an explicit template. |

## Troubleshooting

Use the package extra to get the compatible pair:

```bash
pip install --upgrade --force-reinstall "memorizz[sandbox-e2b]"
```

Run `memorizz capabilities` to see both installed SDK versions and whether the
credential is configured. An execution failure is returned as a failed
`ExecutionResult`; provider construction/configuration errors fail earlier.
