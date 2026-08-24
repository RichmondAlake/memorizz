# Daytona provider

The Daytona adapter executes code in a remote development environment. In the
current MemoRizz adapter, each operation creates and removes its own environment.
Consequently, `sandbox_write_file` followed by `sandbox_read_file` is not a
coherent round trip; use E2B when a stateful provider filesystem is required.

## Install and configure

```bash
pip install "memorizz[sandbox-daytona]"
export DAYTONA_API_KEY="<secret>"
```

Optional deployment settings:

```bash
export DAYTONA_API_URL="https://app.daytona.io/api"
export DAYTONA_TARGET="us"
```

## Usage

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_sandbox(
        {
            "provider": "daytona",
            "api_url": "https://app.daytona.io/api",
            "target": "us",
        }
    )
    .build()
)
```

Direct provider use:

```python
from memorizz.sandbox.providers.daytona_provider import DaytonaSandboxProvider

provider = DaytonaSandboxProvider()
result = provider.execute_code("print(6 * 7)")
assert result.stdout == ["42"]
```

Each call provisions a Daytona environment, runs `process.code_run`, normalizes
the output, and removes the environment in a `finally` block.

## Configuration reference

| Setting | Default | Meaning |
|---|---|---|
| `api_key` | `DAYTONA_API_KEY` | Daytona credential. |
| `api_url` | `DAYTONA_API_URL` or provider default | API endpoint. |
| `target` | `DAYTONA_TARGET` or `us` | Deployment target. |

Provider capabilities, limits, regions, and pricing can change independently of
MemoRizz. Validate them against the Daytona deployment you operate.
