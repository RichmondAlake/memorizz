# Daytona Provider

Daytona provides **full cloud development environments** as sandboxes, with unlimited runtime, GPU support, and the fastest cold starts (~90ms) among cloud providers.

## Overview

| Feature | Details |
|---------|---------|
| **Type** | Cloud-hosted |
| **Isolation** | Docker containers with configurable resources |
| **Cold Start** | ~90ms |
| **Session Limit** | Unlimited |
| **Languages** | Python, JavaScript, TypeScript |
| **GPU** | Supported (configurable) |
| **Filesystem** | Full read/write with Git support |
| **Internet** | Available from inside sandbox |
| **Pricing** | ~$0.08/hour, $200 free credits |
| **Website** | [daytona.io](https://daytona.io) |

## Setup

### 1. Install the SDK

```bash
pip install "memorizz[sandbox-daytona]"
```

### 2. Get an API Key

1. Sign up at [daytona.io](https://daytona.io)
2. Navigate to your dashboard
3. Generate an API key

### 3. Set the Environment Variable

```bash
export DAYTONA_API_KEY="your-api-key"

# Optional: configure region and API URL
export DAYTONA_TARGET="us"                        # or "eu"
export DAYTONA_API_URL="https://app.daytona.io/api"
```

## Usage

### Basic

```python
from memorizz.memagent.core import MemAgent

agent = MemAgent(
    model=my_llm,
    sandbox_provider="daytona",
    instruction="You are a Python developer. Execute code to solve problems.",
)

response = agent.run("Sort a list of 1000 random numbers and time the operation")
```

### With Explicit Configuration

```python
agent = MemAgent(
    model=my_llm,
    sandbox_provider={
        "provider": "daytona",
        "api_key": "your-api-key",
        "api_url": "https://app.daytona.io/api",
        "target": "us",
    },
    instruction="You are a helpful coding assistant.",
)
```

### Direct Provider Usage

```python
from memorizz.sandbox.providers.daytona_provider import DaytonaSandboxProvider

provider = DaytonaSandboxProvider(api_key="your-api-key")
result = provider.execute_code("print('hello from Daytona!')")
print(result.output)     # "hello from Daytona!"
print(result.success)    # True
```

## How It Works

Each `execute_code` call follows this lifecycle:

1. **Create** — A Daytona sandbox is provisioned with the configured resources (~90ms)
2. **Execute** — Code is sent via `process.code_run()` for language-agnostic execution
3. **Capture** — The execution result, stdout, and exit code are collected
4. **Remove** — The sandbox is deleted to free resources

## When to Use Daytona

Daytona is the best choice when you need:

- **Unlimited session duration** — No 24-hour limit (unlike E2B)
- **GPU access** — Run ML training, inference, or data processing with GPU
- **Full dev environments** — Git, package managers, configurable CPU/memory/disk
- **Enterprise features** — Self-hosted deployment, SSH access, environment snapshots

## Daytona vs E2B

| Aspect | E2B | Daytona |
|--------|-----|---------|
| **Best for** | Quick AI agent tasks | Full dev workflows |
| **Cold start** | ~150ms | ~90ms |
| **Session limit** | 24 hours | Unlimited |
| **GPU** | No | Yes |
| **Isolation** | Firecracker microVMs | Docker containers |
| **Git support** | Limited | Built-in |
| **Self-hosting** | Open-source | Enterprise |
| **Free credits** | $100 | $200 |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `DAYTONA_API_KEY` env var | Daytona API key |
| `api_url` | `str` | `https://app.daytona.io/api` | API endpoint URL |
| `target` | `str` | `us` | Target region (`us`, `eu`) |

## Limitations

- **Cloud-only** — Requires internet connectivity and an API key
- **Docker isolation** — Uses Docker containers, not microVMs (less isolated than E2B's Firecracker)
- **Stateless per call** — Variables do not persist between `execute_code` calls

## Troubleshooting

### "daytona SDK is not installed"

```bash
pip install daytona
```

### "Daytona API key not provided"

```bash
export DAYTONA_API_KEY="your-api-key"
```

### Sandbox creation timeouts

If sandbox creation is slow, check your target region:
```python
sandbox_provider={"provider": "daytona", "target": "eu"}  # Try a different region
```
