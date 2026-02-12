# GraalPy Provider

GraalPy is a **local sandbox provider** that executes code on your machine using Oracle's GraalVM Python runtime. It requires **no cloud services, no API keys, and no billing** — making it ideal for offline, air-gapped, or cost-sensitive environments.

## Overview

| Feature | Details |
|---------|---------|
| **Type** | Local (runs on your machine) |
| **Isolation** | OS process boundaries (subprocess) or JVM sandbox (java_wrapper) |
| **Cold Start** | N/A (local process) |
| **Session Limit** | Unlimited |
| **Languages** | Python only |
| **GPU** | Not supported |
| **Filesystem** | Configurable (restricted in java_wrapper mode) |
| **Internet** | Depends on mode and policy |
| **Pricing** | Free |
| **Website** | [graalvm.org/python](https://www.graalvm.org/python/) |

## Setup

### 1. Install GraalPy

GraalPy is a system dependency, not a pip package.

Download a release archive for your OS/CPU from:

- [https://github.com/oracle/graalpython/releases](https://github.com/oracle/graalpython/releases)

Then extract and expose the binary (example layout):

```bash
# Example location after extraction:
# $HOME/.local/graalpy-community-<version>-<platform>/bin/graalpy

mkdir -p "$HOME/.local/bin"
ln -sf "$HOME/.local/graalpy-community-<version>-<platform>/bin/graalpy" "$HOME/.local/bin/graalpy"
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Verify Installation

```bash
graalpy --version
# Expected: GraalPy <version> (Python 3.x compatible)
```

### 3. Install MemoRizz

```bash
pip install memorizz
# No additional sandbox pip dependency needed for GraalPy
```

### 4. Configure Memorizz

Set one of the following:

- `graalpy` on your shell `PATH`
- `GRAALPY_PATH` environment variable
- `graalpy_path` in `sandbox_provider` config

```bash
export GRAALPY_PATH="/absolute/path/to/graalpy"
```

In the UI, configure this under **Settings -> Sandbox -> GRAALPY_PATH**.

In the Memorizz UI Sandbox settings, when **Default Sandbox Provider = graalpy**:

- **GraalPy Internet Access** checked: uses `subprocess` mode (network allowed)
- **GraalPy Internet Access** unchecked: uses `java_wrapper` mode with `UNTRUSTED` policy
  and requires `GRAALPY_JAVA_WRAPPER_JAR`

## Two Execution Modes

GraalPy offers two modes with different security/complexity tradeoffs:

### Subprocess Mode (Default)

Code runs via `graalpy -c "<code>"` in an OS subprocess. Simple to set up, with process-level isolation.

```python
from memorizz.memagent.core import MemAgent

agent = MemAgent(
    model=my_llm,
    sandbox_provider="graalpy",
    instruction="You are a helpful coding assistant.",
)
```

If the executable cannot be resolved, agent/provider creation fails fast with a
`ValueError` that includes setup guidance.

**Security**: The code runs in a separate OS process but has the same filesystem and network access as the parent process.

### Java Wrapper Mode (Production)

Code runs inside a JVM with GraalVM's `SandboxPolicy.UNTRUSTED`, which restricts:

- Filesystem access
- Network access
- CPU and memory consumption
- System calls

```python
agent = MemAgent(
    model=my_llm,
    sandbox_provider={
        "provider": "graalpy",
        "mode": "java_wrapper",
        "java_wrapper_jar": "/path/to/graalpy-sandbox.jar",
        "sandbox_policy": "UNTRUSTED",
    },
    instruction="You are a secure coding assistant.",
)
```

**Requirements**: JVM installed + a wrapper JAR that uses GraalVM's Polyglot API.

**Security**: Full JVM-level sandboxing. The `UNTRUSTED` policy is designed for executing code from untrusted sources.

## Mode Comparison

| Aspect | Subprocess | Java Wrapper |
|--------|-----------|-------------|
| **Setup complexity** | Low (just GraalPy) | Medium (JVM + JAR) |
| **Security** | OS process isolation | JVM SandboxPolicy |
| **Filesystem access** | Full (same as parent) | Restricted |
| **Network access** | Full | Restricted |
| **CPU/memory limits** | OS-level only | JVM-enforced |
| **Best for** | Development, testing | Production, multi-tenant |

## Usage

### Basic (Subprocess Mode)

```python
agent = MemAgent(
    model=my_llm,
    sandbox_provider="graalpy",
    instruction="Execute code to solve problems.",
)

response = agent.run("What's the factorial of 100?")
```

### With Explicit Path

If `graalpy` is not on your PATH:

```python
agent = MemAgent(
    model=my_llm,
    sandbox_provider={
        "provider": "graalpy",
        "graalpy_path": "/opt/graalvm/bin/graalpy",
    },
)
```

### With Working Directory

Restrict file operations to a specific directory:

```python
agent = MemAgent(
    model=my_llm,
    sandbox_provider={
        "provider": "graalpy",
        "working_dir": "/tmp/sandbox_workspace",
    },
)
```

### Direct Provider Usage

```python
from memorizz.sandbox.providers.graalpy_provider import GraalPySandboxProvider

provider = GraalPySandboxProvider()
result = provider.execute_code("print('hello from GraalPy!')")
print(result.output)     # "hello from GraalPy!"
print(result.success)    # True
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graalpy_path` | `str` | `graalpy` (on PATH) | Path to the GraalPy binary |
| `mode` | `str` | `"subprocess"` | Execution mode: `"subprocess"` or `"java_wrapper"` |
| `java_wrapper_jar` | `str` | `None` | Path to the Java wrapper JAR (java_wrapper mode only) |
| `sandbox_policy` | `str` | `"UNTRUSTED"` | GraalVM sandbox policy (java_wrapper mode only) |
| `working_dir` | `str` | System temp dir | Working directory for execution |

### Sandbox Policies (Java Wrapper Mode)

| Policy | Security Level | Description |
|--------|---------------|-------------|
| `TRUSTED` | Low | No restrictions (default GraalVM behavior) |
| `CONSTRAINED` | Medium | Additional security restrictions |
| `UNTRUSTED` | High | Maximum isolation for untrusted code |

## When to Use GraalPy

| Scenario | GraalPy? | Why |
|----------|----------|-----|
| Offline / air-gapped environments | Yes | No cloud dependency |
| Development and testing | Yes | Free, fast, no setup friction |
| Cost-sensitive deployments | Yes | No per-hour billing |
| Multi-tenant production (java_wrapper) | Yes | JVM-level isolation |
| Need GPU support | No | Use Daytona |
| Need multi-language execution | No | Use E2B |
| Need cloud-scale concurrency | No | Use E2B or Daytona |

## Limitations

- **Python only** — GraalPy is a Python runtime; it doesn't support JavaScript, R, etc.
- **Subprocess mode has limited isolation** — The code runs as a regular OS process
- **Java wrapper requires JVM** — Additional setup complexity for the secure mode
- **No cloud features** — No remote access, no snapshots, no horizontal scaling

## Troubleshooting

### "GraalPy not found"

If setup is incomplete, Memorizz now surfaces this early during provider
creation (programmatic and UI usage), for example:

- `GraalPy sandbox requires a working 'graalpy' executable...`
- `Sandbox unavailable: ...`

Make sure GraalPy is installed and resolvable:
```bash
graalpy --version
```

Then set one of:

1. Shell `PATH` includes the executable
2. `GRAALPY_PATH` environment variable
3. Explicit `graalpy_path` in config

```python
sandbox_provider={"provider": "graalpy", "graalpy_path": "/full/path/to/graalpy"}
```

### "GraalPy execution timed out"

Increase the timeout (default is 30 seconds):
```python
result = agent.sandbox_manager.execute_code(code, timeout=120)
```

### "Java runtime not found" (java_wrapper mode)

Install a JVM with GraalVM support:
```bash
sdk install java 25.0.1-graal
```

### Fallback to subprocess mode

If `java_wrapper_jar` is not provided, the provider automatically falls back to subprocess mode with a warning.
