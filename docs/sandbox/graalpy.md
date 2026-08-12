# GraalPy Execution Provider

GraalPy executes Python locally using Oracle's GraalVM Python runtime. It
requires no cloud service or API key. The default subprocess mode is a
**bounded execution provider, not a security sandbox**. Use it only for trusted
code. The Java wrapper with `SandboxPolicy.UNTRUSTED` is the supported local
boundary for untrusted code.

## Overview

| Feature | Details |
|---------|---------|
| **Type** | Local (runs on your machine) |
| **Isolation** | Bounded host process (trusted code) or JVM `UNTRUSTED` sandbox |
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

In the MemoRizz UI Sandbox settings, when **Default Sandbox Provider = graalpy**:

- **GraalPy Execution Mode = Subprocess**: bounded execution for trusted code;
  egress is denied by default
- **GraalPy Internet Access** checked: explicit egress for trusted subprocess code
- **GraalPy Internet Access** unchecked: egress is denied by `sandbox-exec`, a
  Linux network namespace, or a fail-closed error when neither enforcer exists
- **GraalPy Execution Mode = Java UNTRUSTED wrapper**: requires a compiled,
  validated `GRAALPY_JAVA_WRAPPER_JAR`; guest IO remains disabled

## Two Execution Modes

GraalPy offers two modes with different security/complexity tradeoffs:

### Subprocess Mode (Default)

Code runs via `graalpy -c "<code>"` under a private `0700` directory. MemoRizz
rejects absolute and parent-traversal file paths, passes only allowlisted
environment variables, enforces POSIX CPU/address-space/process/file-size
limits, and denies network access by default. These controls reduce accidents;
the process still has the host user's OS identity and is not safe for hostile
code.

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

**Security**: inspect `result.metadata.security_boundary`; subprocess results
report `bounded_execution_provider_not_strong_sandbox`. Egress must be enabled
explicitly for trusted workloads.

### Java Wrapper Mode (Untrusted Code)

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

**Requirements**: JVM installed, the package-shipped
`MemorizzGraalSandbox.java` compiled with the matching GraalVM SDK into an
executable JAR, and a POSIX host for mandatory resource limits. Provider
validation refuses a missing/malformed JAR, the wrong wrapper main class, a
non-`UNTRUSTED` policy, a missing JVM, or unavailable host-limit support.

The shipped wrapper disables host/class/native access, child-process creation,
environment access, and all guest IO. It sets GraalVM's mandatory CPU, heap,
AST-depth, stack-frame, thread, stdout, and stderr limits; the host independently
applies CPU, address-space, process-count, file-size, and wall-clock limits.
MemoRizz ships source—not a compatibility-fragile prebuilt JAR—so validate the
compiled wrapper against the exact GraalVM release used in production.

## Mode Comparison

| Aspect | Subprocess | Java Wrapper |
|--------|-----------|-------------|
| **Setup complexity** | Low (just GraalPy) | Medium (JVM + JAR) |
| **Security** | Not a security boundary | GraalVM `UNTRUSTED` + host limits |
| **Filesystem access** | MemoRizz API paths confined; guest OS access is not a strong boundary | Guest IO disabled |
| **Network access** | Denied by default or fails closed; explicit opt-in for trusted code | Guest IO disabled |
| **CPU/memory limits** | POSIX host limits | POSIX host limits plus JVM policy |
| **Best for** | Trusted development and batch execution | Validated untrusted-code deployments |

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

### With a Private-Directory Parent

Choose where MemoRizz creates its private, random per-provider directory:

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
| `allow_network` | `bool` | `False` | Explicit egress opt-in for trusted subprocess code |
| `env_allowlist` | sequence | Safe locale/path keys | Only host/guest environment keys passed through |
| `max_memory_mb` | `int` | `512` | Address-space limit (minimum 128 MB) |
| `max_cpu_seconds` | `int` | `30` | CPU time limit |
| `max_processes` | `int` | `32` | Child/process-count limit |
| `max_file_bytes` | `int` | `10000000` | Per-process output-file size limit |

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
| Multi-tenant production (validated java_wrapper) | Conditional | Requires a compiled/tested wrapper and defense in depth |
| Need GPU support | No | Use Daytona |
| Need multi-language execution | No | Use E2B |
| Need cloud-scale concurrency | No | Use E2B or Daytona |

## Limitations

- **Python only** — GraalPy is a Python runtime; it doesn't support JavaScript, R, etc.
- **Subprocess mode is not isolation** — The code retains the host user's OS identity
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

There is no silent fallback from `java_wrapper` to subprocess mode. A missing
wrapper or invalid policy fails validation so an untrusted workload cannot be
downgraded accidentally.
