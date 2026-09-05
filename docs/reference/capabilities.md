# Capabilities and Preflight

Streaming is now the default delivery path. See the
[streaming contract, defaults and compatibility modes](../guides/streaming.md)
for the SDK event iterator, CLI opt-out, UI lifecycle and opt-in MCP answer events.
Full-answer completion validators still buffer until acceptance; Python `run()`
retains its complete-string return contract.

Package versions do not prove that optional providers are installed,
configured, or reachable. Use capability reports at startup and provider
preflight before serving traffic.

## Package report

```bash
memorizz capabilities --json
```

```python
import memorizz

report = memorizz.capabilities()
print(report["version"])
print(report["features"]["headless_runtime"])
print(report["features"]["mcp_server"])
print(report["features"]["meta_harness"])
```

The report is secret-free and includes feature states, dependency versions,
and provider readiness. A feature may be implemented but not ready—for example,
E2B can be installed while `E2B_API_KEY` is absent.

The MetaHarness feature report describes the stable SDK/CLI/UI/MCP surface and
its single-node durable coordination boundary. Use `memorizz harness doctor`
for executable-, authentication-, and adapter-specific readiness; package
capabilities do not start or authenticate an external harness.

## Agent report

```python
report = agent.capability_report(preflight=True)

if report["agent"].get("browser_control_error"):
    raise RuntimeError(report["agent"]["browser_control_error"])
```

The agent report adds its effective LLM, memory, sandbox, browser, MCP,
semantic-cache, tool-routing, skill-retrieval, learning, approval, and
delegation state. `preflight=True` calls the provider's preflight hook when one
exists.

## Oracle preflight

```bash
memorizz oracle preflight --index-policy lazy --json
```

Treat `ok=false` as a deployment failure. In particular, resolve embedding
dimension mismatches before writing any vectors. See the
[Oracle provider guide](../memory-providers/oracle.md).

## Deployment pattern

Run these checks in a readiness command or startup hook:

```python
package = memorizz.capabilities()
if not package["features"]["headless_runtime"]["available"]:
    raise RuntimeError("MemoRizz runtime is unavailable")

agent.validate_configuration()
agent_report = agent.capability_report(preflight=True)
```

Capability reports describe state; they do not make external calls on behalf
of the model or expose credentials.
