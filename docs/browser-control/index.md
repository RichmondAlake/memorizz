# Browser Control

MemoRizz 0.5 provides a provider-neutral `browser_control` capability. When it
is enabled, a MemAgent receives one bounded natural-language browser tool. The
first provider is [Browser Use](https://github.com/browser-use/browser-use).

Browser automation can navigate, click, type, upload, submit, publish, buy, or
delete. MemoRizz therefore marks every model-initiated browser call as
nondeterministic, side-effecting, and durable-approval-required. There is no
model-visible `approved` or `confirm` argument.

## Install Browser Use separately

Browser Use currently targets Python 3.11+ and its package dependency line can
conflict with MemoRizz's MCP 2.x runtime. Install it as an isolated tool
instead of into the MemoRizz environment:

```bash
uv tool install --python 3.12 browser-use
browser-use install
browser-use doctor
```

MemoRizz resolves the isolated Python interpreter from the `browser-use` entry
point installed by `uv tool`/`pipx`, then uses the official Python Agent API. It
writes a fixed wrapper to a private temporary directory, sends the model's task
only as JSON data, and never executes model-authored Python or shell commands.
The worker process and browser are bounded and closed on success, failure, and
timeout. If a custom launcher has no absolute Python shebang, configure
`MEMORIZZ_BROWSER_USE_PYTHON_COMMAND` explicitly.

Set the API key for the Browser Use LLM provider you select:

```bash
export OPENAI_API_KEY=<secret>
# Or ANTHROPIC_API_KEY, GOOGLE_API_KEY, or BROWSER_USE_API_KEY.
```

## SDK setup

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_memory_provider(memory_provider)
    .with_llm_config({"provider": "openai", "model": "gpt-4.1-mini"})
    .with_browser_control(
        {
            "provider": "browseruse",
            "llm_provider": "openai",
            "model": "gpt-4.1-mini",
            "headless": True,
            "use_vision": True,
            "allowed_domains": ["example.com", "*.notion.so"],
            "prohibited_domains": ["admin.example.com"],
            "block_ip_addresses": True,
            "max_steps": 25,
            "task_timeout": 600,
        }
    )
    .build(validate=True, persist=True)
)
```

Equivalent direct construction and runtime changes are available:

```python
from memorizz import MemAgent

agent = MemAgent(
    model=model,
    memory_provider=memory_provider,
    browser_control={"provider": "browseruse", "allowed_domains": ["example.com"]},
)

agent.with_browser_control({"provider": "browseruse"})
agent.with_browser_control(None)  # detach and remove the tool
```

`agent.capability_report()` reports configured/readiness state without exposing
credentials. Persisted browser configuration contains policy and environment
variable names only, never API-key values.

## Approval lifecycle

If the model calls:

```json
{
  "name": "browser_control",
  "arguments": {
    "task": "Open example.com and submit the support form",
    "max_steps": 8
  }
}
```

MemoRizz stores an `ApprovalProposal` containing the exact tool name,
arguments, canonical argument hash, policy reason, agent/thread/checkpoint
identity, creation/expiry times, and later the approver identity and audit
timestamp. The provider is not called yet.

A trusted host decides and resumes the exact checkpoint:

```python
pending = agent.list_approval_proposals(status="pending")
proposal_id = pending[0]["proposal_id"]

agent.approve(
    proposal_id,
    approver_id="operator@example.com",
    reason="Destination and form values reviewed",
)
resume = agent.resume_approval(proposal_id)
print(resume.tool_result)         # exact BrowserControlResult payload
print(resume.assistant_response)  # optional model continuation

# Deterministic hosts can omit the model continuation:
resume = agent.resume_approval(proposal_id, continue_model=False)
```

Alternatives are `agent.reject(...)` and `agent.cancel_approval(...)`.
Proposals expire, bind to the stored argument hash, and are consumed atomically
once. Resume does not ask the model to reconstruct the task.

The local playground renders the proposal, arguments/hash, expiry, and
Approve/Reject controls. The interactive CLI exposes the same lifecycle:

```text
/approvals pending
/approvals approve PROPOSAL_ID operator@example.com reviewed
/approvals resume PROPOSAL_ID
```

## CLI and environment

Browser control is explicit opt-in. Merely setting an LLM API key does not add
the tool.

```bash
memorizz chat --browser-control
memorizz run --browser-control "Read the title at example.com"
```

Inside the REPL:

```text
/browser status
/browser on
/browser off
/browser run Read the title at https://example.com
```

`/browser run` is a direct trusted-host action: the user at the terminal is the
authorizing host. Model-initiated calls still create a durable proposal.

Environment configuration:

| Variable | Default | Purpose |
|---|---:|---|
| `MEMORIZZ_BROWSER_CONTROL_PROVIDER` | disabled | Set `browseruse` to opt in globally |
| `MEMORIZZ_BROWSER_USE_COMMAND` | `browser-use` | Entry point for the isolated Browser Use installation |
| `MEMORIZZ_BROWSER_USE_PYTHON_COMMAND` | discovered from entry point | Optional isolated Python executable override |
| `MEMORIZZ_BROWSER_USE_LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `google`, or `browseruse` |
| `MEMORIZZ_BROWSER_USE_MODEL` | provider default | Browser-agent model |
| `MEMORIZZ_BROWSER_USE_HEADLESS` | `true` | Hide/show the local browser window |
| `MEMORIZZ_BROWSER_USE_VISION` | `true` | Enable screenshot understanding |
| `MEMORIZZ_BROWSER_USE_ALLOWED_DOMAINS` | empty | Comma-separated navigation allowlist |
| `MEMORIZZ_BROWSER_USE_PROHIBITED_DOMAINS` | empty | Comma-separated denylist |
| `MEMORIZZ_BROWSER_USE_BLOCK_IP_ADDRESSES` | `true` | Block direct IP navigation |
| `MEMORIZZ_BROWSER_USE_MAX_STEPS` | `25` | Host maximum, clamped to 1–100 |
| `MEMORIZZ_BROWSER_USE_TASK_TIMEOUT` | `600` | Wall timeout, clamped to 10–3600 seconds |
| `MEMORIZZ_BROWSER_USE_CLOUD` | `false` | Use Browser Use cloud browser |
| `MEMORIZZ_BROWSER_USE_CDP_URL_ENV` | `BU_CDP_URL` | Name of the allowlisted CDP URL variable |

The UI Settings page exposes the common fields, while each agent editor and
playground panel controls whether the capability is attached.

## Result contract

The tool returns structured JSON:

```json
{
  "success": true,
  "task": "Read the title at https://example.com",
  "output": "Example Domain",
  "urls": ["https://example.com"],
  "actions": ["go_to_url", "done"],
  "errors": [],
  "steps": 2,
  "duration_seconds": 3.4,
  "metadata": {
    "provider": "browseruse",
    "execution_boundary": "isolated_browser_use_python",
    "max_steps": 8,
    "headless": true,
    "block_ip_addresses": true
  }
}
```

Timeouts and provider failures remain structured failures rather than partial
untyped text. The isolated worker process group is terminated on timeout,
Browser Use receives an environment allowlist, and the wrapper always stops its
browser in a `finally` block.

## Security guidance

- Prefer a narrow `allowed_domains` list. An empty list is unrestricted;
  wildcard `*` is rejected.
- Treat page text as untrusted prompt-injection content. Approval is an action
  gate, not a guarantee that page instructions are safe.
- Use separate browser profiles/accounts with least privilege; do not expose a
  personal administrator profile to a multi-tenant agent.
- Keep direct IP blocking enabled and deny sensitive domains explicitly.
- Review the complete task and destination before approval, especially for
  forms, messages, purchases, uploads, account changes, and destructive work.
- Browser Use is isolated from MemoRizz at the Python dependency boundary and
  runs a private fixed worker per task. This is lifecycle isolation, not an OS
  security sandbox; the browser retains the network and account authority you
  configure.

## Custom providers

Equivalent providers implement `BrowserControlProvider.run_task(...)`, return a
`BrowserControlResult`, expose secret-free `get_config()`, and register an
explicit name. The MemoRizz manager preserves the same single governed tool
and approval policy regardless of provider.

```python
from memorizz.browser_control import (
    BrowserControlProvider,
    BrowserControlResult,
    register_provider,
)

class InternalBrowserProvider(BrowserControlProvider):
    provider_name = "internal"

    def run_task(self, task, *, max_steps=None, timeout=None):
        return BrowserControlResult(success=True, task=task, output="done")

register_provider("internal", InternalBrowserProvider)
```
