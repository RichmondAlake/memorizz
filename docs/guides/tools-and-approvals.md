# Tools, Safety, and Human Approval

Tools give a MemAgent authority outside the model. Treat their Python callable,
schema, credentials, and approval policy as host code—not prompt content.

The examples use environment-backed model credentials. Later snippets assume
the `provider` and tool functions created by your application are still in
scope.

## Register a typed tool

Type annotations and defaults become a strict JSON Schema. Keep the function
small, return JSON-serializable data, and describe the contract in its
docstring.

```python
from typing import Literal

from memorizz import MemAgentBuilder, governed_tool


@governed_tool(deterministic=True, domains=("inventory",))
def inventory_lookup(
    sku: str,
    warehouse: Literal["london", "new-york"] = "london",
) -> dict:
    """Return current inventory for one SKU and warehouse."""
    return {"sku": sku, "warehouse": warehouse, "available": 12}


agent = (
    MemAgentBuilder()
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_tools([inventory_lookup])
    .build()
)
```

MemoRizz rejects undeclared arguments and binds the Python signature before
dispatch. Never accept arbitrary shell, SQL, path, URL, or Python text unless a
separate policy validates and confines it.

## Classify side effects explicitly

```python
from memorizz import governed_tool


@governed_tool(
    deterministic=False,
    side_effects=True,
    requires_approval=True,
    approval_reason="Creates a customer in the production CRM",
    domains=("crm",),
)
def create_customer(name: str, email: str) -> dict:
    """Create one CRM customer after host approval."""
    return application_crm.create_customer(name=name, email=email)
```

Replace `application_crm` with a least-privileged client supplied by your host
application; never let the model construct that client or its credentials.

Side-effecting tools bypass semantic-cache admission. The model never receives
an `approved` or `confirm` argument.

## Durable approval lifecycle

When a governed tool requires approval, `agent.run()` pauses with an exact,
serializable proposal. A trusted host decides it and resumes the stored
checkpoint:

```python
pending = agent.list_approval_proposals(status="pending")
proposal_id = pending[0]["proposal_id"]

agent.approve(
    proposal_id,
    approver_id="operator@example.com",
    reason="Customer and destination reviewed",
)
result = agent.resume_approval(proposal_id, continue_model=False)

print(result.proposal)
print(result.tool_result)
print(result.consumed)  # True; the proposal cannot be reused
```

The host may instead call `reject()` or `cancel_approval()`. Proposals bind the
tool name, exact arguments, argument hash, policy reason, owner, thread and
checkpoint, expiry, approver identity, and audit time.

## Progressive disclosure

Large toolboxes should not send every schema on every turn. Attach a
deterministic `Toolbox` and let the semantic router select a scoped top-k set:

```python
from memorizz import ContextPolicy, MemAgentBuilder, Toolbox

toolbox = Toolbox.from_functions(
    [inventory_lookup, create_customer],
    memory_provider=provider,
    agent_id="operations-agent",
    augment=False,
)

agent = (
    MemAgentBuilder()
    .with_name("Operations")
    .with_memory_provider(provider)
    .with_tools([inventory_lookup, create_customer])
    .with_toolbox(toolbox)
    .with_context_policy(ContextPolicy(tool_top_k=4))
    .build_and_save()
)
```

The model receives stable discovery/invocation handles plus the selected strict
schemas. Scope and allowlists are applied before top-k retrieval. Trusted
callables are rebound from an explicit registry or reviewed import reference;
MemoRizz never unpickles executable code.

## Large tool results

```python
from memorizz import ToolResultPolicy

agent = (
    MemAgentBuilder()
    .with_tool_result_policy(
        ToolResultPolicy(offload_above_chars=8_000)
    )
    .build()
)
```

Small results stay inline. Large results are stored once and replaced by an
auditable pointer containing a digest and identifiers. Expansion tools are
never re-offloaded, preventing pointer-to-pointer loops.

## Report structured tool outcomes

A tool can complete without producing a clean primary-provider success. Use a
`ToolResult` when the host knows that execution was empty, degraded, or served
by a fallback:

```python
from memorizz import ToolOutcome, ToolOutcomeStatus, ToolResult


def inventory_lookup(sku: str) -> ToolResult:
    try:
        return ToolResult(
            primary_inventory.lookup(sku),
            ToolOutcome(
                status=ToolOutcomeStatus.SUCCESS,
                provider="primary_inventory",
                result_count=1,
            ),
        )
    except TimeoutError:
        rows = inventory_replica.lookup(sku)
        return ToolResult(
            rows,
            ToolOutcome(
                status=ToolOutcomeStatus.FALLBACK,
                reason_code="primary_timeout",
                primary_provider="primary_inventory",
                fallback_provider="inventory_replica",
                retryable=True,
                result_count=len(rows),
            ),
        )
```

MemoRizz unwraps the value before sending it to the model. The tool's existing
result schema therefore stays unchanged, while the SDK, CLI, MCP server, UI,
trace analyzer, and learning control plane receive content-free outcome
evidence. Non-clean outcomes bypass semantic-cache admission for that turn, so
an empty, degraded, fallback, or failed provider response is not replayed later
as though it were a clean primary-provider success.

| Outcome | Meaning | Counts as usable |
|---|---|---|
| `success` | Primary path completed normally | Yes |
| `empty` | A read completed but returned no records | Yes |
| `degraded` | The result is usable with reduced capability | Yes |
| `fallback` | A usable alternate path completed the operation | Yes |
| `provider_error` | No provider path produced a usable result | No |
| `error` | A non-provider execution or validation failure | No |

Fallback and degradation can coexist. `status="fallback"` remains the primary
display state while `degraded=True` preserves the quality signal. Do not label
an unavailable-provider placeholder as a successful fallback: if the user's
operation was not satisfied, return `provider_error`.

Legacy tools do not have to change immediately. MemoRizz safely recognizes
`ok: false`, `success: false`, structured error codes, empty `results`/`items`/
`matches` collections, and existing `fallback_used`/`degraded` retrieval
diagnostics. It does not infer outcomes from arbitrary prose.

After a turn, trusted SDK hosts can inspect the same evidence:

```python
for outcome in agent.last_tool_outcomes:
    print(outcome["tool_name"], outcome["status"], outcome.get("reason_code"))
```

## Choose the right execution boundary

| Need | Capability | Boundary |
|---|---|---|
| Call a typed application function | Python tool | Your application process |
| Call Notion, Calendar, or another MCP service | [MCP client](mcp-connectivity.md) | Remote/local MCP server plus durable mutation gate |
| Navigate websites | [Browser control](../browser-control/index.md) | Isolated Browser Use process plus approval |
| Execute code | [Sandbox](../sandbox/index.md) | Provider-specific; read each security contract |

Approval limits *when* authority is exercised. It does not make untrusted code,
web content, or credentials safe. Apply least privilege and input validation at
the underlying boundary.
