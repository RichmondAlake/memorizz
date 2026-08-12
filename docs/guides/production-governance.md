# Production Governance in 0.5

MemoRizz 0.5 makes tool execution, caching, compaction, and delegation explicit
runtime contracts. This guide covers the controls that matter when an agent can
reach Notion, Google Calendar, business systems, or a code-execution provider.

## Compose the complete runtime

The builder now covers the same production surface as `MemAgent`; no
post-build mutation is required:

```python
from memorizz import (
    ContextPolicy,
    MemAgentBuilder,
    SQLiteApprovalStore,
    ToolResultPolicy,
    governed_tool,
)
from memorizz.long_term.procedural.toolbox import Toolbox
from memorizz.sandbox.providers.e2b_provider import E2BSandboxProvider


@governed_tool(
    deterministic=False,
    side_effects=True,
    requires_approval=True,
    approval_reason="Creates an external customer record",
    domains=("crm",),
)
def create_customer(name: str, email: str) -> dict:
    ...


toolbox = Toolbox.from_functions(
    [create_customer],
    memory_provider=provider,
    agent_id="ops-agent",
    augment=False,
)

agent = (
    MemAgentBuilder()
    .with_name("Operations")
    .with_model(model)
    .with_memory_provider(provider)
    .with_tools([create_customer])
    .with_toolbox(toolbox)
    .with_mcp_servers(notion_and_calendar_servers)
    .with_sandbox(
        E2BSandboxProvider(
            template="memorizz-bounded",
            allow_internet_access=False,
            cpu_count=2,
            memory_mb=1024,
        )
    )
    .with_context_policy(ContextPolicy(tool_top_k=5))
    .with_tool_result_policy(ToolResultPolicy(offload_above_chars=8_000))
    .with_approval_store(SQLiteApprovalStore("/var/lib/memorizz/approvals.db"))
    .with_skills(course_skills, persistence="skillbox")
    .with_skill_retrieval(enabled=True, top_k=2)
    .with_continual_learning(enabled=False)
    .with_delegation(delegates, mode="deterministic", plan=workflow_plan)
    .build(validate=True, persist=True)
)
```

`Toolbox.from_functions(..., augment=False)` is deterministic: it does not
create an LLM or global embedding client. Persisted schemas keep required
fields, defaults, enums, nested structures, and
`additionalProperties: false`. Executable Python is never unpickled; trusted
callables are rebound explicitly or by a reviewed `module:qualified_name`
reference.

## Progressive tool disclosure

At the start of each turn, `SemanticToolRouter` applies agent, user, and Toolbox
scope before selecting top-k tools. The model receives the selected strict
schemas plus stable `discover_tools` and `invoke_tool` handles—not the entire
Toolbox. Invocation is allowlisted, aliases and deprecated arguments are
normalized, the Python signature is bound strictly, and duplicate/retry budgets
are enforced.

Setting `ContextPolicy(progressive_tool_disclosure=False)` restores direct
schema exposure for a small, trusted tool set. It should be an explicit
compatibility choice.

## Durable human approval

A side-effecting governed tool pauses execution by creating a serializable
`ApprovalProposal` containing:

- the exact logical tool name and arguments;
- a SHA-256 argument digest and policy reason;
- a single-use proposal/checkpoint ID and thread ID;
- creation and expiry timestamps; and
- the eventual approver identity, decision, and audit timestamp.

`approved` and `confirm` are not model-visible arguments. A host lists and
decides proposals, then resumes the stored checkpoint:

```python
pending = agent.list_approval_proposals(status="pending")
proposal_id = pending[0]["proposal_id"]

agent.approve(proposal_id, approver_id="operator@example.com")
result = agent.resume_approval(proposal_id)  # exact stored call, once only

# Alternatives:
agent.reject(proposal_id, approver_id="operator@example.com")
agent.cancel_approval(proposal_id, approver_id="operator@example.com")
```

The local Playground renders the exact proposal with **Approve & resume** and
**Reject** controls. The JSON routes live under
`/api/agents/{agent_id}/approvals`. MCP-specific CLI/UI approval flows remain
available and use the same durable store semantics.

## Size-aware tool results

Small tool results remain inline. A full result is persisted exactly once only
when it crosses `offload_above_chars` or `offload_above_tokens`; the returned
pointer includes its log ID, SHA-256 digest, size, identifiers, and audit time.
`retrieve_tool_log_entry` and other configured expansion tools are never
offloaded, so expansion cannot create pointer-to-pointer loops. Missing or
invalid expansions return structured failures.

## Semantic cache correctness

Semantic similarity is a candidate-reuse signal; **it does not establish that
an answer is fresh or operationally correct**. MemoRizz therefore admits
read-only, deterministic turns by default and automatically bypasses cache
writes for side-effecting tools. Entries include user/memory/session scope and
model, prompt, tool-schema, and data-version fingerprints, with domain-specific
freshness limits and hit provenance.

```python
stats = agent.semantic_cache_stats()
# hits, misses, bypasses, writes, evictions, size, last-hit provenance

removed = agent.invalidate_semantic_cache(
    domains=["inventory"],
    tags=["warehouse-7"],
    data_version="catalog-2026-08-12",
)
```

Invalidate a domain when its source data changes; do not use a high similarity
threshold as a substitute for invalidation or a freshness limit.

## Oracle compaction and operations

Oracle summaries persist canonical `source_message_ids`, `period_start`,
`period_end`, and `memory_units_count`, with normalized summary-to-message links
and a conversation `summary_id` marker. Summary creation and original-message
marking occur in one transaction. `retrieve_by_id(SUMMARIES, summary_id)` and
`expand_summary()` reconstruct the compacted history losslessly.

Use package-owned bootstrap and diagnostics instead of notebook setup code:

```python
from memorizz import LocalOracleRuntime, OracleProvider

runtime = LocalOracleRuntime.from_env(provision_if_missing=True)
runtime.ensure_ready()

provider = OracleProvider.from_env(
    provision_if_missing=True,
    index_policy="lazy",  # none | lazy | selected | eager
)
report = provider.preflight()
```

Preflight reports service/version/PDB state, privileges, embedding models and
dimensions, vector columns and indexes, `VECTOR_MEMORY_SIZE`, a sizing
recommendation, and exact-search fallback. Scope cleanup is transactional:

```python
counts = provider.delete_scope(
    memory_id="course-run-17",
    user_id="student-42",
    agent_ids=["planner", "executor"],
)
```

## Sandboxes and delegation

E2B uses one bounded `Sandbox.create(...)` session, so write → execute → read is
coherent. It fails at construction without an API key, normalizes SDK result
shapes, terminates through `close()`/context-manager semantics, and reports
provider and resource policy in `ExecutionResult.metadata`. See the
[E2B guide](../sandbox/e2b.md).

GraalPy subprocess mode is deliberately labeled as an execution provider, not
a strong sandbox. It confines MemoRizz file APIs, allowlists environment keys,
applies resource limits, and denies network or fails closed by default. Use the
package-shipped Java wrapper source only after compiling it against the matching
GraalVM release and validating the resulting JAR in your deployment. MemoRizz
verifies the wrapper class/manifest and supplies GraalVM's mandatory resource
limits; no prebuilt cross-version JAR is claimed. See the [GraalPy
guide](../sandbox/graalpy.md).

Multi-agent planning and decomposition use the configured `LLMProvider` and
model; there is no direct OpenAI call or hard-coded model. Deterministic plans
are supported, delegates are operational, tenant/request/tool/trace context is
propagated, shared memory is workflow-scoped, and partial/dependency failures
are returned in the orchestration report.

## Governed browser control

`browser_control` is a provider-neutral, opt-in capability. The Browser Use
implementation runs in a separately installed CLI environment and receives a
fixed MemoRizz wrapper through that tool environment's Python interpreter; the
model supplies only a JSON-encoded task and bounded step count. The worker
inherits an allowlist of environment variables, runs from a private temporary
directory, is terminated as a process group on timeout, and always closes its
browser.

Every model call is classified as nondeterministic and side-effecting, so it
enters the same durable approval lifecycle described above. Domain allowlists,
denylists, direct-IP blocking, headless/vision controls, step limits, and wall
timeouts are host policy—not model arguments. See [Browser
Control](../browser-control/index.md).

## Deployment checks

Do not infer feature availability from `memorizz >= 0.5.0` alone:

```bash
memorizz capabilities --json
memorizz oracle preflight --json
```

The Python equivalents are `memorizz.capabilities()` and
`agent.capability_report(preflight=True)`. They report package/provider
versions and effective feature states, including optional MCP, Oracle, E2B,
GraalPy, and isolated Browser Use dependencies.

For a repository/source deployment, run the release verifier against a staging
environment after loading secrets through environment variables:

```bash
PYTHONPATH=src python scripts/verify_production_050.py --json
python -m pytest -q
mkdocs build --strict
```

The verifier exercises the configured Oracle AI Database, in-database vectors,
OpenAI/Anthropic/Tavily, E2B's stateful file round trip, a durably approved
Browser Use task constrained to `example.com`, the first-party MCP stdio and
authenticated HTTP transports, tenant isolation, and hosted MCP authentication
boundaries. OAuth-success paths still require operator-owned Notion/Google
OAuth grants; an unauthenticated `authorization_required` result is a boundary
test, not a successful OAuth test. Use `--skip-browser` only when the isolated
CLI/browser is intentionally absent from that deployment.
