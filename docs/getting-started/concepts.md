# Core Concepts

A useful mental model is:

```text
MemAgent = model + memory provider + runtime policies + optional capabilities
```

The model reasons, the provider persists, and the runtime decides what may be
retrieved, exposed, executed, cached, compacted, or learned.

## Identity and scope

Pass scope on every request in multi-user applications.

| Identifier | Meaning | Example |
|---|---|---|
| `agent_id` | Persisted agent definition | `support-agent-v2` |
| `memory_id` | Application or memory namespace | `acme-support` |
| `user_id` | Tenant/end-user isolation boundary | `user-42` |
| `thread_id` | One conversation or workstream | `ticket-918` |

`None` is an anonymous/legacy user scope, not a wildcard. Provider APIs that
support an omitted-filter sentinel distinguish “do not filter by user” from
“only rows whose `user_id` is null.” See
[Multi-Tenant Applications](../guides/multi-tenant.md).

## Memory types and providers

Memory types describe semantics. Providers implement persistence and query
behavior.

| Family | Units capture | Common runtime use |
|---|---|---|
| Episodic | Messages, conversations, summaries | Continuity and historical recall |
| Semantic | Knowledge, entities, personas | Facts and structured profiles |
| Procedural | Tools, workflows, skills | Reusable actions and learned procedures |
| Short-term | Working records and semantic-cache entries | Current-turn efficiency |
| Shared | Workflow-scoped coordination state | Multi-agent hand-offs |
| Operational | Tool logs, traces, agent definitions | Audit, expansion, and lifecycle |

All providers should preserve canonical identifiers, timestamps, metadata, and
scope. Semantic retrieval must apply tenant/agent/toolbox filters before vector
top-k selection—not after it.

## Storage is not context

Persisting a record does not mean sending it to the model. On each turn the
runtime retrieves candidates, applies scope and policy, deduplicates them,
selects a bounded set, and assembles a stable prompt prefix. Large tool results
can be replaced with an auditable pointer; summaries compact old history while
retaining links to their source messages.

This separation lets an application keep rich durable history without paying
to place every record in every prompt.

## Turn lifecycle

```text
host request and scope
  → semantic-cache admission / lookup
  → scoped memory and skill retrieval
  → bounded context assembly
  → model call and governed tool loop
  → response and operational evidence
  → memory, cache, summary, trace, and learning writes
```

Side-effecting tools bypass semantic-cache admission. Cache keys and matches
incorporate ownership, model/prompt/tool/data fingerprints, domains, TTL, and
freshness policy; vector similarity alone does not establish freshness.

## Application modes

`assistant`, `workflow`, and `deep_research` choose default memory families.
They do not change the provider security boundary or automatically enable
external capabilities. Explicit `memory_types` can override mode defaults.

## Trust boundaries

- The **host** owns credentials, tenant scope, approval decisions, tool
  registration, and policy configuration.
- The **model** may propose actions but cannot approve its own mutations.
- **MCP, browser, internet, and sandbox providers** are separate authority and
  execution boundaries; enable only the least privilege required.
- **Continual learning** consumes host-verified evidence and promotes workflows
  through explicit lifecycle policy. A repeated model claim is not verified
  success.
- **Observability** may contain sensitive operational metadata. Use content
  redaction, authentication, audit logging, and scoped access.

Next, build an agent in the [SDK quickstart](python-sdk-quickstart.md), then add
[tools and approval](../guides/tools-and-approvals.md) or review
[context efficiency](../guides/context-efficiency.md).
