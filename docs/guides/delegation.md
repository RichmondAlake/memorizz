# Safe parallel delegation

These contracts are available in Memorizz 0.9.0 and later.
Use `MemAgent.delegate(..., plan=plan, return_report=True)` for a host-approved
dependency graph. No second planner is needed for an explicit plan.

```python
from memorizz import CompletionPolicy
from memorizz.streaming import CancellationToken

# coordinator already has registered document/slides agents and a memory provider.
plan = [
    {"task_id": "document", "description": "Create the document from the bound source",
     "assigned_agent_id": "document-worker", "dependencies": []},
    {"task_id": "slides", "description": "Create slides from the verified document",
     "assigned_agent_id": "slides-worker", "dependencies": ["document"]},
]

stop = CancellationToken()  # the host's Stop handler calls stop.cancel()

def execute(coordinator, user_id, source_manifest, verify_receipt, collect_receipt, *, stop):
    result = coordinator.delegate(
        "Create the requested artifacts",
        memory_id="project-memory", thread_id="parent-thread", user_id=user_id,
        plan=plan, return_report=True,
        allow_root_fallback=False, thread_strategy="task", max_workers=2,
        timeout=120, cancellation=stop,
        context={"source_manifest": source_manifest},
        tool_context={"source_manifest": source_manifest},
        task_completion_policy=CompletionPolicy(enabled=True, validator=verify_receipt),
        on_task_result=collect_receipt,
    )
    return result
```

Configure `delegation={"mode": "deterministic", "consolidation_strategy":
"deterministic", "evidence_context": False}` on the coordinator when only
explicitly bound source data should guide this artifact workflow. Independent
branches can use empty dependency lists. Configure each worker's memory policy
and tool allowlist for the intended source boundary; thread separation is not
authorization and does not disable authorized knowledge-base retrieval.

## Choose delegates in the local UI

On an agent's create or edit page, **Delegates** lists your other saved agents.
Tick the ones this agent may hand work to. Each request is then split among them
by the agent's model (`mode="auto"`), each delegate works on its part as a
subagent, and the agent combines their results. The section also sets:

- **Delegate requests**: off keeps the delegates saved but answers every request
  directly (`enabled`).
- **Answer itself when a request can't be split** (`allow_root_fallback`, on by
  default here; the SDK default is off).
- **Combine results**: the agent's model writes one answer (`model`), or each
  delegate's result is listed in order without an extra model call
  (`deterministic`).
- **Delegates working at once**: 1–8 (`max_workers`); blank means all of them.

The form saves `delegates` (agent IDs) and `delegation_config`, and keeps other
configuration set in code, such as a fixed `plan`. It refuses an agent as its own
delegate and any delegate whose own delegates lead back to this agent. Loading
an agent also skips a delegate already in the chain being loaded, so a loop
saved by other means cannot hang it. Every delegate makes its own model calls,
so delegated requests cost more. In a `memagent` harness run each delegate shows
as a subagent in the run's trace.

## Harnesses as delegates

A delegate can be a coding harness instead of another model. An agent whose
meta-harness mode is `runtime` runs every task it is given on its default
harness, so a coordinator hands it parts of a request like any other delegate:

```python
from memorizz import MemAgent

reviewer = MemAgent(
    name="Codex reviewer",
    llm_config=llm_config,
    memory_provider=provider,
    meta_harness=True,
    meta_harness_mode="runtime",
    default_harness="codex",
    harness_config={"model": "gpt-6-luna"},  # optional: the harness default otherwise
)
coordinator = MemAgent(
    llm_config=llm_config,
    memory_provider=provider,
    delegates=[reviewer, writer],  # mix harness-backed and model-backed agents
    delegation={"enabled": True, "mode": "auto"},
)
```

In the local UI, open the coordinator's **Delegates** section and use **Add a
harness as a delegate**: pick a harness (Codex, Claude Code, pi, Hermes,
OpenHands, DeepSeek), optionally its model from the same dropdown the harness
page uses (the newest models of each provider it runs, or **Other model…**),
and a name. MemoRizz saves a new agent that runs on that harness and ticks it in
the list; save the form to keep it as a delegate. Harness-backed delegates are
marked "runs on Codex · gpt-6-luna" in the list. The **Agent harness runtime**
section's **Harness model** sets the same thing for any agent (saved as
`harness_config["model"]`; blank uses the harness default).

When to use which: a harness delegate brings its own agent loop, tools and
sandbox (shell, file edits, web search), so it suits code and file work in a
workspace. A model delegate runs on its own model with MemoRizz's tools and
memory, so it suits research, analysis and writing, and costs less per task.
Mixing both works well. A harness delegate created in the UI has no workspace of
its own: in a harness run (Agent Harnesses) it works in that run's workspace and
may use what that run was approved for.

In the playground, a coordinator with harness delegates shows a **Harness
delegates** card in the inspector's Overview tab, and a chip by the message box
says what they may use. By default each conversation gets its own fresh folder
and no web access (a delegate with its own workspace setting keeps using that).
To give them a folder of yours, web access or edits, fill in the card and press
**Allow**: web access and edits need your name, the folder must sit inside the
allowed workspace roots, and the grant is recorded as an approved
`metaharness.delegate_access` approval that lasts eight hours. Every message in
that conversation then passes it to the delegates, so their harness runs start
without asking again; **Stop using this access** goes back to the conversation's
own folder. When a grant has expired, the reply says so and the card asks you to
allow it again. The playground lists each delegate as it starts and finishes,
and each delegate's harness run appears on Agent Harnesses.

The same works over HTTP: `POST /api/agents/{agent_id}/harness-access`
(`{"workspace", "network": "none"|"full", "write", "approver_id"}`) returns the
grant, `GET /api/agents/{agent_id}/harness-access/{grant_id}` checks it (404 once
expired), `POST /api/agents/{agent_id}/harness-workspace` makes a fresh folder,
and the playground stream takes `harness_grant` or `harness_workspace` form
fields. In Python, `MetaHarness.grant_delegate_access()` creates a grant and
`MemAgent.run(..., tool_context={"harness_parent": grant})` passes it on.

## Admission and completion

The entire plan is checked before executing children: nonempty IDs/descriptions,
registered assignments, unique IDs, existing dependencies, no cycles, and only
pending tasks without prior results. Invalid explicit/callable plans produce a
failed report and never invoke the root. Automatic empty decomposition can run
the root only with `allow_root_fallback=True`; this defaults to false. Task
failures and recording failures never trigger a rerun of the original request.

Provider/runtime errors, unexpected model formats, exhausted iteration budgets,
and rejected completions fail the child. Approval-required children are `waiting`;
their consumers are `blocked`. Explicit unsuccessful `ToolResult` worker returns
also fail. Ordinary strings and application JSON are not business-verification
evidence: Memorizz does not infer successful payment or artifact creation from
English prose or an application-specific field.

Bind the existing `CompletionPolicy` to a trusted verifier that reads the saved
artifact and checks its owner, task, source version and nonempty contents.
Worker `completion_policy` supports bounded correction in the same model/tool
loop, including `require_tool_calls=True`. Reuse the existing artifact during
correction; the library does not provide application billing/idempotency leases.
The runtime `task_completion_policy` is an additional once-only worker-result
gate, not a retry loop. Its candidate metadata includes workflow/task/thread/user
IDs; it does not assert a tool-call count for arbitrary custom workers, so put
`require_tool_calls` on the worker's own policy. Both validators run in the worker.

## Scope and lifecycle callbacks

Task-scoped child threads are the default. Their IDs are stable for the same
workflow, user, memory, parent thread and task. Pass a unique host workflow ID in
`tool_context["workflow_id"]` to correlate a specific operation. This is not a
durable resume token: resubmitting a plan can execute it again. The old shared
conversation behavior is an explicit `thread_strategy="shared"` compatibility
option. Conversation history and episodic recall stay child-thread scoped.

The host user, workflow, task, trace and parent/child thread IDs are passed through
`get_tool_context()`; caller-supplied reserved values cannot override them.
Request/tool dictionaries are copied for each child. Cached agent memory-ID
lists are not modified by default. `persist_participants=True` explicitly opts
into registering/mutating participant associations and is unsuitable for shared
multi-tenant singleton agents without host coordination.

Each worker receives a separate copy of the caller's ContextVars, including the
cooperative cancellation token. The parent answer stream is deliberately cleared
so child prose cannot enter its chat stream. Use lifecycle callbacks for progress:

- `on_task_event(event)` receives `task_started` and `task_settled` in the worker;
  `task_finished`, blocked/cancelled events and `workflow_finished` run in the
  coordinator. Events carry `execution_context` and correlation IDs. Different
  workers can invoke callbacks concurrently; marshaling to a UI loop is the host's job.
- `on_task_result(event)` runs once in the worker's settlement path, including
  failures. It can read thread-local receipts and return JSON-safe metadata (or
  `None`). Metadata is explicitly transferred into `report["tasks"][i]["metadata"]`
  and the shared ledger. This is not a synchronous coordinator-thread hook.

Callbacks, cancellation tokens and task completion validators are runtime-only
arguments; they cannot be stored in the agent's `delegation` configuration.
Keep callbacks fast, thread-safe and nonblocking. Exceptions become recording
warnings without rerunning tools or discarding a successful result.

## Stop, deadlines and outcomes

Call `stop.cancel()` from the host. The coordinator uses bounded polling and
stops admitting work; cooperative workers check cancellation between model/tool
iterations. `timeout` is in seconds. Already running arbitrary Python/network
tools cannot be forcibly killed. Such a child is reported as `cancelling`, with
`execution_still_running=True`, and remains supervised until it settles.
Its late receipt is delivered by `on_task_result` and recorded as `task_settled`;
the original returned report is a snapshot, not a mutable completion promise.

The process permits at most 32 active delegated workers across workflows.
Lingering workers retain their slots; capacity exhaustion blocks additional
tasks. Use provider/tool timeouts too. Planning, callbacks, consolidation and
storage calls are synchronous and need their own timeouts; a deadline does not
interrupt arbitrary code or guarantee a hard wall-clock return bound. Process
shutdown still waits for Python executor threads; durable cross-process job
supervision, leases and external action cancellation belong to the host.

`outcome` is `succeeded`, `partial`, `failed`, `waiting` or `cancelled`.
`status` is `completed` for success (compatibility), otherwise the outcome.
`ok` means execution succeeded; `partial` requires at least one completed child
and an unsuccessful overall outcome. All-failed workflows are not partial.
`counts`, status and outcome are derived once for the report and persisted session.
No consolidation model is invoked if no child completed.

Nested MemAgent delegation propagates unsuccessful inner workflows to the parent
instead of returning success-shaped text. Its failure result retains the inner
`workflow_report`. Registered participants can atomically register sub-agents;
only the owning root commits the overall shared-session outcome. Nested reports
and ledger entries describe their own subgraph, not the parent's aggregate counts.

Inspect `recording_warnings` and `reconciliation_required` separately from `ok`.
They expose failed shared-memory writes, callback/recording errors and unfinished
workers. A successful paid effect with failed bookkeeping must be reconciled,
not repeated. Warnings arriving after the returned snapshot require the host's
late-settlement handling and operational logs.

## Atomic shared storage and testing

Shared blackboard appends, participant registration and status updates use the
provider's `compare_and_swap_shared_memory(id, expected_content, content)`.
MongoDB uses a conditional single-document update; Oracle uses exact CLOB
comparison; filesystem storage uses a SQLite cross-process lock and atomic file
replacement on a local filesystem. Unsupported providers fail explicitly rather
than falling back to unsafe read/replace. Custom providers must implement this
contract and preserve the exact `content` snapshot returned by reads.

Conflicts retry up to 16 times. `add_blackboard_entry(..., entry_id=stable_id)`
makes identical retries idempotent; reusing that ID for different content fails.
These guarantees require all shared-session writers to use the CAS contract;
unconditional external writes can still overwrite state. Network filesystem
locking and a shared ledger beyond provider document-size limits are not promised.

Run the permanent regression and artifact tests:

```sh
PYTHONPATH=src:. python -m pytest tests/integration/test_delegation_contract.py tests/integration/test_delegation_e2e.py tests/integration/test_shared_memory_atomic.py -q
```

Database variants are opt-in with `MEMORIZZ_DELEGATION_LIVE=1`. Use the existing
isolated MongoDB launcher with `--suite delegation`, or supply
`MEMORIZZ_OBS_TEST_ORACLE_USER`, `MEMORIZZ_OBS_TEST_ORACLE_PASSWORD` and
`MEMORIZZ_OBS_TEST_ORACLE_DSN` for a dedicated `MEMORIZZ_OBS_TEST_*` schema and
select `-k oracle`. The Oracle tests create/drop only their uniquely named tables.
No production accounts, paid model calls or host application are exercised.
Local artifact tests use persisted JSON document/slide fixtures, not rendered
office files. Validate real host delivery, ownership, billing and UI separately
before enabling production parallel artifact creation.
