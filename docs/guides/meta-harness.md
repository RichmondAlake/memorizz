# Memory-First Meta-Harness

MemoRizz can run a workspace task through Codex, Claude Code, OpenHands, or a
native MemAgent while keeping memory, policy, approval, observability, and
continual-learning evidence in one control plane.

This is a *meta-harness*: it does not replace the vendor agent loop. It places a
stable host contract around several harnesses and normalizes their inputs,
events, results, and safety boundaries.

For a runnable learning path, see
[`examples/metaharness`](https://github.com/RichmondAlake/memorizz/tree/main/examples/metaharness).
Its six notebooks progress from an offline adapter contract and durable
approvals to a real MemAgent runtime and a Codex + Claude Code multi-agent
review team. The final two notebooks audit the centralized evaluation protocol
and construct its direct-harness and coordinated-MemAgent topology from
scratch.

```text
application / MemAgent / CLI / UI / MCP
                  |
           MemoRizz MetaHarness
      memory -> route -> policy -> approval
                  |
      Codex | Claude Code | OpenHands | MemAgent
                  |
       normalized events + host verification
                  |
       traces + verified learning evidence
```

## When a MemAgent runs on a harness

Yes, a MemAgent running on a harness is a useful separation of concerns. Two
modes are available:

| Mode | Owns the main reasoning loop | Use it when |
|---|---|---|
| `runtime` | Codex, Claude Code, or OpenHands | The external coding agent should complete the whole workspace turn |
| `delegate` | Native MemAgent | The MemAgent should decide when to call a bounded specialist harness tool |

In both modes, MemoRizz owns tenant scope, memory retrieval, the durable run
record, exact approval envelopes, cancellation, host-side verification, and
learning evidence. A vendor checkpoint is diagnostic in this release; approval
resume is supported, but resuming a vendor's conversational session is not yet
part of the public contract.

## Install the harnesses

Install only the CLIs that you intend to operate:

- [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Claude Code programmatic mode](https://code.claude.com/docs/en/headless)
- [OpenHands headless mode](https://docs.openhands.dev/openhands/usage/cli/headless)

The CLIs remain external executables; MemoRizz does not vendor or silently
install them. Confirm readiness with:

```bash
memorizz harness doctor
memorizz harness doctor codex
```

Claude runs with `--bare`, so OAuth and keychain authentication are deliberately
not read. Set `ANTHROPIC_API_KEY`, or configure one of Claude Code's supported
cloud-provider modes in the process environment. Codex may use its normal
`CODEX_HOME` authentication. OpenHands may use its normal CLI configuration,
but it will not be routed until the operator configures an external isolation
boundary.

### Missing credentials and login expiry

Capability probes and failed runs use the same secret-free error contract:
`error_code`, `error`, and `remediation`. Credential values are never included.

| Harness | A missing environment key means |
|---|---|
| Claude Code | Not ready: `--bare` disables OAuth/keychain fallback. Set `ANTHROPIC_API_KEY` or enable Bedrock, Vertex, or Foundry mode. |
| Codex | Not necessarily an error: an authenticated `CODEX_HOME` session may be used. A rejected/expired login is normalized when execution starts. |
| OpenHands | Not necessarily an error: provider or CLI configuration may supply credentials. MemoRizz separately requires an external isolation boundary. |

The result is identical across form factors:

- SDK: `probe()` returns capability details and `run()` returns a failed
  `HarnessResult` with `error_code="authentication_required"`;
- CLI: `harness doctor` and `harness run --json` print the same fields and a
  failed run exits non-zero;
- UI: the harness card displays setup guidance and the launch API returns HTTP
  409 with the durable failed run and structured error; and
- MCP: `memorizz_list_harnesses` reports adapters requiring authentication and
  `memorizz_start_harness_run` returns `ok=false`, the durable run, and the same
  structured error.

Start diagnosis with:

```bash
memorizz harness doctor claude-code --json
memorizz harness doctor codex --json
memorizz harness doctor openhands --json
```

Runtime responses such as HTTP 401, invalid key, login required, or expired
credentials are normalized to `authentication_required`; raw vendor output is
kept out of the public error message and secrets remain redacted from events.

## SDK: run one bounded task

Trusted-host read-only runs do not require an approval unless they request
secrets, expanded tools, network access, or governed MCP writes. A verification
command supplied through a model-visible delegate or MCP tool also requires
approval because it executes as host code; a verification command configured
directly by the SDK, CLI, or authenticated UI is already a host decision.

```python
from pathlib import Path

from memorizz import (
    HarnessPermissions,
    HarnessTask,
    MetaHarness,
    VerificationSpec,
)

repo = Path("./my-project").resolve()

with MetaHarness.from_env(allowed_workspace_roots=[str(repo)]) as harness:
    result = harness.run(
        HarnessTask(
            task="Inspect the authentication flow and report likely defects.",
            workspace=str(repo),
            harness="auto",
            memory_id="engineering",
            user_id="alice",
            thread_id="auth-review",
            permissions=HarnessPermissions(
                workspace_mode="read_only",
                network="none",
                mcp_access="read_only",
            ),
            verification=VerificationSpec(command="python -m pytest -q"),
        )
    )

print(result.status.value, result.verified)
```

`HarnessResult.ok` is true only when execution succeeded and any required host
verification passed.

For machine-consumed results, pass `output_schema={...}` on `HarnessTask`, or
place the same key in `with_execution_harness(..., config=...)`. MemoRizz binds
the schema into the durable task/approval envelope and uses each supported
CLI's native structured-output gate (`--output-schema` for Codex and
`--json-schema` for Claude Code). Routing fails with
`output_schema_unsupported` instead of silently treating prompt-only JSON as a
contract.

Token budgets apply to the provider's reported usage for the complete harness
turn, including reasoning and tool iterations—not just the visible final
paragraph. Choose limits from observed traces, keep a hard wall-time and action
limit, and treat `budget_exceeded` as a failed run even if the vendor process
produced a useful draft. Numeric usage fields remain observable; credential and
session-token values remain redacted.

When filesystem memories were created without vectors, MetaHarness retrieval
uses tenant-, memory-, and thread-scoped lexical fallback. A globally available
embedder therefore cannot silently hide an exact grounding record merely
because that record predates vector generation.

## Evaluate harness wrapping and coordination without confounding them

[`05_single_vs_multi_harness_evaluation.ipynb`](https://github.com/RichmondAlake/memorizz/blob/main/examples/metaharness/05_single_vs_multi_harness_evaluation.ipynb)
audits the earlier adaptive protocol and its committed artifact.
[`06_fair_harness_comparison_results.ipynb`](https://github.com/RichmondAlake/memorizz/blob/main/examples/metaharness/06_fair_harness_comparison_results.ipynb)
then constructs every adapter and MemAgent from scratch. Its corrected
`paired_factorial_v1` protocol crosses two memory providers with six execution
strategies:

1. direct `MetaHarness.run()` to Codex;
2. MemAgent to Codex;
3. direct `MetaHarness.run()` to Claude Code;
4. MemAgent to Claude Code;
5. a mandatory MemAgent panel that executes Codex and Claude Code; and
6. an adaptive MemAgent panel that executes Codex first and Claude only for
   host-identified structured-coverage gaps.

Here *direct* means direct use of MemoRizz's `MetaHarness`, without a MemAgent
wrapper. It is not a raw, ungoverned vendor CLI. Retaining the same scoped
memory context, permissions, schema, and host verification on both sides is
what makes direct-versus-wrapper a controlled estimate.

The panel delegates are runtime-backed MemAgents. The root uses deterministic
structured consolidation and has no LLM provider, so it cannot introduce a
hidden third model call. The full panel must record exactly one Codex and one
Claude run. The adaptive panel must prove Codex was its primary task and may
record one or two runs.

The notebook records independently judged quality, known-defect coverage,
unsupported claims, host verification, memory grounding, success, end-to-end
wall latency, usage tokens, normalized action lifecycles, output size, cost, and
cost per judge point. Each arm gets an isolated `memory_id` and identical base
requirements. Candidate identities, source IDs, and temporary paths are hidden
from the judge, and every candidate is scored in a separate stateless call to
avoid cross-candidate anchoring. The judge score remains secondary to
deterministic coverage, grounding, call-policy, and verification evidence.

The completed
[paid factorial evaluation](https://github.com/RichmondAlake/memorizz/blob/main/eval/results/2026-08-22-metaharness-factorial-paid.md)
demonstrates this boundary in practice. All arms reached 100% required-
criterion recall, but the scalar judge repeatedly used construct-irrelevant
style factors. MemoRizz retained those scores for audit, marked them invalid for
comparison, and based its efficiency conclusions on deterministic task gates.
Future failed judge attempts preserve their call and cost evidence before
validation.

Cost evidence is explicit about its basis. Claude Code supplies its reported
cost. Codex supplies token telemetry but not authoritative dollars, so the
evaluator calculates an API-equivalent estimate from a dated pricing snapshot.
Judge cost is evaluation overhead rather than an execution-arm cost. Unknown
model pricing remains `null` instead of being guessed.

Budget parity is per executed vendor harness. It does not mean every strategy
uses the same number of calls: a panel that escalates pays for a second harness,
while either baseline has exactly one. This is an orchestration comparison, not
equal-compute model benchmarking.

The comparison is fail-closed: every required worker and the panel root must be
grounded, every worker must report usage and cost, the schema and call policy
must hold, and host verification must pass before any candidate reaches the
judge. Host verification is an execution-health gate; it is not an answer
correctness oracle. A failed draft can remain in a diagnostic artifact but is
not ranked. The default notebook path makes no external calls; first inspect a
dry manifest:

```bash
python eval/metaharness/factorial_comparison.py \
  --profile factorial \
  --output /tmp/memorizz-factorial-dry-run.json
```

Then opt in to the paid path with:

```bash
MEMORIZZ_RUN_HARNESS_EVALUATION=1 \
python -m jupyter lab examples/metaharness
```

The paid cell runs the evaluator with two reverse-order-balanced repeats. It
creates both an isolated Filesystem provider and the configured Oracle provider
in the same paired run. Oracle preflight runs before any paid model call, its
version/vector/index capabilities are recorded in the artifact, provider-neutral
evidence fingerprints must match, and synthetic Oracle scopes are deleted at
cleanup. Treat Filesystem and Oracle results as separate provider strata rather
than pooling them silently.

One repeat is useful for adapter smoke testing only. More importantly, repeated
executions of one fixture are not independent task samples. The fixture is the
unit of inference, so a publishable comparison needs preregistered task
families, enough held-out tasks for task-level intervals, fixed model snapshots,
warm- and cold-cache conditions, another judge family, and blinded human
adjudication of disagreements. The evaluator therefore sets
`winner_allowed=false` for this fixture.

## Build a lean memory-first panel

More harnesses do not automatically produce a better answer. A panel is useful
when its workers have complementary responsibilities and the synthesis step is
grounded in the same authoritative evidence. MemoRizz therefore gives every
stage in a coordinated workflow one ranked, tenant-scoped evidence snapshot and
reuses that byte-identical snapshot across parallel delegates. The prompt keeps
the stable host contract and memory prefix first so provider prompt caching can
also reuse the common prefix. The root synthesizer reuses the same snapshot when
all delegate fingerprints match and it fits the root's evidence budget;
otherwise it performs a fresh scoped retrieval. Shared procedural evidence is
resolved under the root agent's ownership; a coincidentally reused workflow key
cannot expose one delegate's private skills or workflows to another agent.

This reuse lives above the `MemoryProvider` interface and applies to filesystem,
MongoDB, and Oracle. The MongoDB regression suite uses a real `mongomock`-backed
`MongoDBProvider` to verify exact `memory_id`/`user_id` scope, a bounded lexical
fallback when Atlas vector search is unavailable, degraded-retrieval
provenance, and no duplicate context build for the second workflow stage.

Configure the coordinator with a deterministic plan and bounded synthesis:

```python
from memorizz import MemAgentBuilder

coordinator = (
    MemAgentBuilder()
    .with_delegation(
        [codex_policy_reviewer, claude_verification_reviewer],
        enabled=True,
        mode="deterministic",
        plan=review_plan,
        evidence_context=True,
        max_dependency_context_chars=6_000,
        max_consolidation_result_chars=8_000,
        consolidation_strategy="model",
    )
    .with_learning_control_plane(
        True,
        config={
            "evidence_token_budget": 900,
            "evidence_max_items": 4,
            "compile_async": False,
        },
    )
    .build()
)

report = coordinator.run(
    task,
    memory_id="review-memory",
    user_id="alice",
    thread_id="change-481",
    context={"cache_data_version": git_commit_sha},
)
```

The four consolidation strategies make the quality/cost trade-off explicit:

| Strategy | Extra model call | Recommended use |
|---|---:|---|
| `model` | One | Merge genuinely complementary findings; the coordinator receives authoritative memory and may not invent claims |
| `structured` | None | Parse, allowlist, deduplicate, and union typed findings by criterion ID without asking another model to rewrite evidence |
| `deterministic` | None | Preserve all worker outputs as a structured report |
| `primary` | None | Return one named primary worker verbatim and keep the remaining reviews as evidence |

For bounded review or evaluation workloads, combine `structured` consolidation
with an adaptive escalation task. The first reviewer must return a JSON
`findings` list whose entries contain `criterion_id`, `title`, `finding`,
`severity`, `evidence`, `missing_tests`, and `source_ids`. MemoRizz checks the
IDs in host code. It invokes the fallback reviewer only when the primary output
is invalid, a primary task fails, or required coverage is missing:

```python
criteria = {
    "authorization": "role and ownership authorization",
    "expiry": "expiry precedence and timezone behavior",
    "verification": "reliable verifier semantics and missing tests",
}

coordinator = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_delegation(
        [codex_reviewer, claude_fallback],
        enabled=True,
        mode="deterministic",
        plan=[primary_task, fallback_task],
        return_report=True,
        consolidation_strategy="structured",
        required_finding_ids=list(criteria),
        adaptive_escalation={
            "enabled": True,
            "escalation_task_ids": ["fallback-review"],
            "criterion_descriptions": criteria,
        },
    )
    .as_ephemeral()
    .build(validate=False)
)
```

`report["execution"]` records whether escalation occurred, primary and
fallback latency, covered/missing IDs, and parse errors. Skipped fallback tasks
remain visible with `status="skipped"`; they do not count as workflow failures.
`report["consolidation"]["coverage"]` contains the deterministic union and
ignored out-of-contract IDs. Use model synthesis only when the task truly
requires prose reasoning across worker outputs.

`as_ephemeral()` suppresses automatic agent/tool configuration registration
for short-lived workers while retaining memory retrieval, run and trace
evidence, verification outcomes, and learning-control-plane records.

Use semantic caching for a *warm repeated workload*, not to improve a cold
one-repeat comparison. Delegation cache admission is fail-closed: the plan must
be deterministic, every delegate must be a verified read-only runtime harness,
network access must be disabled, and the caller must provide an explicit data
version such as a Git commit. The plan, delegate instructions, harness policy,
tool/model configuration, tenant, and version all contribute to the cache
identity. Configure the embedding provider explicitly; otherwise the default
global embedder may use a metered service.

Memory-first means selecting the right memory mechanism for the workload, not
turning every mechanism on for every call:

| Capability | Coordinated review policy |
|---|---|
| Ranked retrieval | Use on every cold run; scope before ranking and share one evidence snapshot |
| Provider prompt cache | Reuse the stable host and memory prefix; keep role-specific instructions later |
| Semantic cache | Use only for exact-policy, versioned, verified read-only repeats |
| Summaries | Retrieve an existing summary when it is the best evidence |
| Compaction | Run between long or repeated threads, never as overhead inside a one-shot review |
| Continual learning | Record verified delegate and root workflow evidence; promote only after repeated successful evidence and shadow gates |
| Forgetting | Apply asynchronously through a reviewed plan; never delete evidence merely to make one turn faster |

### Why the earlier panel cost more

The first valid one-repeat notebook diagnostic measured the Codex-only arm at
98/100 and about $0.00852 API-equivalent cost, the Claude-only arm at 98/100 and
$0.091038 reported cost, and the original panel at 87/100 and $0.141206 mixed
reported/estimated cost. These are local smoke-test measurements, not
leaderboard results.

The panel paid for two overlapping full reviews and then a third coordinator
call. Its Claude worker produced 7,867 output tokens versus 5,345 alone (about
47% more), because the adversarial role encouraged unrelated API and typing
tangents. The coordinator then synthesized delegate prose without independently
retrieving the original requirements, omitted one supported finding, and added
unsupported ones. Panel wall time also included the slowest parallel worker plus
the serial synthesis call.

The first optimization assigned bounded roles, shared one evidence pack, and
removed the coordinator-model call. It demonstrated useful adaptive behavior,
but it did not execute Claude in the panel rows and therefore could not estimate
two-harness coordination value.

The provider-comparison runner goes one step further: its
`structured_adaptive_v1` protocol uses the same structured output contract and
per-harness resource envelope for every arm, removes the coordinator model
call, and pays for Claude fallback only when Codex misses structural coverage.
It also records `phase_timings_ms` for context, MCP and output-schema setup,
adapter execution, verification, workspace finalization, and total runtime. Each context pack has
both an identity fingerprint and a provider-neutral `content_fingerprint`, so
the Filesystem/Oracle comparison fails closed if their panel workers did not
receive equivalent evidence content.

The corrected two-repeat run exercised that protocol end to end. Both panels
covered all six required findings without escalating to Claude or invoking a
synthesis model. Relative to the earlier panel implementation, mean execution
cost fell 87.1% on Filesystem and 85.8% on Oracle; gold coverage was maintained
or improved. Latency did not improve against the earlier stochastic sample, so
the report treats cost, quality, and latency as separate outcomes. See the
[optimized provider-comparison report](https://github.com/RichmondAlake/memorizz/blob/main/eval/results/2026-08-22-metaharness-provider-comparison-optimized.md)
for the complete protocol, results, limitations, and evidence boundary.
The companion
[`06_fair_harness_comparison_results.ipynb`](https://github.com/RichmondAlake/memorizz/blob/main/examples/metaharness/06_fair_harness_comparison_results.ipynb)
constructs both direct interfaces, both one-harness MemAgents, and the full and
adaptive model-less roots before building the corrected 12-arm matrix. It
reconstructs the historical four-arm table only to narrow its interpretation;
it does not invent corrected scores before a new paid artifact exists.

Notebook 06 now also includes the independent-task
[Claude Opus 5 small-suite report](https://github.com/RichmondAlake/memorizz/blob/main/eval/results/2026-08-23-metaharness-opus5-small-suite.md).
That regression holds Filesystem memory constant, pins GPT-5.6 Luna and Claude
Opus 5 to medium effort, uses one fresh workspace and scoped memory record per
task-arm, and scores 32 held-out task-native checks without an LLM judge. The
risk-routed panel uses one call per task; the always-on panel uses two, so their
compute difference remains explicit. The observed router result is descriptive:
one routed Opus execution passed a check that another Opus execution missed,
and one repeat cannot attribute that variance to routing.

## SDK: exact approval for edits

A direct-edit task pauses before the adapter starts. The proposal binds the
tool name, complete task arguments, workspace fingerprint, permissions,
budgets, verification command, scope, and expiry to a single-use ID.

```python
from memorizz import HarnessPermissions, HarnessTask, MetaHarness

harness = MetaHarness.from_env(allowed_workspace_roots=[str(repo)])
pending = harness.run(
    HarnessTask(
        task="Fix the failing parser tests without changing the public API.",
        workspace=str(repo),
        harness="codex",
        permissions=HarnessPermissions(
            workspace_mode="direct",
            network="none",
            mcp_access="read_only",
        ),
        verification={"command": "python -m pytest -q tests/test_parser.py"},
    )
)

proposal_id = pending.checkpoint["proposal_id"]
harness.approve(proposal_id, approver_id="release-owner@example.com")
result = harness.resume_approval(proposal_id)
```

The original checkpoint is executed; the model is never asked to reconstruct
approved arguments. A changed workspace invalidates the approval. Canceling a
pending or already-approved run prevents later execution. An existing dirty Git
worktree is rejected for write runs unless `allow_dirty_workspace=True` is part
of the approved envelope.

## Configure a MemAgent

Runtime mode returns the external harness's final response from `agent.run()`:

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_name("Repository maintainer")
    .with_memory_ids("maintainer-memory")
    .with_execution_harness(
        "codex",
        config={
            "workspace": str(repo),
            "permissions": {
                "allowed_roots": [str(repo)],
                "mcp_access": "read_only",
            },
            "verification": {"command": "python -m pytest -q"},
        },
    )
    .build_and_save()
)

answer = agent.run(
    "Find the regression and fix it.",
    memory_id="maintainer-memory",
    user_id="alice",
    thread_id="issue-481",
)
```

Delegate mode registers `run_harness_task`, `get_harness_run`, and
`list_agent_harnesses` as governed MemAgent tools:

```python
agent = (
    MemAgentBuilder()
    .with_name("Engineering coordinator")
    .with_meta_harness(mode="delegate", default_harness="auto")
    .build()
)
```

The native recursion guard prevents a MemAgent from routing either a delegated
task or a full turn back into itself. A different saved MemAgent can still be
selected explicitly.

## CLI

```bash
# Create the secret-free adapter configuration.
memorizz harness init
memorizz harness config --json

# Read-only analysis.
memorizz harness run \
  "Find the cause of the failing tests" \
  --workspace "$PWD" \
  --harness auto \
  --mcp-access read_only \
  --verify "python -m pytest -q" \
  --timeout 900 \
  --max-steps 80

# Run one saved MemAgent explicitly through the same durable contract.
memorizz harness run \
  "Review the release evidence" \
  --workspace "$PWD" \
  --harness native \
  --agent-id AGENT_ID

# A write run returns pending_approval.
memorizz harness run "Implement the fix" --workspace "$PWD" --write --json
memorizz harness approvals --status pending
memorizz harness approve PROPOSAL_ID --approver operator@example.com

# Operations and evidence.
memorizz harness runs --json
memorizz harness show RUN_ID --events --json
memorizz harness events RUN_ID --after 10 --json
memorizz harness cancel RUN_ID
memorizz harness retry RUN_ID
```

Create a persisted harness-backed agent without post-build mutation:

```bash
memorizz agents create \
  --name "Repository maintainer" \
  --no-llm \
  --harness-mode runtime \
  --default-harness codex \
  --harness-workspace "$PWD" \
  --json
```

Agent creation validates that the selected adapter is registered, but it does
not require that vendor CLI or its credentials to exist on the machine creating
the record. This keeps persisted agent configuration portable across developer,
CI, and production hosts. `memorizz harness doctor` reports host readiness, and
every execution still fails closed with the structured remediation above when
the executable or authentication is unavailable.

## Local UI

Start the UI and open **Agent Harnesses**:

```bash
memorizz ui
```

The page exposes adapter health, a bounded launch form, the host approval
queue, cancellation, a durable run ledger, normalized event inspection, usage,
workspace changes, and verification evidence. Agent create/edit pages persist
runtime or delegate mode, the default harness, and a workspace allowlist.

Set `MEMORIZZ_UI_AUTH_TOKEN` before exposing the UI beyond a trusted local
machine. `MEMORIZZ_UI_READ_ONLY=true` blocks all launch, decision, resume, and
cancel endpoints, not only their visible controls.

## First-party MCP server

The MemoRizz MCP server exposes model-usable discovery and execution tools:

- `memorizz_list_harnesses`
- `memorizz_start_harness_run`
- `memorizz_get_harness_run`
- `memorizz_list_harness_runs`
- `memorizz_get_harness_events`
- `memorizz_cancel_harness_run`

Approval and rejection remain host-side decisions. The model-visible start
schema has no `approved` or `confirm` Boolean. Configure workspace roots and
enable execution explicitly for remote HTTP:

```bash
memorizz mcp serve \
  --transport streamable-http \
  --allow-writes \
  --allow-agent-execution \
  --allow-harness-execution \
  --harness-workspace-root /srv/workspaces/project-a
```

Every remote run is bound to the authenticated MCP principal as `user_id`.
Run lookup, event lookup, listing, and cancellation apply the same tenant
filter.

## Adapter policy matrix

| Adapter | Local file boundary | Network modes | Cost/token budgets | Important constraint |
|---|---|---|---|---|
| Codex | Codex `read-only` or `workspace-write` sandbox | `none` | token telemetry; no host cost estimate | User config is ignored; host overrides disable project hooks, web search, extra writable roots, egress, and non-MemoRizz MCP servers |
| Claude Code | Hard-denied Bash; explicit read/edit/web tool rules | `none`, `full` | cost and token telemetry plus a native turn cap | `--bare` requires environment/cloud authentication |
| OpenHands | Operator-provided isolated wrapper | wrapper-defined | wall time and normalized action count | Headless mode is always-approve and never runs locally by default |
| Native MemAgent | Existing MemAgent tool/provider policies | `none` declaration | configured MemAgent step limit; no generic cost/token telemetry | Standalone surfaces require explicit `harness=native` plus a saved `agent_id`; self-routing is rejected |

The in-process native adapter is a composition convenience, not a process
isolation boundary. Cancellation is guaranteed before the turn starts; once a
model-provider call is in flight it is cooperative, and the provider owns its
timeout. Use Codex, Claude Code, or an isolated OpenHands worker when the host
must be able to terminate a process independently.

`restricted` network mode is reserved for custom or externally isolated
adapters that can enforce an allowlist. Bundled Codex and Claude adapters do not
claim this mode. Model-provider API traffic is part of the harness control
channel; the network policy governs agent tools and workspace processes.

OpenHands must be configured with a wrapper executable that actually enters a
Docker, remote, or sandbox boundary. A task's `execution_backend="docker"`
label alone is rejected:

```json
{
  "version": 1,
  "adapters": {
    "openhands": {
      "command": "/usr/local/bin/openhands-isolated",
      "enabled": true,
      "external_isolation": true
    }
  }
}
```

The wrapper must preserve OpenHands CLI arguments and enforce filesystem,
process, CPU, memory, and egress policy itself.

## Configuration

`memorizz harness init` writes `$MEMORIZZ_HOME/harnesses.json` with mode `0600`.
It contains no credentials:

```json
{
  "version": 1,
  "allowlist": ["memagent", "codex", "claude-code", "openhands"],
  "preference": ["memagent", "codex", "claude-code", "openhands"],
  "allowed_workspace_roots": ["/srv/workspaces"],
  "context_max_chars": 24000,
  "adapters": {
    "codex": {"command": "codex", "enabled": true, "model": null},
    "claude-code": {"command": "claude", "enabled": true, "model": null},
    "openhands": {
      "command": "openhands",
      "enabled": true,
      "model": null,
      "external_isolation": false
    }
  }
}
```

When no root is configured, environment-backed services restrict workspaces to
their current working directory. An explicit SDK-constructed `MetaHarness`
may supply its own roots.

| Variable | Purpose |
|---|---|
| `MEMORIZZ_CODEX_COMMAND` / `MEMORIZZ_CODEX_MODEL` | Codex executable and default model |
| `MEMORIZZ_CLAUDE_CODE_COMMAND` / `MEMORIZZ_CLAUDE_CODE_MODEL` | Claude executable and default model |
| `MEMORIZZ_OPENHANDS_COMMAND` / `MEMORIZZ_OPENHANDS_MODEL` | OpenHands wrapper and default model |
| `MEMORIZZ_OPENHANDS_EXTERNAL_ISOLATION` | Operator attestation that the wrapper enforces isolation |
| `MEMORIZZ_HARNESS_ALLOWLIST` | Comma-separated adapter allowlist |
| `MEMORIZZ_HARNESS_CONTEXT_MAX_CHARS` | Upper bound for rendered memory context |

Pin Claude Code reasoning effort per task through the SDK harness metadata:

```python
agent = (
    MemAgentBuilder()
    .with_execution_harness(
        "claude-code",
        config={
            "model": "claude-opus-5",
            "metadata": {"claude_effort": "medium"},
        },
    )
    .build(validate=False)
)
```

Accepted values are `low`, `medium`, `high`, `xhigh`, and `max`.
Invalid values fail before the CLI launches. Record effort alongside the model
ID in evaluation manifests because Claude Opus 5 defaults to high effort in
Claude Code, which materially changes cost, latency, and output-token use.

For MCP-specific enablement and roots, see the [MCP server guide](mcp-server.md).

## Memory and continual learning

Before execution, MemoRizz retrieves a bounded, tenant-scoped context pack from
summaries, conversation, entity, workflow, Skillbox, and knowledge-base memory.
Records must match `memory_id`, exact `user_id`, and compatible `thread_id`
before they enter the prompt. Credentials, embeddings, raw checkpoints, and
oversized records are removed or bounded.

After execution, normalized events and the outcome enter observability. A run
also produces immutable run, workflow, and outcome events in the learning
control plane. Only host-verified outcomes are authoritative for continual
learning and workflow-to-skill promotion. An unverified success remains useful
operational evidence but cannot establish that a procedure is correct.

## Durability and deployment boundary

The built-in run store is transactional SQLite with WAL mode, cross-process
event sequencing, persistent cancellation requests, heartbeats, and one writer
lease per workspace. It is a reliable single-node/self-hosted boundary, not a
distributed queue. Use one worker owner for execution, keep the database on
durable local storage, and call `recover_interrupted_runs()` only when that
worker is known to be stopped.

MongoDB and Oracle remain the memory and learning-evidence providers; harness
run coordination itself stays deliberately small and local in this release.
