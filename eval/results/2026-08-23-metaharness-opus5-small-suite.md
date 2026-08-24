# MemoRizz MetaHarness small-suite evaluation with Claude Opus 5

## Outcome

On this bounded four-task regression, the **MemoRizz Risk-routed Panel** was
the only configuration to pass all 32 held-out checks. It used one harness call
per task, cost $0.389566, and completed in 236.718 seconds.

This supports a narrow operational claim: for these fixtures and these observed
executions, preregistered risk routing produced the best measured
cost-quality frontier. It does not establish that MemoRizz universally improves
model accuracy. The standalone and routed Opus calls differed on one identical
approval task, so the observed 3.125-point quality difference can be stochastic.

The source of truth is the
[sanitized JSON artifact](2026-08-23-metaharness-opus5-small-suite.json).

## Results

| Configuration | What it means | Model configuration | Memory | Hidden checks | Full tasks | Total latency | Cost | Tokens | Calls |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| MemAgent + Codex | Every task goes through one memory-grounded MemAgent to Codex | GPT-5.6 Luna, medium effort | Filesystem | 31/32 (96.875%) | 3/4 | 268.064 s | $0.041852 | 474,610 | 4 |
| MemAgent + Claude Code | Every task goes through one memory-grounded MemAgent to Claude Code | Claude Opus 5, medium effort | Filesystem | 31/32 (96.875%) | 3/4 | 190.259 s | $0.592389 | 94,996 | 4 |
| MemoRizz Always-on Panel | Luna repairs, then Opus reviews the resulting workspace on every task | GPT-5.6 Luna → Claude Opus 5, medium effort each | Filesystem | 31/32 (96.875%) | 3/4 | 610.007 s | $0.950665 | 566,302 | 8 |
| MemoRizz Risk-routed Panel | A fixed risk rule chooses Luna for ordinary contracts and Opus for security or tenant-isolation tasks | GPT-5.6 Luna or Claude Opus 5, medium effort | Filesystem | 32/32 (100%) | 4/4 | 236.718 s | $0.389566 | 215,873 | 4 |

The preregistered decision rule required the routed panel to equal or exceed
the better singleton, strictly exceed Codex-only, and cost less than
Opus-only. It passed:

- Accuracy was 3.125 percentage points above both singleton observations.
- Cost was 34.24% below Opus-only.
- It was 46.459 seconds slower than Opus-only, so it did not dominate Opus on
  latency.
- Compared with Codex-only, it was 11.69% faster and used 54.52% fewer reported
  tokens, but cost $0.347714 more.
- Compared with the always-on panel, it reduced cost by 59.02%, latency by
  61.19%, tokens by 61.88%, and calls by 50%, while passing one additional
  hidden check.

The always-on panel was not the better option here. More model calls did not
improve accuracy and materially increased every efficiency measure.

## Fairness and validity controls

The protocol was frozen before the valid run:

- Four independent synthetic repair fixtures were the units of observation.
- Every task-arm began from identical fixture bytes in a fresh workspace.
- Every arm used the MemAgent runtime interface, Filesystem memory, a unique
  tenant-scoped memory ID, and the same source-linked requirements.
- Both vendors were pinned to medium effort, 180 seconds, 12 steps, and 12,000
  output tokens per call. Claude additionally had a $0.35 per-call stop.
- All writes used a durable single-use host approval bound to the exact
  workspace fingerprint and execution envelope.
- Every run required host-side public verification.
- The primary score came from 32 held-out task-native checks. There was no LLM
  judge, coordinator model, or style score.
- Semantic-cache reuse was disabled because this was a cold comparison.
  Summarization and compaction were not triggered because the tasks were
  single-turn repairs.
- The risk router used fixture risk labels before execution: ordinary tasks
  went to Luna; security and tenant-isolation tasks went to Opus.

All 20 harness calls were grounded, approved, verified, and cost-accounted.
The runtime reported Claude Opus 5 directly. Codex was pinned through the CLI
argument to GPT-5.6 Luna.

## What separated the configurations

All three 31/32 configurations missed the same held-out approval behavior:
an empty approver identity raised ApprovalStateError rather than TypeError or
ValueError. The routed Opus execution handled this check, while the separate
Opus-only execution and the Opus stage of the always-on panel did not. With one
repeat, that is execution variance, not evidence that the router changed Opus
reasoning.

This is still useful platform evidence. The routed panel chose a cheap model
for ordinary work, reserved Opus for higher-risk work, preserved the same
memory/approval/verification contract, and avoided the guaranteed duplication
of the always-on panel. A production routing claim should be confirmed on a
larger held-out set with at least two counterbalanced repeats.

## Cost accounting

The valid medium-effort run cost $1.974472 across 20 harness calls. No judge
calls were made.

Before that run, one high-effort attempt was rejected after three records
because Opus used 9,347 output tokens against the original 6,000-token host
limit. Its repair passed 8/8 hidden checks, but the arm remained invalid because
the budget was exceeded before public verification. That diagnostic attempt
cost $0.359510 and is retained in the
[failed-attempt artifact](2026-08-23-metaharness-opus5-small-suite-failed-high-effort.json).
Known spend across the failed and valid attempts was therefore $2.333982.

Codex cost is an API-equivalent estimate from token telemetry and the recorded
GPT-5.6 Luna price snapshot. Claude cost is reported by Claude Code. The two
bases are explicit and should not be treated as identical billing evidence.

## Reproduce or inspect

The safe default writes a dry-run manifest and makes no external calls:

    python eval/metaharness/opus5_small_suite.py \
      --output /tmp/memorizz-opus5-dry-run.json

Paid execution is opt-in and keeps credentials outside arguments and artifacts:

    MEMORIZZ_RUN_HARNESS_EVALUATION=1 \
    MEMORIZZ_EVAL_CODEX_MODEL=gpt-5.6-luna \
    MEMORIZZ_EVAL_CLAUDE_MODEL=claude-opus-5 \
    python eval/metaharness/opus5_small_suite.py \
      --execute \
      --max-observed-cost-usd 3 \
      --output /tmp/memorizz-opus5-result.json

Notebook 6 in the MetaHarness examples shows the result table near the
beginning and reconstructs it from the JSON artifact at the end. The variables
OPUS5_RESULT_ROWS, OPUS5_RESULT_TABLE, and OPUS5_RESULT_REFERENCE make the table
and its artifact hash reusable in later notebook analysis.

## Claim boundary

This is a credible small regression, not a statistically powered benchmark,
not a Terminal-Bench or SWE-bench result, and not a model leaderboard. A safe
statement is:

> On four preregistered synthetic code-repair tasks, the MemoRizz risk-routed
> panel achieved 32/32 held-out checks at 34% lower observed cost than
> Opus-only and 59% lower observed cost than the always-on two-harness panel.
> One repeat is insufficient to attribute the quality difference to routing.
