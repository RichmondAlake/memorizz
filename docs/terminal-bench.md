# Terminal-Bench

MemoRizz includes an optional Harbor custom-agent adapter for local
Terminal-Bench evaluation. Install the benchmark dependencies first:

```bash
pip install "memorizz[terminal-bench]"
```

Create a zero-call budget forecast before starting Harbor:

```bash
memorizz eval terminal-bench forecast \
  --model openai/gpt-5.6-terra \
  --per-trial-spend-guard-usd 1.75 \
  --total-budget-usd 1000 \
  --output eval/results/terminal-bench-forecast.json
```

Without a pilot, the command reports a cost scenario from the dated public
Codex/Terra reference and deliberately withholds MemoRizz accuracy and rank.
Give it a normalized `--pilot-json` containing at least ten distinct tasks to
obtain a Wilson interval and an explicitly unofficial point-rank interpolation:

```json
{
  "trials": [
    {
      "task_id": "example-task",
      "reward": 1,
      "cost_usd": 0.74,
      "duration_seconds": 410
    }
  ]
}
```

Run a ten-task local Terminal-Bench 2.1 calibration without uploading results:

```bash
harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  -a memorizz.benchmarks.terminal_bench:MemorizzHarborAgent \
  -m openai/gpt-5.6-terra \
  --ak reasoning_effort=max \
  --ak max_cost_usd=1.75 \
  --ak max_wall_time_seconds=840 \
  --ak finalization_reserve_seconds=120 \
  -l 10 \
  -k 1 \
  -n 1
```

The adapter:

- runs MemoRizz and the model provider on the host;
- exposes bounded execution and final-verification tools backed by Harbor's
  isolated container;
- creates a separate filesystem memory store inside each trial log directory;
- writes cumulative token and estimated cost data to `memorizz-run.json`;
- stops issuing model calls after the configured per-trial spend guard;
- warns the model as its own wall-clock budget approaches, caps terminal calls
  to the remaining work window, and reserves time for a tool-free final answer;
- writes a native ATIF v1.7 `trajectory.json` for audit and comparison;
- never copies the model API key into the benchmark container.

The final response is now guarded by a host-side `CompletionPolicy`. The model
must make `terminal_verify` its last tool call, that command must be substantive,
and its return code must be zero. Rejected candidates remain in the same tool
loop with their original arguments and workspace state. The policy is bounded,
audited in `memorizz-run.json`, and fails closed after its retry budget. This
prevents a run from stopping after admitting that a measurable acceptance
criterion still fails.

Semantic caching is enabled for observability, but terminal tools are explicitly
governed as nondeterministic and side-effecting. The cache therefore bypasses
workspace-mutating turns instead of replaying stale engineering outcomes.

The spend guard is checked between model calls. A call already in flight can
place the final total slightly above the configured value. Cost is an estimate
using a small versioned registry for the selected model; unknown model pricing
fails closed. Pricing source and date are written to the ATIF and MemoRizz run
metadata, while the provider invoice remains authoritative.

Harbor currently enforces task timeouts around custom agents but does not pass
the resolved timeout into their constructors. The adapter therefore defaults to
an 840-second internal budget with a 120-second finalization reserve, which
finishes before the common 900-second Terminal-Bench limit. This does not change
the benchmark timeout or resources. Override the internal budget only when a
known task has a materially different official timeout. Cancellation and forced
finalization are recorded in both the run report and trajectory metadata.

The benchmark adapter selects MemoRizz's Responses API tool loop so GPT-5.6
function tools can run with non-zero reasoning effort, including `max`.

For leaderboard-style measurements, use the canonical dataset digest
`sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`,
do not override dataset timeouts or task resources, run all 89 tasks at least
five times (445 trials), retain all failed/error trials, and retain native ATIF
trajectories for successful trials. A $1.75 guard gives a nominal $778.75 upper
total and $221.25 headroom below a $1,000 budget, but in-flight overshoot means
this is not a hard invoice cap.

As of 23 August 2026, community submissions were closed and additions were
maintainer-run only. The forecast records that dated status; always check the
[current submission policy](https://github.com/harbor-framework/terminal-bench-2-1/blob/main/leaderboard/SUBMIT.md)
before paying for a complete run. The sanitized readiness evidence is in
`eval/results/2026-08-23-terminal-memory-benchmark-readiness.md`.
