# SWE-bench Lite

This adapter runs MemoRizz inside the official per-instance Docker image,
captures an official `model_patch` prediction, and invokes the unmodified
SWE-bench Docker grader. Gold and test patches are never placed in the agent's
prompt or tool context.

```bash
python eval/swe_bench_lite/evaluate_memorizz.py \
  --official-root /path/to/SWE-bench \
  --output-dir /tmp/memorizz-swe-lite \
  --instance-id sympy__sympy-20590 \
  --memory-backend filesystem
```

The agent uses task-scoped filesystem memory, workflow/tool logs, a safe
semantic cache (workspace-mutating turns bypass admission), post-run summary
compaction, a repeated read-only cache probe, and the host completion gate.
One-instance runs are smoke tests, not leaderboard scores.

The control-plane path is enabled by default and records bounded evidence,
immutable tool/workflow events, the host terminal-verification outcome, and an
incremental compilation report. Use `--legacy-memory-path` for a matched
legacy run or `--memory-backend oracle` for the same harness on Oracle.
