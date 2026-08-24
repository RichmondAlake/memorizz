# MetaHarness evaluation protocols

These runners answer different questions. Do not merge their rows into one
leaderboard.

| Runner | Intended question | Evidence boundary |
|---|---|---|
| `provider_comparison.py` | Can a Codex-first adaptive panel stop when required structured coverage is complete, and is that behavior provider-stable? | Historical one-fixture diagnostic; the committed panel rows each executed one harness |
| `factorial_comparison.py` | What are the paired wrapper, mandatory coordination, adaptive-routing, and memory-provider effects? | One-fixture factorial diagnostic; no general winner or paper claim |
| `opus5_small_suite.py` | Can a preregistered one-call risk router improve the observed cost-quality frontier relative to Codex-only, Opus-only, and an always-on panel? | Four independent synthetic repair tasks, one repeat, Filesystem only, held-out task checks; descriptive rather than paper-comparable |

## Safe dry run

The default path writes the complete protocol manifest and makes zero external
model calls:

```bash
python eval/metaharness/factorial_comparison.py \
  --profile factorial \
  --output /tmp/memorizz-factorial-dry-run.json
```

The `smoke` profile prepares six Filesystem arms for one repeat. The
`factorial` profile prepares six strategies over Filesystem and Oracle for two
reverse-order-balanced repeats.

The Opus 5 suite has its own zero-cost manifest:

    python eval/metaharness/opus5_small_suite.py --output /tmp/memorizz-opus5-dry-run.json

## Paid execution

Authenticate the Codex and Claude Code CLIs, configure Oracle when using the
factorial profile, and keep credentials in the process environment. Review the
manifest and estimated call counts before opting in:

```bash
MEMORIZZ_RUN_HARNESS_EVALUATION=1 \
python eval/metaharness/factorial_comparison.py \
  --execute \
  --profile factorial \
  --max-observed-cost-usd 5 \
  --output /tmp/memorizz-factorial-result.json
```

The cost stop uses telemetry observed after completed calls, so it limits the
next call rather than guaranteeing a preflight quote. By default, missing
execution or judge cost invalidates the run.

For the bounded task-native comparison:

    MEMORIZZ_RUN_HARNESS_EVALUATION=1 MEMORIZZ_EVAL_CODEX_MODEL=gpt-5.6-luna MEMORIZZ_EVAL_CLAUDE_MODEL=claude-opus-5 python eval/metaharness/opus5_small_suite.py --execute --max-observed-cost-usd 3 --output /tmp/memorizz-opus5-result.json

This runner pins medium effort, uses no scalar judge, and validates exact
write-task readiness through the durable approval path before model spend.

## Interpretation

“Direct” means `MetaHarness.run()` without a MemAgent wrapper. This preserves
the same memory retrieval, policy envelope, output schema, and host verification
needed for a controlled wrapper comparison; it is not a raw vendor CLI arm.

Every candidate is judged in its own blinded call. Deterministic schema,
grounding, verification, known-defect coverage, usage, cost, and call-policy
evidence remain first-class. The synthetic fixture is the unit of inference;
reruns are repeated measurements, not independent tasks. Add held-out task
families, a second judge family, and human disagreement adjudication before
making broader claims.

The small Opus 5 suite uses independent tasks and task-native checks, closing
those two specific gaps. It still has one execution per task-arm, so differences
between two calls to the same model remain stochastic. Its safe claim is a
bounded observed frontier, not a causal or universal model-ranking claim. See
the [committed report](../results/2026-08-23-metaharness-opus5-small-suite.md).
