# Continual Learning: Workflows → Skills

MemoRizz agents can **learn from their own successful behavior**. Every
tool-calling run is captured as a workflow trajectory; when the same
procedure keeps succeeding across different queries, it is distilled into a
reusable **skill** — a SKILL.md document stored in the new `skillbox` memory
store — and injected into future runs that match. Skills are monitored for
their whole life and demoted when they stop working.

```
Agent run ─→ Workflow memory (per-run trajectory, canonical hash)
                 │  aggregation: frequency × success × recency × diversity
                 ▼
         Promotion Engine (gates → LLM distillation → validation)
                 │
                 ▼
         Skillbox (SKILL.md documents + applicability embedding)
                 │  vector search against the incoming query
                 ▼
         Context injection (top of the per-turn context block)
                 │
                 ▼
         Outcome monitoring → drift detection → demotion
                 │
                 └──── demoted skills release their workflows back to retrieval
```

## Enabling it

```python
from memorizz.memagent import MemAgent

agent = MemAgent(
    memory_provider=provider,
    llm_config={"provider": "openai", "model": "gpt-5-mini"},
    continual_learning=True,
    continual_learning_config={
        "min_executions": 5,          # gate: how many runs before eligible
        "min_success_rate": 0.80,     # gate: how reliable the procedure must be
        "min_distinct_queries": 2,    # gate: repeated *intent*, not one cached query
        "require_shadow": True,       # recommended initial posture (see below)
        "promotion_every_n_runs": 25, # 0 = only manual run_promotion_cycle()
    },
)
```

Or with the builder:

```python
agent = (
    MemAgentBuilder()
    .with_memory_provider(provider)
    .with_continual_learning(True, config={"require_shadow": True})
    .build()
)
```

`MEMORIZZ_CONTINUAL_LEARNING=1` flips the default on for every agent in the
process. Enabling the feature force-activates `workflow_memory` (the loop
cannot observe runs without it) and the `skillbox` store.

### Define business success

By default, a workflow succeeds when its tool loop finishes without an
exception. That is not enough for many applications: a payment tool can return
`{"accepted": false}` without raising, for example. Pass a domain callback so
promotion and drift monitoring use the same business outcome as the product:

```python
def evaluate_workflow(workflow):
    relevant_steps = list(workflow.steps.values())
    return all(
        step.get("result", {}).get("accepted", True)
        for step in relevant_steps
        if isinstance(step.get("result"), dict)
    )


agent = MemAgent(
    # ...model, tools, and memory_provider...
    continual_learning=True,
    workflow_outcome_evaluator=evaluate_workflow,
)
```

The evaluator receives the completed `Workflow` before it is stored. It may
return `True`/`False`, `WorkflowOutcome`, `"success"`/`"failure"`, or `None`
to preserve the execution outcome. An exception or unsupported return value
records a learning failure without breaking the user-facing run, and an
execution failure cannot be upgraded to success.

The callback is deliberately runtime-only because closures and service clients
are not safely serializable. Supply it again when loading an agent:

```python
agent = MemAgent.load(
    agent_id,
    provider,
    workflow_outcome_evaluator=evaluate_workflow,
)
```

The builder exposes the same behavior through
`.with_workflow_outcome_evaluator(evaluate_workflow)`.

## Five invariants

The whole system is governed by five rules:

1. **Always write, selectively retrieve.** Every run writes a workflow
   document — even runs already covered by a promoted skill. Workflow
   *retrieval* excludes covered trajectories (the skill is their compiled
   form); the fresh writes are the drift-detection evidence stream.
2. **Frequency alone never promotes.** Promotion requires executions ×
   success rate × recency × query diversity to clear explicit gates. A
   50×-repeated 60%-success trajectory is a bug report, not a skill.
3. **No skill gains context authority without passing validation.**
   Distillation is LLM generalization and can be wrong; a wrong skill with
   elevated prompt position *authoritatively misleads*.
4. **Skills are strong priors, not mandates.** Injected skills carry
   explicit precondition-check framing so the agent deviates on partial
   matches instead of forcing a near-miss procedure.
5. **Every skill is monitored for life and has a demotion path.**
   Continual learning requires forgetting.

## How trajectory identity works

Two runs are "the same procedure" when their **canonical signatures**
match: the ordered sequence of `(tool name, argument-key shape, error
class)` units, with retries collapsed. Argument *values*, results, and
timestamps are dropped — `lookup_order(order_id="A1")` and
`lookup_order(order_id="Z9")` are the same unit, and a
fail→fail→success retry burst on one tool is one unit. The signature's
sha256 is stored on every workflow document as `canonical_hash`.

Existing deployments should backfill hashes once before enabling
promotion:

```bash
python scripts/backfill_canonical_hashes.py --provider mongodb --dry-run
python scripts/backfill_canonical_hashes.py --provider mongodb
```

(Aggregation also hashes unhashed rows on the fly, so the backfill is an
optimization, not a correctness requirement.)

## The promotion pipeline

Each cycle (every N stored runs, or manually via
`agent.continual_learning_manager.run_promotion_cycle()`):

1. **Staleness sweep** — any ACTIVE skill referencing a tool that no
   longer resolves is deprecated immediately.
2. **Gates** — trajectory classes must pass ALL of: `min_executions`,
   `min_success_rate`, `min_distinct_queries` (a single verbatim query
   repeated 20 times is a semantic-cache candidate, not a skill), and a
   recency window (`max_days_since_last_seen`).
3. **Ranking** — eligible classes are scored
   `success_rate × log1p(executions) × 0.5^(days_since / halflife)`; the
   score never rescues a gate failure. Top
   `max_promotions_per_cycle` proceed.
4. **Distillation** — the most recent successful runs (plus up to two
   failures for the "failure modes" section) are sent to the LLM, which
   must emit a SKILL.md with YAML frontmatter (`name`, `description`,
   `preconditions`, `tools`), parameter-shape procedure steps (never
   literal values from source runs), and a "When NOT to apply" line — or
   the exact string `INSUFFICIENT`.
5. **Validation gate** — machine checks, all mandatory: frontmatter
   parses; every declared tool resolves in the toolbox; no undeclared tool
   appears in procedure steps; the description is intent, not a tool-name
   dump; content under the size cap; no literal argument value leaked from
   successful or failed source runs; and an LLM-judge check that sees both
   groups and asks whether the skill reproduces successes while avoiding or
   correctly handling failures. Rejections are logged with reasons and the
   trajectory stays eligible next cycle.
6. **Stamping** — on ACTIVE promotion, every workflow document of the
   class gets `promoted_skill_id`, which suppresses it from
   `retrieve_workflows_by_query` (pass `exclude_promoted=False` to see
   them for audit or debugging). New runs of the class are stamped at
   write time.

### Shadow mode

With `require_shadow=True` (the recommended initial posture), new skills
land as `SHADOW`: they are stored, monitored, and attributable, but never
injected and never suppress workflow retrieval. Review the distilled
SKILL.md by hand, then activate:

```python
manager = agent.continual_learning_manager
for skill in manager.skillbox.list_skills():
    print(skill.status, skill.name, "\n", skill.content)
manager.activate_skill(skill_id)   # SHADOW → ACTIVE + stamp backfill
```

## Injection is prompt-cache safe

MemoRizz assembles prompts stable-prefix → volatile-tail (see
[Context Efficiency & Prompt Caching](context-efficiency.md)). Learned
skills respect that:

- The **system prompt** gains one *static* section describing the
  learned-skills contract (precondition checking, priors-not-mandates).
  It never changes turn to turn, so the prompt cache keeps hitting.
- The **retrieved skills themselves** are rendered at the *top of the
  per-turn volatile block* — above retrieved memories — in the final user
  message, where per-turn variation invalidates nothing.

Retrieval matches **applicability, not mechanism**: the skill embedding is
generated only from its name, description, preconditions, and sampled
trigger queries — never from the procedure body or tool sequence. The
similarity threshold (`retrieval_min_similarity`, default **0.70**) is
deliberately stricter than any other retrieval in MemoRizz because skills
carry instruction authority: a false-positive costs more than a miss.

## One representation per prompt

MemoRizz never puts a learned skill and the raw workflows compiled into that
skill in the same model context:

- Automatic pre-inference retrieval considers knowledge-base and conversation
  memory, not workflow memory.
- `Workflow.retrieve_workflows_by_query()` excludes every promoted workflow by
  default. `exclude_promoted=False` is an explicit audit/debug escape hatch,
  not a prompt-injection setting.
- The final prompt assembler is a second correctness boundary. When a skill is
  rendered, it removes workflow context matching the skill's
  `promoted_skill_id`, canonical trajectory hash, or source workflow ID.
  Workflow context with missing provenance is also rejected because MemoRizz
  cannot prove that it is distinct.

Direct memory-provider reads bypass the workflow retrieval API. If application
code uses those reads, pass the result through normal MemAgent context assembly
rather than concatenating it into a prompt.

## Monitoring, drift, and demotion

Every stored run records `skills_activated` — the skill IDs that were in
its context. The monitor updates each skill's stats:

- **success / failure** — by run outcome;
- **deviation** — the skill was in context but the run took a different
  canonical path (signal that retrieval is miscalibrated; surfaced as a
  review flag when deviations exceed 50%);
- **rolling success rate** — a ring buffer of the last
  `drift_window_activations` outcomes.

When the rolling rate drops below `baseline − demotion_success_delta`
(baseline is frozen at promotion time), the skill is **demoted**: it stops
retrieving, and its workflows' stamps are cleared so the raw trajectories
are retrievable again. If the trajectory later re-qualifies — counting
only post-demotion runs — it is re-distilled as a new version, with the
old demotion reason fed into the distillation prompt.

## Inspecting learned skills

Learned skills surface through the same tools as file-based skills:
`list_skills` (tagged `"source": "learned"`) and `read_skill`. From code:

```python
manager = agent.continual_learning_manager
manager.skillbox.list_skills()                 # all, any status
manager.last_report                            # most recent PromotionReport
manager.run_promotion_cycle()                  # manual cycle
manager.promote_class(canonical_hash)          # gated single-class promotion
manager.activate_skill(skill_id)               # SHADOW → ACTIVE (+ stamp backfill)
manager.demote_skill(skill_id, reason="...")   # manual demotion
```

## The local UI: human-in-the-loop promotion

The local UI (`memorizz ui`) exposes the whole loop:

- **Memory → Workflows** groups runs into **trajectory classes** by
  canonical hash, showing per-class executions, success rate, distinct
  queries, and a pass/fail chip for each promotion gate. Eligible classes
  get a **"Distill now"** button; each agent section has a
  **"Run promotion cycle"** button, and the resulting promotion report
  (promoted / rejected-with-reasons / skipped) renders on the page.
  Classes covered by an active skill show a `PROMOTED` badge and their
  runs are marked `suppressed`.
- **Memory → Skills** shows every learned skill with its lifecycle badge
  (candidate / shadow / active / deprecated / demoted), version,
  activation stats, baseline, preconditions, and the full distilled
  SKILL.md — plus **Activate** (shadow → active, with workflow stamp
  backfill) and **Demote** (releases the suppressed workflows) buttons.

UI promotion is human-*triggered*, never human-*exempted*: "Distill now"
still runs the full eligibility gates and the distillation validation
gate — hand-picking a class can't buy it out of the evidence
requirements.

## Configuration reference

All knobs live in `continual_learning_config` (see `PromotionConfig`):

| Key | Default | Meaning |
| --- | --- | --- |
| `min_executions` | 5 | Gate: runs of a class before eligible |
| `min_success_rate` | 0.80 | Gate: class success rate |
| `min_distinct_queries` | 2 | Gate: distinct normalized user queries |
| `max_days_since_last_seen` | 30 | Gate: recency window |
| `recency_halflife_days` | 14 | Ranking decay half-life |
| `max_promotions_per_cycle` | 3 | Cap per cycle |
| `distill_sample_size` | 5 | Success runs fed to the LLM |
| `include_failure_samples` | 2 | Failure runs fed to the LLM |
| `skill_max_content_chars` | 4000 | SKILL.md size cap |
| `require_shadow` | False | New skills land as SHADOW |
| `retrieval_min_similarity` | 0.70 | Injection threshold |
| `max_skills_in_context` | 2 | Skills injected per turn |
| `include_exemplar` | False | Deprecated compatibility key; raw source runs are never co-injected |
| `drift_window_activations` | 10 | Rolling-rate window |
| `demotion_success_delta` | 0.25 | Demote when rolling < baseline − delta |
| `min_activations_before_drift_check` | 5 | Grace period |
| `promotion_every_n_runs` | 25 | Cycle cadence (0 = manual) |

## Measuring whether skills help

Don't take the feature's value on faith — measure it. The example notebook
(`examples/continual_learning/continual_learning_guide.ipynb`) ends with a
paired primary comparison plus a positive control:

- `capture_only` and `continual_learning` receive identical coached seed
  runs and the same later production prompt;
- only `continual_learning` receives the reviewed learned skill; and
- `explicit_sop_control` receives the full SOP directly but no learned skill.

Accuracy is graded from exact tool paths and tool-side business events, never
by an LLM judge. The notebook reports provider inference and end-to-end
latency separately, token use, Wilson intervals, and a descriptive paired
bootstrap interval.

In the saved live GPT-5.6 run, passive capture scored 4/10, the learned
skill scored 9/10, and the explicit-SOP control scored 9/10. The defensible
interpretation is that skill compilation recovered a taught SOP that was
deliberately absent from the production prompt. It is a purpose-built,
within-domain retention/selection test, not a preregistered or general claim
that continual learning improves every agent. The notebook records these
validity limits alongside all case-level outputs.

## Rollout recommendation

1. Enable behind `require_shadow=True` on one agent.
2. Backfill canonical hashes on existing deployments; check that top
   trajectory classes have sane counts (heavy fragmentation means the
   canonicalization rules need tuning before trusting promotion).
3. Review the first `PromotionReport` and every distilled SKILL.md by
   hand.
4. Loosen to `require_shadow=False` once distillation quality is trusted.
