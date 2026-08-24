# Continual Learning: Workflows → Skills

MemoRizz agents can **learn from their own successful behavior**. Every
tool-calling run is captured as a workflow trajectory; when the same
procedure keeps succeeding across different queries, it is distilled into a
reusable **skill** — a SKILL.md document stored in the new `skillbox` memory
store — and injected into future runs that match after any required review.
Active skills are monitored and demoted when they stop working. Shadow skills
can instead be evaluated passively on later production workflows without
being injected or executing anything again.

When the [memory-first learning control plane](learning-control-plane.md) is
enabled (the default alongside continual learning), workflow and skill
lifecycle transitions are also captured as immutable events. The control
plane does not replace promotion or Skillbox; it supplies a bounded recall
path, trace-linked outcome evidence, deterministic projections, and reversible
forgetting around the existing loop.

```
Agent run ─→ Workflow memory (per-run trajectory, canonical hash)
                 │  aggregation: frequency × success × recency × diversity
                 ▼
         Promotion Engine (gates → LLM distillation → validation)
                 │
                 ▼
         Skillbox (SKILL.md documents + applicability embedding)
            ┌────┴─────────────────────────────────────────────┐
            │ SHADOW (optional passive post-run evaluation)    │
            │ no prompt injection, LLM judge, or tool call     │
            └────┬─────────────────────────────────────────────┘
                 │ explicit activation review
                 │ vector search against the incoming query
                 ▼
         Trust-aware injection (user context or reviewed developer instruction)
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
        "skill_injection_role": "user", # "user" (default) or "developer"
        "shadow_evaluation_enabled": True, # opt-in passive observations
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

### Memory-provider support

The complete capture → canonicalize → promote → retrieve → monitor → demote
loop uses the `MemoryProvider` contract and works with filesystem, MongoDB,
and Oracle. Filesystem and MongoDB persist `injection_role` in the skill
document. Oracle persists it in the relational `skillbox.injection_role`
column and validates `user | developer`.

Fresh Oracle setup includes the column. For an existing manually managed
schema, run:

```sql
@src/memorizz/memory_provider/oracle/migrations/002_add_skill_injection_role.sql
@src/memorizz/memory_provider/oracle/migrations/003_add_shadow_evaluations.sql
```

Legacy skill records on every provider default to `user`, preserving the
pre-upgrade trust level. Migration 003 adds the JSON-constrained
`workflow_memory.shadow_evaluations` CLOB used as passive evaluation's
auditable source of truth. Filesystem and MongoDB serialize the same workflow
field without a schema migration.

Passive retrieval filters lifecycle, agent, and user scope before final top-k
selection on all first-party providers: filesystem filters documents before
cosine ranking; MongoDB supplies `status`, `agent_id`, and `user_id` to
`$vectorSearch` and reconciles those Skillbox index filter fields; Oracle puts
the same predicates into its vector-search SQL. A third-party provider written
against the older contract remains compatible through an over-fetch-and-filter
fallback. This retrieval runs only in the background worker.

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
3. **No generated skill gains application authority without validation and
   review.** Distillation is LLM generalization and can be wrong. MemoRizz
   rejects `skill_injection_role="developer"` unless `require_shadow=True`;
   activation is the explicit approval boundary.
4. **Authority is scoped, not absolute.** User-authority skills are strong
   priors. Developer-authority skills are reviewed application procedures,
   but system policy, preconditions, current facts, and tool results still
   win. A partial match must never force a near-miss procedure.
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
land as `SHADOW`. They are stored but never injected, never written to
`skills_activated`, never included in active drift statistics, and never
suppress workflow retrieval.

Passive shadow evaluation is separately opt-in:

```python
continual_learning_config={
    "require_shadow": True,
    "shadow_evaluation_enabled": True,
    "shadow_evaluation_max_candidates": 3,
    # None means use retrieval_min_similarity:
    "shadow_evaluation_min_similarity": None,
    "shadow_evaluation_queue_size": 100,
    "shadow_evaluation_recent_window": 50,
    "shadow_readiness_min_observations": 10,
    "shadow_readiness_min_trajectory_match_rate": 0.80,
    "shadow_readiness_min_matched_success_rate": 0.80,
}
```

After a tool-calling workflow is stored, the response path performs only a
bounded `put_nowait()` into an in-process queue. A single daemon worker then:

1. retrieves semantically matching `SHADOW` skills under exact `agent_id` and
   `user_id` scope;
2. excludes source workflows and every workflow created at or before the
   skill;
3. deterministically compares the expected and observed canonical hashes plus
   the already-recorded business outcome;
4. appends a versioned record to `Workflow.shadow_evaluations`; and
5. reconciles `Skill.stats["shadow"]` from those workflow records.

The worker does not build model context, call an LLM judge, call a tool, invoke
a second agent, or alter lifecycle status. Semantic retrieval can use the
configured embedding provider, but that work happens only after the response
path has enqueued the snapshot. Queue-full, retrieval, persistence, and worker
failures are logged without raw queries, skill bodies, or tool results and
never reach the user-facing run. Use the bounded drain only in tests or
controlled shutdown:

```python
manager = agent.continual_learning_manager
manager.drain_shadow_evaluations(timeout=5.0)
readiness = manager.get_shadow_readiness(skill_id)
print(readiness)
```

A readiness response is advisory:

```python
{
    "ready": False,
    "observations": 7,
    "trajectory_match_rate": 0.86,
    "matched_success_rate": 1.0,
    "reasons": ["observations 7 < 10"],
}
```

`trajectory_mismatch` is not automatically a skill failure. A different
successful canonical path may be a valid alternative procedure, so success
and failure counters apply only when the observed trajectory matches the
skill's source trajectory. These are observational metrics: they do not prove
that injecting the skill caused an improvement. Establish causal benefit with
a reviewed canary or A/B test before broad activation.

Review the distilled SKILL.md and its evidence by hand, then activate:

```python
manager = agent.continual_learning_manager
for skill in manager.skillbox.list_skills():
    print(skill.status, skill.name, "\n", skill.content)
manager.activate_skill(skill_id)   # SHADOW → ACTIVE + stored role + stamp backfill

# Or, instead of the call above, explicitly choose developer authority at review:
# manager.activate_skill(skill_id, injection_role="developer")
```

## Trust-aware instruction authority

MemoRizz assembles prompts stable-prefix → volatile-tail (see
[Context Efficiency & Prompt Caching](context-efficiency.md)). Every skill
stores an `injection_role`:

- `user` is the default and the upgrade behavior for legacy skill records. The
  skill is rendered at the top of the final user's volatile context, above
  ordinary recalled memories. Treat it as selectively retrieved guidance.
- `developer` is opt-in and requires `require_shadow=True`. After an explicit
  activation review, MemoRizz emits the matching skill as a separate
  developer message. It is application instruction below the system message
  and above the user's request—not system policy and not trusted database
  state.

The **system prompt** contains only a static contract explaining these trust
boundaries, precondition checks, and monitoring. Retrieved skill text never
changes that system prompt.

Provider mapping is explicit:

- The official OpenAI API receives the native `developer` role. This follows
  OpenAI's [message-role guidance](https://developers.openai.com/api/docs/guides/text#message-roles-and-instruction-following)
  and [instruction-hierarchy research](https://openai.com/index/the-instruction-hierarchy/).
- Anthropic's Messages API has no portable developer role, so MemoRizz maps
  reviewed developer skills into its
  [top-level `system` parameter](https://platform.claude.com/docs/en/api/messages/create).
- Ollama and custom OpenAI-compatible endpoints normalize developer messages
  into system instructions; Hugging Face and MLX templates use their
  system-equivalent path.

That compatibility mapping preserves the logical authority distinction as far
as each provider permits, but it is not a claim that every model implements
identical instruction precedence. Test the target model. Developer-authority
skills can also reduce prompt-cache reuse on providers that hoist them into a
top-level system field; measure the token and latency effect before rollout.

Only application-owned, validated, explicitly reviewed skills should receive
developer authority. Automatically generated, imported, user-authored,
shadow, or merely retrieved content should remain at user authority.

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

Every stored run records `skills_activated` — ACTIVE skill IDs that were
actually in its context. The active monitor ignores every other lifecycle
status, even if application code manually supplies a SHADOW ID. It updates:

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

Passive evidence never enters these counters or rolling windows. It is stored
only in `Workflow.shadow_evaluations` and the derived
`Skill.stats["shadow"]` aggregate.

## Inspecting learned skills

Learned skills surface through the same tools as file-based skills:
`list_skills` (tagged `"source": "learned"`) and `read_skill`. From code:

```python
manager = agent.continual_learning_manager
manager.skillbox.list_skills()                 # all, any status
manager.last_report                            # most recent PromotionReport
manager.run_promotion_cycle()                  # manual cycle
manager.promote_class(canonical_hash)          # gated single-class promotion
manager.get_shadow_readiness(skill_id)         # advisory passive metrics
manager.drain_shadow_evaluations(timeout=5.0)  # tests / controlled shutdown
manager.activate_skill(skill_id)               # SHADOW → ACTIVE (+ stamp backfill)
manager.demote_skill(skill_id, reason="...")   # manual demotion
```

## The local UI: human-in-the-loop promotion

The local UI (`memorizz ui`) exposes the whole loop:

- **Create/Edit Agent** exposes **Authority for newly learned skills** with
  `user` and `developer` options plus **Require shadow review before
  activation** and **Passively evaluate shadow skills on new workflows**.
  Selecting developer checks the review control, and the server rejects an
  unsafe developer-without-shadow configuration.
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
  persisted authority, activation stats, baseline, preconditions, and the
  full distilled SKILL.md. SHADOW cards also show passive observation count,
  trajectory-match rate, matched-trajectory success rate, last evaluation,
  and advisory readiness reasons — plus the existing explicit **Activate as
  user/developer** action. ACTIVE cards retain **Demote**.

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
| `shadow_evaluation_enabled` | False | Opt in to passive post-store evaluation of SHADOW skills |
| `shadow_evaluation_max_candidates` | 3 | Maximum tenant-scoped SHADOW candidates per new workflow |
| `shadow_evaluation_min_similarity` | None | Passive threshold; `None` uses `retrieval_min_similarity` |
| `shadow_evaluation_queue_size` | 100 | Bounded non-blocking background queue |
| `shadow_evaluation_recent_window` | 50 | Maximum recent evaluations cached in skill stats |
| `shadow_readiness_min_observations` | 10 | Advisory readiness observation floor |
| `shadow_readiness_min_trajectory_match_rate` | 0.80 | Advisory readiness trajectory-match floor |
| `shadow_readiness_min_matched_success_rate` | 0.80 | Advisory readiness success floor for matching trajectories |
| `skill_injection_role` | `"user"` | Injection authority: `"user"` or `"developer"`; developer requires `require_shadow=True` |
| `retrieval_min_similarity` | 0.70 | Injection threshold |
| `max_skills_in_context` | 2 | Skills injected per turn |
| `include_exemplar` | False | Deprecated compatibility key; raw source runs are never co-injected |
| `drift_window_activations` | 10 | Rolling-rate window |
| `demotion_success_delta` | 0.25 | Demote when rolling < baseline − delta |
| `min_activations_before_drift_check` | 5 | Grace period |
| `promotion_every_n_runs` | 25 | Cycle cadence (0 = manual) |

## Measuring whether skills help

Don't take the feature's value on faith—measure it. The example notebook
(`examples/continual_learning/continual_learning_guide.ipynb`) ends with
three controlled arms:

- a baseline that replays three successful raw workflows;
- the exact same reviewed skill injected at user authority; and
- that same skill injected at developer authority.

All arms use the same model, tools, generic production instruction, held-out
requests, and independently ingested Oracle state. The code asserts that both
skill arms retrieve zero raw workflows, that only the developer arm contains a
developer message, and that the two stored skill bodies are byte-identical.

Styled pandas DataFrames report each case and aggregate exact workflow
accuracy, deterministic answer accuracy, combined task accuracy,
prompt/completion/total tokens, LLM-call count, provider inference latency,
end-to-end latency, raw-workflow prompt rate, developer-message rate, and each
treatment's delta from the baseline. It is a small within-domain smoke test,
not a general claim that higher authority or continual learning always helps.

## Rollout recommendation

1. Enable behind `require_shadow=True` on one agent. Optionally enable passive
   evaluation, observe fresh (never source) workflows, and review readiness
   reasons.
2. Backfill canonical hashes on existing deployments; check that top
   trajectory classes have sane counts (heavy fragmentation means the
   canonicalization rules need tuning before trusting promotion).
3. Review the first `PromotionReport` and every distilled SKILL.md by
   hand.
4. Run a reviewed canary or A/B test; passive observational metrics are not a
   causal treatment comparison and never activate a skill automatically.
5. For user-authority skills only, consider loosening to
   `require_shadow=False` once distillation quality is trusted. Developer
   authority always requires shadow review and explicit activation.
