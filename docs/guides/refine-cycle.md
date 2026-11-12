# Refine cycle: upkeep as one ordered cycle

An agent's memory upkeep has several jobs: consolidating entities, promoting
workflows to skills, learning for the persona, planning retention. Each used
to fire on its own counter (every 25 stored runs, every 20 turns or 24
hours), in no fixed order, and only when a user's turn happened to trigger
it. A refine cycle runs the same jobs once, in a fixed order, and writes one
report.

```python
agent = MemAgent(..., refine_cycle=True)   # the per-turn counters stop
report = agent.refine()                 # run one cycle now
report = agent.refine(dry_run=True)     # plan every job, apply nothing
```

`refine_cycle=True` only changes when the jobs run. With it on, the counters for
consolidation, promotion and persona learning are off, so nothing runs
twice. With it off (the default) nothing changes.

## The order

Later jobs read what earlier ones cleaned, so the order is fixed.

| # | Job (`name`) | What it runs |
|---|---|---|
| 1 | Verify notes (`verify`) | Check stored notes against the conversations they cite; flag what is not supported |
| 2 | Entities (`entities`) | Consolidation: merge duplicates and review attributes, under the [source rule](memory-consolidation.md#source-levels) |
| 3 | Stated facts (`stated_facts`) | Consolidation: add facts the user stated, each quoted from the transcript, and count the ones stated again |
| 4 | Stale references (`propagation`) | Flag the skills and notes that still state a value the entity jobs replaced |
| 5 | Tool review (`toolbox`) | [Tool learning](tool-learning.md): propose revisions from the tool log, list usage flags |
| 6 | Skill staleness (`skill_staleness`) | Deprecate skills whose tools no longer resolve |
| 7 | Promotion (`promotion`) | Gates, ranking, distillation and validation |
| 8 | Skill review (`skill_review`) | The deviation flag, the [holdout](continual-learning.md#the-holdout) flag, and skills with an open stale reference |
| 9 | Persona learning (`persona_learning`) | A goals sentence, and [standing rules](persona-learning.md) for their owner |
| 10 | Compact and re-embed (`compact`) | Embed again what changed, so vectors match the text |
| 11 | Retention (`retention`) | A retention plan |

Verification is first, so no later job builds on a note its transcript no
longer supports. Stale references follow the entity jobs, because that is
when values are replaced. Compaction comes after everything that edits
text. Jobs 2 and 3 come from one consolidation plan per memory, so the
memory is read once. Capture, routing, reflection and conversation
compaction are not part of the cycle; they still run after each turn.

Each job keeps its own apply rule:

| Job | What it may do without a person |
|---|---|
| Verify notes | Open a flag. It deletes and rewrites nothing |
| Entities, stated facts | Apply when consolidation's `auto_apply` is on; otherwise the plan waits |
| Stale references | Open a flag, and close one whose text was fixed. It rewrites nothing |
| Tool review | Nothing with `refine_tool_review="propose"` (the default): each revision waits. `"auto"` applies them |
| Promotion | Leave a new skill in shadow when `require_shadow` is on |
| Persona learning | Apply a goals sentence when its `auto_apply` is on. A standing rule always waits for its owner |
| Compact and re-embed | Write vectors. It changes no text |
| Retention | Nothing: a plan always waits for a named person |

A cycle never changes a persona's
[guardrails](persona-learning.md#guardrails).

The persona, its standing rules and the active skills are what a model
reads at the top of every prompt. With `refine_cycle` on they are changed
by the cycle and nowhere else, so those changes land together, once a
cycle. If you rely on a prompt cache, that is the point at which it is
invalidated: one batch a cycle, not one edit at a time.

## Verify notes

Memory an agent wrote from a conversation is only as good as the
conversation. The job checks three kinds of note against the transcript each
one names:

| Note | What it cites | How it is checked |
|---|---|---|
| A stated fact | The user's words, kept when consolidation added it | The quote is still in a user message. Code only |
| A captured fact | The conversation an entity value came from | The value's words are in the conversation: code. Otherwise the background model is asked, at most `refine_verify_model_checks` (3) times a cycle, and its yes counts only with words copied from the conversation |
| A summary | The conversation it summarises | The conversation still exists |

What fails is flagged as `unsupported` (the conversation is there and does
not say it), `source_missing` (the conversation is gone) or `source_sealed`
(the conversation was sealed, so the fact can no longer be checked). A
summary of a sealed conversation is what sealing is for and is not flagged.
A note is flagged once; a captured fact the model confirmed is remembered
and not asked about again. Facts the budget did not reach wait for the next
cycle.

```python
agent.verify_notes(memory_id, dry_run=True)       # the same check, on its own
agent.unverified_notes()                          # open flags
agent.resolve_unverified_note(flag_id, by="dev")  # the note was fixed
agent.dismiss_unverified_note(flag_id, by="dev")  # the note is right
```

Stated facts added before this release have no stored quote and are checked
as captured facts when they name a conversation, or not at all.

## Stale references

The entity jobs replace values. This job then looks for the skills and
notes that still state an old one, with the conservative matching described
under [stale references](memory-consolidation.md#stale-references-when-a-fact-changes), and lists every flag raised
since the cycle started, including those raised while the entity jobs were
applying. A flag whose skill or note no longer states the old value is
closed by the cycle. A skill with an open flag is also listed under Skill
review, beside the other reasons to open a skill.

## Tool review

The job calls `agent.review_tools()` once. A tool with no new calls since
its last review costs no model call, and an agent without a toolbox skips
the job. Each revision is listed with what it would add and is left
`proposed`; it appears in the cycle's waiting list as `tool_revision`, and
`agent.apply_tool_revision(revision_id, approved_by=...)` applies it.
Usage flags (offered and never called, or never offered) are listed for a
person and propose nothing.

## Compact and re-embed

Retrieval finds a record by its vector, so after text has changed the
vectors have to follow. The job, for each memory of the cycle:

1. embeds again every live entity record that changed since the cycle
   started, or whose vector is missing, partial or the wrong size for the
   current embedder. An entity's vector is built from its name, type and
   current values; earlier and retired values are never in it, so a search
   for the old value stops finding the record;
2. embeds again every live knowledge-base note or summary whose vector is
   missing or the wrong size;
3. with `refine_drop_suppressed_vectors=True` only, removes the vector of a
   suppressed note and of an entity record a merge replaced. The row and its
   text stay; a row restored later is embedded again by the next cycle.

It deletes no record and rewrites no text, so nothing is lost: a vector can
always be computed again. At most 200 records are embedded in one cycle. A
provider that builds its own vectors is skipped. An attribute's history is
already bounded where it is written (the newest 20 earlier values, one
entry per value offered and not applied), so there is nothing to fold.

## On a schedule

```python
agent.schedule_refine_cycle()            # every refine_interval_hours (24)
agent.schedule_refine_cycle(interval_hours=12)
```

This adds a `refine_cycle` job to the [automation](automations.md) worker,
which gives it the lock, retries and time limit every job has, and turns
`refine_cycle` on. A scheduled cycle is skipped when no conversation has a
message newer than the last cycle. Without a worker, call `agent.refine()`
yourself, or leave `refine_cycle` off and keep the counters.

## The report

Every cycle stores one `RefineRun`:

```text
{
  "run_id": "refine-…", "trigger": "schedule", "status": "done",
  "started_at": "…", "finished_at": "…",
  "changed": 5, "waiting": 1, "tokens": 11130,
  "jobs": [
    {"name": "entities", "status": "done",
     "changes": {"merge_entities": 1, "retire_attribute": 2},
     "waiting": 0, "tokens": 4210},
    …
  ],
  "pending": [
    {"kind": "retention", "job": "retention", "plan_id": "…",
     "summary": "4 memory record(s) could be suppressed"}
  ],
}
```

`pending[].kind` is one of `consolidation`, `persona`, `persona_rule`,
`tool_revision`, `skill_activation` and `retention`. Flags are not in
`pending`: they are for a person to read, not plans to approve.

A dry run reports the same jobs with `would_` and `planned_` counts
(`would_flag_source_missing`, `would_flag_note`, `planned_tool_revisions`,
`would_re_embed_entities`), stores no flag, revision or vector, and leaves
nothing waiting. A run stored before these jobs existed lists the seven it
ran and still opens.

Each job also carries `items`: one line per change with what it did, why,
and whether it was applied, for example "Merged CFU and Counter Fraud Unit"
or "user.city: Lisbon → Porto" with the user's words it rests on. The local
UI shows them under "What changed", and each earlier cycle opens to show
its own. The MCP tools leave `items` out, because they quote memory content.

`tokens` comes from the background model's own usage report when it has
one; otherwise it is estimated from the text and `tokens_estimated` is true.
`agent.refine_runs()` lists stored cycles, newest first.

Plans left for review are stored with the run. A consolidation plan that
waits no longer lives only in the process that made it:
`agent.apply_memory_consolidation("<plan id>", approved_by="…")` finds it
after a restart.

## CLI, MCP and the local UI

```bash
memorizz learning refine --agent-id AGENT [--dry-run]
memorizz learning refine-runs --agent-id AGENT
```

In the chat CLI, `/refine` runs a cycle, `/refine dry` plans one, `/refine runs`
lists past cycles and `/refine on` or `/refine off` switches the setting.

Over MCP, `memorizz_refine(agent_id, dry_run=False)` runs a cycle and
`memorizz_list_refine_runs(agent_id)` lists them. A real cycle needs the
write and execute scopes; a dry run needs write only.

In the local UI, *Continual learning → Control plane* has a **Refine cycles**
section: what started each cycle, its status, the changes and tokens of each
job, the plans waiting for review, and **Run now**, **Dry run** and **Run on
a schedule**.

## Settings

| Key | Where | Default | Meaning |
|---|---|---|---|
| `refine_cycle` | Agent | `False` | Run upkeep as one ordered cycle in place of the separate counters |
| `refine_interval_hours` | Agent | `24` | Gap between scheduled cycles |
| `refine_verify` | `memory_consolidation` | `True` | Run the verify job |
| `refine_verify_model_checks` | `memory_consolidation` | `3` | Captured facts the verify job may ask the background model about in one cycle; `0` keeps it to code |
| `refine_tool_review` | `memory_consolidation` | `"propose"` | `"propose"` leaves tool revisions for a person, `"auto"` applies them, `"off"` skips the job |
| `refine_compact` | `memory_consolidation` | `True` | Run the compact job |
| `refine_drop_suppressed_vectors` | `memory_consolidation` | `False` | Also remove the vectors of suppressed rows |
| `propagate_supersessions` | Agent attribute | `True` | `False` skips the stale-references job and the watch on entity writes |

All but the last are saved with the agent's consolidation settings.
`agent.with_refine_cycle(True, verify=..., verify_model_checks=...,
tool_review=..., compact=..., drop_suppressed_vectors=...)` sets them.
