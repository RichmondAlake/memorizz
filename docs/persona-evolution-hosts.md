# Account-owned persona evolution

Memorizz's native `Persona.update()` records versioned changes; it is not a
daily reflection scheduler. A persona attached to one shared `MemAgent` is
configuration for that agent, not automatically a different persona for every
`user_id`.

## Request-local snapshots

A multi-tenant host can load an authenticated account's canonical persona and
bind it for the lifetime of a call:

```python
with agent.persona_manager.use_snapshot(account_persona_dict):
    answer = agent.run(
        message,
        user_id=verified_user_id,
        memory_id=account_memory_id,
        thread_id=verified_thread_id,
    )
```

For streaming, keep the context open while consuming the stream. MemAgent's
copied-context stream worker inherits the snapshot. Passing `None` explicitly
uses no persona for that request. The scope resets even after cancellation or
an exception. Input dictionaries are copied and rehydrated without embeddings
or storage calls.

The manager's mutation APIs reject changes while a request snapshot is bound.
Hosts still own authentication, evidence access, policy, persistence and
approval. This is not a sandbox for arbitrary Python code mutating objects.
The host must not use `set_persona()` to switch a shared singleton between
accounts.

`configuration_persona` exposes persistent agent configuration. Saving a
MemAgent during a request never saves the temporary account persona into its
shared agent record. Native turn-start traces record a hashed `persona_id` and
`persona_version`; semantic-cache identity includes the rendered persona,
not just its name/version representation.

## Operator UI adapter

The `/persona-evolution` page displays the host's canonical account profile,
before/after changes, supporting memory excerpts, conversation-trace links,
reflection attempts and version history. It is separate from the saved-persona
picker in the generic agent configuration form.

```python
app = create_app(
    persona_evolution=host_persona_adapter,
    persona_evolution_actions=False,  # inspection only unless explicitly enabled
)
```

The adapter is synchronous and implements:

- `view(user_id)` → canonical UI state.
- `reflect(user_id, *, actor_id)` → reviewed proposal or a no-change result.
- `decide(user_id, *, proposal_id, revision, accept, actor_id)` → approve/dismiss.
- `pause(user_id, *, revision, paused)`.
- `undo(user_id, *, revision, actor_id)`.
- Optional: `schedule(user_id, *, revision, daily_enabled)`; expose
  `daily_enabled` in state only when the host supports this control.

These actions return the latest canonical state. The initial host UI contract
is for a learning-goals profile: `persona` has the standard Persona fields and
`evolution_history`; `pending` has `id`, `reason`, `changes.goals.{old,new}` and
`evidence` (`id`, `thread_id`, `timestamp`, `text`). State also carries
`revision`, `paused`, `review_state`, `last_review_at`, `reviews` and
`history_limit`. The consuming application supplies the concrete adapter.
The UI does not directly write a provider's global PERSONAS collection.
Provide the canonical `agent_id` in the host state so evidence links can select
the agent and conversation timeline. Persisted turn-start events retain the
persona version for trace inspection and exports.

MongoDB conversation writes now preserve `agent_id`, including bulk writes.
Older versions stripped that metadata. Hosts using those older records must
handle missing agent attribution explicitly inside their authenticated account
and application memory bucket; never remove account scoping to make a review
find evidence. Native conversation timestamps are naive local time, so host
review cutoffs and forgetting watermarks must match the writer's convention.

Authentication is required when an adapter is configured. Only unrestricted
administrator accounts may inspect or change these private profiles; existing
restricted trace-only accounts gain no permissions. Reads and requested
actions require the local audit log to be writable. Audit records include a
keyed target-account pseudonym, not the raw account ID. Operator identity comes
from the authenticated principal, not from an action's JSON body.

When `persona_evolution_actions=True` is explicitly configured, only the host
persona action endpoint is exempted from UI read-only mode. Other provider,
agent and trace mutations remain blocked. Cross-origin actions are rejected.
Use named administrator accounts to distinguish operators in change history;
the legacy shared UI token identifies its operator as `local`.

The page contains private profile/evidence content. Do not enable the adapter
on an unauthenticated/public UI or treat this as metadata-only trace access.
No model training or automatic application is started by this page. A manual
button must never bypass the host's ownership and version checks.

## Memory-first trace contract

`memorizz.observability.references.persona_reference(persona)` returns a typed,
versioned `memory` reference with `role="persona_style"`. It contains neither
the goals nor a user-chosen persona identifier. The same reference connects
an approved version to subsequent model inputs. Persona supply is separate
from retrieved candidates: an account snapshot is bound directly, not found
by similarity search. It has its own supply/approximate prompt-token category.

Supplying a style is **not** proof the model followed it. The reference-stage
counter explicitly leaves behavioral application unmeasured. Evaluate actual
answers separately; do not call a persona unused because its text was not
quoted in the answer. Scoped virtual sources likewise mean registration is
not inspected, not that the agent was necessarily never saved.

Hosts can use `ObservabilityRecorder` for reflection outside a MemAgent run:
memory-supply references → measured model call → validated proposal → explicit
decision. Use a dedicated account-scoped review thread/root, provider tokens,
measured duration, optional list-rate cost estimate and decision output
references. Keep private goal/excerpt text out of these events. A review/entry
may carry `trace` with the canonical `thread_id` and
`root_trace_id`; the operator page links to that trace. A link is correlation,
not a guarantee that every telemetry write succeeded. Telemetry remains
fail-soft and cannot replay or veto the business operation.

If a host resets version numbers when forgetting, it must also rotate the
persona identity. Reusing both identity and version for different goals makes
historical lineage ambiguous.

## Optional host scheduling

Memorizz does not implement a daily account-reflection worker through this
adapter. Expose the optional daily controls only when your host implements the
UI's displayed contract: opt-in review no earlier than every 24 hours, only
after new conversations, with a separate approval before applying changes.
This is not a promise of midnight execution; busy workers may run later.

Hosts should enforce account ownership, compare-and-swap revisions, bounded
model calls, concurrent-review exclusion, pause/forget cancellation and retry
policy in their own service. Do not rely on disabled UI buttons for these
guarantees. Review database indexes and query plans for your actual storage
schema; the generic page neither installs indexes nor migrates profile data.

For unchanged evidence, an adapter may return `notice="no_new_evidence"`; the
page explains that the approved persona was unchanged and no model call was
made. Only return that notice when both statements are true.

Delegated host workers can use the parent's already-approved snapshot through
`use_snapshot`, keeping style in ephemeral context rather than saving it as a
worker's shared configuration. Hosts still need their own end-to-end tests for
account boundaries, persistence, scheduling and approval before deployment.
