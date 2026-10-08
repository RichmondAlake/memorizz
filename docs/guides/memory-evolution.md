# Memory evolution and historical context

Playground → Memory → **Memory evolution** shows a horizontally scrollable
timeline of creations, corrections, deletions and explicit source relationships.
Select a node to inspect its actor, originating source, changed fields, version
hashes and run identifiers. Filter by memory type, actor or change, and load
older pages without replacing the already loaded history.

The date-and-time ruler follows the recorded changes and stays visible while
scrolling vertically. It shows execution run identifiers when available and
labels elapsed gaps between changes. Choose your browser's local timezone or
UTC. Cards are spaced for readability; their distance is not a proportional
duration scale. Existing records with unknown history do not receive invented
execution times. The Playground uses the same ruler in its compact view, and
the full view pins memory-type labels while scrolling horizontally.

Search finds names, writers, sources, memory types and identifiers in the loaded
history. Load older changes to extend the search. Cards wrap long titles and
show concise change summaries; the detail panel keeps IDs and hashes inside
**Technical details**. Select a change to emphasize its recorded connections,
or choose **Related changes** to focus on that connected history. Click a source
or a previous/next recorded version in the detail panel to navigate to it.
Related changes includes connected versions and sources outside the display
filters, within the loaded, authorized history. The controls retain their values;
leaving the related view restores those filters. Changing a filter exits the
related view. Load older pages to reveal connections that are not loaded yet.

For teams, choose **Group by → Writer** to give each contributor a fixed row.
Saved agent names replace IDs when available; every card still shows its memory
type. Atomic shared-blackboard updates retain the contributing agent as writer
and an outer coordinator as initiator when present. The coordinator's timeline
contains its owned shared session; delegate-private conversations remain on
their own agent timelines. This is write provenance, not proof that an agent
read or used a memory. Explicit sources remain necessary for derivation links.

The **MemAgent memory history** strip identifies the current agent and lists
its currently configured delegates. Click a delegate, a writer row, or the
writer in a change's details to open that MemAgent's own history across its
conversations. The previous timeline link restores the parent view's agent,
namespace and run scope. Nested navigation keeps a bounded trail of previous
views. **Memory inspector** opens that agent's Playground directly on the
Memory tab; **All agent history** expands a conversation or run view to its
agent's full history. Tenant and application scopes remain in effect.

Agent-scoped existing-record observations exclude records explicitly owned by
other agents, even in a common namespace. Unattributed older namespace records
remain marked as history unknown. Configured delegates describe today's saved
configuration, not an inferred historical delegation tree.

## Developer investigation

The developer review found these gaps in the original presentation:

| Gap | Current behavior |
| --- | --- |
| Details below a tall graph | Desktop selection opens an adjacent, scrollable inspector. Narrow and embedded views stack the inspector and bring it into view. |
| No chronological investigation controls | Previous/next change buttons show the position among visible changes. Focused cards support Left/Right and Home/End; recorded-version links remain separate. |
| Execution IDs without a path to evidence | Filter loaded executions and open the selected change's trace. Captured-turn links select the correct conversation and historical turn in the Context tab. Missing requests are stated explicitly. |
| Writer and ownership are ambiguous | The inspector distinguishes writer, owner, initiator, namespace and shared coordination. Missing initiators and execution IDs are marked as not recorded. Ownership does not establish exclusive visibility. |
| Unknown records dominate recorded history | They are excluded initially and can be included explicitly. Counts distinguish creations, updates and deletions; coverage notes say when older pages remain unloaded. |
| Hashes are presented without useful value evidence | A scoped current-value preview states whether it matches the selected version. Deleted or reassigned records, hidden content and unavailable historical values have explicit states. New changes classify fields as added, modified or removed without storing their values. |
| Pinned labels clip execution times | Partially covered ruler labels are hidden rather than showing a fragment of a timestamp. |

To investigate a correction, select the updated memory, inspect its writer and
owner, follow its previous recorded version, and open **View current stored
value**. If the current value differs from the selected version, it is clearly
marked as a later/current record rather than a historical snapshot. Follow
**Open execution trace** or **Captured turn context** when those identities were
recorded. Captured requests establish what was sent to the model; they do not
prove that a particular memory influenced the answer.

The value preview observes trace content permissions, removes credential-shaped
fields, and is bounded to 16,000 characters. Configuration values stay in the
agent's Settings inspector. A record reassigned to a different tenant,
application or owner is unavailable from its earlier change. Trace-only
accounts are not offered links into the Playground.

Historical before/after values are not retained by this journal. No UI can
recover deleted or older values from hashes alone. Unrecorded reads, missing
attribution and direct backend writes remain evidence gaps, and are not filled
by inference.

Oracle saves delegate relationships using its internal agent foreign keys and
restores public agent IDs when loading or listing agents. If an older save
omitted those relationships, re-save the coordinator with its original delegate
configuration; the timeline does not guess missing configuration from writers.

The same view is available from an agent's Observability page, a harness run's
**Memory evolution** evidence tab, and links in harness comparisons. The full
timeline is `/traces/memory-evolution`; agent, memory namespace and harness run
filters can be combined.

Solid edges connect recorded versions of the same memory. Dashed edges follow
explicit identifiers such as supersession, derivation and a summary's source
messages. Similarity never establishes a branch. Records predating capture are
shown as existing observations with unknown historical attribution.

## SDK

Recording is optional for standalone providers and enabled by the UI, MCP host
and harness service. Attribute your SDK writes explicitly:

```python
from memorizz import MemoryHistory, MemoryType

history = MemoryHistory(provider)
with history.recording(actor="release-owner", source="release-review", agent_id=agent_id):
    provider.update_by_id(record_id, {"content": "Corrected fact"}, MemoryType.KNOWLEDGE_BASE)

page = history.timeline(agent_id=agent_id, limit=200)
events = page["events"]
if page["next_cursor"]:
    older = history.timeline(agent_id=agent_id, limit=200, cursor=page["next_cursor"])

# Read an authorized change, then inspect the current target separately.
change = history.get_change(events[-1]["record_id"], agent_id=agent_id)
current = history.current_record(change) if change else None
# current['matches_selected_version'] distinguishes a current value from
# the selected change's recorded after-version; this is not a historical diff.
```

For a native agent, pass `capture_memory_history=True`. `agent.memory_history`
provides the same reader; supply `agent_id=agent.agent_id` to scope its queries.
SDK attribution is a caller assertion, not an authentication mechanism. Passing
`user_id=None` selects anonymous records; leaving it unspecified is an
administrative read. Application and agent filters further restrict the result.
Direct `agent.delegate()` calls attribute coordinator writes to the coordinator.
Delegate turns and shared contributions retain their own writer and the enclosing
initiator when recording is enabled on the provider. Existing unknown writers
are not inferred or rewritten.
Oracle shared sessions continue to support both logical session UUIDs and
physical archive row UUIDs, preserving the exact content used by atomic
blackboard writes.

Timeline filters apply before pagination where supported. Older providers and
post-filtered Notion pages are followed until the requested page is filled or
the history ends, so an empty source page does not interrupt the filtered result.
Keep the same scope and filters when following a continuation cursor.

`history.observations(agent_id=agent_id, memory_ids=namespaces, limit=200)`
returns bounded current-record observations with unknown write history. The UI
excludes already displayed change targets before applying that observation
limit. Filesystem uses cached ownership metadata; MongoDB and Oracle push scope
and limits into queries; Notion uses scoped, bounded queries with its configured
page budget. Old filesystem index entries are upgraded lazily in memory on
their first read. Third-party providers can implement
`query_memory_observations()`; the compatibility fallback uses `list_all()`.

The private journal stores field names and before/after hashes rather than
copies of memory content. It does not restore deleted data. History survives
provider restarts, but recording is best-effort: direct database writes, earlier
writes and failed journal writes can leave gaps. It is not a transactional audit
log. Current filesystem, MongoDB, Oracle and Notion providers use the same
portable journal contract.

## Previous model requests

The Context tab's left/right arrows navigate captured requests by turn. Select
individual model calls when a turn used several, including tool-loop requests.
It shows the fitted messages, tool schemas, estimated token count and change
from the previous captured turn. Older request pages load as you navigate.

The current context preview is an estimate. Historical requests are copied
after prompt fitting and before provider-specific wire transformations. Older
turns without snapshots are explicitly unavailable; turns answered without a
model call, such as semantic-cache hits, show zero requests.

Standalone SDK agents opt in with `capture_context_snapshots=True`:

```python
from memorizz import ContextSnapshots

page = ContextSnapshots(provider).page(agent_id=agent_id, memory_id=memory_id)
request = ContextSnapshots(provider).get(
    page["items"][-1]["record_id"], agent_id=agent_id, memory_id=memory_id
)
```

Requests are separate private observability records and can contain per-call
context. The UI enforces trace permissions and full/redacted/metadata content
policy. Deleting a playground thread also removes its captured requests.

## MCP and CLI

`memorizz_get_memory_timeline` accepts agent, namespace, run, memory type, actor,
action, limit and cursor filters. It requires `memorizz:read` and applies the
authenticated caller's exact tenant and configured agent allowlist.

```sh
memorizz memory timeline --agent-id AGENT_ID --json
memorizz memory timeline --run-id RUN_ID --actor release-owner
```

The CLI uses the configured provider and supports `--cursor`, `--limit`,
`--memory-id`, `--type`, `--action` and an exact `--user-id` filter.

## Working example

Run `examples/memory_evolution/demo.py` to create fictional launch briefs,
correct the owner, derive a revised launch time, generate a real local-model
summary, and expire a working note. It prints the playground, timeline and
observability URLs. Its README includes filesystem and Oracle instructions.
Each run creates a fresh agent and namespace. A filesystem UI picks up records
written by another process on its next read, so no reconnection is needed.

`examples/memory_evolution/delegates.py` runs a researcher and reviewer on the
local model, records their shared blackboard, then publishes their actual
results as linked team memories through the SDK. It prints a timeline grouped
by writer and also supports Oracle. These publication writes are performed by
the example script, not autonomous model tool calls.

## Forgetting on the timeline

Approved retention plans show up as recorded changes: a suppression is an
`updated` card whose writer is the approver and whose source is
`forgetting:<plan-id>`, with `retention_state` among the changed fields; a
restore is the next recorded version of the same memory. Suppressed records
stay on the timeline because nothing was deleted. See the
[forgetting mechanism](forgetting-mechanism.md).
