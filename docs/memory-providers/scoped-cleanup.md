# Delete a Scope

`delete_scope` deletes the data MemoRizz owns inside one exact scope: a memory,
a user, a set of agents, or a mix of them. Use it to clean up after a test or an
evaluation run, to erase a tenant, or to tear down agents. It is not for broad,
unscoped deletion.

```python
result = provider.delete_scope(
    memory_id="eval-run-17",
    user_id="grader",
    agent_ids=["eval-agent"],
    keep_traces=True,
)
print(result["counts"], result["total_deleted"])
```

The result holds `counts` per store, `total_deleted`, and the `scope` it used,
`keep_traces` included. `agent.close(cleanup_scope={...})` and
`agent.lifecycle(cleanup_scope={...})` pass the same arguments through.

## Which rows it deletes

- Give at least one of `memory_id`, `user_id` and `agent_ids`. An empty scope
  raises `ValueError`.
- A row is deleted when every scope part its store records matches. A part the
  store doesn't record is ignored there. A store that records none of the given
  parts is skipped.
- `user_id=None` selects rows without a user.
- `agent_ids` also deletes those agents' records and their automation jobs,
  with the jobs' runs and deliveries.
- `memory_id` also removes that memory from every agent's memory list.
- A trace record is scoped by the turn it traced: that turn's memory, user and
  agent. Its own record id is not a memory id.

| Store | Memory | User | Agent |
|---|---|---|---|
| `conversation_memory`, `tool_log`, `workflow_memory`, `summaries`, `semantic_cache`, `entity_memory`, `knowledge_base`, `short_term_memory`, `toolbox` | `memory_id` | `user_id` | `agent_id` |
| `skillbox` | — | `user_id` | `agent_id` |
| `personas` | `memory_id` | — | `agent_id` |
| `shared_memory`, trace records | the traced memory | the traced user | the traced agent |
| `shared_memory`, everything else | `memory_id` | — | `owner_agent_id` |

Trace records are the shared-memory records whose type starts with
`observability_`: trace bundles, outcomes and grader verdicts, feedback,
recommendations, experiments and context snapshots. Other shared-memory records,
such as learning events and memory history, follow the last row: they are
keyed by their own id, so an agent scope reaches them and a memory scope
doesn't.

## Keep the traces

`keep_traces=True` deletes the memories and keeps the trace records, so a
finished run stays on the Traces page. Use it when an evaluation deletes each
run's memories but you still want to read the runs.

It keeps:

- every trace record in shared memory, as listed above;
- legacy trace bundles saved as `tool` rows in conversation memory.

It still deletes everything else in the scope:

- **Messages.** A turn's message text is conversation memory, so it goes. The
  timeline keeps the turn's events: its start and result, model calls, tool
  calls and results, memory supply, tokens and cost, run tags and verdicts.
- **Tool logs.** The trace bundle keeps each tool call and its result.
- **Agent records.** A deleted agent's runs appear under **Unregistered runtime
  traces** in the navigator. `agent.trace_url()`, links that name the agent id,
  and tag filters still open them.

A trace bundle alone carries what Traces needs to list and open a run, so
nothing else is kept. Context snapshots (`capture_context_snapshots=True`) hold
the model's input; they are trace records too. Delete without `keep_traces`
when they must go.

## Per provider

- **Oracle** deletes in one transaction; a failure rolls it all back. It reads a
  trace record's scope from its JSON. Automation jobs, runs and deliveries have
  their own counts.
- **MongoDB** runs one `delete_many` per collection with native filters. The
  collections are deleted one after another, not in a transaction. After a
  failure, run it again to finish.
- **Filesystem** reads each store and deletes the matching files, with one
  index write per store. It is not transactional. After a failure, run it
  again to finish.
- **Notion** uses the same portable sweep, one request per page. Bulk deletes
  can partly succeed; run it again to finish. Notion has no automation store.
- **Custom providers** inherit the portable sweep from `MemoryProvider`. It uses
  `list_all`, `delete_by_id`, `delete_memagent` and
  `update_memagent_memory_ids`. Override `delete_scope` for a native or
  transactional version.

No provider touches the optional native span index
(`MEMORIZZ_OBSERVABILITY_DUAL_WRITE`). Its retention runs separately; see the
[production governance guide](../guides/production-governance.md).
