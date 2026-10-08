# Memory evolution

This example runs a real local Ollama agent, corrects a launch brief, branches a
revised brief from its source, generates a conversation summary, and deletes a
temporary note. The timeline shows the actors and sources of these changes.
It uses a fresh namespace and leaves existing agents and memories untouched.

Install `memorizz[ui,ollama]` and pull `qwen2.5:7b` and `nomic-embed-text`.
Start `memorizz ui --port 8766` and connect it to your example filesystem store:

```sh
python examples/memory_evolution/demo.py --store /tmp/memorizz-evolution/memory \
  --portal http://127.0.0.1:8766 --output /tmp/memorizz-evolution/evidence.json
```

Reconnect the UI after an external SDK process writes to a filesystem store.
Open the printed timeline URL, or Playground → Memory → Memory evolution.
Observability also embeds the selected agent's timeline. Harness evidence has
a Memory evolution tab, and comparison tables link each run to its history.

## Delegates and shared memory

Run two real local delegates and publish their outputs as linked team memories:

```sh
python examples/memory_evolution/delegates.py --store /tmp/memorizz-evolution/memory \
  --portal http://127.0.0.1:8766 --output /tmp/memorizz-evolution/team-evidence.json
```

The researcher reads the initial launch brief; the reviewer receives that
result and applies an explicit correction. Their shared blackboard updates are
recorded atomically, with the contributing agent as writer. The script then
publishes their actual results as an initial finding, a superseding review and
a sourced team decision. These publication steps are SDK writes, not autonomous
model tool calls. No other agents or memories are changed.

Open the printed coordinator timeline and choose **Group by → Writer** to see
the researcher, reviewer and coordinator in separate fixed rows. Every card
still identifies its memory type. Select a finding, then **Related changes** to
focus on recorded versions and explicit derivations. Source links in the detail
panel navigate to the contributing change. Search applies to loaded history;
load older changes to extend it.

Click the researcher or reviewer in the delegate strip, writer row, or change
details to enter that agent's own memory timeline. Use **Previous timeline** to
return to the coordinator. **Memory inspector** opens the selected agent's
Playground on its Memory tab, including its captured conversation history.

The coordinator timeline includes its owned shared blackboard and published
team records. A delegate's private conversation stays on that delegate's own
timeline. Recording captures writes, not proof that another agent read or used
a memory. Namespace and agent filters intersect; querying a common namespace
without an agent filter includes its scoped contributors. The shared blackboard
has its own namespace, so use the coordinator timeline for this example.

The same script supports `--backend oracle` with the credentials described below.

For Oracle AI Database, use a separate development schema and set `ORACLE_USER`,
`ORACLE_PASSWORD`, and `ORACLE_DSN` securely in your environment. Add the `oracle`
extra, connect the UI to that schema, and run the example with `--backend oracle`.
The example uses local 768-dimensional embeddings; use a schema with compatible
vector columns. No credentials are written to the evidence report.

SDK recording is opt-in:

```python
from memorizz import MemoryHistory, MemoryType

history = MemoryHistory(provider)
with history.recording(actor="operator-7", source="launch-review", agent_id=agent_id):
    provider.update_by_id(record_id, {"content": "Corrected fact"}, MemoryType.KNOWLEDGE_BASE)
page = history.timeline(agent_id=agent_id, user_id="account-7", limit=200)
```

For native agents, set `capture_memory_history=True` and
`capture_context_snapshots=True`. `agent.get_context_history(memory_id)` returns
captured request metadata. `ContextSnapshots(provider).get(...)` reads an exact
request with agent, namespace, tenant and application filters.
`ContextSnapshots(provider).page(agent_id=..., cursor=...)` traverses older
request metadata. The playground loads older turns as you navigate left.

The MCP tool is `memorizz_get_memory_timeline`. CLI access:

```sh
memorizz memory timeline --agent-id AGENT_ID --json
memorizz memory timeline --run-id HARNESS_RUN_ID --limit 200
```

Pass `next_cursor` to continue through older journal entries. Explicitly passing
`user_id=None` in the SDK selects anonymous records; omitting it is an
administrative read. MCP reads always use the authenticated caller's tenant.

The journal retains changed field names and version hashes, not memory content.
Deleting memory leaves content-free deletion metadata. Captured model requests
are separate private observability records and may contain per-call context;
they are not conversation memory. Deleting a playground thread removes its
request snapshots. Snapshot access follows the UI's trace content policy.

Existing memories appear as observations with unknown historical attribution.
Branches use explicit source IDs or supersession links; similarity alone never
creates a lineage edge. Writes made directly in a backend, failed audit writes,
and writes before recording was enabled can leave history gaps. The journal is
best-effort and is not a transactional compliance log.
