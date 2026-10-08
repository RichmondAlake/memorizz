# Real memory lifecycle demo

This example creates a fictional Harbor launch assistant and runs four actual
conversations through the playground SSE endpoint. Generation uses Ollama
`qwen2.5:7b`; embeddings use `nomic-embed-text`. It does not deploy anything.

Use an isolated filesystem store and portal so every record is easy to inspect
and its embedding model is consistent. Install the `filesystem`, `ui` and
`ollama` package extras if your environment does not already have them.

In one terminal, start the portal:

```sh
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
export MEMORIZZ_HOME=/tmp/memorizz-memory-demo/home
export MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=ollama
export MEMORIZZ_DEFAULT_EMBEDDING_MODEL=nomic-embed-text
memorizz ui --port 8766
```

In another terminal, from the repository root:

```sh
python examples/memory_lifecycle/demo.py \
  --store /tmp/memorizz-memory-demo/memory \
  --portal http://127.0.0.1:8766 \
  --output /tmp/memorizz-memory-demo/evidence
python examples/memory_lifecycle/verify.py \
  --store /tmp/memorizz-memory-demo/memory \
  --portal http://127.0.0.1:8766 \
  --output /tmp/memorizz-memory-demo/evidence
```

The script connects the specified portal to the specified store. Use a dedicated
portal; this connection replaces that portal's current provider connection.
If a run is interrupted, pass `--resume` with the same arguments to continue
from the saved `evidence.json`.

## What is seeded and what is generated

Seeded inputs are a persona, one knowledge document, the Harbor owner entity,
an explicitly authored launch-readiness skill and native tool definitions.
The skill is authored, rather than learned or promoted by this demo.

The actual agent generates conversation records, a tool execution workflow,
tool logs and semantic-cache entries. The local model generates a conversation
summary; a subsequent turn consumes it. No agent answer is mocked. The verifier
requires successful tool results, actual skill activation, memory-supply traces,
the expected answer and an actual cache-hit trace.

This exercises 10 of the 12 memory types (excluding the agent configuration
record). Short-term scratch memory and shared agent coordination are not
exercised. Observability's internal shared-memory records do not count as
shared memory consumed by the agent.

## Inspect the result

`evidence.json` contains the agent ID, conversation memory ID, playground and
observability links, raw SSE events and timings. `verification.json` records
the checks; `trace-events.json` and `playground-thread.json` preserve the
portal's actual evidence. Earlier failed attempts remain distinguishable in
the trace export.

In the playground, open **Context** for the instruction, persona, tools and
conversation, **Memory** for stored workflows, skills, entities, summaries,
knowledge and tool logs, and **Overview** for the context token breakdown.
Observability records which memory reached each model call and which tools ran.
The sidebar's memory pages show personas, toolbox, conversations, workflows,
knowledge, entities, summaries, skills and cache entries. Tool logs appear in
the playground Memory panel.

The expected readiness answer is: budget USD 1,200, committed spend USD 940,
headroom USD 260, owner Mina, earliest launch after 18:00 UTC, **HOLD** because
rollback verification has not been completed. After compaction, the answer to
the timing question must still be after 18:00 UTC. Repeating that same timing
question must produce a semantic-cache hit with no new model call.
