# Context Efficiency & Prompt Caching

MemoRizz assembles every agent turn to be **prompt-cache friendly** and
**deduplicated**, so multi-turn conversations and tool loops are billed at
each provider's cached-input rate instead of full price, and the model never
pays twice for the same memory.

## The prompt layout

Prompt caching (OpenAI and Anthropic alike) is a **byte-exact prefix match**:
the first changed byte invalidates the cache for everything after it.
MemoRizz therefore orders every request stable-prefix → volatile-tail:

```
┌─────────────────────────────────────────────────────────────┐
│ tools           deterministic order (sorted by name)        │  stable
│ system prompt   frozen for the session                      │  prefix
│ history         append-only, chunk-evicted                  │  (cached)
├─────────────────────────────────────────────────────────────┤
│ final user message                                          │  volatile
│   <memorizz:context>                                        │  tail
│     retrieved memories (deduplicated)                       │  (never
│     entity memory facts                                     │  cached,
│     tool-log digest                                         │  never
│     per-call request context                                │  breaks
│   </memorizz:context>                                       │  the
│   the user's query                                          │  prefix)
└─────────────────────────────────────────────────────────────┘
```

Everything that changes per turn — retrieved memories, entity facts, the
tool-log digest, the M2 `context={...}` request payload — is rendered into
the **final user message**, where it can't invalidate the cached prefix.
The volatile block is never persisted to `conversation_memory`; only the raw
query is recorded.

### Append-only history with chunked eviction

Instead of recomputing a token-fitted history window every turn (which moves
the window start every turn and breaks the prefix), MemoRizz evicts history
in chunks of 20 messages: the window start stays byte-stable for many turns,
then jumps a whole chunk at once — one cache miss per chunk instead of one
per turn.

## Provider support

### Anthropic

The `Anthropic` LLM provider attaches `cache_control` breakpoints
automatically (`enable_prompt_caching=True` by default):

- on the **system prompt** — caches tool schemas + system text for every
  tool-loop iteration and follow-up turn;
- on the **final message** — the read point for the next iteration of the
  same turn's tool loop;
- on the **second-to-last message** — the cross-turn read point that lets
  the growing conversation accrue incremental cache hits.

Request construction is isolated from MemAgent's reusable history: MemoRizz
annotates provider-owned deep copies, so cache metadata cannot leak into a
later tool-loop iteration. It also enforces
[Anthropic's request-wide maximum of four breakpoints](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
across tools, system content, and messages. Existing caller-supplied
breakpoints are preserved and consume the budget before MemoRizz adds its
own; an already-invalid request containing more than four fails locally
before an API call.

In live testing this serves ~99% of the prompt from cache from turn 2
onward. Reads bill at ~0.1× input price; disable with
`Anthropic(..., enable_prompt_caching=False)`.

### OpenAI

[OpenAI caches automatically](https://developers.openai.com/api/docs/guides/prompt-caching)
for prompts ≥ 1024 tokens, keyed on the exact prefix. MemoRizz maximizes hits
by:

- pinning a **`prompt_cache_key`** per conversation thread
  (`memorizz:{agent_id}:{memory_id}:{thread_id}`), so requests route to the
  same cache shard;
- optional **`prompt_cache_retention="24h"`**
  (`OpenAI(..., prompt_cache_retention="24h")`) for long-lived agents on
  models that support extended retention.

Unlike the Anthropic integration, MemoRizz does not add cache annotations to
OpenAI message blocks. It relies on OpenAI's default implicit server-side
breakpoint, so there is no reusable-message mutation or client-side
breakpoint accumulation. GPT-5.6 and later also support explicit
`prompt_cache_breakpoint` blocks with a four-write limit, but MemoRizz does
not currently emit those blocks. `prompt_cache_retention` is the legacy
control for models before GPT-5.6; newer model families use
`prompt_cache_options.ttl`.

Cache parameters are only sent to the official endpoint — local
OpenAI-compatible servers (llama.cpp, LM Studio, vLLM) are left untouched.

### Verifying cache hits

`agent.model.get_last_usage()` now surfaces cache metrics on both providers:

```python
agent.run("first turn")
agent.run("second turn")
print(agent.model.get_last_usage())
# {'prompt_tokens': 7844, 'completion_tokens': 20, 'total_tokens': 7864,
#  'cached_tokens': 7748, 'cache_read_input_tokens': 7748}
```

If `cached_tokens` stays 0 across turns, something is mutating your prefix —
the usual suspects are a per-turn timestamp in a custom instruction or a
tool set that changes between requests.

## Pre-inference deduplication

Everything retrieved for a turn (episodic recall, knowledge-base chunks)
passes through a dedup/selection pipeline
(`memorizz.memagent.utils.context_dedup`) before it reaches the context
window:

1. **Exact dedup** — normalized-content fingerprints across all sources.
2. **History dedup** — anything already present (verbatim or contained) in
   the conversation window going into this prompt is dropped.
3. **Near-dup dedup** — cosine similarity ≥ 0.95 between stored embeddings
   (vectors come back with the rows, so this costs no extra embedding
   calls).
4. **MMR selection** — maximal marginal relevance (relevance blended with
   recency decay) picks a small, diverse subset.
5. **Stable ordering** — selected memories render oldest-first with
   deterministic tie-breaks, so the block doesn't reshuffle between turns.

Write-path guards complement it: identical consecutive `(query, response)`
pairs (semantic-cache hits, client retries) are not double-written, and
entity upserts that add no new information skip the re-embed and re-store
entirely.

## Choose automatic retrieval explicitly

Memory types describe what an agent can store and expose through tools.
`RetrievalPolicy` independently controls what semantic matches are injected
before inference:

```python
from memorizz import MemAgent, MemoryType, RetrievalPolicy

agent = MemAgent(
    memory_provider=provider,
    memory_types=[MemoryType.CONVERSATION_MEMORY, MemoryType.KNOWLEDGE_BASE],
    retrieval_policy=RetrievalPolicy(
        conversation_scope="thread",
        knowledge_base_scope="namespace",
        knowledge_base_namespaces=("agent-harness",),
    ),
)
```

Use `RetrievalPolicy.disabled()` when the application already has explicit,
tenant-scoped search tools. Exact current-thread history still loads normally;
only automatic semantic recall is disabled.

Semantic response caching is session-scoped by default. Its fingerprint also
includes per-request context, so the same query on a different page or quoted
selection is a miss rather than a stale replay.

## Fewer embedding calls

- `EmbeddingManager` memoizes text → vector (LRU): the same query used by
  the semantic cache, episodic recall, KB recall, and entity search within
  one turn is embedded **once**.
- Conversation rows are written with `embedding=None` and backfilled by a
  single background worker, so the user-facing turn never blocks on the
  embedding API. Disable with `MEMORIZZ_DISABLE_CONVERSATION_EMBEDDINGS=1`.
- Automatic context summarization runs on a background thread
  ("sleep-time" consolidation) instead of blocking the turn that crossed
  the threshold.
