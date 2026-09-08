# Usage, cost and memory analytics

Open **Observability → Usage & memory analytics** (`/traces/usage`) for daily
token/charge trends, agent and interaction tables, memory prompt-share charts,
and per-type retrieval latency. Selected traces also have a collapsible
**Tokens, charges & memory** panel using exactly that trace window. Evalground
shows recorded quality, latency and API-cost estimates alongside run history.

The page supports agent, model, date and IANA timezone filters. Date inputs are
UTC; the timezone controls daily grouping. Trace/thread/turn filters survive
navigation and JSON export. Interaction links reopen the corresponding trace.
Missing dates use an explicit `Unknown date` bucket; missing measurements are
not inferred from response length.

## What the numbers mean

| Metric | Source and limitations |
| --- | --- |
| Input/output tokens | Provider response usage, including usage-bearing streaming events. Missing usage is unknown. |
| Calculated charge | Measured token counts multiplied by a versioned rate card, calculated with `Decimal`. Not a provider invoice. |
| Memory prompt tokens | Approximate `ceil(rendered_characters / 4)`, per memory type and model call. Not exact tokenizer output or billed-token allocation. |
| Retrieval latency | Measured wall time around instrumented memory reads, including failures. Not causal attribution of model generation latency. |
| Model latency | Recorded call duration; streaming follows the existing provider-stream timing contract. |
| Evalground cost | Benchmark-reported estimate, not invoice reconciliation. Local hardware cost is not included. |

Memory types include history, episodic recall, knowledge base, entity memory,
summary references, learned skills, personalization, persona style, tool-log digests and mixed
evidence packs. Measurements count **rendered** memory text after selection,
history trimming and skill/workflow filtering. System instructions (apart from
the separately measured persona style), tool schemas and ephemeral request
context are not classified as stored memory. Common
retrieval-section headers and message framing are not allocated to individual
memory types. Multimodal/non-text history is not assigned a text-token estimate.

Supplying the same memory in two model calls counts twice; retrieving it once
counts as one retrieval. Cache discounts cannot reliably be allocated to each
memory type from provider totals, so the UI does not invent per-type dollar
charges. Mixed personalization/evidence-pack rendering stays in a mixed bucket.
Host-provided personalization, explicit memory tools, compression and other
uninstrumented operations may have no per-type retrieval samples. Add host
instrumentation where needed; historical measurements cannot be reconstructed.

### Pricing coverage

The offline default catalog covers standard direct OpenAI text-model pricing,
verified September 7, 2026. It includes input, output, cache reads, GPT-5.6/GPT-6
cache writes and applicable long-context multipliers. Rates come from
[official OpenAI pricing](https://developers.openai.com/api/docs/pricing),
[prompt-caching usage semantics](https://developers.openai.com/api/docs/guides/prompt-caching),
and the corresponding model pages, such as
[GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra).

Unknown providers/models/tiers or incomplete token details yield `cost_usd: null`
and a reason. This includes local models: unconfigured pricing does not mean
free hardware. OpenAI-compatible custom endpoints are identified as
`openai_compatible` and never inherit OpenAI list pricing automatically.

Actual response `service_tier` is retained. If absent, the default standard tier
is explicitly assumed in the quote's `cost_reason`. Configure separate cards
for priority/flex/batch, regional premiums, other providers or contract pricing.
The built-in catalog does not discover an account's region, discounts or billing
contract. Tool fees, image/audio modalities, embeddings, taxes, unobserved
provider retries, credits and billing reconciliation are outside these totals.

New model results persist a pricing version, date, scheme-free public source
reference and calculated charge. These snapshots are not repriced when the
catalog changes. Older results without a snapshot can be calculated using the
currently configured catalog; the API/UI labels that basis `current_catalog`,
not historical invoiced spend. Known charges exclude unpriced calls, with the
priced/total call count shown beside every subtotal.

## Developer APIs

No UI or network access is required:

```python
from memorizz.observability import aggregate_usage, query_usage

# Inspect the latest turn in the same execution context as the completed run.
latest = agent.last_usage_analytics(timezone_name="Europe/London")

# Aggregate normalized events from your existing recorder/export pipeline.
summary = aggregate_usage(events, timezone_name="UTC", max_events=10_000)
print(summary["totals"]["cost_usd"])  # Decimal string, or None
print(summary["totals"]["unpriced_calls"])
print(summary["memory"])

# Query historical records. The host must authorize these identity filters.
summary = query_usage(
    provider,
    filters={
        "application_id": "workspace-7",
        "user_id": "user-42",
        "agent_ids": ["researcher"],
        "start_time": "2026-09-01T00:00:00Z",
        "end_time": "2026-09-07T23:59:59Z",
    },
    max_events=10_000,
    timezone_name="Europe/London",
)
```

SDK aggregation is a data utility, not an authorization boundary. Scope inputs
to the authorized tenant first. The UI performs this enforcement using the
existing trace principal and audit controls and serves analytics with
`Cache-Control: no-store`.

### Custom rate cards

```python
from memorizz.observability import DEFAULT_PRICING, PricingRegistry, RateCard
from memorizz.ui.app import create_app

# Illustrative contract rates; replace with your actual provider agreement.
custom = RateCard(
    provider="private_provider",
    model="reader-v1",
    input_per_million="1.00",
    cached_input_per_million="0.10",
    output_per_million="3.00",
    source_url="https://example.com/public-pricing",
    as_of="2026-09-07",
    version="reader-contract-v1",
    service_tier="default",
)
registry = PricingRegistry([*DEFAULT_PRICING.cards.values(), custom])
agent.usage_pricing = registry  # Set at application setup, before concurrent runs.
app = create_app(usage_pricing=registry)
```

Registries and rate cards are immutable. Pass `pricing=registry` to either SDK
aggregation function. Configure each service-tier card separately. A custom
`max_input_tokens` can refuse unsupported context sizes; alternatively use
`long_context_threshold`, `long_input_multiplier` and `long_output_multiplier`.
Source URLs must be public HTTPS URLs without credentials, queries or fragments.
Do not put sensitive contract identifiers in versions or source references.

The pricing contract expects `input_tokens` to **include** cached and cache-write
tokens; output tokens include billable reasoning tokens already reported by the
provider. Normalize provider-specific usage before quoting: some APIs report
uncached input and cache reads/writes separately. Never double-count reasoning
tokens or add provider totals to input/output totals.

### Host memory instrumentation

The public recorder accepts content-free memory metrics:

```python
recorder.record_event(
    "knowledge.read",
    kind="memory_retrieval",
    duration_ms=elapsed_ms,
    attributes={"memory_type": "knowledge_base"},
)
recorder.record_event(
    "knowledge.supplied",
    kind="memory_supply",
    attributes={
        "memory_type": "knowledge_base",
        "memory_chars": rendered_chars,
        "memory_tokens_estimate": (rendered_chars + 3) // 4,
        "token_estimation_method": "rendered_chars_div_4",
    },
)
```

Use the same tenant/agent/turn context and distinct event IDs. Emit a supply
measurement per model call, but emit retrieval duration once per operation.
Do not put prompts, memories, account addresses or credentials in attributes.
With the model filter selected, only supply events carrying that model identity
are included; pre-model retrieval timings cannot be attributed to a model and
are omitted rather than guessed.

## Efficiency and completeness

Agent instrumentation counts text already rendered for the prompt, uses
`perf_counter` around existing reads and emits bounded scalar metadata. It does
not make extra memory reads, tokenize the full prompt, or fetch price pages.
Memory counters are isolated per execution context and reset every turn.

The aggregator processes a bounded iterable once, deduplicates scoped result
spans, then sorts bounded groups/latency samples for tables and percentiles. It
does not sum start events or parent rollups. Defaults cap aggregation at 10,000
events; the SDK hard ceiling is 100,000. The UI defaults to 5,000 and caps at
10,000. Native provider event-index queries are preferred; legacy conversation
and tool stores are queried through existing bounded pagination adapters.
Synchronous analytics routes run reads in FastAPI's worker pool, not its event
loop. Charts are server-rendered SVG/HTML with text/table equivalents, no CDN,
browser chart library, polling or new runtime dependency.

Inspect `coverage.read_complete`, per-store read results and
`coverage.aggregation_truncated` before treating a daily total as a complete
recorded window. A bounded page is **not** complete historical accounting;
filter by agent/time or aggregate your authorized event-export pipeline for
larger workloads. Even a complete read cannot prove uninstrumented calls never
occurred. The latest-turn and legacy task/thread-memory paths explicitly leave
overall read completeness unestablished.

## Local verification

```bash
python -m pytest tests/unit/test_usage_analytics.py tests/unit/test_observability_operator_security.py
python tests/browser/usage_app.py
# In another terminal, with Node 18+ and Playwright available:
node tests/browser/usage.cjs
```

Browser fixtures use temporary synthetic data and require no paid provider calls.
Set `MEMORIZZ_PLAYWRIGHT_MODULE` to a local Playwright module path if necessary.
