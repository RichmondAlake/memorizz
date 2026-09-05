# Observability rollout and operator guide

This guide covers the implemented Memorizz roadmap: native trace indexes,
selection explanations, coverage, secure operator workflows and rollout gates.
It does not configure, repair or deploy any host application. In particular,
no OpenSpeech integration or production data change is part of this work.

Passing repository tests is not an end-to-end incident all-clear. The independent
5 September audit identified additional semantic, legacy-scope and operator-UI
defects; the remediation is covered by
`tests/unit/test_observability_audit_remediation.py` and both browser gates below.
Host source-to-artifact and parser/browser evidence, configured lookup hooks,
the deployed release identity and production rollout checks remain separate
acceptance requirements. Missing historical evidence cannot be manufactured.

## Reading evidence without overclaiming coverage

Analysis, health and inspector responses distinguish three independent facts:

| Field | Meaning |
|---|---|
| `read_completeness` | Whether the selected stored-event window was loaded without truncation or normalization/query errors; never a guarantee that unwritten events exist |
| `instrumentation_coverage` | `unknown` without declared profiles, `partial` when required stages are missing, or `complete` when all selected task scopes declare profiles whose stages were observed |
| `outcome_verification` | `recorded` when verified outcome evidence is present, otherwise `unknown`; recorded evidence may describe failure and does not itself establish end-to-end success |

An uninstrumented legacy trace displays “All stored events loaded; end-to-end
coverage unknown.” Absent metrics use `value: null` and
`state: no_recorded_evidence` for a complete read, or `state: unknown` for an
incomplete/untrusted read. Summary numeric counts remain **recorded-event
counts**, not assertions that no artifact was created or no browser delivery
occurred. The low-level paginated query `coverage` field remains a read-page
compatibility alias; never use it as a task-success signal. Event exports include
`evidence_coverage` with the separate semantics.

Detail, analysis and health use the same authorized agent-or-registered-memory
selection, conjoined with tenant/thread scope. Their combined store selection is
`conversation`, `tool_log`, `trace`; `/traces/events.json` defaults explicitly to
`trace` only. To export the combined bounded snapshot, use `store=all` with the
same agent/thread/root/turn/task/thread-memory selection, including run and
child-event time bounds. Combined export is metadata-only, up to
1,000 normalized events from 500 source rows per store; it is not a historical
pagination API. Use the per-store cursor API for longer histories.
Per-store exports reject task/thread-memory filters they cannot apply; they do
not silently broaden those requests. Failure to load registered agent-memory
associations returns HTTP 503, not a falsely complete narrower selection.

Narrowed root/turn counts use distinct recorded source provenance. Parent query
counts and deduplication are exposed separately as `parent_query_metadata`;
unknown selection-local deduplication stays null. A source row may contribute to
multiple selections, so selected-root counts are not necessarily additive.
Comparison applies its 1,000-event cap **after** selecting each root independently;
unrelated roots cannot consume the selected root's event budget. The parent
source-row window is still bounded and its truncation remains visible.

Finder results group events by root/task and preserve thread/root/turn/task/event
identity in navigation. The selected event is highlighted; health/comparison
links preserve the selection. The agent/thread overview is collapsed and bounded
when an incident is selected. Empty lineage/artifact/contract panels explain the
missing fields and host recording boundaries, and memory-only traces retain
aggregate evidence without inventing source bindings.

`GET /traces/capabilities.json` reports permission-and-hook availability. The UI
disables account resolution, current-artifact lookup and replay controls when
their `identity_resolver`, `artifact_resolver` or `resource_authorizer` callback
is absent. It does not install callbacks or connect to an application database.
The existing audited, tenant-aware `create_app(...)` hook contract still applies.
Legacy replay source arrays are decoded and type-checked centrally; `[]`,
JSON-encoded `"[]"` and null generate no references. Malformed arrays are rejected
before resource authorization, never interpreted as characters.
The same parser protects direct diagnostic/lineage calls; malformed raw list
representations raise `ValueError`. Stored malformed source/task metadata is
reported as untrusted normalization evidence.

Reveal, inspection and inert replay preserve thread, turn, task, run, memory and
time selection. A reused event ID must resolve to exactly one event for reveal;
otherwise the endpoint returns HTTP 409 and requests a narrower turn/run.
Replay freezes the selected evidence window, not necessarily the entire root.
Unknown earlier pages, untrusted metadata and non-advancing/duplicate pagination
cannot become a complete replay window just because the last page is complete.
Incomplete replay windows fail before invoking a host resource authorizer.

## Native storage and rollout

Canonical source envelopes remain private shared-memory `trace_bundle` v2
records containing legacy or typed v3 children. New Memorizz trace writes are
immutable: the first writer owns a tenant-scoped event/turn ID. An external-ID
retry preserves the original timestamp and evidence. A changed observation,
artifact revision or new attempt requires a new external ID/turn. Recommendations
and experiment drafts retain their existing mutable lifecycle.

| Provider | Native storage | Provisioning |
|---|---|---|
| Filesystem | `_observability/index.sqlite3`, outside semantic recall | Explicit `initialize()`, WAL, private directory/database permissions |
| MongoDB | Dedicated `observability_spans`, `observability_bundles`, `observability_previews`, `observability_state` collections | Explicit compound indexes for time, tenant, agent/thread, root, kind/status, resource, memory, job and error |
| Oracle | Dedicated `obs_spans`, `obs_bundles`, `obs_resources`, `obs_previews`, `obs_state` tables | Additive `008_observability.sql`, schema-qualified binds/CLOBs, explicit initialization |
| Other providers | Existing bundle compatibility queries | No automatic native-index support is claimed |

Reads never provision tables or modify connection/session schema. The Oracle
migration ignores only “object already exists”; other DDL failures propagate.
Use a migration identity with DDL permission, then a restricted runtime identity.
SQLite/Oracle writes are transactional and serialized; Mongo event replacements
are atomic with a durable pending checkpoint until all children are written.
Pending/failed index writes make coverage untrusted. Canonical source writes
survive index failure; inspect process counters and backfill to recover.

```python
from memorizz.observability import ObservabilityMaintenance

# provider is the explicitly selected Memorizz provider, not an application DB.
maintenance = ObservabilityMaintenance(provider)
maintenance.initialize()  # deliberate, additive provisioning

# Review one page before making index writes.
print(maintenance.backfill(since="2026-09-01T00:00:00Z"))  # dry run
cursor = None
while True:
    result = maintenance.backfill(
        since="2026-09-01T00:00:00Z", limit=100,
        cursor=cursor, dry_run=False,
    )
    if result["failed"]:
        raise RuntimeError("Backfill failed; inspect the content-free error codes")
    cursor = result["next_cursor"]
    if not cursor:
        break

check = maintenance.parity(start_time="2026-09-01T00:00:00Z")
assert check["passed"], check
```

Use these independent deployment flags:

| Setting | Default | Meaning |
|---|---|---|
| `MEMORIZZ_OBSERVABILITY_DUAL_WRITE` | false | Store the canonical envelope and write its normalized native index |
| `MEMORIZZ_OBSERVABILITY_READ_PATH` | bundles | `bundles` compatibility path or explicitly selected `index` path |

Enable dual-write only after provisioning. Start it **before** backfilling so
newly arriving events are covered; repeat the final parity check on a quiescent
window or with producers briefly paused. Cut over reads only after parity and
the live-provider/reliability gates pass. The flags do not grant permission to
migrate or delete data. Index readiness by itself is not proof of full backfill.

Parity compares exact, content-free event projections, not just row counts.
It is bounded by `max_pages` (default 100 pages of 1,000 events); a larger or
incomplete window fails closed. Narrow the time/tenant scope or increase the
bound explicitly. Do not compare an expired index against unexpired source
history and interpret deliberate retention differences as capture failures.

Rollback: set `MEMORIZZ_OBSERVABILITY_READ_PATH=bundles`. Backfill never rewrites
the originals. Leave dual-write enabled while investigating or disable it
explicitly. Once old source envelopes are expired, rollback for that historical
window requires the archive; the live bundle path alone cannot reconstruct it.

### Query contract

```python
page = provider.query_trace_events(
    agent_ids=["agent-1"], memory_ids=["memory-1"],
    application_id="app-1", user_id="user-1", thread_id="thread-1",
    resource_refs=["analysis-1"], limit=250,
)
next_page = provider.query_trace_events(
    agent_ids=["agent-1"], memory_ids=["memory-1"],
    application_id="app-1", user_id="user-1", thread_id="thread-1",
    resource_refs=["analysis-1"], limit=250, cursor=page["next_cursor"],
) if page["next_cursor"] else None
```

Agent and memory are alternatives; application, user and thread scope are
conjunctive. Omitting `user_id` is an unscoped SDK query; `user_id=None` selects
anonymous evidence only. SDK callers must supply authorized scope. UI and MCP
tools bind scope independently and never let model-provided IDs override it.
Additional filters: `root_trace_id`, `run_id`, `turn_id`, `event_kinds`, `statuses`,
`tool_name`, `success`, `query` (exact identifier), `start_time`, `end_time`.
Native cursors bind timestamp plus event key and query scope; reject cursor reuse
under different filters. New head insertions do not duplicate continuation rows.

`index.summaries(...)` aggregates event/bundle counts in the database without
deserializing historical source payloads. The UI uses bounded aggregate groups
in index mode (first 1,000 agent/thread groups), and separately loads a bounded
event window. Overview and timeline counts consequently have distinct scopes.
`window_complete=False` on a continuation page is deliberate; its absence of
errors does not make it a complete incident.

## Separate retention and source expiry

Default native policies are 7 days for previews, 30 for metadata, and 365 for
verified outcomes. Policies require `previews <= metadata <= outcomes`.

```python
print(maintenance.retention())  # dry-run counts only
# After reviewing the policy and target:
maintenance.retention(previews_days=7, metadata_days=30,
                      outcomes_days=365, dry_run=False)
```

New dual-written canonical envelopes are metadata-only. Raw previews have one
separately expirable native copy. Content is capped at 16,000 characters and
is never included in index queries or summaries. An idempotent retry cannot
repopulate a purged preview while its bundle checkpoint exists. Historical
backfill can reintroduce expired telemetry; use a recent `since` and reapply
retention before enabling operator access after a rebuild.

Old/non-dual-written bundles may still contain previews. Native retention does
**not** scrub those immutable sources. Review and archive/expire their exact
envelopes separately:

```python
plan = maintenance.plan_source_expiry(before="2026-08-01T00:00:00Z", limit=100)
# Review plan entries and ensure indexed metadata/outcomes are retained first.
# archive(payload) must durably preserve this envelope and return exactly True.
# The archive needs its own access controls and retention policy.
result = maintenance.apply_source_expiry(
    plan, confirmation=plan["confirmation"], archive=archive,
)
```

This is an explicitly destructive maintenance API, not a UI action or automatic
background job. It rejects changed fingerprints, requires archival acknowledgement,
and compare-and-deletes the exact source record (including Mongo physical IDs).
Only trace bundles in the reviewed plan are eligible; conversation/semantic
memory is not deleted. A multi-record failure may leave an archived, partly
applied batch; inspect the provider and generate a fresh plan. Archives, backups
and old replicas are outside the native preview purge boundary.

## Explainable memory selection and worker coverage

Runtime deduplication records up to 64 candidates: typed opaque resource ref,
provider rank/score, selected flag and the actual selection/rejection reason.
Reasons distinguish MMR/provider ranking, conversation history, exact/near/parent
duplicates and budget exhaustion. The runtime does not invent provenance or
claim to have observed candidates rejected inside a remote provider.

Hosts can use `SelectionDecision` and `recorder.record_selection(...)` to record
canonical-source, wrong-thread/tenant, stale-version or application policy decisions.
The Memory Lineage view separates expected, retrieved, supplied and output-bound
refs. Explicit reference metadata is required; prose/model claims are not proof.

```python
from memorizz.observability import register_coverage_profile, ObservabilityRecorder

register_coverage_profile(
    "document_workflow", required={"intent_plan", "artifact_persisted"},
    optional={"ui_delivery"},
)
recorder.record_intent("document", expected_artifact_types=["doc"],
                       coverage_profile="document_workflow")

# After the host validates job ownership and obtains an authorized fallback scope:
worker = ObservabilityRecorder.for_worker(
    provider, job_trace_carrier, fallback_context=authorized_context,
)
```

Register profiles during application startup in each process. Replacing an
existing profile requires explicit `replace=True`. Required/optional stages are
disjoint and bounded; missing required stages mean partial coverage. A missing
or invalid worker carrier emits `trace_context_missing` in the authorized
fallback scope, or raises if no fallback was supplied. Carriers never authorize
resource access. Intent, parser, browser acknowledgement, current ownership and
artifact persistence remain host-reported facts; the SDK cannot observe these
without host integration.

## Scoped operator roles and host hooks

Existing `MEMORIZZ_UI_AUTH_TOKEN` remains an unscoped administrator token for
backward compatibility. For multiple operators, configure
`MEMORIZZ_UI_AUTH_ACCOUNTS` as a JSON object of operator names to token, role,
optional `application_id`, and optional `user_id`:

```json
{
  "reader-1": {
    "token": "replace-with-a-long-random-secret",
    "role": "viewer",
    "application_id": "app-1",
    "user_id": "user-1"
  }
}
```

Remove the legacy admin token when it is not needed. Tokens must be unique and
at least 16 characters; use generated secrets in deployment. A present null
`user_id` binds an anonymous account. Omitted scope is broad access within the
other configured constraints—do not omit it accidentally. Cookie sessions are
signed, expire and resolve the operator's current configured role. Configure the
provider with the host/admin connection flow before handing scoped operators a
trace URL; they cannot use connection/settings or generic memory pages.

| Role | Metadata trace reads | Current artifact lookup / replay draft | Transient account resolution | Raw reveal |
|---|---|---|---|---|
| viewer | yes | no | no | no |
| analyst | yes | yes | no | no |
| operator | yes | yes | yes | no |
| admin | yes | yes | yes | yes, if content mode allows |

Any tenant-bound account is limited to trace routes. `MEMORIZZ_UI_READ_ONLY=true`
also blocks source/index/draft writes, including nested native-index APIs;
read-only users can export an inert replay plan. Explicit query scopes may only
narrow configured scope. Cross-origin trace POSTs are rejected. Metadata is the
default on the page; content is fetched only by an explicit, audited reveal.
The existing export endpoint remains an explicit administrator content action
when its content mode allows it. Configure a writable private audit destination:
privileged lookups, reveals, replay and raw export fail closed if auditing fails.

Embed host integrations through `create_app(identity_resolver=...,
artifact_resolver=..., resource_authorizer=...)`. Hooks can be sync or async:

| Hook | Input | Required result |
|---|---|---|
| identity resolver | transient email, authorized application/user scope | bounded opaque `user_id` and optional `application_id` in scope |
| artifact resolver | typed resource ref, authorized scope | `ownership_verified=True`; optional `exists`, `version`, `title_fingerprint` |
| resource authorizer | frozen typed ref/version, trace tenant scope | exactly `True` after checking current access |

An absent hook returns 501; lookup failures return content-free errors. Emails
are sent only in POST bodies, cleared from the input, not stored as traces or in
audit/query strings, and never echoed in validation errors. Hosts/reverse proxies
must also disable sensitive request-body logging. The SDK does not connect to an
account directory or application artifact database automatically.

### Operator endpoints

| Endpoint | Purpose |
|---|---|
| `GET /traces/find.json` | Bounded metadata-only Incident Finder, exact opaque `q`, IDs/time filters and cursor |
| `GET /traces/inspect.json` | Captured lineage/artifact/contract inspection for agent + root |
| `GET /traces/artifact.json` | Authorized current artifact state through the host hook |
| `POST /traces/account/resolve` | Transient `{"email": ...}` resolution |
| `POST /traces/reveal` | Explicit event preview; agent, root and event ID required |
| `POST /traces/replays` | Authorized agent/root selection to an inert Evalground draft |
| `GET /traces/replay-plan.json` | Same authorization/frozen refs without persistence |
| `GET /traces/health[.json]` | Capture/query/index/registration health and deterministic alerts |
| `GET /traces/compare[.json]` | Structural baseline/candidate differences, never execution |

Replay freezes content-free refs and an evidence fingerprint. It requires a
complete, trusted, single-tenant trace and current authorization for every ref,
including rejected selection candidates. Unversioned refs are counted explicitly;
they are not immutable content snapshots. Drafts have `execution_enabled=false`,
network/side effects disabled, and sandbox/operator-approval requirements. They
appear as reviewable `observability_experiment` records; creating one does not
invoke a model, tool, queue or application action.

The optional MCP `memorizz_query_traces` tool is off by default. Enable
`MEMORIZZ_MCP_SERVER_ALLOW_TRACE_QUERIES=true` with the existing read scope and
agent exposure policy. Tenant identity comes from the authenticated principal
(explicit anonymous scope for an anonymous local caller), not tool arguments.
`explain=true` adds captured lineage and deterministic analysis. There is no raw
reveal or replay execution API over this tool.

## Health and release gates

Health reports schema/instrumentation versions, child counts, missing links,
registration mismatches, truncation, query latency and index readiness. Process
counters expose source/index write failures, dropped metadata and callback failures,
with bounded latency samples. No counters from another process means unknown,
not zero. Hosts can forward `pipeline_health(provider)`, `recorder.health`, and
deterministic health alerts to their monitoring service. Instrumentation deployment
versions must be supplied by the host in allowlisted event attributes.

Run the normal regression suite and the offline workload:

```bash
PYTHONPATH=src python -m pytest tests/unit tests/integration -q
PYTHONPATH=src python examples/observability/host_workflow.py
PYTHONPATH=src python examples/observability/benchmark.py --events 10000 --max-p95-ms 250
```

The synthetic SQLite benchmark uses 500-child bundles, verifies 250-event bounded
pages, aggregate/event parity, no cursor duplicates and no normalization errors.
The unit gate also asserts that summaries never deserialize trace payloads and
native reads never expand old source bundles. The audit-remediation rerun recorded
p50 10.677 ms, p95 18.673 ms and 1.321 seconds total indexing for 10,000 events. These are local
measurements, not production SLAs or MongoDB/Oracle performance claims.

### Live databases (mandatory before native production cutover)

The integration gate is opt-in and never starts Docker or discovers endpoints.
Supply isolated disposable targets using environment variables:

```bash
export MEMORIZZ_OBSERVABILITY_LIVE=1
export MEMORIZZ_OBS_TEST_MONGODB_URI="mongodb://127.0.0.1:27018"
export MEMORIZZ_OBS_TEST_ORACLE_USER="MEMORIZZ_OBS_TEST_RELEASE"
export MEMORIZZ_OBS_TEST_ORACLE_DSN="127.0.0.1:1522/FREEPDB1"
# Set MEMORIZZ_OBS_TEST_ORACLE_PASSWORD through your secret manager.
PYTHONPATH=src python -m pytest tests/integration/test_observability_live.py -q
```

Mongo creates and drops one randomly named test database. Oracle requires an
empty dedicated `MEMORIZZ_OBS_TEST_*` schema and removes only the five index
tables it creates. Never supply a production or host-application schema. With
the live gate enabled, missing configuration fails rather than silently skips.
Without opt-in, the eight cases (four per database) explicitly skip. To validate
one supplied target at a time, add `-k mongodb` or `-k oracle`; the selected
backend still fails on missing configuration.

For a local MongoDB gate without Docker or an existing server, obtain a trusted
standalone `mongod` binary for your platform and verify its vendor checksum.
The [official macOS tarball instructions](https://www.mongodb.com/docs/v8.0/tutorial/install-mongodb-on-os-x-tarball/)
describe the available standalone package. No system installation is necessary:

```bash
PYTHONPATH=src python tests/integration/run_observability_mongodb.py \
  --mongod /absolute/path/to/mongod
```

This launcher does not download software, start Docker or load a system config.
It starts one loopback-only child on an ephemeral port with temporary storage,
checks both the server PID and database path before tests can write, and runs
only the MongoDB cases. It overrides any inherited MongoDB test endpoint, stops
only its own child (including on failure), and removes the synthetic test data.
The binary is retained for reuse; no application database or service is used.

On 5 September 2026, all four cases passed against standalone MongoDB **8.0.26**
on macOS arm64/Python 3.12.13: the 74-event two-tenant contract, concurrent
idempotent retries and scope-bound cursors, durable interrupted-write recovery,
and typed resource lookup with dry-run/apply retention. Nine offline launcher
tests cover ownership checks, endpoint isolation, cleanup and failure paths.
This is real-server standalone validation, not Atlas/replica-set failover or a
production workload benchmark. The four Oracle cases subsequently passed on the
newly authorized isolated local Oracle instance and disposable
`MEMORIZZ_OBS_TEST_LOCAL` schema. See
`examples/observability/ORACLE-LOCAL.md` for its connection and repeatable test
commands. Docker was already running; only the new isolated container was
created. OpenSpeech and all existing application containers remain untouched.

Actual deployed runtime
credentials, source-store compatibility, concurrent producers, failover and
retention/backup policy must additionally be checked in the target environment.

### Browser gate

With a local Playwright installation and its Chromium browser available:

```bash
PYTHONPATH=src python tests/browser/observability_app.py
# In another terminal (Node >=18):
node tests/browser/observability.cjs
# Stop the fixture, then exercise unconfigured hooks and legacy evidence:
MEMORIZZ_BROWSER_TEST_HOOKS=off PYTHONPATH=src python tests/browser/observability_app.py
# In another terminal:
node tests/browser/observability_audit.cjs
```

The server binds localhost:8779, uses a temporary filesystem store and synthetic
data, and does not load or connect to application services. The browser validates
search/navigation, lineage, lane filtering, artifact lookup, explicit redacted
reveal, inert replay creation and mobile layout. Override the module/browser path
with `MEMORIZZ_PLAYWRIGHT_MODULE` / `MEMORIZZ_BROWSER_EXECUTABLE` if needed. Stop
the fixture server after validation; its temporary data is cleaned up.

Both fixtures include a busy 40-thread overview. The configured-hook gate also
uses a reused event ID in a second turn, checks the actual reveal/replay request
scope and verifies that the draft excludes the other turn's resources.
The audit browser gate checks
unknown end-to-end coverage, explicit missing-evidence panels, disabled hooks,
scoped navigation round trips, and incident placement within the first 1,200px
on desktop / 1,400px on mobile. Width and absence of JavaScript errors alone are
not treated as sufficient mobile usability evidence.

Before release, record the exact commit/wheel, provider versions, parity window,
load distribution, browser result, migration identity and rollback decision.
Repository implementation and synthetic validation do not certify production
deployment, repair historical artifacts, or fill host instrumentation gaps.
