# Scheduled Automations

Automations persist cron, interval, or one-shot MemAgent queries and execute
them through a worker. Results return to conversation memory and can optionally
be delivered through Twilio WhatsApp.

## Authority model

- A direct SDK call to `create_automation()` is trusted host code.
- When the model proposes creating, pausing, resuming, deleting, or immediately
  running a job, Memorizz emits a durable approval proposal. The exact tool,
  arguments, hash, owner, checkpoint, reason, and expiry are stored before a
  host approves and resumes it.
- There is no model-visible `confirm` or `approved` argument.
- Listing jobs is read-only and does not require approval.

Automations are currently **agent-owned**, not independently user-owned. A job
stores an `agent_id` and target `memory_id`, but not a `user_id`. In a
multi-tenant application, use a dedicated agent/provider scope per tenant or
keep scheduling in a tenant-aware host service. Do not attach user-specific
automations to one globally shared agent.

## Create a job with the SDK

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_name("Operations scheduler")
    .with_llm_config({"provider": "openai", "model": "gpt-4o-mini"})
    .with_memory_ids("operations-automations")
    .with_default_timezone("Europe/London")
    .build_and_save()
)

if not agent.has_automations():
    raise RuntimeError("The selected memory provider does not support automations")

job = agent.create_automation(
    name="Weekday operations digest",
    schedule_type="cron",
    cron_expr="0 8 * * 1-5",
    timezone="Europe/London",
    query_template="Summarize open operational issues for {today_iso}.",
    memory_id="operations-digest",
)

print(job.job_id, job.next_run_at)
```

Filesystem, MongoDB, and Oracle providers support automation storage. Oracle
deployments must run the current schema setup/migrations first.

## Schedule types

| Type | Required field | Behavior |
|---|---|---|
| `cron` | five-field `cron_expr` | Runs in the configured IANA timezone |
| `interval` | `interval_seconds` | Runs after each fixed interval |
| `one_shot` | optional `start_at` through lower-level job APIs | Runs once, then disables |

Timezone resolves from the method argument, `agent.default_timezone`, then
`MEMORIZZ_DEFAULT_TIMEZONE`. A missing or invalid timezone fails before the job
is stored.

Query templates support `{today_iso}`, `{scheduled_for_iso}`, `{now_utc_iso}`,
and `{timezone}`. Treat the template as persistent instructions: review it for
prompt injection, excessive authority, secrets, and unstable external data.

## Manage jobs

```python
jobs = agent.list_automations(enabled=True)
selected = agent.get_automation(job.job_id)
agent.pause_automation(job.job_id)
agent.resume_automation(job.job_id)
agent.trigger_automation(job.job_id)
runs = agent.list_automation_runs(job.job_id, limit=20)
deleted = agent.delete_automation(job.job_id)
```

`trigger_automation()` makes the job due; a worker still performs the run.
Retries and misfire behavior are bounded by each job's policy.

## Run the worker

```bash
export MEMORIZZ_BACKEND="filesystem"
memorizz automations run --poll-interval 5 --lease-seconds 120 --concurrency 2
```

Use `MEMORIZZ_BACKEND=mongodb` with `MONGODB_URI`, or
`MEMORIZZ_BACKEND=oracle` with `ORACLE_USER`, `ORACLE_PASSWORD`, and
`ORACLE_DSN`. The worker and the process that created the agent must resolve the
same provider and `MEMORIZZ_HOME`.

The local UI can run an embedded worker when enabled in Settings. A dedicated
worker is easier to supervise and scale in a deployed environment.

## WhatsApp delivery

Set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_WHATSAPP_FROM`, then
pass `whatsapp_to=["+15551234567"]`. Credentials remain in the worker
environment. Review recipient consent, data residency, retention, retry, and
provider messaging requirements before sending agent output.

## Operate safely

- Keep workers single-purpose and least-privileged; separate high-authority
  tools from routine digest agents.
- Use stable job IDs/client request IDs where the model-facing tool offers them
  so retries are idempotent.
- Bound execution time, retries, concurrency, and delivery fan-out.
- Monitor job/run status and provider errors; an accepted schedule does not
  prove that later model or tool calls succeeded.
- Record application-verified task outcomes when automation results feed
  continual learning.
- Pause jobs before changing model, tool, prompt, tenant, or data-version
  assumptions.

The local UI exposes job creation, history, pause/resume, run-now, and deletion
under **Automations**. Model-originated changes use the same durable approval
lifecycle described in [Tools, Safety, and Human Approval](tools-and-approvals.md).
