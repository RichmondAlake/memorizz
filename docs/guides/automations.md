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
| `one_shot` | optional `interval_seconds`, or `start_at` through lower-level job APIs | Runs once, after that delay, then disables |

Timezone resolves from the method argument, `agent.default_timezone`, then
`MEMORIZZ_DEFAULT_TIMEZONE`. Only a cron time depends on it, so `interval` and
`one_shot` jobs fall back to this machine's timezone. A cron job created in Python
or on a `multi_tenant` agent has to name one. So does one on a machine whose
timezone can't be read, as on Windows or in some containers. Otherwise an agent
working for the person at this machine uses this machine's. An invalid timezone
fails before the job is stored.

Query templates support `{today_iso}`, `{scheduled_for_iso}`, `{now_utc_iso}`,
and `{timezone}`. Treat the template as persistent instructions: review it for
prompt injection, excessive authority, secrets, and unstable external data.

A `one_shot` job runs once, after `interval_seconds` if you give it, and straight
away if you don't.

## Timers

For a countdown someone wants to watch, such as a pomodoro or a tea timer, an
automation is the wrong tool: it needs a worker and approval, and it runs the
agent when it ends. An agent has `start_timer`, `list_timers` and `cancel_timer`
instead. A timer needs no approval, runs nothing when it ends, and makes no model
call. You can also start one without the agent:

```text
/timer 25 focus     25 minutes
/timer 90s tea      90 seconds
/timer 1h           one hour
/timer              list what's running
/timer cancel       stop the only one running, or give its id
```

In the CLI, a timer counts down in the status bar, and when it ends you get a
line above the prompt and a terminal bell. In the local UI it's a live card above
the chat box, with a chime and a button to cancel or dismiss it.

Timers belong to the person at this machine. They live in
`~/.memorizz/timers.json`, so the CLI and the UI show the same ones, and one keeps
counting with both closed. If it ends then, the CLI tells you at the next launch,
and the UI shows it as finished. A timer can run for up to 24 hours. Agents with
`multi_tenant=True` don't get these tools.

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
