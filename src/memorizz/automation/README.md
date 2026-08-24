# Automations — Durable Scheduled Agent Execution

Automations let you schedule MemAgent queries to run on a recurring basis (or once) without a human in the loop. Results are stored in the agent's conversation memory and can optionally be delivered via WhatsApp.

## Architecture

```
automation/
├── __init__.py       # Public exports (AutomationJob, AutomationRun, AutomationDelivery, etc.)
├── models.py         # Pydantic data models and type aliases
├── schedule.py       # Cron parsing, interval computation, timezone validation
├── runner.py         # Single-job execution and delivery dispatch
├── worker.py         # Background worker loop (claim → run → reschedule)
├── store/
│   ├── base.py       # AutomationStore protocol (interface)
│   ├── factory.py    # Provider-aware store resolution
│   ├── filesystem.py # Filesystem-backed implementation
│   ├── mongodb.py    # MongoDB-backed implementation
│   └── oracle.py     # Oracle-backed implementation
└── README.md         # This file
```

**Flow:** `worker.py` polls the store for due jobs → calls `runner.py` to execute each job via `MemAgent.run()` → reschedules the job based on its `schedule_type`.

## Requirements

- A filesystem, MongoDB, or Oracle memory provider
- For Oracle, provision the automation tables with
  `memorizz oracle setup-schema`
- `automations_enabled=True` on the agent (this is the default)

## Three Ways to Create Automations

### 1. SDK (Programmatic)

The recommended approach for integrations and scripts.

```python
from memorizz import MemAgent

# Load or create an agent with an Oracle memory provider
agent = MemAgent.load("my-agent-id", memory_provider=oracle_provider)

# Create a cron-scheduled automation
job = agent.create_automation(
    name="Morning News Digest",
    schedule_type="cron",
    cron_expr="0 8 * * 1-5",           # Weekdays at 8 AM
    timezone="America/New_York",
    query_template="Summarize today's top 5 tech news for {today_iso}.",
)
print(f"Created: {job.job_id}, next run: {job.next_run_at}")

# Create an interval-scheduled automation
job = agent.create_automation(
    name="Hourly Stock Check",
    schedule_type="interval",
    interval_seconds=3600,
    timezone="UTC",
    query_template="What are the current prices for AAPL, GOOGL, MSFT?",
)

# Create a one-shot automation (runs once)
job = agent.create_automation(
    name="Migration Report",
    schedule_type="one_shot",
    timezone="UTC",
    query_template="Generate a full migration status report.",
)

# Manage automations
jobs = agent.list_automations()                # List all jobs for this agent
jobs = agent.list_automations(enabled=True)    # Only enabled jobs
job  = agent.get_automation(job.job_id)        # Get a specific job
job  = agent.pause_automation(job.job_id)      # Pause
job  = agent.resume_automation(job.job_id)     # Resume
ok   = agent.delete_automation(job.job_id)     # Delete (returns bool)
job  = agent.trigger_automation(job.job_id)    # Trigger immediate run

# View run history
runs = agent.list_automation_runs(job.job_id, limit=10)
for run in runs:
    print(f"  {run.status} at {run.scheduled_for}")
```

### 2. Web UI

Navigate to **Automations** in the Memorizz dashboard (`/automations`) or from an agent's playground page. The form provides:

- Quick Schedule presets (every 30 min, hourly, daily at a specific time, etc.)
- Custom cron/interval configuration
- Query template editor with variable hints
- Delivery channel selection (In Chat or WhatsApp)

### 3. Agent Conversation (Tool-Based)

When `has_automations()` is `True`, six tool functions are automatically registered on the agent. You can ask the agent in natural language:

```
"Schedule a daily briefing every morning at 9 AM New York time
 that summarizes my portfolio performance."
```

The agent calls `automation_create_job` internally. Mutating automation tools
pause as durable, single-use approval proposals containing the exact arguments
and checkpoint. A trusted UI/CLI/host approves or rejects the proposal and
resumes the stored call; the model never receives a `confirm` or `approved`
argument.

**Registered tools:** `automation_create_job`, `automation_list_jobs`, `automation_pause_job`, `automation_resume_job`, `automation_delete_job`, `automation_run_now`.

## SDK Reference

All methods raise `ValueError` if automations are unavailable (unsupported
provider or `automations_enabled=False`). Check availability first with
`agent.has_automations()`.

### `create_automation(name, schedule_type, *, query_template, ...)`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Human-readable job name |
| `schedule_type` | `str` | Yes | `"cron"`, `"interval"`, or `"one_shot"` |
| `query_template` | `str` | Yes | Prompt template (see [Query Templates](#query-templates)) |
| `cron_expr` | `str` | If cron | 5-field cron expression |
| `interval_seconds` | `int` | If interval | Seconds between runs |
| `timezone` | `str` | No* | IANA timezone (e.g. `"America/New_York"`) |
| `memory_id` | `str` | No | Thread memory ID (auto-generated if omitted) |
| `whatsapp_to` | `list[str]` | No | WhatsApp recipient numbers for delivery |
| `max_run_seconds` | `int` | No | Max execution time per run (default: 900) |
| `retry_max_attempts` | `int` | No | Retry attempts on failure (default: 1) |
| `retry_backoff_seconds` | `int` | No | Backoff between retries (default: 60) |
| `misfire_policy` | `str` | No | `"skip"` (default) or `"run"` for missed runs |

*Timezone resolution cascade: explicit param → `agent.default_timezone` → `MEMORIZZ_DEFAULT_TIMEZONE` env var.

Returns: `AutomationJob`

### `list_automations(*, enabled=None)`
Returns: `list[AutomationJob]` — all jobs for this agent. Pass `enabled=True` or `enabled=False` to filter.

### `get_automation(job_id)`
Returns: `AutomationJob | None`

### `pause_automation(job_id)` / `resume_automation(job_id)`
Returns: updated `AutomationJob`

### `delete_automation(job_id)`
Returns: `bool` — `True` if deleted, `False` if not found.

### `trigger_automation(job_id)`
Sets `next_run_at` to now and enables the job. The worker picks it up on its next poll cycle.
Returns: updated `AutomationJob`

### `list_automation_runs(job_id, *, limit=50)`
Returns: `list[AutomationRun]` — execution history, most recent first.

## Models Reference

### AutomationJob

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `job_id` | `str` | — | Unique identifier (UUID) |
| `agent_id` | `str` | — | Owning agent ID |
| `name` | `str` | — | Human-readable name |
| `enabled` | `bool` | `True` | Whether the job is active |
| `schedule_type` | `ScheduleType` | — | `"cron"`, `"interval"`, or `"one_shot"` |
| `cron_expr` | `str?` | `None` | 5-field cron expression |
| `interval_seconds` | `int?` | `None` | Seconds between runs |
| `timezone` | `str` | — | IANA timezone name |
| `next_run_at` | `datetime` | — | Next scheduled execution (UTC) |
| `last_run_at` | `datetime?` | `None` | Last execution time (UTC) |
| `action_type` | `str` | — | Currently only `"agent_query"` |
| `action_config` | `dict` | `{}` | Contains `query_template` and optional `memory_id` |
| `delivery_type` | `str?` | `None` | `None`, `"in_chat"`, or `"whatsapp_twilio"` |
| `delivery_config` | `dict` | `{}` | Delivery-specific config (e.g. `whatsapp_to`) |
| `misfire_policy` | `str` | `"skip"` | `"skip"` or `"run"` |
| `max_run_seconds` | `int` | `900` | Execution timeout |
| `retry_max_attempts` | `int` | `1` | Max retry attempts |
| `retry_backoff_seconds` | `int` | `60` | Backoff between retries |

### AutomationRun

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Unique run identifier |
| `job_id` | `str` | Parent job ID |
| `status` | `RunStatus` | `"running"`, `"succeeded"`, `"failed"`, or `"canceled"` |
| `scheduled_for` | `datetime` | Scheduled execution time |
| `started_at` | `datetime?` | Actual start time |
| `finished_at` | `datetime?` | Completion time |
| `error` | `str?` | Error message (if failed) |
| `result_payload` | `dict?` | Contains `response`, `rendered_query`, `memory_id` |

### AutomationDelivery

| Field | Type | Description |
|-------|------|-------------|
| `delivery_id` | `str` | Unique delivery identifier |
| `run_id` | `str` | Parent run ID |
| `channel` | `str` | Delivery channel (e.g. `"whatsapp"`) |
| `provider` | `str` | Provider name (e.g. `"twilio"`) |
| `recipient` | `str` | Recipient address |
| `status` | `DeliveryStatus` | `"queued"`, `"sent"`, or `"failed"` |
| `provider_message_id` | `str?` | External message ID |
| `error` | `str?` | Error message (if failed) |

## Schedule Types

### Cron

Standard 5-field cron syntax: `minute hour day_of_month month day_of_week`

```
0 9 * * *       # Every day at 9:00 AM
0 9 * * 1-5     # Weekdays at 9:00 AM
30 */2 * * *    # Every 2 hours at :30
0 0 1 * *       # First of every month at midnight
```

### Interval

Fixed delay in seconds between runs.

```python
interval_seconds=1800     # Every 30 minutes
interval_seconds=3600     # Every hour
interval_seconds=86400    # Every 24 hours
```

### One-Shot

Runs once immediately (or at `start_at`), then the job is automatically disabled.

## Query Templates

The `query_template` string supports these placeholders, which are rendered at execution time:

| Placeholder | Example Value | Description |
|-------------|--------------|-------------|
| `{today_iso}` | `2025-01-15` | Today's date in the job's timezone |
| `{scheduled_for_iso}` | `2025-01-15T09:00:00-05:00` | Scheduled execution time |
| `{now_utc_iso}` | `2025-01-15T14:00:00+00:00` | Current UTC time |
| `{timezone}` | `America/New_York` | The job's configured timezone |

Example:
```
Summarize the market performance for {today_iso}.
Focus on tech stocks and any breaking news since {scheduled_for_iso}.
```

## Delivery Channels

### In Chat (default)

No configuration needed. The agent's response is stored in its conversation memory and visible in the playground's Automations panel.

### WhatsApp (Twilio)

Requires Twilio credentials as environment variables:

| Env Var | Description |
|---------|-------------|
| `TWILIO_ACCOUNT_SID` | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | Twilio auth token |
| `TWILIO_WHATSAPP_FROM` | Sender number (e.g. `whatsapp:+14155238886`) |

```python
job = agent.create_automation(
    name="Portfolio Alert",
    schedule_type="cron",
    cron_expr="0 9 * * *",
    timezone="UTC",
    query_template="Check my portfolio and alert if any position is down >5%.",
    whatsapp_to=["+15551234567", "+15559876543"],
)
```

## Running the Worker

The worker is a long-running process that polls the store for due jobs and executes them.

**CLI:**
```bash
memorizz automations run --poll-interval 5 --lease-seconds 120 --concurrency 2
```

Backend environment variables:

| Env Var | Description |
|---------|-------------|
| `ORACLE_USER` | Oracle database user |
| `ORACLE_PASSWORD` | Oracle database password |
| `ORACLE_DSN` | Oracle connection DSN |
| `ORACLE_SCHEMA` | (Optional) Oracle schema name |

Set `MEMORIZZ_BACKEND=filesystem` for the zero-configuration local store,
`MEMORIZZ_BACKEND=mongodb` plus `MONGODB_URI`, or
`MEMORIZZ_BACKEND=oracle` plus the Oracle values above. The worker must resolve
the same provider and Memorizz home as the process that saved the agent.

The web UI can also run an embedded worker when automations are enabled in settings.

## Troubleshooting

**"Automations are not available"**
- Ensure you're using a filesystem, MongoDB, or Oracle memory provider
- Check that `automations_enabled=True` (this is the default)
- On Oracle, run `memorizz oracle setup-schema` to create automation tables

**"timezone is required"**
- Pass `timezone` explicitly to `create_automation()`
- Or set `default_timezone` on the agent: `MemAgent(..., default_timezone="UTC")`
- Or set `MEMORIZZ_DEFAULT_TIMEZONE` env var

**WhatsApp delivery failures**
- Verify `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_WHATSAPP_FROM` env vars
- Ensure recipients are in `whatsapp:+<number>` format
- Check Twilio sandbox status for development numbers

**Jobs not executing**
- Ensure a worker is running (`memorizz automations run` or UI embedded worker)
- Check that the job is `enabled=True`
- Verify `next_run_at` is in the past (the worker only picks up due jobs)
