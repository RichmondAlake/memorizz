# Failures and alerts

**Observability → Failures** shows what failed across your agents, and alerts tell
you when failures pass a limit you set.

## See what failed

The Failures tab lists every failure in a time range, from the last hour to 90
days or a custom range:

- **Agents**, worst first. Agents with a turn that stopped come before agents
  that carried on, so you know where to look. Each name opens that agent's own
  Failures panel, for the same range.
- **What is failing**, with one line per failure however many agents hit it.

**Download CSV** gives you the same list for a ticket or a spreadsheet, and
`/traces/failures.json` gives it to your own tools.

A tool can fail and the agent still finish the turn, for example by trying
something else. Each failure says which happened: **Stopped the turn** or **Turn
finished anyway**.

Each agent's Traces page opens on its own Failures panel, with a range of its
own. **Show the error** reveals the message the tool returned. Accounts that can't
read trace content see the reason only.

An account limited to one application or user sees only that scope, on the page
and in both exports.

## Set an alert

Admins add rules at the bottom of the Failures tab. A rule is a limit, what to
count, a time window and an agent:

| Setting | Meaning |
|---|---|
| **Alert at** | The count that fires the rule |
| **Count** | Failures, or only turns that stopped |
| **In** | The last hour, 24 hours or 7 days |
| **Agent** | One agent, or any agent |
| **Webhook** | Where to send a message (optional) |

Rules are checked every minute **while the MemoRizz UI is running**. If it isn't
running, nothing checks them.

A rule that fires shows under Needs attention on the Overview. With a webhook, it
sends one message when it fires and one when it clears, not one a minute. If a
delivery fails, the rule says so and tries again at the next check. **Send a test**
checks the address.

### Webhooks

MemoRizz posts JSON with a `text` field, which Slack and most chat tools show as
it is, and the details in `memorizz`:

```json
{
  "text": "MemoRizz: 12 failures in the last hour on Billing, reaching your limit of 5. Most often: mcp_call_tool (protocol_error). https://…",
  "memorizz": {
    "event": "failure_alert.firing",
    "rule_id": "…",
    "agent_id": "…",
    "window": "1h",
    "threshold": 5,
    "count": 12,
    "top": [{"name": "mcp_call_tool", "code": "protocol_error", "count": 9}],
    "url": "https://…",
    "at": "2026-11-01T22:10:00+00:00"
  }
}
```

`event` is `failure_alert.firing`, `.resolved` or `.test`. Messages hold names,
codes and counts, never conversation content.

A webhook address is a credential, so MemoRizz shows only its host and keeps it in
`~/.memorizz/failure_alerts.json`, which only your user can read.

The address must be public and use https. MemoRizz refuses this computer and
private networks, and doesn't follow redirects. To send to a service on your own
network, set `MEMORIZZ_ALERT_ALLOW_PRIVATE_WEBHOOKS=1`.
