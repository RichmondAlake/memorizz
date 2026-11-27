# Observability Providers

Every turn an agent takes is traced: its model calls, tool calls, memory
reads, timings, token counts and how it ended. Two arguments on `MemAgent`
decide whether that happens and where the traces go.

```python
from memorizz import MemAgent

# Traced, and kept in MemoRizz. This is also what you get with neither argument.
agent = MemAgent(llm_config=llm, memory_provider=provider, observability=True)

# Traced, kept in MemoRizz and sent to Sentry.
agent = MemAgent(
    llm_config=llm,
    memory_provider=provider,
    observability=True,
    observability_provider="sentry",
)

# Not traced at all.
agent = MemAgent(llm_config=llm, memory_provider=provider, observability=False)
```

`observability` is the switch. `None` (the default) and `True` record traces;
`False` records none, so Traces, Failures and Usage show nothing for that
agent. `observability_provider` names a service that also receives each turn.
Without one, the provider is MemoRizz itself: traces are stored in the agent's
memory provider and read in the local UI
([Observability and Trace Inspection](../observability-ui.md)).

A provider adds a destination, it does not replace MemoRizz's own store. With
`observability_provider="sentry"` the MemoRizz copy is still written, because
the Traces and Failures pages, usage analytics and the learning loop read it.

The provider can be a name, a dict of settings, or a provider object, and can
be given as `observability` itself:

```python
MemAgent(observability="sentry")
MemAgent(observability_provider={"provider": "sentry", "org": "acme"})
MemAgent(observability=False, observability_provider="sentry")  # kept, switched off
```

Both settings are saved with the agent. Credentials are not: they are read
from the environment, and settings that hold a DSN or a key are refused.

## Providers

| Provider | Sends traces to | Needs |
|---|---|---|
| `memorizz` (default) | The agent's memory provider | Nothing |
| `sentry` | Sentry, as well as MemoRizz | `SENTRY_DSN` and `pip install "memorizz[sentry]"`, or no DSN under [`sentry local`](#without-a-sentry-account-sentry-local) |

`agent.observability_status()` says whether the agent's provider is ready and
names what is missing:

```python
agent.observability_status()
# {"provider": "sentry", "label": "Sentry", "enabled": True, "ready": False,
#  "missing": ["SENTRY_DSN"], "dashboard_url": "https://acme.sentry.io/explore/traces/", ...}
```

A provider that is not ready never fails a turn. The agent logs one warning
and carries on; the MemoRizz copy of each trace is still written.

## Sentry

Set the DSN of your Sentry project in the environment, or in the local UI
under **Settings → Observability**:

```bash
pip install "memorizz[sentry]"
export SENTRY_DSN="https://<key>@o<org>.ingest.sentry.io/<project>"
export SENTRY_ORG="acme"   # optional: your organisation's slug, for links
```

`MEMORIZZ_SENTRY_DSN` is read before `SENTRY_DSN`, for an application whose
own errors go to a different Sentry project.

Each finished turn is sent as one Sentry transaction named
`invoke_agent <agent name>`. Its spans follow Sentry's conventions for AI
agents, so the turn appears in Sentry's trace explorer and agent views:

| MemoRizz step | Sentry span | Carries |
|---|---|---|
| The turn | `gen_ai.invoke_agent` | agent name, conversation ID, total tokens, run labels as tags |
| A model call | `gen_ai.chat` | model, provider, input, output and cached tokens, finish reason |
| A tool call | `gen_ai.execute_tool` | tool name, status, duration |
| A memory read, a cache decision and the other steps | `memorizz.<step>` | the step's counts and status |

The Sentry trace ID is the turn's `root_trace_id` without its dashes, so a
trace in MemoRizz and its trace in Sentry name each other. A turn that fails
is also reported as an error on the same trace, grouped by agent and error
code, which is what Sentry alerts on. A turn somebody stopped, or one waiting
for an approval, is not an error.

MemoRizz sends through a Sentry client of its own and never calls
`sentry_sdk.init`, so your application's Sentry setup, its integrations and
its global scope are left alone. If no DSN is set and your application has
initialised Sentry, MemoRizz sends through that client.

### Without a Sentry account: `sentry local`

Sentry's CLI has a local development server that receives what the SDK sends
and shows it in the terminal. It needs no DSN, no account and no sign-in, and
nothing leaves your machine. Install the CLI, then run your script or the UI
under it:

```bash
curl -fsSL https://cli.sentry.dev/install | bash
pip install "memorizz[sentry]"

sentry local run -- python my_agent.py
sentry local run -- memorizz ui
```

`sentry local run` starts the server and sets `SENTRY_SPOTLIGHT` for the
command it runs. An agent with `observability_provider="sentry"` then sends
each turn there, and the terminal prints a line per turn:

```text
[TRACE] [SERVER] [gen_ai] invoke_agent Ada [gpt-5] [4496ms] [6 spans] [trace:a0d44940]
```

`sentry local -f ai` in another terminal shows only agent spans, and the
address the CLI prints (`https://local.sentry.dev/?url=...`) opens a page that
reads from your local server. You can also start the server yourself with
`sentry local` and set `SENTRY_SPOTLIGHT=http://localhost:8969/stream` (or
`SENTRY_SPOTLIGHT=1`) in the agent's environment.

Without a DSN the turns go to the local server only:
`agent.observability_status()` reports `"local_only": True`, the provider is
shown as **Sentry (local)**, its link opens the local page, and the Traces
page offers no **Open in Sentry** link for those turns, since there is no
trace on sentry.io to open. With a DSN set as well, turns go to both.

### Settings

| Setting | Meaning |
|---|---|
| `org` | Your organisation's slug (`acme` in `acme.sentry.io`). Shapes the link to the agent's traces. Falls back to `SENTRY_ORG`. |
| `project` | The project ID in links. Read from the DSN when unset. |
| `url` | The link itself, for a saved search or a dashboard. |
| `environment` | The Sentry environment traces are filed under. |
| `sample_rate` | The share of turns sent, from 0 to 1. Every turn when unset. |
| `include_content` | Also send tool arguments, tool results and the user ID. Off by default. |

By default only names, timings, token counts and statuses leave your process.
Prompts and replies are not sent with either setting.

```python
MemAgent(
    observability_provider={
        "provider": "sentry",
        "org": "acme",
        "environment": "production",
        "sample_rate": 0.25,
    },
)
```

`examples/observability/sentry_provider.py` runs an agent with one tool and
sends its turn to Sentry. Without a DSN it starts a stand-in for Sentry on
your machine and prints the trace it received, so you can see what is sent
before pointing it at a real project.

A short script should wait for the last turn to be sent before it exits:

```python
agent.run("Summarise today's incidents")
agent.flush_observability()
```

## The link to the provider

`agent.observability_url()` is where a person opens the agent's traces. For
Sentry it is the organisation's trace explorer; for the built-in provider it
is the Traces page of the local UI.

In the local UI, an agent's **Observability** section on its form sets the
switch and the provider. An agent whose traces go to Sentry shows a
**Sentry ↗** link beside its name on the Agents page, in its detail panel and
at the top of its playground, next to **Traces**.

On the **Traces** page, each turn in the waterfall that was sent to Sentry has
an **Open in Sentry ↗** link that opens that turn's trace there. MemoRizz
records on its own copy of a turn that the turn was sent, so the link shows
only for turns that reached Sentry: not for turns taken before the provider
was set, sampled out, or taken while the DSN was missing. A turn sent earlier
keeps its link after the agent goes back to MemoRizz only.

`provider.trace_url(root_trace_id)` builds the same link in code:

```python
context = agent.get_trace_context()
agent.observability_provider.trace_url(context["root_trace_id"])
# "https://acme.sentry.io/explore/traces/trace/689837bf201e408681a76d3894afaaa6/"
```

## Changing it later

```python
agent.with_observability(provider="sentry")   # start sending to Sentry
agent.with_observability(False)               # stop tracing
agent.with_observability(provider="memorizz") # back to MemoRizz only
agent.save()
```

With the builder:

```python
from memorizz import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_llm_config(llm)
    .with_memory_provider(provider)
    .with_observability("sentry", org="acme")
    .build()
)
```

## On the command line

```bash
memorizz agents create --name Ada --observability-provider sentry --observability-org acme
memorizz agents update <agent-id> --no-observability               # stop tracing
memorizz agents update <agent-id> --observability                  # trace again, same provider
memorizz agents update <agent-id> --observability-provider memorizz  # MemoRizz only
```

Naming a provider turns tracing on. `--no-observability` keeps the provider
for when tracing is switched back on. `memorizz agents show <agent-id> --json`
reports `observability`, `observability_provider` and `observability_org`.

## Over MCP

The MemoRizz MCP server's `memorizz_create_agent` and `memorizz_update_agent`
tools take `observability`, `observability_provider` and `observability_org`,
with the same rules. The DSN is never passed through a tool; it is set in the
environment of the process that runs the agent.

## Your own provider

Subclass `ObservabilityProvider`, give it a `name`, and implement
`export_turn`. It receives the turn's trace events in order and the context
that names the agent, the conversation and the `root_trace_id`.

```python
from memorizz.observability import ObservabilityProvider

class LogfileObservability(ObservabilityProvider):
    name = "logfile"
    label = "Log file"

    def export_turn(self, trace_context, events, *, agent_name=None):
        with open("turns.log", "a") as log:
            log.write(f"{agent_name} {trace_context['root_trace_id']} {len(events)}\n")
        return trace_context["root_trace_id"]

    def dashboard_url(self, *, agent_id=None, agent_name=None):
        return "https://logs.example.com/agents"

agent = MemAgent(llm_config=llm, observability_provider=LogfileObservability())
```

An exception raised by `export_turn` is logged by its type and never reaches
the turn. To let saved agents name your provider in another process, register
the class there with `register_observability_provider(LogfileObservability)`
before loading them; an agent that names a provider nobody registered loads
with MemoRizz's own.
