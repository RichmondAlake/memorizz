# CLI Guide

Streaming is now the default delivery path. See the
[streaming contract, defaults and compatibility modes](../guides/streaming.md)
for the SDK event iterator, CLI opt-out, UI lifecycle and opt-in MCP answer events.
Full-answer completion validators still buffer until acceptance; Python `run()`
retains its complete-string return contract.

The Memorizz CLI turns the library into a tool: an interactive, Claude-Code-style
terminal agent backed by the full Memorizz harness (memory backends, providers,
tools). It streams replies token-by-token, supports `/` slash commands, and — by
default — keeps **one persistent agent whose memory carries across sessions**.

It also runs a **100% local stack** (Ollama LLM + Ollama embeddings + on-disk
memory, no API key) and can launch the [Local UI](local-ui.md).

## Install

The CLI ships in the base package, so the `memorizz` command works straight away:

```bash
uv tool install --python 3.12 memorizz       # recommended
pipx install memorizz
pip install memorizz
```

For the **fully-local Ollama stack** (Ollama SDK + FAISS filesystem vector
store), add the `local` extra:

```bash
uv tool install --python 3.12 "memorizz[local]"  # == memorizz[ollama,filesystem]
```

Other providers are extras: `memorizz[anthropic]`, `memorizz[mongodb]`,
`memorizz[oracle]`, `memorizz[ui]`, `memorizz[mcp]`, or everything with
`memorizz[all]`. The npm bootstrapper installs the MCP extra because it ships
the complete CLI surface.

!!! note "Homebrew & npm"
    A Homebrew tap (`brew install RichmondAlake/memorizz/memorizz`) and an npm
    bootstrapper (`npm i -g memorizz`, which installs the real tool via `uv`) are
    also available for non-Python-first workflows.

## Update notices

Interactive sessions (`memorizz` and `memorizz chat`) check PyPI in the
background for a newer stable release compatible with your Python version.
When one is available, a notice above the prompt shows the installed and
latest versions and an upgrade command for your installer (pip, uv, pipx,
npm, or Homebrew). The CLI never installs updates automatically.

Successful checks are cached for 24 hours in
`~/.memorizz/update-check.json` (or under `MEMORIZZ_HOME`). An available update
is shown on each launch until you upgrade. Network failures stay quiet and
the check never holds up startup or exit. One-shot commands, JSON output, MCP
stdio, help, and version output do not perform update checks.

To disable the check, including its network request:

```bash
export MEMORIZZ_NO_UPDATE_CHECK=1
```

Checks are also skipped in CI and when output is redirected.

## Quickstart

### Option A — Local, no API key (Ollama)

```bash
# 1. install + run Ollama, then pull a tool-capable chat model + an embedder
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 2. launch the REPL
memorizz
```

With no cloud key set and an Ollama daemon running, Memorizz auto-selects Ollama
for the LLM, Ollama (`nomic-embed-text`) for embeddings, and an on-disk
filesystem store under `~/.memorizz/memory`.

### Option B — Cloud (OpenAI / Anthropic)

```bash
export OPENAI_API_KEY=sk-...      # or ANTHROPIC_API_KEY=...
memorizz
```

Cloud keys are auto-detected (Anthropic → OpenAI → Azure → local Ollama). You can
also save a key from inside the REPL with `/login`.

### Choose a memory provider (including Notion)

Memory storage is independent of the LLM. `--provider` and `/provider` select
the **LLM**; use these commands for memory:

```bash
memorizz memory configure          # guided filesystem / MongoDB / Oracle / Notion
memorizz notion connect            # shortcut to the guided Notion setup
memorizz notion connect --project  # explicitly save to this directory's .env
```

Inside the REPL, use `/memory-provider notion`. The wizard prompts for the
token with hidden input, accepts a Notion database URL or data-source ID,
checks access and managed columns, and lets you choose a separate vector
backend and embedding model. Choose **create** to provision an isolated area
below a page you have shared with the connection; nothing is created until
you confirm. Keep the printed resource IDs and local `notion-setup-*.json`
manifest; do not blindly rerun creation after a partial failure.

Setup checks Notion read access/schema, not vector connectivity, model
availability or write permissions on an existing library. It makes no paid
embedding calls or model downloads. After restart, run `memorizz notion status`
and then `memorizz notion sync` (which can incur embedding charges).
The `none` vector backend disables semantic search. See the
[Notion provider guide](../memory-providers/notion.md) for limits and installation extras.

Saved defaults do not switch the current agent, migrate existing memories,
or rewrite saved agent model configurations. `/memory [id]` still selects a
conversation's memory ID, not a provider.

During tool-using turns, the REPL displays the normalized terminal state beside
each tool: **Completed**, **Completed · no results**, **Completed with
limitations**, **Completed via fallback**, **Provider error**, or **Failed**.
One-shot execution prints non-clean outcomes after the answer so headless logs
do not silently treat an unavailable primary provider as a normal success.

## Create and inspect agents

Create a persisted agent without entering the REPL:

```bash
memorizz agents create \
  --name "Research Assistant" \
  --instruction "Research carefully and preserve useful findings." \
  --application-mode deep_research \
  --semantic-cache \
  --memory-id research-team \
  --set-default

memorizz agents list
memorizz agents show AGENT_ID --json
```

Add `--tool-cache` to reuse results of repeated calls to cacheable tools (see
[Tool call cache](../guides/context-efficiency.md#tool-call-cache)).

The command uses the configured memory provider and therefore defaults to the
filesystem store under `~/.memorizz/memory`. LLM settings are auto-detected;
pass `--llm-provider openai --model gpt-4o-mini` to select one explicitly, or
`--no-llm` to create a configuration that will be completed later in the SDK or
UI. `--set-default` makes the new agent the one opened by `memorizz chat`.

### Headless use

The non-UI CLI, one-shot runner, SDK, and MCP server need no window system. They
work in containers, CI, SSH sessions, and servers with `DISPLAY` and
`WAYLAND_DISPLAY` unset, and they do not import the optional UI application.

```bash
unset DISPLAY WAYLAND_DISPLAY
memorizz agents create --name "Headless Agent" --no-llm --json
memorizz capabilities --json
memorizz mcp serve --transport stdio
```

Use `memory_provider=False` only for an intentionally stateless SDK agent;
otherwise headless agents use the normal filesystem default.

## Run Codex, Claude Code, OpenHands, DeepSeek, pi, Hermes, or MemAgent

The `harness` command group places installed agent CLIs behind MemoRizz's
durable memory, policy, approval, cancellation, verification, and learning
contract. It covers what the **Agent Harnesses** page does: single runs,
staged plans, comparisons (each harness on its own model), conversations,
reruns, deletion, model choices and harness delegates.

```bash
memorizz harness doctor
memorizz harness run \
  "Inspect the failing tests and report the cause" \
  --workspace "$PWD" \
  --harness auto \
  --read-only \
  --verify "python -m pytest -q" \
  --json
```

A write run returns `pending_approval` before the external process starts:

```bash
memorizz harness run "Implement the verified fix" --workspace "$PWD" --write --json
memorizz harness approvals --status pending --json
memorizz harness approve PROPOSAL_ID --approver operator@example.com
memorizz harness show RUN_ID --events --json
```

Initialize the secret-free adapter allowlist with `memorizz harness init`.
OpenHands remains unavailable until its configured command is an
operator-provided isolation wrapper. See the
[Memory-First Meta-Harness guide](../guides/meta-harness.md) for adapter
policies, budgets, runtime/delegate modes, and deployment limits.

`memorizz harness doctor NAME --json` reports typed `error_code`, `error`, and
`remediation` fields. A missing Claude bare-mode credential or missing Codex
key/login therefore fails before launch with an actionable, secret-free
message; `harness run` preserves the same fields in its durable failed result.

To run a saved MemAgent, use `--harness memagent --agent-id AGENT_ID`
(`native` is an alias). Without `--agent-id` it runs the agent last used with
a harness, else the newest saved one. MemAgent execution is explicit-only so
`auto` cannot accidentally recurse into the coordinating MemAgent. If that
agent has harness delegates, they work in the run's folder with what the run
was approved for, and `memorizz harness cancel` stops them too.

```bash
memorizz harness plan "Fix the rounding bug" --stage plan:pi \
  --stage implement:codex:edit --stage review:claude-code \
  --stage-model review=claude-sonnet-5-5
memorizz harness compare "Where can totals lose precision?" \
  --harness codex --harness claude-code --harness-model claude-code=claude-sonnet-5-5
memorizz harness continue RUN_ID "Now add tests for that"
memorizz harness rerun-workflow WORKFLOW_ID
memorizz harness delete RUN_ID            # also: delete-workflow, delete-conversation
memorizz harness models                   # the models each harness can run
memorizz harness delegate create --harness codex --coordinator AGENT_ID --attach
```

Unknown IDs and refused requests end with a message and exit code 1.

## Run memory evaluations

The `eval` group uses versioned protocol manifests and never upgrades a smoke
run into a paper claim merely because it completed:

```bash
memorizz eval list
memorizz eval protocol show beam
memorizz eval dataset sync beam
memorizz eval dataset verify beam --data-path /data/BEAM --variant 128k
memorizz eval terminal-bench forecast \
  --per-trial-spend-guard-usd 1.75 \
  --total-budget-usd 1000

memorizz eval run beam \
  --data-path /data/BEAM \
  --variant 128k \
  --profile smoke \
  --memory-provider filesystem
```

The default `--evaluation-mode retrieval` is a normalized retriever/reader
diagnostic. To exercise an actual agent's automatic memory path, export a
secret-free `MemAgentModel` JSON template and run:

```bash
memorizz eval run locomo-plus \
  --data-path /data/Locomo-Plus/data \
  --variant cognitive \
  --evaluation-mode memagent \
  --agent-template ./agent-template.json \
  --top-k 6 \
  --candidate-pool-size 256
```

Useful retrieval controls are `--query-expansion/--no-query-expansion`,
`--lexical-weight`, and `--rerank-weight`. The lexical and rerank controls apply
to diagnostic fusion; full MemAgent reports them as not applied. Use
`--reader-repair/--no-reader-repair` to control the one bounded repair of
malformed JSON-like output.

Profiles are `smoke` (one case/category), `regression` (fixed stratified
subset), and `paper` (full split). Add `--strict-paper` when any mismatch with
the official runner, scorer, source revision, models, prompts, or runtime must
fail the command. The report always separates retrieval metrics, grounded
retrieved-evidence answers, the gold-evidence reader ceiling, and the scorer.

Use `--memory-provider oracle` for a matched Oracle run. Use
`--no-oracle-reader` only when you explicitly accept losing the reader/retriever
decomposition, and `--no-corpus-cache` for a deliberate cold-ingestion trial.
See the [Evaluation Suite](../evaluation-suite.md) for the protocol support
matrix and interpretation rules.

The Terminal-Bench forecast makes no model call. Without at least ten distinct
pilot tasks it reports only cost/headroom and withholds accuracy and rank. This
prevents a public agent's cost or score from being presented as MemoRizz
performance.

## The REPL

Running `memorizz` with no arguments launches the interactive loop:

- Type plain text to chat; the reply streams live and renders as Markdown.
- For **reasoning models** (e.g. `qwen3`, `deepseek-r1`), the model's thinking is
  shown dimmed above the answer, and tool activity is shown as it happens.
- **Ctrl-C** during a reply aborts just that reply (you stay in the REPL).
- **Ctrl-C** at the prompt, **Ctrl-D**, or `/exit` saves the agent and quits.
- Press **Tab** to autocomplete slash commands.

## Slash commands

| Command | Description |
|---|---|
| `/help` | List all commands + the current mode/model. |
| `/models [name]` | Show the current model, list installed Ollama models, or switch models (keeps the provider). |
| `/provider [name]` | Switch provider: `openai`/`anthropic`/`ollama`/`azure`/`huggingface`/`mlx`. |
| `/ollama [list\|pull <tag>\|host <url>]` | List/pull Ollama models or set `OLLAMA_HOST`. |
| `/web [on\|off\|tavily\|firecrawl]` | Enable/disable internet search (Tavily/Firecrawl). |
| `/code [on\|off]` | Toggle coding tools (file read/write + bounded commands, scoped to cwd). |
| `/browser [status\|on\|off\|run <task>]` | Inspect/configure Browser Use or run an explicit host browser task. |
| `/approvals [status/action]` | List, approve, reject, cancel, or resume durable generic tool proposals. |
| `/memory [id]` | Show or switch the active memory id. |
| `/history` | Print the current conversation history. |
| `/conversations [search]` | Open a searchable picker and resume a saved conversation thread. |
| `/forget <id>` | Delete a single stored memory by id. |
| `/new` | Start a fresh conversation thread (keeps long-term memory). |
| `/clear` | **Erase the agent's entire stored memory** (asks to confirm). |
| `/cls` | Clear the terminal screen. |
| `/harnesses` | List the external harnesses (Codex, Claude Code, OpenHands, pi, Hermes, ...) and whether each is ready. |
| `/harness <name\|auto\|delegate\|off>` | Run the following turns on that harness with MemoRizz memory, traces and approvals; `auto` lets MemoRizz pick, `delegate` keeps your model in charge and lets it hand parts to its harness delegates, `off` returns to native execution. Session only. |
| `/compare <harness> <harness> <task>` | Run one read-only task on two or more harnesses side by side and print each verdict; saved as a workflow. |
| `/agents` | List saved agents. |
| `/agent <id>` | Load a saved agent by id. |
| `/persona [name \| goals \| background]` | Show or set the agent's persona. |
| `/persona-reset` | Clear the persona (revert to default). |
| `/tools` | List the agent's registered tools. |
| `/ingest <file>` | Ingest a file into the knowledge base. |
| `/ui [--port N] [--host H]` | Launch the local web UI. |
| `/login [provider\|ENV_VARIABLE]` | Save a credential using hidden input. `/login notion` writes `NOTION_TOKEN`; it does not select memory storage. |
| `/config [get\|set\|path\|keys]` | Inspect active/saved settings, show the save target, or edit defaults. |
| `/memory-provider [filesystem\|mongodb\|oracle\|notion]` | Guided memory setup for the next launch; add `--project` for this project's `.env`. |
| `/docs [cli\|ui]` | Open the documentation in your browser. |
| `/exit` | Save the agent and quit. |

### Switch harnesses

The chat can hand whole turns to an installed coding agent while MemoRizz
keeps the memory. The prompt shows the active harness next to the mode:

```text
memorizz> /harnesses
Harnesses (active: off)
  codex        codex-cli 0.160.0   ready
  claude-code  2.1.293             ready
  deepseek                         not ready  DEEPSEEK_API_KEY is not set.
memorizz> /harness codex
Harness → codex  (this session only; /harness off to return to native execution)
memorizz·codex> summarise what this repository does
...
memorizz·codex> /harness off
Harness off (was codex); turns run natively again.
```

Each turn is a governed harness run: MemoRizz retrieves the relevant memories
for the harness, exposes the permitted MCP access, records the run with its
events, and saves the answer to the conversation. A turn that needs a write is
held as an approval; the chat prints the waiting proposal ids after the turn
so you can `/approvals approve <id> <name>` and `/approvals resume <id>`
without leaving the terminal. The choice lives on the session, not the saved
agent: `/harness off` restores exactly what the agent had. Launch straight
into a harness with `memorizz chat --harness codex`, or run one turn with
`memorizz run "..." --harness codex`.

Two multi-harness forms live in the chat as well:

```text
memorizz> /harness delegate
Harness → delegate  (this session only; your model plans, harnesses do the parts)
Harness delegates (2, delegation on):
  Codex reviewer (codex)   4f1c…
  pi summariser (pi)       9a0e…
Tools added for this session: run_harness_task, get_harness_run, list_agent_harnesses
memorizz·delegate> review mathlib.py for bugs and summarise README.md

memorizz> /compare codex claude-code where can totals lose precision in billing.py?
Comparing codex vs claude-code on: where can totals lose precision in billing.py?
[1/2] codex: running
[2/2] claude-code: running
...
┌ Comparison 3c2e… · succeeded ───────────────────────────────┐
│ Harness      Status     Verified  Cost     Time   Answer      │
│ codex        succeeded  -         $0.0000  14.2s  Totals …   │
│ claude-code  succeeded  -         $0.0312  21.7s  The sum …  │
└──────────────────────────────────────────────────────────────┘
Saved as workflow 3c2e…: memorizz harness show-workflow 3c2e… or the Agent Harnesses page.
```

`/harness delegate` is the coordinator pattern: your agent's own model keeps
the turn, splits the work across its harness delegates (create one with
`memorizz harness delegate create --harness codex --coordinator <agent-id>
--attach`) and can call any harness directly through `run_harness_task`.
`/compare` takes the leading words that name configured harnesses and treats
the rest as the task; it is read-only, runs in the current folder, Ctrl-C
cancels it, and the comparison is kept as a workflow. A compared `memagent`
runs another saved agent (the chat's own agent cannot hand a task to itself):
`memagent=<agent-id>` names it, otherwise the agent last used with a harness,
else the newest, is picked and announced.

### Switch models

At the `memorizz>` prompt, `/models` shows the current model and, when using
Ollama, lists installed models. Pass a name to switch for the current session:

```text
/models
/models qwen2.5:7b
```

Download an Ollama model first with `/ollama pull <tag>` if it is not installed.
Use `/provider ollama` to select Ollama when switching from another provider.
Model switches keep the current conversation and memories. `/model` remains
available as a compatibility alias; help and Tab completion use `/models`.

## Modes

**Memory assistant (default).** A conversational agent with persistent long-term
memory — it remembers facts you share and recalls them in later turns and later
sessions.

**Coding mode.** Launch with `memorizz chat --code`, or type `/code` in the REPL, to
enable the agent's self-aware tools: read/write files and run a bounded set of
commands, scoped to the current working directory (writes on, deletes off).

**Browser-control mode.** Launch with `memorizz chat --browser-control` (or use
`/browser on`) to attach the configured Browser Use provider. Model-initiated
browser calls pause as durable proposals; a model cannot set an `approved` or
`confirm` argument. `/browser run <task>` is a direct, explicit host action and
therefore does not represent a model approval.

## Internet access

Give the agent web search + page reading via [Tavily](https://tavily.com) or
[Firecrawl](https://firecrawl.dev). No extra install is needed — the providers
call the REST APIs directly.

```bash
export TAVILY_API_KEY=tvly-...      # or FIRECRAWL_API_KEY=fc-...
memorizz                            # internet tools auto-enable when a key is set
```

Or manage it from the REPL:

```
/login tavily       # save the key AND enable internet immediately
/web                # show status  (also: /web on | off | tavily | firecrawl)
```

When enabled, the agent gains `internet_search` (web search) and `open_web_page`
(fetch + read a full page). It can search, then open the most relevant result for
deeper detail (Tavily runs at advanced search depth for ~5x richer results). Use
a 7B+ model for reliable multi-step web use.

## Persistent agent & memory

Unlike a stateless chat, the CLI reuses **one** persistent agent across launches.
The default agent id and active memory/thread pair are stored in
`~/.memorizz/state.json`, so the selected conversation resumes after relaunch.

| Action | Command | Effect |
|---|---|---|
| New conversation, keep long-term memory | `/new` | Starts a fresh thread; past facts still recalled semantically. |
| Resume a previous conversation | `/conversations` | Search by title, preview, memory ID, or thread ID; select with ↑/↓ and Enter. |
| Forget one memory | `/forget <id>` | Deletes a single stored entry. |
| Wipe everything | `/clear` | Erases all stored memory after confirmation; keeps persona + tools. |

## One-shot mode

Run a single prompt and print the reply (pipe-friendly, no REPL):

```bash
memorizz run "Summarize what you remember about my project."
memorizz run --code "Add a docstring to utils.py and run the tests."
memorizz run --browser-control "Read the title of example.com."
```

One-shot turns share the same persistent agent + memory as the REPL.

## Configuration

Memorizz centralizes config under `~/.memorizz/`:

| Path | Purpose |
|---|---|
| `~/.memorizz/.env` | API keys and `MEMORIZZ_*` defaults. |
| `~/.memorizz/memory/` | Default filesystem memory store. |
| `~/.memorizz/state.json` | Persistent agent id + active memory/thread ids. |
| `~/.memorizz/history` | REPL input history. |

Overrides: `MEMORIZZ_HOME` (the home dir) and `MEMORIZZ_ENV_FILE` (the env file).
A project-local `./.env` is still honored. Precedence is **process exports >
project `.env` > canonical Memorizz `.env`**. The default save target is
`$MEMORIZZ_ENV_FILE`, otherwise `$MEMORIZZ_HOME/.env`, otherwise
`~/.memorizz/.env`. CLI, UI and MCP use this shared loader; SDK applications
must load their environment or supply explicit configuration themselves.

You do not need to find or edit `.env` manually:

```bash
memorizz config path
memorizz config keys
memorizz config set MEMORIZZ_BACKEND notion
memorizz config set MEMORIZZ_NOTION_DATA_SOURCE_ID YOUR_DATA_SOURCE_UUID
memorizz config set NOTION_TOKEN  # hidden prompt; never put the token here
memorizz config get NOTION_TOKEN # status/source only; credential stays hidden
memorizz config set MEMORIZZ_BACKEND notion --project
```

The REPL equivalents are `/config path`, `/config keys`, `/config set KEY [VALUE]`
and `/config get KEY`. Omit VALUE to be prompted. Known ordinary settings use
visible input; credentials and unknown integration variables use hidden input.
Do not put tokens in command arguments, chat, screenshots or shell history.
`/login` accepts provider shortcuts or any environment-variable name;
use `/config set` for ordinary settings. An old `NOTION` entry is not a token
alias: run `/login notion` again to save `NOTION_TOKEN` correctly.

New config commands save **for the next launch**. `/config` reports the active
memory/vector providers; `/config get KEY` distinguishes the current process's
value/source from the value saved in the target file. `/login` and UI Settings
also update the current process environment for compatibility, but cached
clients still require reconnection. Existing `/login tavily` / `firecrawl`
behavior continues to attach internet access immediately.

Use `--project` to edit a project override explicitly, or `--env-file PATH` for
another file. A custom file is not automatically loaded unless it is selected
by `MEMORIZZ_ENV_FILE` at launch. Set `MEMORIZZ_HOME` and `MEMORIZZ_ENV_FILE` in
the launch environment, not through `config set`. Warnings explain when a
project value or export may hide your saved change.

Writes are atomic and serialized across CLI/UI processes, preserve unrelated
comments and multiline values, and set owner-only file permissions on POSIX
(0600). On Windows, also secure the directory with your account's ACLs. Symlink
targets and malformed existing dotenv files are rejected without replacement.
Literal `${...}` input is rejected because dotenv would expand it on reload;
use a secret manager/process export for such credentials. Files are plaintext,
not encrypted: never commit them. Sensitive slash commands and abbreviations
are excluded from new REPL history and filtered out when recalling old history;
this does not erase existing history files or your shell's history.

Useful commands:

```bash
memorizz init           # interactive key wizard
memorizz init --local   # configure the local Ollama stack
memorizz config         # show paths, providers, embeddings, and learning mode
memorizz capabilities   # report installed features/provider readiness
memorizz agents create --help
memorizz oracle preflight --index-policy lazy
memorizz learning --help # inspect/compile/govern durable learning records
```

### Browser control

Browser control is explicit opt-in; an LLM API key alone never enables it. Keep
Browser Use in its own tool environment because its MCP dependency line can
differ from MemoRizz's:

```bash
uv tool install --python 3.12 browser-use
browser-use install
browser-use doctor

export MEMORIZZ_BROWSER_CONTROL_PROVIDER=browseruse
export MEMORIZZ_BROWSER_USE_COMMAND=browser-use
# Optional only when the entry point has no discoverable Python shebang:
# export MEMORIZZ_BROWSER_USE_PYTHON_COMMAND=/opt/browser-use/bin/python
export MEMORIZZ_BROWSER_USE_LLM_PROVIDER=openai
export MEMORIZZ_BROWSER_USE_ALLOWED_DOMAINS="example.com,*.notion.so"
export MEMORIZZ_BROWSER_USE_MAX_STEPS=25
export MEMORIZZ_BROWSER_USE_TASK_TIMEOUT=600
```

The provider resolves the isolated interpreter behind the Browser Use entry
point, runs a private fixed worker for each bounded task, passes only an
environment allowlist, blocks direct IP navigation by default, and closes the
worker/browser on success, error, or timeout. Configure the matching
credential (`BROWSER_USE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or
`GOOGLE_API_KEY`) in the environment; credentials are not serialized into the
agent record.

```text
/browser status
/browser on
/browser run Read the title at https://example.com
/browser off
```

For a model-requested task, use the generic approval lifecycle:

```text
/approvals pending
/approvals approve PROPOSAL_ID operator@example.com reviewed
/approvals resume PROPOSAL_ID
# Or execute the approved call without a subsequent model continuation:
/approvals resume PROPOSAL_ID --no-model
```

See the [Browser Control guide](../browser-control/index.md) for SDK, builder,
UI, domain-policy, result, and custom-provider details.

To run the CLI against Oracle with an existing external-vector schema, keep
the embedding mode and dimensions aligned with the schema:

```bash
export MEMORIZZ_BACKEND=oracle
export ORACLE_USER=memorizz_user
export ORACLE_PASSWORD=...
export ORACLE_DSN=localhost:1521/FREEPDB1
export MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING=false
export MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai
export MEMORIZZ_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
export MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=256
```

Enable workflow capture and learned-skill retrieval for one CLI launch:

```bash
MEMORIZZ_CONTINUAL_LEARNING=1 memorizz chat --code
```

To keep it enabled, add the setting to `~/.memorizz/.env`, then restart the
CLI:

```dotenv
MEMORIZZ_CONTINUAL_LEARNING=1
```

The startup banner and `/config` report the live learning state. Continual
learning needs tool-calling runs to produce workflow trajectories; ordinary
text-only chat has no procedure to promote. Repeat a successful tool workflow
with at least two distinct queries, then use `memorizz ui` → **Continual
Learning** to inspect trajectory classes, run a promotion cycle, and review or
activate learned skills.

The agent form also selects the authority for newly promoted skills. Keep the
default `user` role for selectively retrieved guidance. Choose `developer`
only for application-owned procedures; MemoRizz requires shadow review and
explicit activation before those skills can enter the provider's
developer/system-equivalent instruction channel.

## Launch the Local UI

```bash
memorizz ui                         # http://127.0.0.1:8765
memorizz ui --host 0.0.0.0 --port 9000
```

See the [Local UI Guide](local-ui.md) for details.

## Command reference

```text
memorizz                 # launch the interactive REPL (default)
memorizz chat [--code] [--browser-control] [--provider P] [--model M]
memorizz run [--code] [--browser-control] "<prompt>"  # one-shot
memorizz ui [--host H] [--port N]
memorizz init [--local]
memorizz mcp serve [--transport stdio|streamable-http]
memorizz config
memorizz mcp --help
memorizz --version
memorizz oracle install|setup|setup-schema|preflight|teardown
memorizz automations run [--poll-interval N] [--lease-seconds N] [--concurrency N]
memorizz learning status --agent-id AGENT [--memory-id ID] [--user-id USER]
memorizz learning events --agent-id AGENT [--limit 50]
memorizz learning compile --agent-id AGENT --memory-id ID --user-id USER
memorizz learning forget-plan --agent-id AGENT --memory-id ID --user-id USER
memorizz learning forget-apply PLAN_ID --agent-id AGENT --approved-by OPERATOR
memorizz capabilities [--json]
memorizz config path|keys|set|get
memorizz memory configure
memorizz notion connect|init|status|sync|repair
memorizz agents create|list|show|update|delete
memorizz harness list|doctor|init|config|models
memorizz harness run|plan|compare|continue|conversation
memorizz harness runs|show|events [--follow]|cancel|retry|delete
memorizz harness workflows|show-workflow|cancel-workflow|rerun-workflow|delete-workflow
memorizz harness delete-conversation
memorizz harness approvals|approve|reject|resume
memorizz harness delegate options|create
memorizz eval list|run|protocol show|dataset verify|dataset sync
memorizz eval terminal-bench tasks|status|run|forecast
memorizz plugin install codex|claude-code [--user NAME] [--no-capture] [--no-summaries]
  [--prompt-recall] [--allow-agents] [--allow-harness --harness-root PATH] [--allow-traces]
  [--remote URL [--token-env NAME] | --local]
memorizz plugin uninstall codex|claude-code
memorizz plugin memory-id [PATH]
memorizz plugin import-session LOG...   # past Codex/Claude Code sessions onto the Harnesses page
memorizz plugin hook session-start|prompt|stop|summarize   # run by the plugins' hooks
memorizz learning retention-plan --agent-id ID [--memory-id M] [--json]   # dry run over primary memories
memorizz learning retention-apply PLAN_ID --agent-id ID --approved-by NAME   # reversible suppression
memorizz learning suppressed --agent-id ID
memorizz learning unsuppress RECORD_ID --memory-type TYPE --agent-id ID --approved-by NAME
# Uses MEMORIZZ_BACKEND=filesystem|mongodb|oracle (filesystem by default)
```

The `learning` commands use `--backend filesystem|mongodb|oracle`, preserve
tenant filters, and emit JSON with `--json`. Forget planning is always a dry
run; application writes reversible tombstones and requires an approver
identity. The `retention-*` commands do the same for primary memories, hiding
records from retrieval with a reversible `retention_state` instead of deleting
them. See the [Learning Control Plane guide](../guides/learning-control-plane.md)
and the [forgetting mechanism](../guides/forgetting-mechanism.md).

## MCP connections

Configure, authorize, inspect, and call local or remote MCP servers without
leaving the terminal:

```bash
# Notion hosted MCP (OAuth)
memorizz mcp add notion --preset notion
memorizz mcp login notion

# Google Calendar hosted MCP (OAuth client from Google Cloud Console)
memorizz mcp add calendar --preset google-calendar \
  --client-id "$GOOGLE_OAUTH_CLIENT_ID" \
  --client-secret "$GOOGLE_OAUTH_CLIENT_SECRET"
memorizz mcp login calendar

# Inspect or invoke
memorizz mcp list
memorizz mcp test notion
memorizz mcp tools notion --json
memorizz mcp call notion search --arguments '{"query":"roadmap"}'
```

Mutating calls return a durable proposal instead of executing immediately:

```bash
memorizz mcp approvals --status pending
memorizz mcp approve PROPOSAL_ID --approver operator@example.com
memorizz mcp resume PROPOSAL_ID
# Or: memorizz mcp reject PROPOSAL_ID --approver operator@example.com
```

The proposal binds the exact tool and argument hash, expires, and can be
consumed once. Secrets supplied to `mcp add` are immediately moved to the
encrypted credential store and omitted from the public per-agent JSON. See the
[MCP Connectivity guide](../guides/mcp-connectivity.md) for transports, policy,
deployment settings, and the UI workflow.

!!! note "Back-compatible commands"
    The earlier forms still work with a deprecation notice: `memorizz run local`
    → `memorizz ui`, `memorizz run automations` → `memorizz automations run`, and
    `memorizz install-oracle` → `memorizz oracle install` (etc.).

## Choosing an Ollama model

The agent always sends tools, so the local model **must support tool-calling**:

- **Recommended:** `qwen2.5:7b`, `qwen2.5:3b`, or `llama3.1:8b` — tool-capable,
  non-reasoning, good quality.
- **Reasoning models** (`qwen3`, `deepseek-r1`, `qwq`, `magistral`) work — Memorizz
  auto-enables their "thinking" so reasoning is surfaced and answers aren't
  truncated — but for a snappy default a non-reasoning instruct model is better.
- **Smaller models** (`3b`, even `0.5b`) give simpler answers but stay snappy —
  plain chat exposes no tools by default, so they no longer loop. `gemma` models
  lack tool-calling in Ollama, so only use them for plain chat (not `/code`).
- **For web search + memory reasoning, prefer 7B+** (`qwen2.5:7b` /
  `llama3.1:8b`): smaller models tend to answer from guesswork instead of reading
  their memory or chaining `internet_search → open_web_page`.

Zero-config auto-selection already prefers tool-capable, non-reasoning families.

## Troubleshooting

**"This model can't tool-call."** Your Ollama model lacks tool support (e.g.
`gemma`). Pull a tool-capable model and switch:

```bash
ollama pull llama3.1:8b
# then in the REPL:
/models llama3.1:8b
```

**No provider configured.** Set a key (`/login` or `export OPENAI_API_KEY=...`)
or start Ollama and pull a model, then relaunch.

**Semantic recall seems weak (local stack).** Pull the embedder so memory uses
vectors instead of brute-force text matching:

```bash
ollama pull nomic-embed-text
```

**Slow startup.** A default install is lean (no PyTorch). If `import memorizz`
feels heavy, ensure you didn't install `memorizz[huggingface]` unless you need
local HuggingFace models/embeddings.
