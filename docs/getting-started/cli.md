# CLI Guide

The Memorizz CLI turns the library into a tool: an interactive, Claude-Code-style
terminal agent backed by the full Memorizz harness (memory backends, providers,
tools). It streams replies token-by-token, supports `/` slash commands, and — by
default — keeps **one persistent agent whose memory carries across sessions**.

It also runs a **100% local stack** (Ollama LLM + Ollama embeddings + on-disk
memory, no API key) and can launch the [Local UI](local-ui.md).

## Install

The CLI ships in the base package, so the `memorizz` command works straight away:

```bash
uv tool install memorizz       # recommended
pipx install memorizz
pip install memorizz
```

For the **fully-local Ollama stack** (Ollama SDK + FAISS filesystem vector
store), add the `local` extra:

```bash
uv tool install "memorizz[local]"     # == memorizz[ollama,filesystem]
```

Other providers are extras: `memorizz[anthropic]`, `memorizz[mongodb]`,
`memorizz[oracle]`, `memorizz[ui]`, or everything with `memorizz[all]`.

!!! note "Homebrew & npm"
    A Homebrew tap (`brew install RichmondAlake/memorizz/memorizz`) and an npm
    bootstrapper (`npm i -g memorizz`, which installs the real tool via `uv`) are
    also available for non-Python-first workflows.

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
| `/model [name]` | Show or switch the chat model (keeps the provider). |
| `/provider [name]` | Switch provider: `openai`/`anthropic`/`ollama`/`azure`/`huggingface`/`mlx`. |
| `/ollama [list\|pull <tag>\|host <url>]` | List/pull Ollama models or set `OLLAMA_HOST`. |
| `/web [on\|off\|tavily\|firecrawl]` | Enable/disable internet search (Tavily/Firecrawl). |
| `/code [on\|off]` | Toggle coding tools (file read/write + bounded commands, scoped to cwd). |
| `/memory [id]` | Show or switch the active memory id. |
| `/history` | Print the current conversation history. |
| `/forget <id>` | Delete a single stored memory by id. |
| `/new` | Start a fresh conversation thread (keeps long-term memory). |
| `/clear` | **Erase the agent's entire stored memory** (asks to confirm). |
| `/cls` | Clear the terminal screen. |
| `/agents` | List saved agents. |
| `/agent <id>` | Load a saved agent by id. |
| `/persona [name \| goals \| background]` | Show or set the agent's persona. |
| `/persona-reset` | Clear the persona (revert to default). |
| `/tools` | List the agent's registered tools. |
| `/ingest <file>` | Ingest a file into the knowledge base. |
| `/ui [--port N] [--host H]` | Launch the local web UI. |
| `/login [provider]` | Log in / save an API key — lists platforms to pick from if none given. |
| `/config` | Show resolved config + paths. |
| `/docs [cli\|ui]` | Open the documentation in your browser. |
| `/exit` | Save the agent and quit. |

## Modes

**Memory assistant (default).** A conversational agent with persistent long-term
memory — it remembers facts you share and recalls them in later turns and later
sessions.

**Coding mode.** Launch with `memorizz --code`, or type `/code` in the REPL, to
enable the agent's self-aware tools: read/write files and run a bounded set of
commands, scoped to the current working directory (writes on, deletes off).

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
deeper detail. Use a 7B+ model for reliable multi-step web use.

## Persistent agent & memory

Unlike a stateless chat, the CLI reuses **one** persistent agent across launches.
The default agent id and the rolling memory id are stored in
`~/.memorizz/state.json`, so a fact you teach it today is recalled tomorrow.

| Action | Command | Effect |
|---|---|---|
| New conversation, keep long-term memory | `/new` | Starts a fresh thread; past facts still recalled semantically. |
| Forget one memory | `/forget <id>` | Deletes a single stored entry. |
| Wipe everything | `/clear` | Erases all stored memory after confirmation; keeps persona + tools. |

## One-shot mode

Run a single prompt and print the reply (pipe-friendly, no REPL):

```bash
memorizz run "Summarize what you remember about my project."
memorizz run --code "Add a docstring to utils.py and run the tests."
```

One-shot turns share the same persistent agent + memory as the REPL.

## Configuration

Memorizz centralizes config under `~/.memorizz/`:

| Path | Purpose |
|---|---|
| `~/.memorizz/.env` | API keys and `MEMORIZZ_*` defaults. |
| `~/.memorizz/memory/` | Default filesystem memory store. |
| `~/.memorizz/state.json` | Persistent agent id + rolling memory id. |
| `~/.memorizz/history` | REPL input history. |

Overrides: `MEMORIZZ_HOME` (the home dir) and `MEMORIZZ_ENV_FILE` (the env file).
A project-local `./.env` is still honored for backwards compatibility. The CLI
and the Local UI read/write the **same** `.env`, so configuring once applies to
both.

Useful commands:

```bash
memorizz init           # interactive key wizard
memorizz init --local   # configure the local Ollama stack
memorizz config         # show resolved paths + detected provider
```

## Launch the Local UI

```bash
memorizz ui                         # http://127.0.0.1:8765
memorizz ui --host 0.0.0.0 --port 9000
```

See the [Local UI Guide](local-ui.md) for details.

## Command reference

```text
memorizz                 # launch the interactive REPL (default)
memorizz chat [--code] [--provider P] [--model M]
memorizz run "<prompt>"  # one-shot
memorizz ui [--host H] [--port N]
memorizz init [--local]
memorizz config
memorizz --version
memorizz oracle install|setup|setup-schema|teardown
memorizz automations run [--poll-interval N] [--lease-seconds N] [--concurrency N]
```

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
/model llama3.1:8b
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
