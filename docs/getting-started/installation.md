# Installation

MemoRizz supports Python 3.10–3.12. Start with the base package, then add only
the integrations your application uses.

!!! note "License"
    MemoRizz uses the PolyForm Noncommercial License 1.0.0. Review the
    repository `LICENSE` before using it in a commercial product.

## Choose an interface

| Goal | Install | Start with |
|---|---|---|
| Build a Python application | `pip install memorizz` | [SDK quickstart](python-sdk-quickstart.md) |
| Use the terminal agent | `uv tool install --python 3.12 memorizz` | [CLI guide](cli.md) |
| Operate agents in a browser | `pip install "memorizz[ui]"` | [Local UI](local-ui.md) |
| Connect to or expose MCP | `pip install "memorizz[mcp]"` | [MCP client](../guides/mcp-connectivity.md) or [server](../guides/mcp-server.md) |

`pipx install memorizz` is equivalent to a uv tool install when pipx uses
Python 3.10 or newer. The explicit uv Python pin is portable across hosts with
older system Python versions. Use a project virtual environment for SDK
applications.

## Base installation

=== "pip"

    ```bash
    python -m pip install --upgrade memorizz
    ```

=== "uv"

    ```bash
    uv add memorizz
    ```

The base install includes the Python SDK, CLI, OpenAI client, and Ollama client.
It does not install databases, the web UI, MCP, remote sandboxes, or benchmark
harnesses.

## Optional extras

| Extra | Adds |
|---|---|
| `filesystem` | FAISS acceleration; filesystem persistence itself works without it |
| `local` | Ollama plus filesystem/FAISS support |
| `anthropic` | Anthropic SDK |
| `mongodb` | PyMongo |
| `oracle` | Oracle Database Python driver |
| `mcp` | MCP client/server, encrypted credential storage, and HTTP runtime |
| `ui` | FastAPI local control plane |
| `sandbox-e2b` | Stateful E2B sandbox adapter |
| `sandbox-daytona` | Daytona execution adapter |
| `ingest-pdf` | PDF extraction with `pypdf` |
| `huggingface` | Local Transformers models and embeddings |
| `mlx` | Apple-Silicon MLX models; requires native arm64 Python |

Install several extras together:

```bash
python -m pip install "memorizz[filesystem,mcp,ui,anthropic]"
```

`memorizz[all]` installs runtime integrations, but intentionally excludes the
large benchmark stacks. Evaluation extras are documented in the
[evaluation suite](../evaluation-suite.md).

## Local, cloud, and headless setups

=== "Fully local"

    ```bash
    python -m pip install "memorizz[local]"
    ollama pull qwen2.5:7b
    ollama pull nomic-embed-text
    memorizz
    ```

=== "Cloud LLM"

    ```bash
    export OPENAI_API_KEY="your-key"
    memorizz
    ```

=== "Headless server"

    ```bash
    python -m pip install "memorizz[mcp]"
    unset DISPLAY WAYLAND_DISPLAY
    memorizz mcp serve --transport stdio
    ```

The SDK, CLI, and MCP server do not require the UI or a display server. The web
UI is also a server process and can be accessed remotely through an approved
private network, but it is never imported by the other form factors.

## Default storage

If no memory provider is supplied, MemoRizz uses the filesystem provider at
`~/.memorizz/memory`. Important paths are:

| Path | Purpose |
|---|---|
| `~/.memorizz/.env` | CLI/UI-managed environment settings |
| `~/.memorizz/memory` | Default filesystem memory |
| `~/.memorizz/state.json` | CLI agent and active conversation |
| `~/.memorizz/approvals.sqlite3` | Default durable approvals |

Set `MEMORIZZ_HOME` to relocate the entire home, or
`MEMORIZZ_MEMORY_ROOT` to relocate only filesystem memory. Pass
`memory_provider=False` only when you explicitly want a stateless SDK agent.

## Verify the installation

```bash
memorizz --version
memorizz capabilities --json
```

`capabilities` reports effective dependency and provider readiness. It is more
reliable than checking only the package version because optional integrations
can be absent or unconfigured.

Next, choose a [model provider](model-providers.md) and build a persistent agent
with the [Python SDK quickstart](python-sdk-quickstart.md).
