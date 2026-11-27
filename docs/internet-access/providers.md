# Internet Access Providers

MemoRizz treats internet tooling as a first‑class capability for Deep Research agents. This page explains how providers are discovered, how to configure them, and what to expect from the built-in integrations.

## Provider Discovery Order

When a MemAgent is created with `ApplicationMode.DEEP_RESEARCH`, MemoRizz automatically attempts to attach an internet provider:

1. **Explicit override** – Any provider passed via `.with_internet_access_provider(...)` takes priority.
2. **Environment hint** – If `MEMORIZZ_DEFAULT_INTERNET_PROVIDER` is set, MemoRizz instantiates that provider (optionally with `MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY`).
3. **Tavily default** – If no override exists but `TAVILY_API_KEY` is present, MemoRizz prefers Tavily for its speed and structured research output.
4. **Firecrawl fallback** – If Tavily is unavailable yet `FIRECRAWL_API_KEY` exists, MemoRizz creates a Firecrawl provider.
5. **Exa** – If neither of those keys is set but `EXA_API_KEY` is, MemoRizz creates an Exa provider.
6. **Offline provider** – When none of the above are configured, MemoRizz falls back to the built-in `offline` provider so the `internet_search` and `open_web_page` tools still exist and inform the agent/user how to enable real access.

Regardless of provider, every Deep Research agent (root, delegates, synthesis) registers two tools:

- `internet_search(query: str, max_results: int = 5)` – returns a list of normalized search results (`title`, `snippet`, `url`, `score`, optional metadata + raw payload).
- `open_web_page(url: str)` – fetches a URL and returns parsed content plus metadata (word count, truncation info, raw body when available).

You can also call `agent.search_internet(...)` or `agent.fetch_url(...)` directly from Python for the same behavior.

## Configuring Providers

| Setting | Purpose |
| --- | --- |
| `MEMORIZZ_DEFAULT_INTERNET_PROVIDER` | Name registered via `register_provider` (e.g., `tavily`, `firecrawl`, `exa`). |
| `MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY` | API key passed to the provider constructed from the env hint. |
| `TAVILY_API_KEY` | Shortcut specifically for the Tavily provider (preferred). |
| `FIRECRAWL_API_KEY` | Shortcut specifically for the Firecrawl provider. |
| `EXA_API_KEY` | Shortcut specifically for the Exa provider. |

To force a provider in code (e.g., for tests), build it manually and pass it to the builder:

```python
from memorizz.internet_access.providers.tavily import TavilyProvider
from memorizz.memagent.builders import create_deep_research_agent

tavily = TavilyProvider(api_key="sk-...")
agent = (create_deep_research_agent("Web scout", internet_provider=tavily)
    .with_memory_provider(memory_provider)
    .build())
```

You can also swap providers on an existing agent via `agent.with_internet_access_provider(new_provider)`.

## Tavily Provider (Preferred)

The Tavily integration is the recommended default. It balances speed and extraction quality, and MemoRizz automatically wires it up for Deep Research agents whenever `TAVILY_API_KEY` exists.

1. Export `TAVILY_API_KEY="<your-key>"`.
2. Optionally configure `MEMORIZZ_DEFAULT_INTERNET_PROVIDER=tavily` to make every Deep Research agent pick it explicitly.
3. (Optional) Pass a config dict to tune options such as `search_depth`, `default_max_results`, and `max_content_chars`.

```python
TavilyProvider(
    api_key="sk-...",
    config={
        "search_depth": "advanced",
        "default_max_results": 8,
        "max_content_chars": 10_000,
        "include_raw_page": False,
    },
)
```

Responses include truncation metadata whenever `max_content_chars` shortens an extracted page so downstream prompts can adapt.

Search results carry Tavily's short `snippet` by default. Set `include_raw_content` to get each page's own text as well, in `result.content` (and `"content"` in the agent's `internet_search` results):

```python
search = TavilyProvider(config={"include_raw_content": True})  # or "text"
for result in search.search("TAP Air Portugal power bank rules"):
    passage = result.content or result.snippet
```

`True` asks for markdown and `"text"` for plain text. Pass `include_raw_content=` to `search()` to choose per call. Each page's text is cut to `max_content_chars` (12,000 by default), with `content_truncated` in `result.metadata` when it was. Page text makes results much longer, so pair it with a `ToolResultPolicy` that offloads large tool results when an agent searches.

## Firecrawl Provider

The Firecrawl integration gives you search + crawl with no extra package:

1. Export `FIRECRAWL_API_KEY="<your-key>"`.
2. Optionally configure `MEMORIZZ_DEFAULT_INTERNET_PROVIDER=firecrawl` to ensure every Deep Research agent uses Firecrawl by default.

### Advanced Configuration

The provider accepts extra keyword arguments via the config dict:

```python
FirecrawlProvider(
    api_key="sk-...",
    base_url="https://api.firecrawl.dev/v1",
    timeout=45,
    config={
        "max_content_chars": 12_000,
        "max_raw_chars": 2_000,
        "include_raw_response": True,
    },
)
```

From the environment, only `MEMORIZZ_DEFAULT_INTERNET_PROVIDER` and its API key are read; to change timeouts or base URLs, pass them in the config you give `create_internet_access_provider`.

### Response Shape

- `internet_search` returns a list of objects containing `url`, `title`, `snippet`, optional `score`, and a `metadata` dict with provider-specific fields (`source`, `published_at`, etc.).
- `open_web_page` returns `title`, parsed `content` (Markdown), `metadata` describing truncation, and a `raw` dict with the full provider payload if `include_raw_response` is enabled.

MemoRizz automatically trims long documents to keep responses within the model’s context window, flagging truncated responses via `metadata["content_truncated"]`.

## Exa Provider

[Exa](https://exa.ai) searches by meaning and returns the passages of each page that answer the query. No extra package is needed:

1. Export `EXA_API_KEY="<your-key>"`.
2. Pick it for an agent (`/web exa` in the CLI, or the internet provider menu in the console), or set `MEMORIZZ_DEFAULT_INTERNET_PROVIDER=exa` to make it the default over Tavily and Firecrawl.

```python
from memorizz.internet_access import ExaProvider

ExaProvider(
    api_key="...",
    config={
        "search_type": "auto",        # Exa's search type; "deep" reads further
        "default_max_results": 5,     # up to 25
        "highlight_chars": 600,       # size of each result's snippet
        "include_text": False,        # also return each page's text as `content`
        "max_content_chars": 12_000,  # cap on page text, in search and open_web_page
    },
)
```

`internet_search` results carry Exa's highlights as `snippet`, with `published_at`, `author` and `site` in `metadata`. `search()` also accepts `include_domains`, `exclude_domains`, `start_published_date`, `end_published_date` and `category`. `open_web_page` reads a page through Exa's contents API and raises with Exa's reason when the page cannot be read.

### Websets and Monitors

With Exa as the provider, an agent also gets Exa's Websets and Monitors as
tools. They are added when the provider is set and removed when it changes.

A **webset** is a dataset Exa builds: it searches for things that match a
description (companies, people, articles, papers), checks each against your
criteria and fills in extra columns. It takes a few minutes and costs far
more than a search, so the agent is told to use it for a list or table of
many things, and `internet_search` for a question or a single lookup.

| Tool | What it does |
| --- | --- |
| `exa_create_webset` | Starts a webset: `query`, `count`, `entity`, up to 5 `criteria`, extra `columns`. |
| `exa_get_webset` | Reads the rows so far; waits up to `wait_seconds` for it to finish. |
| `exa_save_webset_to_memory` | Stores the rows in the agent's knowledge base, a chunk per row, attached to the agent so any conversation can recall them. Rows saved before are skipped. |
| `exa_create_monitor` | Has a search run again every `every` (`"6h"`, `"1d"`). Needs approval. |
| `exa_list_monitors`, `exa_monitor_results`, `exa_stop_monitor`, `exa_resume_monitor` | List them, read what the latest runs found (or run one now), pause or delete, set running again. |

An agent can read, save and stop only the websets and monitors it made; it
keeps a note of them in `<MEMORIZZ_HOME>/exa/<agent>.json`. A webset asks for
at most `webset_max_count` rows (25 by default).

Exa's API requires every monitor to have a public HTTPS address to call
when a run ends. MemoRizz reads the runs from Exa itself, so the address
only has to accept the calls:

- Set `EXA_MONITOR_WEBHOOK_URL` where MemoRizz runs (or pass
  `monitor_webhook_url` to `ExaProvider`) to use your own.
- With none set, MemoRizz asks webhook.site for a temporary address, which
  is what Exa's dashboard does when the field is left blank. Only that one
  request goes to webhook.site, without the Exa key. Anyone holding the
  address can read what the runs found, and a free address expires, so use
  your own for private or long-lived monitors.
- Pass `temporary_webhooks=False` in the Exa config to refuse instead; then
  `exa_create_monitor` says what is missing and creates nothing.

The same structures are available without an agent:

```python
from memorizz.internet_access.providers.exa import ExaProvider
from memorizz.internet_access.exa_tools import webset_table, table_text, monitor_view

exa = ExaProvider()
webset = exa.create_webset(
    "AI agent memory startups",
    count=10,
    entity="company",
    criteria=["builds memory infrastructure for AI agents"],
    enrichments=["Year founded"],
)
webset = exa.wait_for_webset(webset["id"], seconds=300)
table = webset_table(webset, exa.list_webset_items(webset["id"]))

table["columns"]   # [{"key", "label", "kind": "field" | "criterion" | "enrichment"}]
table["rows"]      # [{"name", "url", "description", "cells", "why"}]
table_text(table)  # flat rows: column label -> value, for a DataFrame or CSV

monitor = exa.create_monitor("Acme launches", period="1d")
monitor_view(monitor, exa.list_monitor_runs(monitor["id"]))
```

In the playground a search shows the pages it found, a webset shows as a
table that fills in while Exa works (with **Save to memory**, **CSV** and
**Open in Exa**), and a monitor shows its schedule and what its latest run
found. In the CLI the same tables are printed under the answer.

### Showing a tool's result

Any tool can give people something to look at beside what the model reads.
Return a `ToolResult` with a `display`: a small mapping with a `kind`. It
travels with the `tool.completed` stream event and the stored trace, and is
never sent to the model.

```python
from memorizz.tooling import ToolOutcome, ToolOutcomeStatus, ToolResult

return ToolResult(
    value={"results": results},
    outcome=ToolOutcome(status=ToolOutcomeStatus.SUCCESS),
    display={"kind": "web_search", "query": query, "results": pages},
)
```

The console draws the kinds `web_search`, `exa_webset` and `exa_monitor`.
A display over 8,000 characters is cut down to its `kind` and `id`.

## Offline Provider

If neither `MEMORIZZ_DEFAULT_INTERNET_PROVIDER` nor `TAVILY_API_KEY`/`FIRECRAWL_API_KEY`/`EXA_API_KEY` is configured, the `offline` provider keeps the internet tools available but responds with helpful error messages. This ensures Deep Research prompts remain stable even on air-gapped machines, while clearly signaling that live browsing is disabled.
