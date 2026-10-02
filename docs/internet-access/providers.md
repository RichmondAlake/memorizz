# Internet Access Providers

MemoRizz treats internet tooling as a first‑class capability for Deep Research agents. This page explains how providers are discovered, how to configure them, and what to expect from the built-in integrations.

## Provider Discovery Order

When a MemAgent is created with `ApplicationMode.DEEP_RESEARCH`, MemoRizz automatically attempts to attach an internet provider:

1. **Explicit override** – Any provider passed via `.with_internet_access_provider(...)` takes priority.
2. **Environment hint** – If `MEMORIZZ_DEFAULT_INTERNET_PROVIDER` is set, MemoRizz instantiates that provider (optionally with `MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY`).
3. **Tavily default** – If no override exists but `TAVILY_API_KEY` is present, MemoRizz prefers Tavily for its speed and structured research output.
4. **Firecrawl fallback** – If Tavily is unavailable yet `FIRECRAWL_API_KEY` exists, MemoRizz creates a Firecrawl provider.
5. **Offline provider** – When none of the above are configured, MemoRizz falls back to the built-in `offline` provider so the `internet_search` and `open_web_page` tools still exist and inform the agent/user how to enable real access.

Regardless of provider, every Deep Research agent (root, delegates, synthesis) registers two tools:

- `internet_search(query: str, max_results: int = 5)` – returns a list of normalized search results (`title`, `snippet`, `url`, `score`, optional metadata + raw payload).
- `open_web_page(url: str)` – fetches a URL and returns parsed content plus metadata (word count, truncation info, raw body when available).

You can also call `agent.search_internet(...)` or `agent.fetch_url(...)` directly from Python for the same behavior.

## Configuring Providers

| Setting | Purpose |
| --- | --- |
| `MEMORIZZ_DEFAULT_INTERNET_PROVIDER` | Name registered via `register_provider` (e.g., `tavily`, `firecrawl`). |
| `MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY` | API key passed to the provider constructed from the env hint. |
| `TAVILY_API_KEY` | Shortcut specifically for the Tavily provider (preferred). |
| `FIRECRAWL_API_KEY` | Shortcut specifically for the Firecrawl provider. |

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

## Offline Provider

If neither `MEMORIZZ_DEFAULT_INTERNET_PROVIDER` nor `TAVILY_API_KEY`/`FIRECRAWL_API_KEY` is configured, the `offline` provider keeps the internet tools available but responds with helpful error messages. This ensures Deep Research prompts remain stable even on air-gapped machines, while clearly signaling that live browsing is disabled.
