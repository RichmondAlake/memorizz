# Tool call cache benchmark

`benchmark.py` measures what the MemAgent tool cache saves, and checks that no
answer changes. A support agent works through eight tickets, each in its own
conversation, against slow (simulated) order and shipping-quote APIs; several
tickets need the same lookup or quote. The queue runs with and without a tool
cache, in A-B-B-A order after a warm-up, with the same model.

```bash
python examples/tool_cache/benchmark.py                       # local Ollama, qwen2.5:7b
MODEL=gemma4:latest python examples/tool_cache/benchmark.py
PROVIDER=openai MODEL=gpt-4.1-mini python examples/tool_cache/benchmark.py
ROUNDS=4 ORDER_API_SECONDS=3 python examples/tool_cache/benchmark.py
```

A result on `qwen2.5:7b` (two rounds, averaged):

| | No cache | Tool cache |
|---|---|---|
| Real API calls | 8 | 4 |
| Tool time saved | — | 5.5 s |
| Queue time | 36.6 s | 28.6 s |
| Correct answers | 8 of 8 | 8 of 8 |

The return-label tool has side effects, so it runs every time. Each run's
tickets, tool calls, cache results and answers are written to
`tool_cache_benchmark.json`.

See [Tool call cache](../../docs/guides/context-efficiency.md#tool-call-cache).
