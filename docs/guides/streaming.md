# Streaming delivery

Streaming is the default for newly created/reloaded agents, Playground (including
the desktop wrapper), the REPL, one-shot CLI and MCP agent execution. Streaming
means incremental **accepted public answer text**, not private reasoning or tool
drafts. Text deltas are not necessarily individual tokenizer tokens.

Python's existing `run()` still returns one complete string. Its compatibility
contract is unchanged; use the event API for incremental delivery:

```python
with agent.run_stream_events("Explain this document", user_id="alice") as events:
    for event in events:
        if event["type"] == "answer.delta":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "run.done":
            print(event["status"], event["persistence"])
```

`arun_stream_events()` is an async iterator with the same events. Close it with
`contextlib.aclosing` if breaking out early. Legacy `run_stream()` still yields
strings and accepts a per-call diagnostic callback.

## Contract and completion

Every JSON-safe v1 event has `run_id`, `seq`, `type` and monotonic `elapsed_ms`.
Sequences start at one; append each `answer.delta.delta` exactly once.
`answer.done` includes the exact UTF-8 SHA-256, character count, answer/attempt
identity and validation state. It ends text delivery **before** remaining
workflow/conversation/cache/trace work. Keep consuming until the single
`run.done` for completed/error/cancelled/approval_required and persistence
outcomes. Unknown write outcomes are not successes.

`run.started` publishes IDs and the selected delivery/capability information.
Deferred UI/service loading starts with `delivery_mode=initializing`; its
`agent_ready` status publishes resolved capabilities. Other events include
`status`, `tool.started`, `tool.completed`, `completion.check`, `usage` and
`approval.required`. Tool arguments, results, provider reasoning and rejected
candidate text are excluded. An error preserves delivered partial text and adds
a safe code in the terminal event, never exception prose in the answer.

| Delivery | Behavior |
|---|---|
| Default, no completion gate | `final_stream`: private tool phase, host-controlled finalize-answer tool, then incremental answer generation with tools disabled |
| Existing/enabled completion policy | `buffered` by default: accept the whole candidate before exposing any text |
| Evidence-only completion policy | Explicit `delivery_mode="final_stream"` is supported |
| Complete-answer validator or forbidden-response patterns | `final_stream` is rejected; required validators are never silently bypassed |
| Harness/delegation/no native provider stream | Explicitly reported buffered fallback; no token-latency promise |

`CompletionPolicy(enabled=True, delivery_mode="final_stream",
require_tool_calls=True)` checks evidence at the finalization boundary.
The reserved `memorizz_finalize_answer` tool must be called alone; application
tools must not use that name. This extra phase can add model work. Cache hits
are revalidated and delivered as an already-available answer; cache compatibility
includes delivery policy. Persisted required runtime validators still fail
closed until rebound by the host.

## CLI and UI

```bash
memorizz run "question"                       # streaming text, flushed stdout
memorizz run --output jsonl "question"        # one canonical event per line
memorizz run --no-stream "question"          # complete-result compatibility
```

`--stream` remains accepted. Progress/diagnostics go to stderr. Exit codes:
0 completed, 1 error/state-save failure, 3 approval, 130 cancelled, 141 closed
pipe. Stop/Ctrl-C retains partial display; partial generation is never cached.
A pipe closing after a completed write cannot undo that already committed write.

Playground displays startup progress, coalesces text painting on animation frames
and preserves partial text on Stop/errors. Answer completion and persistence are
separate; a follow-up stays serialized behind prior state commits. SSE supports
UTF-8 fragments, CR/LF/CRLF, multiline data, IDs and heartbeat comments. EOF
without a terminal frame is interruption. Responses use `no-cache, no-transform`
and `X-Accel-Buffering: no`; production proxies still need a flushing test.

## MCP

Ordinary `memorizz_execute_agent` calls now consume the canonical stream and
return the usual complete response plus IDs/outcomes. Standard progress is sent
when the client supplies a progress callback/token. A client ignoring progress
still receives the final tool result. Explicit `event_format=null` selects the
synchronous compatibility path.

For actual answer events, opt into `event_format="memorizz.events.v1"`.
This is a **Memorizz extension, not a standard MCP token feature**. It places
the canonical envelope in progress-notification
`_meta["io.memorizz/stream-event"]`, preserving standard progress token/value.
The adapter echoes a per-request `io.memorizz/stream-request` correlation ID.
Use the compatible client dispatcher:

```python
from mcp import Client
from memorizz.mcp.streaming import MCPEventDispatcher

dispatcher = MCPEventDispatcher()
async with Client(transport, message_handler=dispatcher.receive) as client:
    result = await dispatcher.call_tool(
        client,
        {"agent_id": "your-agent", "message": "question"},
        lambda event: print(event["delta"], end="", flush=True)
        if event["type"] == "answer.delta" else None,
    )
```

Callbacks must be synchronous and quick; enqueue bounded renderer work if needed.
Sequence validation, a bounded reorder buffer and request/run correlation prevent
silent drops or cross-request delivery. Each server stream retains execute/write
authorization, exposed-agent checks, principal-local conversation IDs, approval
handling and its agent lock through state saving.

Explicit MCP cancellation reaches the provider. HTTP disconnection alone is not
interpreted as cancellation. Native stream resumption/polling is not implemented.
External MCP tools' progress remains tool activity, never Memorizz answer text.
Arbitrary host applications still need an extension renderer to animate text.

## Providers, resources and timing

| Provider | Native text | Tools | Notes |
|---|---|---|---|
| OpenAI Chat / Responses | Yes | Yes | Native event loops, terminal status/usage, closeable streams |
| Azure Chat | Yes | Yes | Shared Chat parser and explicit implementation |
| Anthropic | Yes | Yes | Content blocks, usage/stop checks, context-manager cleanup |
| Ollama | Yes | Yes | Final done required, usage and iterator cleanup |
| Hugging Face | With tokenizer | No | Bounded streamer channel, cancellation stopping criterion, supervised worker |
| MLX | Yes | No | Exact whitespace, cooperative iterator cleanup |

The default channel has 64 events (configurable 1–1024); frames are capped at
64 KiB. Backpressure blocks the producer rather than dropping answer deltas.
At most 32 owned stream/provider workers are admitted. Close requests cooperative
cancellation and waits up to two seconds; an uncooperative third-party call
remains supervised and retains capacity until it exits. Cancellation cannot
roll back a tool's completed external side effect.

Elapsed statuses distinguish agent/context readiness, provider request/first
delta, public delta, answer completion and run completion. There are no per-token
database writes. Compare client receipt/paint intervals separately rather than
subtracting unsynchronized client/server clocks. Synthetic barrier tests prove
ordering, not live model performance or production latency SLAs.

Adapter verification uses OpenAI SDK 2.54.0 and MCP SDK 2.0.0. The implementation
follows [OpenAI's typed Responses streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses),
[Transformers' streamer and stopping APIs](https://huggingface.co/docs/transformers/main/en/internal/generation_utils),
and the MCP [progress](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/progress),
[transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
and [cancellation](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/cancellation)
contracts. Tested SDK versions are evidence, not a claim that every older provider
SDK or every model/deployment has been exercised live.
