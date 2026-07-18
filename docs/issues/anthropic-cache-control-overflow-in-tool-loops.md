# Bug report: Anthropic tool loops exceed the `cache_control` block limit

## Triage

- **Status:** Resolved in `0.2.1`
- **Severity:** High
- **Suggested priority:** P1 for the Anthropic provider
- **Component:** `memorizz.llms.anthropic.Anthropic`
- **Affected paths:** Streaming and non-streaming MemAgent tool loops
- **Confirmed revision:** `main` at `cff9e45`
- **Confirmed package manifest version:** `0.2.0`
- **Environment used:** Python 3.12.13, `anthropic` 0.116.0
- **Prompt-caching introduction:** `git blame` attributes the relevant
  `_apply_cache_control` implementation to `88c2b95`

## Resolution

The provider now fixes the request-ownership invariant rather than masking the
failure by disabling caching:

1. Streaming and non-streaming generation share one Anthropic request builder.
2. Message content, tool arguments, tool schemas, system blocks, and request
   options are request-owned copies before cache metadata is attached.
3. Cache budgeting counts valid API-level breakpoints across tools, system
   content, and message content without miscounting identically named keys in
   tool schemas or tool inputs.
4. Caller-supplied breakpoints are preserved and MemoRizz spends only the
   remaining slots. A caller request already exceeding Anthropic's limit fails
   locally before network I/O.
5. Regression coverage exercises both generation paths, string and block
   content, an established post-tool request, five repeated tool iterations,
   pre-annotated inputs, over-budget inputs, and disabled caching.

The original deterministic reproduction now reports an unchanged input and
three cache breakpoints on both the first and post-tool requests.

## Summary

When Anthropic prompt caching is enabled, MemoRizz mutates the reusable
MemAgent message history while adding `cache_control` fields. Those fields
survive into the next tool-loop iteration. MemoRizz then adds three more
breakpoints to the next request, producing five `cache_control` blocks even
though Anthropic accepts at most four.

The tool itself can complete successfully, but Anthropic rejects the
post-tool model request before the assistant can use the result or produce
its final answer.

## Customer-visible failure

The confirmed OpenSpeech sequence was:

1. The user asked the assistant to ingest a YouTube video and create two
   LinkedIn posts.
2. The `ingest_url` tool completed and returned `done`.
3. The next Anthropic request failed with HTTP 400:

```text
Error code: 400 - {
  'type': 'error',
  'error': {
    'type': 'invalid_request_error',
    'message': 'A maximum of 4 blocks with cache_control may be provided. Found 5.'
  },
  'request_id': 'req_011Cd9zeTN94T9tuVtSXh1AB'
}
```

This is not caused by the YouTube URL, ingestion result, or user prompt. The
tool completed; request construction for the following model call is invalid.

## Impact and trigger conditions

The failure is reproducible when all of the following are true:

1. The Anthropic provider is used with `enable_prompt_caching=True`, which is
   the default.
2. The request has enough existing conversation content for MemoRizz to mark
   both the final and second-to-last non-system messages.
3. The model calls a tool and MemAgent performs another model request with the
   tool result.

This makes ordinary tool use unreliable in established Anthropic-backed
threads. Both execution paths are affected because
[`generate`](https://github.com/RichmondAlake/memorizz/blob/cff9e45/src/memorizz/llms/anthropic.py#L193-L229)
and
[`generate_stream`](https://github.com/RichmondAlake/memorizz/blob/cff9e45/src/memorizz/llms/anthropic.py#L235-L369)
use the
same conversion and cache-control sequence.

A brand-new thread with only one user message can remain exactly at the
four-block limit after one tool call. That does not make the implementation
safe: normal conversation history triggers the five-block request.

## Deterministic reproduction

This reproduction uses the provider's request-building helpers and requires
no API key or network call. Run it from the repository root:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import copy
import json

from memorizz.llms.anthropic import Anthropic


def count_cache_control(value):
    if isinstance(value, dict):
        own = 1 if "cache_control" in value else 0
        return own + sum(count_cache_control(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_cache_control(item) for item in value)
    return 0


provider = Anthropic.__new__(Anthropic)
provider._enable_prompt_caching = True

messages = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "earlier answer"},
    {
        "role": "user",
        "content": "ingest this URL and write two LinkedIn posts",
    },
]
before = copy.deepcopy(messages)

system, api_messages = provider._split_system(messages)
converted = provider._convert_messages(api_messages)
plain_message_is_aliased = converted[0] is messages[1]
first_request = {"system": system, "messages": converted}
provider._apply_cache_control(first_request)

messages.extend(
    [
        {
            "role": "assistant",
            "content": "I will ingest it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "ingest_url",
                        "arguments": {"url": "https://youtu.be/example"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "done",
        },
    ]
)

system, api_messages = provider._split_system(messages)
converted = provider._convert_messages(api_messages)
post_tool_request = {"system": system, "messages": converted}
provider._apply_cache_control(post_tool_request)

print(
    json.dumps(
        {
            "plain_message_is_aliased": plain_message_is_aliased,
            "input_mutated_after_first_call": messages[:4] != before,
            "first_model_call_cache_blocks": count_cache_control(first_request),
            "post_tool_model_call_cache_blocks": count_cache_control(
                post_tool_request
            ),
            "anthropic_maximum": 4,
        },
        indent=2,
    )
)
PY
```

Confirmed output:

```json
{
  "plain_message_is_aliased": true,
  "input_mutated_after_first_call": true,
  "first_model_call_cache_blocks": 3,
  "post_tool_model_call_cache_blocks": 5,
  "anthropic_maximum": 4
}
```

## Root cause

There are three interacting behaviors:

1. MemAgent builds `messages` once and reuses the same mutable list for every
   tool-loop iteration. See the streaming loop in
   [`core.py`](https://github.com/RichmondAlake/memorizz/blob/cff9e45/src/memorizz/memagent/core.py#L3628-L3735)
   and the
   non-streaming loop in
   [`core.py`](https://github.com/RichmondAlake/memorizz/blob/cff9e45/src/memorizz/memagent/core.py#L4178-L4263).
2. `_split_system` returns references to the original non-system message
   dictionaries. `_convert_messages` creates new dictionaries for tool
   messages, but its plain-message branch uses `converted.append(msg)`. See
   [`anthropic.py`](https://github.com/RichmondAlake/memorizz/blob/cff9e45/src/memorizz/llms/anthropic.py#L499-L591).
3. `_apply_cache_control` mutates those dictionaries. For string content it
   replaces `msg["content"]`; for block content it replaces `content[-1]`.
   See
   [`anthropic.py`](https://github.com/RichmondAlake/memorizz/blob/cff9e45/src/memorizz/llms/anthropic.py#L375-L443).

The aliasing means provider-specific request metadata leaks back into
MemAgent's provider-neutral conversation state.

The breakpoint totals then progress as follows:

| Request | System | Inherited from first request | Newly marked messages | Total |
| --- | ---: | ---: | ---: | ---: |
| Initial request | 1 | 0 | 2 | 3 |
| Request after tool result | 1 | 2 | 2 | **5** |

The second row exceeds Anthropic's enforced maximum of four.

## Expected behavior

- Provider request construction must not mutate the caller's message objects.
- Every outbound Anthropic request must contain at most four
  `cache_control` blocks.
- After a tool returns, the next model call should consume the result and
  produce a final answer.
- Prompt caching should retain its intended system/final/second-to-last
  breakpoints where the request budget allows them.

## Recommended fix

### 1. Make Anthropic conversion request-local

The minimum fix is to ensure plain messages and nested content blocks are
owned by the converted request before `_apply_cache_control` runs. A deep
copy is required because `_apply_cache_control` also updates nested lists.

For example, the plain-message branch of `_convert_messages` could append a
deep copy instead of the original dictionary. An equivalent solution is to
deep-copy the messages before conversion/cache annotation. The important
invariant is that both `generate` and `generate_stream` leave their input
objects unchanged.

### 2. Enforce a hard request-level breakpoint budget

As defense in depth, count existing `cache_control` fields across tools,
system content, and message content before sending a request. The provider
must never add markers beyond Anthropic's four-block limit.

Engineering should define how caller-supplied markers are handled:

- preserve them and spend only the remaining budget; or
- remove only MemoRizz-managed stale markers on the request-local copy, then
  add the desired breakpoints.

Do not remove or rewrite fields on the caller's original objects.

### 3. Consider sharing request assembly

`generate` and `generate_stream` currently duplicate the split, conversion,
kwargs, and cache-control sequence. A shared request-building helper would
make the immutability and marker-budget guarantees easier to test once.

## Missing regression coverage

The current cache-control tests in
[`test_context_efficiency.py`](https://github.com/RichmondAlake/memorizz/blob/cff9e45/tests/unit/test_context_efficiency.py#L423-L493)
all pass:

```text
3 passed, 27 deselected in 0.41s
```

They each annotate one fresh request. They do not assert that the input is
unchanged or simulate a second model call after a tool result, so they cannot
detect this accumulation.

Add tests covering:

1. `generate` does not mutate input messages with string or block content.
2. `generate_stream` does not mutate input messages with string or block
   content.
3. An established conversation followed by an assistant tool call and tool
   result produces no more than four markers on every model request. Under
   the current three-breakpoint design, each independently built request
   should contain three.
4. Multiple tool-loop iterations remain within the limit.
5. Pre-annotated/caller-supplied cache markers cannot make the final request
   exceed four.
6. `enable_prompt_caching=False` produces zero markers and remains a no-op.

The Anthropic client can be mocked; no live API request is necessary.

## Acceptance criteria

- The original `messages` value is deeply equal before and after request
  construction in both streaming and non-streaming paths.
- No outbound request contains more than four `cache_control` blocks.
- An established Anthropic-backed thread can execute `ingest_url` and then
  receive the assistant's final response.
- Existing cache-usage extraction and prompt-caching tests continue to pass.
- A regression test fails on `cff9e45` with five markers and passes with the
  fix.

## Workaround for version 0.2.0

Construct the provider with prompt caching disabled:

```python
Anthropic(..., enable_prompt_caching=False)
```

This avoids the invalid request in the affected release because
`_apply_cache_control` becomes a no-op. It also gives up Anthropic
prompt-cache savings, so it should be used only until a release containing
the provider fix is installed.
