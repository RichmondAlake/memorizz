"""Provider stream normalization; provider objects never become public events."""

from contextlib import contextmanager
from types import SimpleNamespace

from .llm_provider import LLMProvider
from .response_metadata import response_metadata


class ProviderStreamError(RuntimeError):
    def __init__(self, code, metadata=None):
        self.code = code
        self.response_metadata = metadata or {}
        super().__init__(code)


def streaming_capabilities(provider):
    method = getattr(provider, "generate_stream", None)
    supported = (
        callable(method)
        and getattr(method, "__func__", method) is not LLMProvider.generate_stream
    )
    name = type(provider).__name__
    reason = None if supported else "provider_has_no_stream"
    if (
        name == "HuggingFaceLLM"
        and getattr(getattr(provider, "_pipeline", None), "tokenizer", None) is None
    ):
        supported, reason = False, "tokenizer_unavailable"
    mode = getattr(provider, "api_mode", "chat_completions")
    return {
        "text_deltas": supported,
        "tool_calls": name not in {"HuggingFaceLLM", "MLXLLM"},
        "provider": name,
        "api_mode": mode if isinstance(mode, str) else "unspecified",
        "fallback_reason": reason,
    }


@contextmanager
def closing_provider_stream(stream):
    """Close on exhaustion/error/consumer cancellation, including network reads."""
    from ..streaming import current_cancellation

    token = current_cancellation.get()
    close = getattr(stream, "close", None)
    unregister = token.register(close) if token and callable(close) else lambda: None
    try:
        yield stream
    finally:
        unregister()
        if callable(close):
            close()


def tool_response(calls, content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content or None,
                    tool_calls=[
                        SimpleNamespace(
                            id=c["id"],
                            type="function",
                            function=SimpleNamespace(
                                name=c["name"], arguments=c["arguments"]
                            ),
                        )
                        for c in calls
                    ],
                )
            )
        ]
    )


def chat_events(provider, stream, *, max_output_tokens=None):
    text, calls, finished, failure = "", {}, False, None
    with closing_provider_stream(stream):
        for chunk in stream:
            provider._last_response_metadata.update(response_metadata(chunk))
            if getattr(chunk, "usage", None) is not None:
                provider._last_usage = provider._extract_usage(chunk)
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.delta
            reason = getattr(choice, "finish_reason", None)
            if reason:
                finished = True
                if reason not in {"stop", "tool_calls", "function_call"}:
                    failure = "provider_" + str(reason)
            # Reasoning is diagnostic only; the core public adapter never emits it.
            reasoning = next(
                (
                    getattr(delta, key, None)
                    for key in (
                        "reasoning_content",
                        "reasoning",
                        "thinking",
                        "reasoning_summary",
                    )
                    if isinstance(getattr(delta, key, None), str)
                    and getattr(delta, key)
                ),
                None,
            )
            if reasoning:
                yield {"type": "reasoning", "content": reasoning}
            content = getattr(delta, "content", None)
            if content:
                text += content
                yield {"type": "content", "content": content}
            if getattr(delta, "refusal", None):
                raise ProviderStreamError("provider_refusal")
            for call in getattr(delta, "tool_calls", None) or []:
                entry = calls.setdefault(
                    call.index, {"id": "", "name": "", "arguments": ""}
                )
                if call.id:
                    entry["id"] = call.id
                if call.function:
                    entry["name"] += call.function.name or ""
                    entry["arguments"] += call.function.arguments or ""
    provider._last_response_metadata.update(
        response_metadata(None, text=text, max_output_tokens=max_output_tokens)
    )
    if not finished or failure:
        usage = provider._last_usage or {}
        metadata = {
            **provider._last_response_metadata,
            **{
                target: usage[source]
                for source, target in (
                    ("prompt_tokens", "input_tokens"),
                    ("completion_tokens", "output_tokens"),
                    ("total_tokens", "total_tokens"),
                )
                if isinstance(usage.get(source), int)
            },
        }
        raise ProviderStreamError(failure or "provider_stream_incomplete", metadata)
    if provider._last_usage:
        yield {"type": "usage", "usage": provider._last_usage}
    if calls:
        ordered = [calls[k] for k in sorted(calls)]
        if any(not c["id"] or not c["name"] for c in ordered):
            raise ProviderStreamError("invalid_tool_call")
        yield {"type": "tool_calls", "response": tool_response(ordered, text)}
    else:
        yield {"type": "done", "content": text}


def responses_events(provider, stream, *, max_output_tokens=None):
    text, calls, finished = "", {}, False
    with closing_provider_stream(stream):
        for event in stream:
            kind = event.type
            if kind == "response.output_text.delta":
                text += event.delta
                yield {"type": "content", "content": event.delta}
            elif kind in {"response.refusal.delta", "response.refusal.done"}:
                raise ProviderStreamError("provider_refusal")
            elif (
                kind == "response.output_item.added"
                and event.item.type == "function_call"
            ):
                item = event.item
                calls[item.id] = {
                    "id": item.call_id,
                    "name": item.name,
                    "arguments": getattr(item, "arguments", "") or "",
                    "index": event.output_index,
                }
            elif kind == "response.function_call_arguments.delta":
                if event.item_id not in calls:
                    raise ProviderStreamError("unknown_tool_call_item")
                calls[event.item_id]["arguments"] += event.delta
            elif kind == "response.function_call_arguments.done":
                if event.item_id not in calls:
                    raise ProviderStreamError("unknown_tool_call_item")
                if calls[event.item_id]["arguments"] != event.arguments:
                    raise ProviderStreamError("tool_argument_delta_mismatch")
            elif kind in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }:
                response = event.response
                provider._last_usage = provider._extract_responses_usage(response)
                provider._last_response_metadata.update(
                    response_metadata(
                        response, text=text, max_output_tokens=max_output_tokens
                    )
                )
                if (
                    kind != "response.completed"
                    or getattr(response, "status", "completed") != "completed"
                ):
                    raise ProviderStreamError("provider_" + kind.split(".")[-1])
                finished = True
            elif kind == "error":
                raise ProviderStreamError("provider_stream_error")
    if not finished:
        raise ProviderStreamError("provider_stream_incomplete")
    if provider._last_usage:
        yield {"type": "usage", "usage": provider._last_usage}
    if calls:
        yield {
            "type": "tool_calls",
            "response": tool_response(
                sorted(calls.values(), key=lambda c: c["index"]), text
            ),
        }
    else:
        yield {"type": "done", "content": text}
