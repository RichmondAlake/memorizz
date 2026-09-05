"""Optional content-free response telemetry shared by provider adapters."""

from ..observability.models import LastResponseMetadata


def value(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def last_response_metadata(provider):
    metadata = dict(getattr(provider, "_last_response_metadata", {}) or {})
    usage = getattr(provider, "_last_usage", None) or {}
    for source, target in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
        ("cached_tokens", "cached_tokens"),
        ("cache_read_input_tokens", "cached_tokens"),
    ):
        item = usage.get(source)
        if type(item) is int and item >= 0:
            metadata[target] = item
    return LastResponseMetadata.model_validate(metadata).model_dump(exclude_none=True)


def response_metadata(response, *, text=None, max_output_tokens=None):
    choices = value(response, "choices") or []
    choice = choices[0] if choices else None
    incomplete = value(response, "incomplete_details")
    reason = (
        value(choice, "finish_reason")
        or value(response, "stop_reason")
        or value(response, "done_reason")
        or value(incomplete, "reason")
        or value(response, "finish_reason")
    )
    if text is None:
        message = value(choice, "message") or value(response, "message")
        text = value(message, "content") or value(response, "output_text")
        if text is None:
            content = value(response, "content")
            if isinstance(content, list):
                text = "".join(
                    value(block, "text") or ""
                    for block in content
                    if value(block, "type") == "text"
                )
    kwargs = {}
    request_id = value(response, "id")
    if isinstance(request_id, str):
        kwargs["request_id"] = request_id[:240]
    if isinstance(reason, str):
        kwargs["finish_reason"] = reason[:240]
    if isinstance(max_output_tokens, int) and max_output_tokens >= 0:
        kwargs["max_output_tokens"] = max_output_tokens
    if isinstance(text, str):
        kwargs.update(
            response_chars=len(text), response_bytes=len(text.encode("utf-8"))
        )
    return LastResponseMetadata(**kwargs).model_dump(exclude_none=True)
