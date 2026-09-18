"""Optional content-free response telemetry shared by provider adapters."""

from ..observability.models import LastResponseMetadata


def value(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def last_response_metadata(provider):
    metadata = {
        **(getattr(provider, "_last_prompt_cache_metadata", {}) or {}),
        **(getattr(provider, "_last_response_metadata", {}) or {}),
    }
    usage = getattr(provider, "_last_usage", None) or {}
    for source, target in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
        ("cached_tokens", "cached_tokens"),
        ("cache_read_input_tokens", "cached_tokens"),
        ("cache_write_tokens", "cache_write_tokens"),
        ("cache_creation_input_tokens", "cache_write_tokens"),
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
    # Preserve the ID reported on the wire, separately from the requested alias
    # (or Azure deployment name). Never infer a snapshot from configuration.
    model = value(response, "model")
    if isinstance(model, str) and model.strip() and len(model) <= 240:
        kwargs["response_model"] = model
    tier = value(response, "service_tier")
    if isinstance(tier, str):
        kwargs["service_tier"] = tier[:240]
    usage = value(response, "usage")
    if usage is not None:
        for sources, target in (
            (("input_tokens", "prompt_tokens"), "input_tokens"),
            (("output_tokens", "completion_tokens"), "output_tokens"),
            (("total_tokens",), "total_tokens"),
        ):
            for source in sources:
                count = value(usage, source)
                if type(count) is int and count >= 0:
                    kwargs[target] = count
                    break
        details = value(usage, "input_tokens_details") or value(
            usage, "prompt_tokens_details"
        )
        for field in ("cached_tokens", "cache_write_tokens"):
            count = value(details, field)
            if type(count) is int and count >= 0:
                kwargs[field] = count
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


def validate_response_completion(provider, response):
    """Reject incomplete/empty model output before accepting answers or tools.

    Shared by MemAgent's streaming and non-streaming request boundaries.
    Tool-only responses are valid; reasoning-only responses are not answers.
    Providers without the optional metadata method remain supported.
    """
    from .streaming import ProviderStreamError

    metadata = {}
    getter = getattr(provider, "get_last_response_metadata", None)
    if callable(getter):
        try:
            recorded = getter()
            if isinstance(recorded, dict):
                metadata = LastResponseMetadata.model_validate(recorded).model_dump(
                    exclude_none=True
                )
        except Exception:
            pass
    metadata.update(response_metadata(response))
    reason = metadata.get("finish_reason")
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        raise ProviderStreamError("provider_length", metadata)
    if reason in {"content_filter", "refusal"}:
        raise ProviderStreamError("provider_refusal", metadata)
    if value(response, "status") in {"incomplete", "failed", "cancelled"}:
        raise ProviderStreamError("provider_incomplete", metadata)

    choices = value(response, "choices") or []
    message = value(choices[0], "message") if choices else value(response, "message")
    if value(message, "tool_calls"):
        return
    text = response if isinstance(response, str) else value(message, "content")
    if (
        response is None
        or isinstance(text, str)
        and not text.strip()
        or message is not None
        and text is None
    ):
        raise ProviderStreamError("empty_response", metadata)
