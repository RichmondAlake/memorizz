"""Content-free evidence of provider cache configuration and stable prefixes."""

import hashlib
import json


def cache_request_metadata(kwargs, provider):
    """Describe the actual outbound request; this does not claim a cache hit."""
    messages = kwargs.get("messages", kwargs.get("input", []))
    instructions = [kwargs.get("system"), kwargs.get("instructions")]
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system",
                "developer",
            }:
                break
            instructions.append(message.get("content"))
    prefix = {
        "instructions": instructions,
        "tools": kwargs.get("tools"),
        **{
            key: kwargs.get(key)
            for key in (
                "thinking",
                "reasoning",
                "tool_choice",
                "response_format",
                "output_config",
            )
        },
    }

    def digest(value):
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        ).hexdigest()

    def marked(field):
        # Inspect API cache boundaries only. A tool's JSON schema or nested
        # user data can contain the same field without enabling caching.
        boundaries = [kwargs] if provider == "anthropic" else []
        for name in ("system", "tools") if provider == "anthropic" else ():
            value = kwargs.get(name)
            if isinstance(value, list):
                boundaries.extend(value)
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    for name in (
                        ("content",)
                        if provider == "anthropic"
                        else ("content", "output")
                    ):
                        content = message.get(name)
                        if isinstance(content, list):
                            boundaries.extend(content)
        if provider == "anthropic":
            # Tool-result blocks may contain nested typed content blocks.
            # Follow only that hierarchy, not tool inputs or JSON schemas.
            for block in boundaries:
                content = block.get("content") if isinstance(block, dict) else None
                if isinstance(content, list):
                    boundaries.extend(content)
        return any(
            isinstance(item, dict) and bool(item.get(field)) for item in boundaries
        )

    if provider == "anthropic":
        enabled = marked("cache_control")
    elif provider == "openai":
        enabled = (kwargs.get("prompt_cache_options") or {}).get(
            "mode"
        ) != "explicit" or marked("prompt_cache_breakpoint")
    else:
        return {}
    metadata = {
        "prompt_cache_enabled": enabled,
        "prompt_cache_prefix": digest(prefix),
    }
    if kwargs.get("prompt_cache_key"):
        metadata["prompt_cache_key"] = digest(kwargs["prompt_cache_key"])
    return metadata
