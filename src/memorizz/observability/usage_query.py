"""Bounded paginated usage reads, shared by SDK consumers and the local UI."""

from ..enums import MemoryType
from .analytics import aggregate_usage
from .normalization import query_trace_events
from .pricing import DEFAULT_PRICING


def query_usage(
    provider,
    *,
    filters=None,
    max_events=5000,
    pricing=DEFAULT_PRICING,
    timezone_name="UTC",
    model=None,
):
    """Read recorded trace events, preferring the provider's native event index.

    ``filters`` uses ``query_trace_events`` filters. Hosts MUST enforce tenant
    restrictions here. Work is bounded across three legacy stores; partial
    reads and normalization failures are exposed, never called complete usage.
    """
    if type(max_events) is not int or not 1 <= max_events <= 100_000:
        raise ValueError("max_events must be between 1 and 100000")
    rows, stores = [], []
    for memory_type in (
        MemoryType.SHARED_MEMORY,
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.TOOL_LOG,
    ):
        cursor = None
        errors = pages = 0
        complete = False
        while len(rows) < max_events and pages < 20:
            kwargs = dict(filters or {})
            kwargs.update(limit=min(1000, max_events - len(rows)), cursor=cursor)
            if memory_type == MemoryType.SHARED_MEMORY:
                kwargs["record_type"] = "observability_trace_bundle"
            native = getattr(provider, "query_trace_events", None)
            page = (
                native(**kwargs)
                if memory_type == MemoryType.SHARED_MEMORY and callable(native)
                else query_trace_events(provider, memory_type, **kwargs)
            )
            pages += 1
            rows.extend(page.get("items") or [])
            errors += int(page.get("normalization_errors") or 0)
            next_cursor = page.get("next_cursor")
            if not next_cursor:
                complete = (
                    not errors
                    and not page.get("truncated")
                    and page.get("read_completeness") not in {"untrusted", "partial"}
                    and page.get("coverage") not in {"untrusted", "partial"}
                )
                break
            if next_cursor == cursor:
                break
            cursor = next_cursor
        stores.append(
            {
                "store": memory_type.value,
                "pages": pages,
                "read_complete": complete,
                "normalization_errors": errors,
            }
        )
    return aggregate_usage(
        rows,
        pricing=pricing,
        timezone_name=timezone_name,
        max_events=max_events,
        model=model,
        coverage={
            "scope": "bounded_recorded_events",
            "stores": stores,
            "read_complete": all(store["read_complete"] for store in stores),
        },
    )
