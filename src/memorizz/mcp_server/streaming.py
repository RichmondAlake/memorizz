"""Request-scoped MCP progress; answer events are an opt-in vendor extension."""

import asyncio

from mcp.types import ProgressNotification, ProgressNotificationParams

FORMAT = "memorizz.events.v1"
EVENT_META = "io.memorizz/stream-event"
REQUEST_META = "io.memorizz/stream-request"


async def execute_stream(service, message, identity, ctx, *, event_format, **kwargs):
    from .runtime import MemorizzServerError

    if event_format not in {"progress", FORMAT}:
        raise MemorizzServerError(
            "unsupported_event_format", "Use progress or memorizz.events.v1"
        )
    request = ctx.request_context
    meta = request.meta
    meta_values = dict(meta or {})
    progress_token = meta_values.get("progress_token", meta_values.get("progressToken"))
    correlation = meta_values.get(REQUEST_META)
    if event_format == FORMAT and progress_token is None:
        raise MemorizzServerError(
            "progress_token_required", "Event streaming requires a progress token"
        )
    await ctx.report_progress(0, message="Preparing agent")
    setup = asyncio.create_task(
        asyncio.to_thread(service.execute_agent_events, message, identity, **kwargs)
    )
    try:
        stream = await asyncio.shield(setup)
    except asyncio.CancelledError:
        # Loading itself may be synchronous; reclaim any later-created stream.
        def reclaim(task):
            if not task.cancelled() and task.exception() is None:
                asyncio.create_task(asyncio.to_thread(task.result().close))

        setup.add_done_callback(reclaim)
        raise

    async def watch_cancel():
        # MCP 2.0 request-scoped dispatch exposes explicit peer cancellation.
        outbound = getattr(ctx.session, "_request_outbound", None)
        requested = getattr(outbound, "cancel_requested", None)
        if requested is not None:
            await requested.wait()
            await asyncio.to_thread(stream.cancellation.cancel)

    watcher = asyncio.create_task(watch_cancel())
    try:
        async for event in stream.async_events():
            kind = event["type"]
            # Never put answer text, tool arguments or reasoning in standard progress.
            readable = event.get("stage") or kind
            if event_format == FORMAT:
                await ctx.session.send_notification(
                    ProgressNotification(
                        params=ProgressNotificationParams(
                            progress_token=progress_token,
                            progress=event["seq"],
                            message=str(readable).replace("_", " "),
                            meta={EVENT_META: dict(event), REQUEST_META: correlation},
                        )
                    ),
                    related_request_id=request.request_id,
                )
            elif kind not in {"answer.delta", "usage", "completion.check"}:
                await ctx.report_progress(
                    event["seq"], message=str(readable).replace("_", " ")
                )
        return stream.result
    finally:
        watcher.cancel()
        # Task cancellation is explicit MCP request cancellation. A dropped HTTP
        # connection alone is handled by the MCP transport, not an abort signal.
        await asyncio.to_thread(stream.close)
