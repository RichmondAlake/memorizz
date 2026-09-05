import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from memorizz.mcp.streaming import MCPEventDispatcher
from memorizz.mcp_server.runtime import MemorizzServerError
from memorizz.mcp_server.streaming import (
    EVENT_META,
    FORMAT,
    REQUEST_META,
    execute_stream,
)


def test_extension_requires_explicit_supported_format_and_progress_token():
    service = NS(execute_agent_events=Mock())
    ctx = NS(request_context=NS(meta={}))
    with pytest.raises(MemorizzServerError, match="requires a progress token"):
        asyncio.run(execute_stream(service, "hi", None, ctx, event_format=FORMAT))
    with pytest.raises(MemorizzServerError, match="Use progress"):
        asyncio.run(execute_stream(service, "hi", None, ctx, event_format="unknown"))
    service.execute_agent_events.assert_not_called()


def test_dispatcher_reorders_and_ignores_other_requests():
    async def run():
        dispatcher = MCPEventDispatcher()
        seen = []

        class Client:
            async def call_tool(self, name, arguments, *, meta, progress_callback):
                key = meta[REQUEST_META]

                async def send(seq, kind, correlation=key):
                    await dispatcher.receive(
                        NS(
                            params=NS(
                                meta={
                                    REQUEST_META: correlation,
                                    EVENT_META: {
                                        "version": 1,
                                        "run_id": "one",
                                        "seq": seq,
                                        "type": kind,
                                    },
                                }
                            )
                        )
                    )

                await send(1, "run.started", "other-request")
                await send(2, "answer.delta")
                await send(1, "run.started")
                await send(3, "run.done")
                return NS(is_error=False)

        await dispatcher.call_tool(Client(), {}, seen.append)
        assert [e["seq"] for e in seen] == [1, 2, 3]
        assert not dispatcher._routes

    asyncio.run(run())


@pytest.mark.parametrize("bad", ["duplicate", "cross_run", "oversize", "gap"])
def test_dispatcher_rejects_corrupt_stream_without_leaking_route(bad):
    async def run():
        dispatcher = MCPEventDispatcher()
        seen = []

        class Client:
            async def call_tool(self, name, arguments, *, meta, progress_callback):
                first = {"version": 1, "run_id": "one", "seq": 1, "type": "run.started"}

                async def send(event):
                    await dispatcher.receive(
                        NS(
                            params=NS(
                                meta={
                                    REQUEST_META: meta[REQUEST_META],
                                    EVENT_META: event,
                                }
                            )
                        )
                    )

                await send(first)
                event = {**first, "seq": 2, "type": "answer.delta"}
                if bad == "duplicate":
                    event["seq"] = 1
                elif bad == "cross_run":
                    event["run_id"] = "another"
                elif bad == "oversize":
                    event["delta"] = "x" * 70000
                else:
                    event["seq"] = 1000
                await send(event)
                await asyncio.sleep(0)
                return NS(is_error=False)

        task = asyncio.create_task(dispatcher.call_tool(Client(), {}, seen.append))
        with pytest.raises(ValueError):
            await task
        assert len(seen) == 1
        assert not dispatcher._routes

    asyncio.run(run())
