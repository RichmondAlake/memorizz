"""Compatible-client adapter for Memorizz's opt-in progress metadata extension.

Pass dispatcher.receive as Client(message_handler=...). Callbacks are synchronous
and must be quick (enqueue bounded UI work); they are never diagnostic log sinks.
"""

import asyncio
import inspect
import json
import uuid

from ..mcp_server.streaming import EVENT_META, FORMAT, REQUEST_META
from ..streaming import MAX_EVENT_BYTES


class MCPEventDispatcher:
    def __init__(self):
        self._routes = {}

    async def receive(self, notification):
        params = getattr(notification, "params", None)
        meta = getattr(params, "meta", None) or {}
        key = meta.get(REQUEST_META)
        if not isinstance(key, str) or key not in self._routes:
            return
        route = self._routes[key]
        try:
            event = meta.get(EVENT_META)
            if (
                not isinstance(event, dict)
                or len(json.dumps(event).encode("utf-8")) > MAX_EVENT_BYTES
            ):
                raise ValueError("Invalid stream event size")
            if event.get("version") != 1 or not isinstance(event.get("seq"), int):
                raise ValueError("Unsupported stream event")
            seq = event["seq"]
            if route["done"].is_set() or seq < route["next"] or seq in route["pending"]:
                raise ValueError("Duplicate/late stream event")
            if seq - route["next"] >= 64:
                raise ValueError("Stream reorder buffer exceeded")
            route["pending"][seq] = event
            while route["next"] in route["pending"]:
                current = route["pending"].pop(route["next"])
                if route["next"] == 1:
                    if current["type"] != "run.started":
                        raise ValueError("Stream must start with run.started")
                    route["run_id"] = current["run_id"]
                if current["run_id"] != route["run_id"]:
                    raise ValueError("Cross-run stream event")
                returned = route["callback"](current)
                if inspect.isawaitable(returned):
                    if inspect.iscoroutine(returned):
                        returned.close()
                    raise TypeError("Use a synchronous, nonblocking event callback")
                route["next"] += 1
                if current["type"] == "run.done":
                    route["done"].set()
        except Exception as exc:
            route["error"] = exc
            route["done"].set()
            route["task"].cancel()

    async def call_tool(self, client, arguments, on_event, *, progress_callback=None):
        key = str(uuid.uuid4())
        route = {
            "callback": on_event,
            "next": 1,
            "pending": {},
            "run_id": None,
            "done": asyncio.Event(),
            "error": None,
            "task": asyncio.current_task(),
        }
        self._routes[key] = route

        async def progress(value, total=None, message=None):
            if progress_callback:
                result = progress_callback(value, total, message)
                if inspect.isawaitable(result):
                    await result

        try:
            result = await client.call_tool(
                "memorizz_execute_agent",
                {**arguments, "event_format": FORMAT},
                progress_callback=progress,
                meta={REQUEST_META: key},
            )
            if not getattr(result, "is_error", False):
                # SDK tees raw notifications in tasks; the result can resolve first.
                await asyncio.wait_for(route["done"].wait(), timeout=2)
            if route["error"]:
                raise route["error"]
            return result
        except asyncio.CancelledError:
            if route["error"]:
                raise route["error"]
            raise
        finally:
            self._routes.pop(key, None)
