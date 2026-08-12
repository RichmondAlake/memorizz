"""Deterministic SDK-based MCP server used by client integration tests."""

import os
from typing import Any

from mcp.server import MCPServer

server = MCPServer("memorizz-test-server")


@server.tool()
def echo(payload: dict[str, Any]) -> dict[str, Any]:
    """Echo arbitrary JSON, including booleans and null values."""
    return payload


@server.tool()
def create_item(name: str) -> dict[str, str]:
    """Pretend to create an item so approval policy can be tested."""
    return {"name": name}


@server.resource("memo://welcome")
def welcome() -> str:
    """Return a deterministic resource."""
    return "hello from MCP"


@server.prompt()
def greeting(name: str) -> str:
    """Return a deterministic prompt."""
    return f"Hello {name}"


if __name__ == "__main__":
    transport = os.environ.get("MCP_TEST_TRANSPORT", "stdio")
    kwargs = {}
    if transport == "streamable-http":
        kwargs = {
            "host": "127.0.0.1",
            "port": int(os.environ.get("MCP_TEST_PORT", "8766")),
        }
    server.run(transport=transport, **kwargs)
