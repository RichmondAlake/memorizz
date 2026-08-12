# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Official-SDK MCP server exposing Memorizz as tools, resources, and prompts."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional

from mcp.server import MCPServer
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .auth import StaticAPIKeyVerifier, current_identity
from .config import MemorizzMCPServerConfig
from .runtime import MemorizzRuntime, MemorizzServerError

logger = logging.getLogger(__name__)
DOCUMENTATION_URL = "https://richmondalake.github.io/memorizz/guides/mcp-server/"


async def _tool_call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except MemorizzServerError as exc:
        raise ToolError(f"{exc.code}: {exc.message}") from exc
    except Exception as exc:
        logger.exception("Unexpected Memorizz MCP server failure")
        raise ToolError(
            "internal_error: Memorizz could not complete the operation"
        ) from exc


def create_memorizz_mcp_server(
    config: Optional[MemorizzMCPServerConfig] = None,
    *,
    runtime: Optional[MemorizzRuntime] = None,
) -> MCPServer:
    """Create the server without starting a transport (useful for embedding/tests)."""
    from .. import __version__

    resolved = config or MemorizzMCPServerConfig.from_env()
    token_verifier = None
    auth = None
    if resolved.auth_required:
        resource_url = f"{resolved.public_url}{resolved.path}"
        token_verifier = StaticAPIKeyVerifier(resolved.api_key_grants, resource_url)
        auth = AuthSettings(
            issuer_url=resolved.public_url,
            resource_server_url=resource_url,
            required_scopes=["memorizz:read"],
            service_documentation_url=DOCUMENTATION_URL,
        )

    service = runtime or MemorizzRuntime(resolved)
    server = MCPServer(
        "memorizz",
        title="Memorizz Memory and Agent Server",
        description="Tenant-scoped access to Memorizz agents, memory, and conversations.",
        instructions=(
            "Use read tools freely. Agent execution and memory writes may be disabled "
            "by server policy and should receive explicit user approval. Never claim a "
            "memory was stored or deleted unless the tool returned ok=true."
        ),
        version=__version__,
        website_url=DOCUMENTATION_URL,
        auth=auth,
        token_verifier=token_verifier,
    )

    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    write = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )
    destructive = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=True,
        idempotent_hint=True,
        open_world_hint=False,
    )
    execute = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=True,
    )

    @server.tool(annotations=read_only)
    async def memorizz_server_info() -> Dict[str, Any]:
        """Return server version, caller scope, policy, and supported memory types."""
        return await _tool_call(service.server_info, current_identity())

    @server.tool(annotations=read_only)
    async def memorizz_list_agents() -> Dict[str, Any]:
        """List Memorizz agents exposed to this MCP caller."""
        return await _tool_call(service.list_agents, current_identity())

    @server.tool(annotations=read_only)
    async def memorizz_get_agent(agent_id: str) -> Dict[str, Any]:
        """Return secret-free metadata for one exposed Memorizz agent."""
        return await _tool_call(service.get_agent, agent_id, current_identity())

    @server.tool(annotations=execute)
    async def memorizz_execute_agent(
        message: str,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a tenant-scoped Memorizz agent turn and return conversation IDs."""
        return await _tool_call(
            service.execute_agent,
            message,
            current_identity(),
            agent_id=agent_id,
            memory_id=memory_id,
            thread_id=thread_id,
            context=context,
        )

    @server.tool(annotations=read_only)
    async def memorizz_list_memories(
        memory_type: str = "knowledge_base",
        memory_id: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """List tenant-scoped records from one Memorizz memory store."""
        return await _tool_call(
            service.list_memories,
            memory_type,
            current_identity(),
            memory_id=memory_id,
            limit=limit,
        )

    @server.tool(annotations=read_only)
    async def memorizz_search_memories(
        query: str,
        memory_type: str = "knowledge_base",
        memory_id: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Semantically search one tenant-scoped Memorizz memory store."""
        return await _tool_call(
            service.search_memories,
            query,
            memory_type,
            current_identity(),
            memory_id=memory_id,
            limit=limit,
        )

    @server.tool(annotations=read_only)
    async def memorizz_get_memory(
        record_id: str, memory_type: str = "knowledge_base"
    ) -> Dict[str, Any]:
        """Read one tenant-scoped memory record by ID."""
        return await _tool_call(
            service.get_memory, record_id, memory_type, current_identity()
        )

    @server.tool(annotations=write)
    async def memorizz_store_memory(
        content: str,
        memory_type: str = "knowledge_base",
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a tenant-scoped knowledge-base or short-term memory record."""
        return await _tool_call(
            service.remember,
            content,
            memory_type,
            current_identity(),
            memory_id=memory_id,
            metadata=metadata,
        )

    @server.tool(annotations=destructive)
    async def memorizz_forget_memory(
        record_id: str,
        memory_type: str = "knowledge_base",
    ) -> Dict[str, Any]:
        """Propose permanent deletion for durable human-host approval."""
        return await _tool_call(
            service.forget,
            record_id,
            memory_type,
            current_identity(),
        )

    @server.tool(annotations=read_only)
    async def memorizz_list_conversations(
        agent_id: str, limit: int = 20
    ) -> Dict[str, Any]:
        """List resumable tenant-scoped conversations for an exposed agent."""
        return await _tool_call(
            service.list_conversations,
            agent_id,
            current_identity(),
            limit=limit,
        )

    @server.tool(annotations=read_only)
    async def memorizz_get_conversation(
        memory_id: str, thread_id: str, limit: int = 100
    ) -> Dict[str, Any]:
        """Read ordered messages from one tenant-scoped conversation."""
        return await _tool_call(
            service.get_conversation,
            memory_id,
            thread_id,
            current_identity(),
            limit=limit,
        )

    @server.resource("memorizz://server")
    async def server_resource() -> str:
        """Machine-readable public server capabilities and caller policy."""
        value = await _tool_call(service.server_info, current_identity())
        return json.dumps(value, ensure_ascii=False, indent=2)

    @server.resource("memorizz://agents/{agent_id}")
    async def agent_resource(agent_id: str) -> str:
        """Secret-free metadata for one exposed agent."""
        value = await _tool_call(service.get_agent, agent_id, current_identity())
        return json.dumps(value, ensure_ascii=False, indent=2)

    @server.resource("memorizz://agents/{agent_id}/conversations")
    async def conversations_resource(agent_id: str) -> str:
        """Recent tenant-scoped conversation summaries for an agent."""
        value = await _tool_call(
            service.list_conversations, agent_id, current_identity(), limit=50
        )
        return json.dumps(value, ensure_ascii=False, indent=2)

    @server.prompt()
    def memorizz_memory_assistant(objective: str) -> str:
        """Guide an MCP client through safe recall, agent execution, and storage."""
        return (
            "Use Memorizz to complete this objective: "
            f"{objective}\n\n"
            "1. Call memorizz_server_info to inspect policy and scopes.\n"
            "2. Discover an exposed agent, or search/list memories directly.\n"
            "3. Use memorizz_execute_agent only with user approval because the agent "
            "may call open-world tools and writes conversation memory.\n"
            "4. Use memorizz_store_memory or memorizz_forget_memory only when the user "
            "explicitly asks to change stored data.\n"
            "5. Preserve the returned memory_id and thread_id to resume a conversation."
        )

    return server


def run_memorizz_mcp_server(
    config: Optional[MemorizzMCPServerConfig] = None,
) -> None:
    from ..cli.config import load_layered_env

    load_layered_env()
    resolved = config or MemorizzMCPServerConfig.from_env()
    server = create_memorizz_mcp_server(resolved)
    if resolved.transport == "stdio":
        server.run(transport="stdio")
        return
    server.run(
        transport="streamable-http",
        host=resolved.host,
        port=resolved.port,
        streamable_http_path=resolved.path,
        stateless_http=resolved.stateless_http,
        max_request_body_size=resolved.max_request_body_size,
    )
