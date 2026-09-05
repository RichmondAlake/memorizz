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
from mcp.server.mcpserver.context import Context
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


def _harden_tool_input_schemas(server: MCPServer) -> None:
    """Forbid undeclared MCP arguments in both JSON Schema and validation.

    The official SDK currently derives function argument models with Pydantic's
    default ``extra=ignore`` behavior. MemoRizz uses strict signatures, so make
    each registered model reject and advertise rejection of unknown fields.
    """
    manager = getattr(server, "_tool_manager", None)
    tools = getattr(manager, "_tools", None)
    if not isinstance(tools, dict):
        raise RuntimeError("The installed MCP SDK does not expose registered tools")
    for tool in tools.values():
        argument_model = tool.fn_metadata.arg_model
        argument_model.model_config = {
            **dict(argument_model.model_config),
            "extra": "forbid",
        }
        argument_model.model_rebuild(force=True)
        tool.parameters = argument_model.model_json_schema(by_alias=True)


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
        description=(
            "Tenant-scoped access to Memorizz agents, memory, conversations, and "
            "governed external agent harness runs."
        ),
        instructions=(
            "Inspect server policy first. Use read tools freely. Agent execution, "
            "configuration changes, compilation, compaction, and memory writes may be "
            "disabled by server policy and should receive explicit user approval. "
            "Never claim a mutation succeeded unless the tool returned ok=true."
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

    @server.tool(annotations=read_only)
    async def memorizz_inspect_agent(
        agent_id: str,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        include_package_capabilities: bool = False,
    ) -> Dict[str, Any]:
        """Inspect capabilities, cache, learning, and scoped observability."""
        return await _tool_call(
            service.inspect_agent,
            agent_id,
            current_identity(),
            memory_id=memory_id,
            thread_id=thread_id,
            include_package_capabilities=include_package_capabilities,
        )

    if resolved.allow_trace_queries:

        @server.tool(annotations=read_only)
        async def memorizz_query_traces(
            agent_id: str,
            thread_id: Optional[str] = None,
            root_trace_id: Optional[str] = None,
            cursor: Optional[str] = None,
            limit: int = 250,
            explain: bool = False,
        ) -> Dict[str, Any]:
            """Query/explain a bounded metadata-only trace page in the caller's tenant."""
            return await _tool_call(
                service.query_traces,
                agent_id,
                current_identity(),
                thread_id=thread_id,
                root_trace_id=root_trace_id,
                cursor=cursor,
                limit=limit,
                explain=explain,
            )

    @server.tool(annotations=read_only)
    async def memorizz_preview_personalization(
        agent_id: str,
        query: str,
        memory_id: Optional[str] = None,
        exclude_thread_id: Optional[str] = None,
        include_conversation_recall: bool = False,
        min_relevance_score: float = 0.65,
        max_conversation_memories: int = 2,
        preferences: Optional[Dict[str, Any]] = None,
        writing_samples: Optional[list[Dict[str, Any]]] = None,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Preview bounded memory personalization and content-safe evidence."""
        return await _tool_call(
            service.preview_personalization,
            agent_id,
            query,
            current_identity(),
            memory_id=memory_id,
            exclude_thread_id=exclude_thread_id,
            include_conversation_recall=include_conversation_recall,
            min_relevance_score=min_relevance_score,
            max_conversation_memories=max_conversation_memories,
            preferences=preferences,
            writing_samples=writing_samples,
            include_content=include_content,
        )

    @server.tool(annotations=write)
    async def memorizz_create_agent(
        name: str,
        instruction: Optional[str] = None,
        application_mode: str = "assistant",
        max_steps: int = 20,
        memory_ids: Optional[list[str]] = None,
        semantic_cache: bool = False,
        continual_learning: bool = False,
        learning_control_plane: bool = False,
        skill_retrieval: bool = False,
        skill_retrieval_top_k: int = 2,
        tool_access: str = "private",
        automations_enabled: bool = True,
        default_timezone: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        meta_harness_mode: Optional[str] = None,
        default_harness: str = "auto",
        harness_workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create and persist a local MemAgent after durable host approval."""
        return await _tool_call(
            service.create_agent,
            name,
            current_identity(),
            instruction=instruction,
            application_mode=application_mode,
            max_steps=max_steps,
            memory_ids=memory_ids,
            semantic_cache=semantic_cache,
            continual_learning=continual_learning,
            learning_control_plane=learning_control_plane,
            skill_retrieval=skill_retrieval,
            skill_retrieval_top_k=skill_retrieval_top_k,
            tool_access=tool_access,
            automations_enabled=automations_enabled,
            default_timezone=default_timezone,
            llm_provider=llm_provider,
            llm_model=llm_model,
            meta_harness_mode=meta_harness_mode,
            default_harness=default_harness,
            harness_workspace=harness_workspace,
        )

    @server.tool(annotations=write)
    async def memorizz_update_agent(
        agent_id: str,
        name: Optional[str] = None,
        instruction: Optional[str] = None,
        application_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        memory_ids: Optional[list[str]] = None,
        tool_access: Optional[str] = None,
        semantic_cache: Optional[bool] = None,
        continual_learning: Optional[bool] = None,
        learning_control_plane: Optional[bool] = None,
        skill_retrieval: Optional[bool] = None,
        skill_retrieval_top_k: Optional[int] = None,
        automations_enabled: Optional[bool] = None,
        default_timezone: Optional[str] = None,
        is_favorite: Optional[bool] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        clear_llm: bool = False,
        meta_harness_mode: Optional[str] = None,
        default_harness: Optional[str] = None,
        harness_workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update safe local agent configuration after host approval."""
        return await _tool_call(
            service.update_agent,
            agent_id,
            current_identity(),
            name=name,
            instruction=instruction,
            application_mode=application_mode,
            max_steps=max_steps,
            memory_ids=memory_ids,
            tool_access=tool_access,
            semantic_cache=semantic_cache,
            continual_learning=continual_learning,
            learning_control_plane=learning_control_plane,
            skill_retrieval=skill_retrieval,
            skill_retrieval_top_k=skill_retrieval_top_k,
            automations_enabled=automations_enabled,
            default_timezone=default_timezone,
            is_favorite=is_favorite,
            llm_provider=llm_provider,
            llm_model=llm_model,
            clear_llm=clear_llm,
            meta_harness_mode=meta_harness_mode,
            default_harness=default_harness,
            harness_workspace=harness_workspace,
        )

    @server.tool(annotations=destructive)
    async def memorizz_delete_agent(
        agent_id: str, cascade: bool = False
    ) -> Dict[str, Any]:
        """Propose local agent deletion for durable operator approval."""
        return await _tool_call(
            service.delete_agent,
            agent_id,
            current_identity(),
            cascade=cascade,
        )

    @server.tool(annotations=execute)
    async def memorizz_execute_agent(
        message: str,
        ctx: Context,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        event_format: Optional[str] = "progress",
        delivery_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a scoped turn. Opt into progress or memorizz.events.v1 notifications."""
        if event_format is not None:
            from .streaming import execute_stream

            try:
                return await execute_stream(
                    service,
                    message,
                    current_identity(),
                    ctx,
                    event_format=event_format,
                    agent_id=agent_id,
                    memory_id=memory_id,
                    thread_id=thread_id,
                    context=context,
                    delivery_mode=delivery_mode,
                )
            except MemorizzServerError as exc:
                raise ToolError(f"{exc.code}: {exc.message}") from exc
        if delivery_mode is not None:
            raise ToolError("delivery_mode requires event_format")
        return await _tool_call(
            service.execute_agent,
            message,
            current_identity(),
            agent_id=agent_id,
            memory_id=memory_id,
            thread_id=thread_id,
            context=context,
        )

    @server.tool(annotations=write)
    async def memorizz_compile_memory(
        agent_id: str,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        mode: str = "fast",
    ) -> Dict[str, Any]:
        """Compile scoped continual-learning events into retrieval artifacts."""
        return await _tool_call(
            service.compile_agent_memory,
            agent_id,
            current_identity(),
            memory_id=memory_id,
            thread_id=thread_id,
            mode=mode,
        )

    @server.tool(annotations=execute)
    async def memorizz_compact_conversation(
        agent_id: str,
        memory_id: str,
        thread_id: Optional[str] = None,
        days_back: int = 7,
        max_memories_per_summary: int = 50,
    ) -> Dict[str, Any]:
        """Summarize and compact one tenant-scoped conversation."""
        return await _tool_call(
            service.compact_conversation,
            agent_id,
            memory_id,
            current_identity(),
            thread_id=thread_id,
            days_back=days_back,
            max_memories_per_summary=max_memories_per_summary,
        )

    @server.tool(annotations=read_only)
    async def memorizz_list_harnesses() -> Dict[str, Any]:
        """List Codex, Claude Code, OpenHands, and native harness capabilities."""
        return await _tool_call(service.list_harnesses, current_identity())

    @server.tool(annotations=execute)
    async def memorizz_start_harness_run(
        task: str,
        workspace: str,
        harness: str = "auto",
        model: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        write: bool = False,
        allow_dirty_workspace: bool = False,
        network: str = "none",
        mcp_access: str = "read_only",
        allowed_env: Optional[list[str]] = None,
        execution_backend: str = "local",
        verification_command: Optional[str] = None,
        timeout_seconds: int = 900,
        max_steps: int = 80,
        max_cost_usd: Optional[float] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start a bounded harness run; risky envelopes pause for host approval."""
        return await _tool_call(
            service.start_harness_run,
            task,
            workspace,
            current_identity(),
            harness=harness,
            model=model,
            agent_id=agent_id,
            memory_id=memory_id,
            thread_id=thread_id,
            write=write,
            allow_dirty_workspace=allow_dirty_workspace,
            network=network,
            mcp_access=mcp_access,
            allowed_env=allowed_env,
            execution_backend=execution_backend,
            verification_command=verification_command,
            timeout_seconds=timeout_seconds,
            max_steps=max_steps,
            max_cost_usd=max_cost_usd,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )

    @server.tool(annotations=read_only)
    async def memorizz_get_harness_run(
        run_id: str, include_events: bool = False
    ) -> Dict[str, Any]:
        """Read one tenant-scoped durable harness run and verification result."""
        return await _tool_call(
            service.get_harness_run,
            run_id,
            current_identity(),
            include_events=include_events,
        )

    @server.tool(annotations=read_only)
    async def memorizz_list_harness_runs(
        status: Optional[str] = None, limit: int = 50
    ) -> Dict[str, Any]:
        """List durable harness runs visible to this tenant."""
        return await _tool_call(
            service.list_harness_runs,
            current_identity(),
            status=status,
            limit=limit,
        )

    @server.tool(annotations=read_only)
    async def memorizz_get_harness_events(
        run_id: str, after: int = 0, limit: int = 100
    ) -> Dict[str, Any]:
        """Read normalized events after a durable per-run sequence cursor."""
        return await _tool_call(
            service.get_harness_events,
            run_id,
            current_identity(),
            after=after,
            limit=limit,
        )

    @server.tool(annotations=execute)
    async def memorizz_cancel_harness_run(run_id: str) -> Dict[str, Any]:
        """Request durable cancellation of one tenant-scoped harness run."""
        return await _tool_call(service.cancel_harness_run, run_id, current_identity())

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

    @server.resource("memorizz://harness-runs/{run_id}")
    async def harness_run_resource(run_id: str) -> str:
        """Durable tenant-scoped harness run, result, and recent events."""
        value = await _tool_call(
            service.get_harness_run,
            run_id,
            current_identity(),
            include_events=True,
        )
        return json.dumps(value, ensure_ascii=False, indent=2)

    @server.prompt()
    def memorizz_memory_assistant(objective: str) -> str:
        """Guide an MCP client through safe recall, agent execution, and storage."""
        return (
            "Use Memorizz to complete this objective: "
            f"{objective}\n\n"
            "1. Call memorizz_server_info to inspect policy and scopes.\n"
            "2. Discover and inspect an exposed agent, or search/list memories directly.\n"
            "3. On local stdio, create, update, or propose deletion of an agent only "
            "when the user asks; host approval resumes the exact call.\n"
            "4. Use memorizz_execute_agent only with user approval because the agent "
            "may call open-world tools and writes conversation memory.\n"
            "5. Compile learning memory or compact a conversation only in the caller's "
            "explicit memory/thread scope.\n"
            "6. Store or forget memory only when the user explicitly asks to change "
            "stored data.\n"
            "7. Preserve returned memory_id and thread_id values to resume a conversation."
            "\n8. For workspace tasks, inspect harness capabilities, start one bounded "
            "run, and poll its run or event record. Never invent approval or verification."
        )

    _harden_tool_input_schemas(server)
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
