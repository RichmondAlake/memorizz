# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Keep the tools a connected MCP server lists in toolbox memory.

Each tool becomes one TOOLBOX row with an embedding, so the agent's tool
router ranks it by what a request means, the way it ranks the agent's own
tools, instead of by the words the two happen to share. One catalog row per
server records what was written. When a server lists the same tools as last
time a sync is a single read; otherwise only new or changed tools are
embedded, rows for tools the server dropped are deleted, and a change of
embedding model re-embeds them all.
"""

import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..enums.memory_type import MemoryType
from ..long_term.procedural.toolbox.indexing import (
    MCP_TOOL,
    MCP_TOOL_CATALOG,
    embed_documents,
    search_text,
)

logger = logging.getLogger(__name__)

PublicName = Callable[[str, str], str]


def tool_row_id(agent_id: str, server_name: str, tool_name: str) -> str:
    return f"{agent_id}:mcp-tool:{server_name}:{tool_name}"


def catalog_row_id(agent_id: str, server_name: str) -> str:
    return f"{agent_id}:mcp-tools:{server_name}"


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class MCPToolIndex:
    """One agent's MCP tools as searchable TOOLBOX rows."""

    def __init__(self, memory_provider: Any, agent_id: str):
        self.memory_provider = memory_provider
        self.agent_id = agent_id

    def sync(
        self,
        server_name: str,
        tools: List[Dict[str, Any]],
        public_name: PublicName,
    ) -> Dict[str, int]:
        """Bring the server's rows in line with its listing; returns counts."""
        rows = {}
        for tool in tools:
            name = str(tool.get("name") or "")
            if name:
                rows[name] = self._tool_row(
                    server_name, name, public_name(server_name, name), tool
                )
        wanted = {name: _fingerprint(search_text(row)) for name, row in rows.items()}
        embedder, model = self._embedder()
        catalog = self._catalog(server_name)
        written = catalog.get("tools") or {}
        current = written if catalog.get("embedder") == model else {}
        changed = [name for name, mark in wanted.items() if current.get(name) != mark]
        removed = sorted(set(written) - set(wanted))
        if not changed and not removed:
            return {"stored": 0, "removed": 0}

        vectors: List[Optional[List[float]]] = [None] * len(changed)
        if embedder is not None and changed:
            try:
                vectors = embed_documents(
                    embedder, [search_text(rows[name]) for name in changed]
                )
            except Exception as exc:
                # Nothing is recorded, so the next sync tries again.
                logger.warning(
                    "MCP tools from %s were not indexed: %s", server_name, exc
                )
                return {"stored": 0, "removed": 0}
        for name, vector in zip(changed, vectors):
            row = rows[name]
            if vector is not None:
                row["embedding"] = vector
            self.memory_provider.store(row, memory_store_type=MemoryType.TOOLBOX)
        for name in removed:
            self._delete(tool_row_id(self.agent_id, server_name, name))
        self.memory_provider.store(
            self._catalog_row(server_name, {"embedder": model, "tools": wanted}),
            memory_store_type=MemoryType.TOOLBOX,
        )
        logger.info(
            "Indexed %s MCP tools from %s (%s removed)",
            len(changed),
            server_name,
            len(removed),
        )
        return {"stored": len(changed), "removed": len(removed)}

    def remove_server(self, server_name: str) -> int:
        """Delete every row written for the server; returns the tools removed."""
        catalog = self._catalog(server_name)
        if not catalog:
            return 0
        names = list((catalog.get("tools") or {}).keys())
        for name in names:
            self._delete(tool_row_id(self.agent_id, server_name, name))
        self._delete(catalog_row_id(self.agent_id, server_name))
        return len(names)

    # ------------------------------------------------------------------ rows

    def _tool_row(
        self, server_name: str, name: str, public: str, tool: Dict[str, Any]
    ) -> Dict[str, Any]:
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        schema = {
            key: value
            for key, value in (schema if isinstance(schema, dict) else {}).items()
            if key != "$schema"
        }
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        description = (
            str(tool.get("description") or "").strip()
            or f"MCP tool '{name}' on server '{server_name}'"
        )
        row_id = tool_row_id(self.agent_id, server_name, name)
        return {
            "_id": row_id,
            "tool_id": row_id,
            "name": public,
            "description": description,
            "docstring": description,
            "signature": f"{public}({', '.join(sorted(properties))})",
            "tool_type": MCP_TOOL,
            "type": MCP_TOOL,
            "parameters": properties,
            "required": list(schema.get("required") or []),
            "input_schema": schema,
            "agent_id": self.agent_id,
            "user_id": None,
        }

    def _catalog_row(self, server_name: str, catalog: Dict[str, Any]):
        row_id = catalog_row_id(self.agent_id, server_name)
        return {
            "_id": row_id,
            "tool_id": row_id,
            "name": f"mcp-tools::{server_name}",
            "description": f"MCP tools indexed from {server_name}",
            "tool_type": MCP_TOOL_CATALOG,
            "type": MCP_TOOL_CATALOG,
            "parameters": catalog,
            "agent_id": self.agent_id,
            "user_id": None,
        }

    def _catalog(self, server_name: str) -> Dict[str, Any]:
        try:
            row = self.memory_provider.retrieve_by_id(
                catalog_row_id(self.agent_id, server_name),
                memory_store_type=MemoryType.TOOLBOX,
            )
        except Exception as exc:
            logger.debug("MCP tool catalog for %s unreadable: %s", server_name, exc)
            return {}
        catalog = (row or {}).get("parameters")
        if isinstance(catalog, str):
            try:
                catalog = json.loads(catalog)
            except ValueError:
                return {}
        return catalog if isinstance(catalog, dict) else {}

    def _delete(self, row_id: str) -> None:
        try:
            self.memory_provider.delete_by_id(
                row_id, memory_store_type=MemoryType.TOOLBOX
            )
        except Exception as exc:
            logger.debug("MCP tool row %s not deleted: %s", row_id, exc)

    # ------------------------------------------------------------ embeddings

    def _embedder(self) -> Tuple[Any, str]:
        """The embedder the provider searches with, and a name for its model.

        A provider that embeds what it stores gets rows without vectors;
        without any embedder the rows are still stored, and are embedded once
        one is configured (the model name changes).
        """
        from ..memory_provider.base import provider_manages_embeddings

        if provider_manages_embeddings(self.memory_provider):
            return None, "provider"
        getter = getattr(self.memory_provider, "_get_embedding_provider", None)
        try:
            embedder = getter() if callable(getter) else None
        except Exception:
            embedder = None
        if embedder is None:
            return None, "none"
        return embedder, _model_name(embedder)


def _model_name(embedder: Any) -> str:
    parts = []
    for attribute in ("provider_type", "get_default_model", "get_dimensions"):
        value = getattr(embedder, attribute, None)
        try:
            value = value() if callable(value) else value
        except Exception:
            value = None
        parts.append(str(getattr(value, "value", value)))
    if not any(part != "None" for part in parts):
        return type(embedder).__name__
    return "/".join(parts)
