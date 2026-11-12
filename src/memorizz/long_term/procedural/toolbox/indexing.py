# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""How toolbox rows are found by meaning.

Shared by the memory providers that embed tool rows themselves and by the MCP
tool index, so a tool is always embedded from the same text however it was
stored.
"""

from typing import Any, Dict, List, Sequence

# A connected MCP server's settings (one row per server).
MCP_SERVER_CONFIG = "mcp_server_config"
# One tool a connected MCP server lists.
MCP_TOOL = "mcp_tool"
# What the MCP tool index last wrote for one server.
MCP_TOOL_CATALOG = "mcp_tool_catalog"

# Rows that describe tools but are not the agent's own Python tools; loading
# an agent must never turn them into callables.
NOT_AGENT_TOOLS = (MCP_SERVER_CONFIG, MCP_TOOL, MCP_TOOL_CATALOG)
# Rows that are bookkeeping, not tools, so nothing should embed them.
NOT_SEARCHABLE = (MCP_SERVER_CONFIG, MCP_TOOL_CATALOG)
# Rows the MCP tool index keeps up to date itself; saving an agent (which
# rewrites the rows mirroring its own tools) must leave them alone.
KEPT_BY_MCP_TOOL_INDEX = (MCP_TOOL, MCP_TOOL_CATALOG)

# Longer tool descriptions add little meaning and cost embedding tokens.
MAX_SEARCH_TEXT = 2000

#: Heading of the usage notes tool learning keeps after a tool's description.
LEARNED_HEADING = "Learned from use:"


def row_type(row: Dict[str, Any]) -> str:
    """The row's tool type, lower-cased (``function`` when unset)."""
    return str(row.get("tool_type") or row.get("type") or "function").strip().lower()


def is_agent_mirror_row(row: Dict[str, Any], agent_id: str) -> bool:
    """A row saving the agent wrote (``<agent_id>:<tool>``), so it may rewrite.

    Rows a Toolbox registered have their own ids and owners; saving the agent
    leaves them alone.
    """
    # Any of its ids: on Oracle ``_id`` is the row's own id and the mirror id
    # is kept as ``tool_id``.
    if not agent_id:
        return False
    return any(
        str(row.get(key) or "").startswith(f"{agent_id}:")
        for key in ("_id", "tool_id", "id")
    )


def search_text(row: Dict[str, Any]) -> str:
    """The text a tool row is embedded from: its name, what it does, the
    names of its arguments (which often say what a terse description omits)
    and the example requests kept for it.

    The description comes first because it is the richer of the two: it
    holds what augmentation wrote, where the docstring is the author's own
    text. Usage notes learned since are left out: they tell the model how to
    call the tool, and say nothing about which requests it is for.
    """
    name = str(row.get("name") or "").strip()
    described = str(row.get("description") or row.get("docstring") or "")
    described = described.split(LEARNED_HEADING, 1)[0]
    about = " ".join(described.split())
    text = f"{name}: {about}" if about else name
    arguments = argument_names(row.get("parameters"))
    if arguments:
        text = f"{text} Arguments: {', '.join(arguments)}."
    queries = example_requests(row.get("queries"))
    if queries:
        text = f"{text} Example requests: {' | '.join(queries)}"
    return text[:MAX_SEARCH_TEXT]


def example_requests(value: Any) -> List[str]:
    """Example requests kept on a tool row, as a list. Older rows hold one
    block of text, which is kept whole."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                import json

                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                value = parsed
        if isinstance(value, str):
            return [" ".join(text.split())] if text else []
    if not isinstance(value, (list, tuple)):
        return []
    return [" ".join(str(item).split()) for item in value if str(item).strip()]


def argument_names(parameters: Any) -> List[str]:
    """Argument names from a JSON schema, a properties map or a parameter list."""
    if isinstance(parameters, dict):
        properties = parameters.get("properties")
        names = properties if isinstance(properties, dict) else parameters
        return sorted(str(name) for name in names)
    if isinstance(parameters, list):
        return sorted(
            str(item["name"])
            for item in parameters
            if isinstance(item, dict) and item.get("name")
        )
    return []


def embed_documents(embedder: Any, texts: Sequence[str]) -> List[List[float]]:
    """Embed ``texts`` as stored documents, in one batch when the embedder can."""
    batch = getattr(embedder, "get_embeddings", None)
    if callable(batch):
        return [list(vector) for vector in batch(list(texts))]
    return [list(embedder.get_embedding(text)) for text in texts]
