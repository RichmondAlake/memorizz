"""Explicit provisioning of the Notion memory library and human-facing views."""

import os
import uuid

from ...enums.memory_type import MemoryType
from .client import NotionClient, NotionError
from .provider import PROPERTIES, rich_text


def provision_notion_workspace(
    parent_page_id, *, token=None, title="Memorizz", create_views=True, client=None
):
    """Create a new, isolated Memorizz area beneath an explicitly chosen page.

    This is intentionally NOT invoked by NotionProvider's constructor. Keep the
    returned IDs; calling it again creates another area. It does not grant access
    or connect Notion AI/MCP. On failure the partial area is left recoverable.
    """
    parent_page_id = str(uuid.UUID(str(parent_page_id)))
    api = client or NotionClient(token or os.environ.get("NOTION_TOKEN"))
    result = {}

    def page(parent, name):
        return api.request(
            "POST",
            "/pages",
            body={
                "parent": {"type": "page_id", "page_id": parent},
                "properties": {"title": {"title": rich_text(name)}},
            },
        )["id"]

    try:
        result["root_page_id"] = page(parent_page_id, title)
        result["memory_page_id"] = page(result["root_page_id"], "Memory")
        result["interface_page_id"] = page(result["root_page_id"], "Agent workspace")
        properties = {name: {kind: {}} for name, kind in PROPERTIES.values()}
        properties["Memory type"] = {
            "select": {"options": [{"name": value.value} for value in MemoryType]}
        }
        database = api.request(
            "POST",
            "/databases",
            body={
                "parent": {"type": "page_id", "page_id": result["memory_page_id"]},
                "title": rich_text("Memorizz memories"),
                "initial_data_source": {"properties": properties},
            },
        )
        sources = database.get("data_sources") or []
        if not sources:
            raise NotionError("Notion did not return the new memory data source")
        result["database_id"] = database["id"]
        result["data_source_id"] = sources[0]["id"]
        result["view_ids"] = []
        if create_views:
            schema = api.request("GET", "/data_sources/" + result["data_source_id"])
            ids = {
                key: schema["properties"][name]["id"]
                for key, (name, _) in PROPERTIES.items()
            }
            visible = {
                "name",
                "memory_type",
                "content",
                "event_time",
                "agent_id",
                "status",
            }
            configuration = {
                "type": "table",
                "wrap_cells": True,
                "properties": [
                    {"property_id": identifier, "visible": key in visible}
                    for key, identifier in ids.items()
                ],
            }
            for memory_type in MemoryType:
                view = api.request(
                    "POST",
                    "/views",
                    body={
                        "database_id": result["database_id"],
                        "data_source_id": result["data_source_id"],
                        "name": memory_type.value.replace("_", " ").title(),
                        "type": "table",
                        "filter": {
                            "property": ids["memory_type"],
                            "select": {"equals": memory_type.value},
                        },
                        "configuration": configuration,
                    },
                )
                result["view_ids"].append(view["id"])
            for name, memory_type in (
                ("Agents", MemoryType.MEMAGENT),
                ("Conversations", MemoryType.CONVERSATION_MEMORY),
                ("Traces", MemoryType.SHARED_MEMORY),
                ("Tool activity", MemoryType.TOOL_LOG),
            ):
                predicate = {
                    "property": ids["memory_type"],
                    "select": {"equals": memory_type.value},
                }
                if name == "Traces":
                    predicate = {
                        "and": [
                            predicate,
                            {
                                "property": ids["record_type"],
                                "rich_text": {"equals": "observability_trace_bundle"},
                            },
                        ]
                    }
                view = api.request(
                    "POST",
                    "/views",
                    body={
                        "create_database": {
                            "parent": {
                                "type": "page_id",
                                "page_id": result["interface_page_id"],
                            }
                        },
                        "data_source_id": result["data_source_id"],
                        "name": name,
                        "type": "table",
                        "filter": predicate,
                        "configuration": configuration,
                    },
                )
                result["view_ids"].append(view["id"])
        result["url"] = "https://www.notion.so/" + result["root_page_id"].replace(
            "-", ""
        )
        return result
    except BaseException as exc:
        # Preserve the original typed exception and recoverable IDs, without
        # logging credentials, request content, or deleting a partially built UI.
        # Interrupts also need recovery IDs when creation already started.
        exc.notion_workspace = dict(result)
        raise
    finally:
        if client is None:
            api.close()
