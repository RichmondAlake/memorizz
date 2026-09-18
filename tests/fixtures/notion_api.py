"""Deterministic Notion REST contract fixture; no account or network required."""

import copy
import json
import uuid
from datetime import datetime
from urllib.parse import unquote, urlparse

from memorizz.memory_provider.notion.provider import PROPERTIES


class Response:
    def __init__(self, status, payload, headers=None):
        self.status_code, self.payload = status, payload
        self.headers = headers or {}

    def json(self):
        return copy.deepcopy(self.payload)


class NotionAPI:
    def __init__(self):
        self.data_source_id = str(uuid.uuid4())
        self.database_id = str(uuid.uuid4())
        self.parent_id = str(uuid.uuid4())
        self.properties = {
            name: {"id": "p:" + key, "type": kind, kind: {}}
            for key, (name, kind) in PROPERTIES.items()
        }
        self.pages, self.calls, self.views = {}, [], []
        self.failures = []
        self.hidden = set()
        self.page_size = 100
        self.closed = False

    def schema(self):
        return {
            "object": "data_source",
            "id": self.data_source_id,
            "parent": {"type": "database_id", "database_id": self.database_id},
            "properties": copy.deepcopy(self.properties),
        }

    @staticmethod
    def normalize_dates(properties):
        # Live Notion date properties discard seconds/subseconds on writes.
        # Rich-text timestamps retain their original precision.
        for prop in properties.values():
            value = prop.get("date")
            if prop.get("type") == "date" and value and value.get("start"):
                parsed = datetime.fromisoformat(value["start"].replace("Z", "+00:00"))
                value["start"] = parsed.replace(second=0, microsecond=0).isoformat(
                    timespec="milliseconds"
                )

    def value(self, page, field):
        prop = next(
            (prop for prop in page["properties"].values() if prop["id"] == field), {}
        )
        kind = prop.get("type")
        if kind == "select":
            return (prop.get(kind) or {}).get("name", "")
        if kind == "date":
            return (prop.get(kind) or {}).get("start")
        return "".join(
            item.get("plain_text", item.get("text", {}).get("content", ""))
            for item in prop.get(kind, [])
        )

    def matches(self, page, query):
        if "and" in query:
            return all(self.matches(page, item) for item in query["and"])
        if "or" in query:
            return any(self.matches(page, item) for item in query["or"])
        field = query.get("property")
        if field is None:
            return True
        operation = next(value for key, value in query.items() if key != "property")
        actual = self.value(page, field)
        if "equals" in operation:
            return actual == operation["equals"]
        if "is_empty" in operation:
            return not actual
        if "on_or_after" in operation:
            return bool(actual) and actual >= operation["on_or_after"]
        if "on_or_before" in operation:
            return bool(actual) and actual <= operation["on_or_before"]
        raise AssertionError(f"Unsupported fixture predicate {query}")

    def edit(self, page_id, key, text):
        from memorizz.memory_provider.notion.provider import rich_text

        kind = PROPERTIES[key][1]
        self.pages[page_id]["properties"]["p:" + key] = {
            "id": "p:" + key,
            "type": kind,
            kind: {"name": text} if kind == "select" else rich_text(text),
        }

    def request(self, method, url, *, data=None, params=None, headers=None, **kwargs):
        body = json.loads(data) if data else None
        path = urlparse(url).path.removeprefix("/v1")
        self.calls.append((method, path, copy.deepcopy(body), copy.deepcopy(params)))
        assert headers["Notion-Version"] == "2026-03-11"
        assert kwargs.get("allow_redirects") is False
        if self.failures:
            failure = self.failures.pop(0)
            if callable(failure):
                failure = failure(method, path, body)
            if isinstance(failure, Exception):
                raise failure
            if failure is not None:
                return failure
        return self.handle(method, path, body, params or {})

    def handle(self, method, path, body, params):
        if path == "/data_sources/" + self.data_source_id and method == "GET":
            return Response(200, self.schema())
        if path == "/data_sources/" + self.data_source_id + "/query":
            assert method == "POST"
            rows = [
                copy.deepcopy(page)
                for page in self.pages.values()
                if page.get("parent", {}).get("data_source_id") == self.data_source_id
                and page["id"] not in self.hidden
                and not page.get("in_trash")
                and self.matches(page, body["filter"])
            ]
            for sort in reversed(body.get("sorts", [])):
                if "property" in sort:
                    rows.sort(
                        key=lambda page: self.value(page, sort["property"]) or "",
                        reverse=sort["direction"] == "descending",
                    )
            offset = int(body.get("start_cursor", 0))
            size = min(body.get("page_size", 100), self.page_size)
            more = offset + size < len(rows)
            return Response(
                200,
                {
                    "object": "list",
                    "results": rows[offset : offset + size],
                    "has_more": more,
                    "next_cursor": str(offset + size) if more else None,
                },
            )
        if path == "/databases" and method == "POST":
            assert (
                body["parent"]["page_id"] == self.parent_id
                or body["parent"]["page_id"] in self.pages
            )
            initial = body["initial_data_source"]
            self.properties = {
                name: {
                    "id": "p:"
                    + next(
                        key for key, (label, _) in PROPERTIES.items() if label == name
                    ),
                    "type": next(iter(value)),
                    **value,
                }
                for name, value in initial["properties"].items()
            }
            return Response(
                200,
                {
                    "object": "database",
                    "id": self.database_id,
                    "data_sources": [{"id": self.data_source_id}],
                },
            )
        if path == "/views" and method == "POST":
            view = {"id": str(uuid.uuid4()), **copy.deepcopy(body)}
            self.views.append(view)
            return Response(200, view)
        if path == "/pages" and method == "POST":
            identifier = str(uuid.uuid4())
            properties = copy.deepcopy(body.get("properties") or {})
            for key, prop in properties.items():
                prop["id"], prop["type"] = key, next(iter(prop))
            self.normalize_dates(properties)
            page = {
                "object": "page",
                "id": identifier,
                "parent": copy.deepcopy(body["parent"]),
                "properties": properties,
                "in_trash": False,
                "url": "https://www.notion.so/" + identifier.replace("-", ""),
            }
            self.pages[identifier] = page
            return Response(200, page)
        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "pages":
            identifier = parts[1]
            if identifier not in self.pages or identifier in self.hidden:
                return Response(404, {"code": "object_not_found"})
            page = self.pages[identifier]
            if len(parts) == 4 and parts[2] == "properties":
                prop = page["properties"][unquote(parts[3])]
                kind = prop["type"]
                offset = int(params.get("start_cursor", 0))
                values = prop[kind]
                size = min(100, self.page_size)
                more = offset + size < len(values)
                return Response(
                    200,
                    {
                        "object": "list",
                        "results": [
                            {kind: value} for value in values[offset : offset + size]
                        ],
                        "has_more": more,
                        "next_cursor": str(offset + size) if more else None,
                    },
                )
            if method == "PATCH":
                for key, value in body.get("properties", {}).items():
                    page["properties"][key] = {
                        "id": key,
                        "type": next(iter(value)),
                        **copy.deepcopy(value),
                    }
                if "in_trash" in body:
                    page["in_trash"] = body["in_trash"]
                self.normalize_dates(page["properties"])
            return Response(200, page)
        raise AssertionError(f"Unexpected Notion request: {method} {path}")

    def close(self):
        self.closed = True
