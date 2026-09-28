# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The memory explorer shows the newest records, readable and without vectors."""

from pathlib import Path
from unittest.mock import patch

import pytest

from memorizz.ui.memory_view import build_memory_view, shape_record


@pytest.mark.unit
def test_records_sort_newest_first_and_undated_last():
    view = build_memory_view(
        [
            {"_id": "old", "content": "a", "timestamp": "2026-09-20T10:00:00"},
            {"_id": "undated", "content": "b"},
            {"_id": "new", "content": "c", "timestamp": "2026-09-28T09:00:00"},
        ]
    )
    assert [row["id"] for row in view["records"]] == ["new", "old", "undated"]
    assert view["newest"] == "2026-09-28 09:00:00"
    assert view["oldest"] == "2026-09-20 10:00:00"


@pytest.mark.unit
def test_embeddings_are_never_shown_and_previews_are_plain():
    row = shape_record(
        {
            "_id": "r1",
            "role": "assistant",
            "content": "Order **A-1001** has `shipped`.\n\nArrives Thursday.",
            "embedding": [0.1] * 1536,
            "query_vector": [0.2] * 64,
            "memory_id": "m1",
            "metadata": {"source": "tool"},
        }
    )
    keys = {field["key"] for field in row["fields"]}
    assert "embedding" not in keys and "query_vector" not in keys
    assert "metadata" in keys and "content" not in keys, "body is shown once"
    assert row["preview"] == "Order A-1001 has shipped. Arrives Thursday."
    assert row["body"].startswith("Order **A-1001**"), "the pane keeps the original"


@pytest.mark.unit
def test_page_is_limited_but_summary_counts_everything():
    items = [
        {
            "_id": str(i),
            "content": "x" * 10,
            "role": "user" if i % 2 else "assistant",
            "memory_id": f"m{i % 3}",
            "timestamp": f"2026-09-{10 + i % 15:02d}T12:00:00",
        }
        for i in range(150)
    ]
    view = build_memory_view(items, limit=100)
    assert len(view["records"]) == 100 and view["total"] == 150 and view["has_more"]
    assert dict(view["roles"]) == {"user": 75, "assistant": 75}
    assert view["memory_ids"] == ["m0", "m1", "m2"]
    assert len(view["series"]) == 14 and sum(view["series"]) > 0
    assert view["mean_chars"] == 10


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.enums.memory_type import MemoryType  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402


@pytest.mark.unit
def test_memory_page_renders_the_explorer(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "ui", lazy_vector_indexes=True)
    )
    records = [
        {
            "_id": "turn-1",
            "role": "user",
            "content": "Remember my **home airport** is SFO.",
            "memory_id": "mem-a",
            "timestamp": "2026-09-28T09:00:00",
            "embedding": [0.5] * 32,
        },
        {
            "_id": "turn-2",
            "role": "assistant",
            "content": "Noted.",
            "memory_id": "mem-b",
            "timestamp": "2026-09-28T09:00:05",
        },
    ]
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    ), patch.object(provider, "list_all", return_value=records) as list_all:
        response = TestClient(create_app()).get("/memory/conversations")
    assert list_all.call_args.args[0] == MemoryType.CONVERSATION_MEMORY
    assert response.status_code == 200
    html = response.text
    assert "Remember my home airport is SFO." in html, "preview is plain text"
    assert "0.5, 0.5" not in html, "embedding vectors never render"
    assert html.index("turn-2") < html.index("turn-1"), "newest first"
    assert 'aria-label="Memory types"' in html
    assert 'data-monitor-tag="assistant"' in html
