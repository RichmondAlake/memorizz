"""Which agents share a memory, shown from the store's own records."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memorizz.enums import MemoryType  # noqa: E402
from memorizz.memagent.models import MemAgentModel  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import agent_status, shared_memory_view, state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

pytestmark = pytest.mark.unit

TEAM = "fern-ember"
TYPES = ["conversation_memory", "entity_memory"]


def _turn(provider, agent_id, memory_id, role, content, second, thread="t"):
    provider.store(
        {
            "role": role,
            "content": content,
            "memory_id": memory_id,
            "thread_id": thread,
            "agent_id": agent_id,
            "user_id": None,
            "timestamp": f"2026-11-26T10:00:{second:02d}+00:00",
        },
        MemoryType.CONVERSATION_MEMORY,
    )


@pytest.fixture()
def shop(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    reader = {"self_identity_key": None, "subjects": "all", "writes": False}
    for agent_id, name, extra in (
        ("mara", "Mara", {}),
        ("cole", "Cole", {"entity_policy": reader}),
        ("sam", "Sam", {"entity_policy": reader, "memory_ids": [TEAM, "support-desk"]}),
        ("ona", "Ona", {"entity_policy": reader, "delegates": ["mara", "cole"]}),
        ("loner", "Loner", {"memory_ids": ["elsewhere"], "memory_types": []}),
    ):
        fields = {"memory_ids": [TEAM], "memory_types": TYPES, **extra}
        provider.store_memagent(MemAgentModel(agent_id=agent_id, name=name, **fields))
    _turn(provider, "mara", TEAM, "user", "Decision: the sale runs 3 to 9 March.", 1)
    _turn(provider, "mara", TEAM, "assistant", "Recorded.", 2)
    _turn(provider, "cole", TEAM, "assistant", "Our sale runs 3 to 9 March.", 3)
    _turn(provider, "sam", "support-desk", "user", "Priya, 12 Elm Row.", 4)
    provider.store(
        {
            "name": "Spring Sale",
            "entity_type": "offer",
            "memory_id": TEAM,
            "user_id": None,
            "attributes": [{"name": "end_date", "value": "9 March 2027"}],
        },
        MemoryType.ENTITY_MEMORY,
    )
    board = {
        "root_agent_id": "ona",
        "status": "completed",
        "outcome": "succeeded",
        "created_at": "2026-11-26T10:01:00+00:00",
        "blackboard": [
            {
                "entry_type": "workflow_start",
                "content": {"original_query": "Launch pack"},
            },
            {
                "entry_type": "task_decomposition",
                "content": {
                    "sub_tasks": [
                        {
                            "task_id": "t1",
                            "description": "Confirm",
                            "assigned_agent_id": "mara",
                        },
                        {
                            "task_id": "t2",
                            "description": "Write",
                            "assigned_agent_id": "cole",
                        },
                    ]
                },
            },
            {
                "entry_type": "task_completion",
                "content": {
                    "task_id": "t1",
                    "status": "completed",
                    "result": "3 to 9 March",
                    "duration_ms": 2400,
                },
            },
        ],
    }
    provider.store(
        {
            "content": json.dumps(board),
            "memory_id": "session-1",
            "owner_agent_id": "ona",
        },
        MemoryType.SHARED_MEMORY,
    )
    agent_status.reset()
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "read_only": False,
    }
    with patch.dict(state._state, values):
        yield provider
    agent_status.reset()
    provider.close()


def test_a_memory_is_shared_when_more_than_one_agent_runs_in_it(shop):
    memories = shared_memory_view.overview(shop, shop.list_memagents())

    by_id = {memory["memory_id"]: memory for memory in memories}
    assert memories[0]["memory_id"] == TEAM  # most shared first
    team = by_id[TEAM]
    assert team["shared"] is True and team["used_by"] == 4
    # The agent that may change the records comes first.
    assert [(a["name"], a["access"]) for a in team["agents"]] == [
        ("Mara", "writes"),
        ("Cole", "reads"),
        ("Ona", "reads"),
        ("Sam", "reads"),
    ]
    assert (team["records"], team["turns"]) == (1, 3)
    # A memory one agent runs in is private, whoever else shares the store.
    assert by_id["support-desk"]["shared"] is False
    assert [a["name"] for a in by_id["support-desk"]["agents"]] == ["Sam"]
    assert by_id["elsewhere"]["agents"][0]["access"] == "none"

    # A harness that ran in the memory counts as sharing it too.
    runs = [{"harness": "codex", "task": {"memory_id": "elsewhere"}}]
    shared = shared_memory_view.overview(shop, shop.list_memagents(), runs)
    elsewhere = next(m for m in shared if m["memory_id"] == "elsewhere")
    assert elsewhere["shared"] is True and elsewhere["harnesses"] == {"codex": 1}


def test_one_memory_shows_its_writers_readers_records_and_tasks(shop):
    memory = shared_memory_view.detail(shop, shop.list_memagents(), TEAM)

    assert [a["name"] for a in memory["writers"]] == ["Mara"]
    assert [a["name"] for a in memory["readers"]] == ["Cole", "Ona", "Sam"]
    ona = next(a for a in memory["agents"] if a["name"] == "Ona")
    assert ona["coordinates"] == ["Mara", "Cole"]
    assert memory["records"][0]["name"] == "Spring Sale"
    assert memory["records"][0]["attributes"] == [
        {"name": "end_date", "value": "9 March 2027"}
    ]
    (board,) = memory["boards"]
    assert (board["coordinator"], board["request"], board["outcome"]) == (
        "Ona",
        "Launch pack",
        "succeeded",
    )
    assert [(t["agent"], t["status"], t["seconds"]) for t in board["tasks"]] == [
        ("Mara", "completed", 2.4),
        ("Cole", "planned", None),
    ]
    # Newest first, and the customer's turn in Sam's own memory is not here.
    assert [turn["agent"] for turn in memory["turns"]] == ["Cole", "Mara", "Mara"]
    assert "Priya" not in json.dumps(memory)
    assert shared_memory_view.detail(shop, shop.list_memagents(), "nobody") is None


def test_the_pages_list_the_memories_and_open_one(shop):
    client = TestClient(create_app(), follow_redirects=False)

    page = client.get("/memory/shared")
    assert page.status_code == 200
    assert 'href="/memory/shared/fern-ember"' in page.text
    assert "Shared by 4" in page.text and "Private" in page.text
    assert 'href="/memory/shared-records"' in page.text

    detail = client.get("/memory/shared/fern-ember")
    assert detail.status_code == 200
    for needed in (
        "Write its records",
        "Read them",
        "Spring Sale",
        "Coordinates Mara, Cole",
        "Launch pack",
    ):
        assert needed in detail.text, needed
    assert client.get("/memory/shared/nobody").status_code == 404
    # The stored records of the type keep their explorer, at its own address.
    assert client.get("/memory/shared-records").status_code == 200


def test_an_agents_sidebar_line_is_its_own_in_a_shared_memory(shop):
    client = TestClient(create_app(), follow_redirects=False)

    agents = client.get(
        "/api/nav/agent-status", params={"ids": "mara,cole,ona"}
    ).json()["agents"]

    assert agents["mara"]["text"] == "Recorded."
    assert agents["cole"]["text"] == "Our sale runs 3 to 9 March."
    # Ona has said nothing: not the newest thing someone else said.
    assert agents["ona"]["text"] == ""


def test_an_answer_a_harness_saved_is_shown_as_written_by_that_harness(shop):
    saved = "[Harness run: codex · succeeded]\nImplemented sale_price.\nHarness run: 0f915a88-1c2d"
    _turn(shop, "metaharness", TEAM, "assistant", saved, 9, thread="harness-run-1")
    runs = [{"harness": "codex", "task": {"memory_id": TEAM}}]

    memory = shared_memory_view.detail(shop, shop.list_memagents(), TEAM, runs)

    newest = memory["turns"][0]
    assert (newest["agent"], newest["harness"], newest["text"]) == (
        "codex",
        True,
        "Implemented sale_price.",
    )
    assert memory["harnesses"] == {"codex": 1}
    assert {"agent": "codex", "harness": True, "turns": 1}.items() <= next(
        row for row in memory["authors"] if row["agent"] == "codex"
    ).items()


def test_a_connected_app_that_saved_here_shares_the_memory(shop):
    """A record saved through the MCP server names the app that saved it."""
    shop.store(
        {
            "content": "200 extra pots arrive on 1 March",
            "memory_id": TEAM,
            "user_id": None,
            "created_via": "Muse",
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    shop.store(
        {
            "content": "A note in a memory no agent runs in",
            "memory_id": "phone",
            "user_id": None,
            "created_via": "Grok Bot",
        },
        MemoryType.KNOWLEDGE_BASE,
    )

    memories = {
        m["memory_id"]: m
        for m in shared_memory_view.overview(shop, shop.list_memagents())
    }
    detail = shared_memory_view.detail(shop, shop.list_memagents(), TEAM)

    assert memories[TEAM]["apps"] == {"Muse": 1} and memories[TEAM]["used_by"] == 5
    assert memories["phone"]["apps"] == {"Grok Bot": 1}
    assert memories["phone"]["shared"] is False
    assert detail["apps"] == {"Muse": 1}
    page = TestClient(create_app(), follow_redirects=False).get(
        "/memory/shared/fern-ember"
    )
    assert "1 connected app" in page.text and "Muse · 1" in page.text


def test_an_agents_latest_thread_preview_is_its_own_in_a_shared_memory(shop):
    from memorizz.ui.routers.agents_crud import _build_agent_recent_messages

    saved = "[Harness run: codex · succeeded]\nImplemented.\nHarness run: 0f915a88-1c2d"
    _turn(shop, "metaharness", TEAM, "assistant", saved, 9, thread="harness-run-1")
    agents = [a for a in shop.list_memagents() if a.agent_id in ("mara", "ona")]

    previews, memories = _build_agent_recent_messages(agents)

    assert [m["content"] for m in previews["mara"]] == [
        "Decision: the sale runs 3 to 9 March.",
        "Recorded.",
    ]
    # Ona has said nothing here: not Cole's line, and not a harness's answer.
    assert "ona" not in previews and memories["ona"] == TEAM
