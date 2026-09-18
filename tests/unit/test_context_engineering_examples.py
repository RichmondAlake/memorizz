"""Test the actual self-contained notebook definitions without paid API calls."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from memorizz import FileSystemConfig, FileSystemProvider, MemoryType
from memorizz.embeddings import get_embedding_manager

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/context_engineering"
NOTEBOOKS = sorted(EXAMPLES.glob("persistent_notes_*.ipynb"))


def notebook_cells(path):
    return json.loads(path.read_text())["cells"]


def source(cell):
    return "".join(cell["source"])


def library_sources(path):
    return [
        source(cell)
        for cell in notebook_cells(path)
        if "memory-library" in cell.get("metadata", {}).get("tags", [])
    ]


@pytest.fixture
def lesson():
    namespace = {}
    for code in library_sources(EXAMPLES / "persistent_notes_filesystem.ipynb"):
        exec(compile(code, "notebook memory cell", "exec"), namespace)
    return namespace


def open_lesson(lesson, path, memory_id="task", user_id="alice"):
    # The suite's embedding stub isolates storage tests; notebook runs use OpenAI.
    embeddings = get_embedding_manager()
    provider = FileSystemProvider(
        FileSystemConfig(root_path=path, use_faiss=False, embedding_provider=embeddings)
    )
    return lesson["open_session"](provider, embeddings, memory_id, user_id)


def test_downloaded_notebooks_are_self_contained_and_have_short_cells():
    assert len(NOTEBOOKS) == 2
    assert library_sources(NOTEBOOKS[0]) == library_sources(NOTEBOOKS[1])
    for path in NOTEBOOKS:
        cells = notebook_cells(path)
        text = "\n".join(source(cell) for cell in cells)
        assert "memory_lab" not in text and "sys.path" not in text
        assert "src/memorizz" not in text and "../../" not in text
        assert text.count("```mermaid") == 2
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        assert source(code_cells[0]).startswith('%pip install -q "memorizz')
        for cell in code_cells:
            code = source(cell)
            assert len(code.splitlines()) <= 25
            if not code.startswith("%pip"):
                ast.parse(code)


def test_notes_and_sources_survive_rollover_and_provider_reopen(lesson, tmp_path):
    session = open_lesson(lesson, tmp_path)
    archive, note = lesson["archive_source"], lesson["write_note"]
    original = archive(session, "Keep order stable.", "user", {"id": "request"})
    first = note(session, "contract", "Keep order stable.", "requirement", [original])
    lesson["rollover"](session, "window-2")
    correction = archive(session, "At most four attempts.", "user", {"id": "steering"})
    second = note(
        session,
        "contract",
        "Stable order; four attempts.",
        "requirement",
        [original, correction],
        expected_revision=1,
    )
    lesson["rollover"](session, "window-3")
    session.provider.close()
    restored = open_lesson(lesson, tmp_path)
    assert restored.state["window"] == "window-3"
    assert lesson["read_notes"](restored) == [second]
    assert lesson["read_notes"](restored, True) == [first, second]
    assert second["supersedes"] == first["note_id"]
    assert lesson["read_source"](restored, original)["content"] == "Keep order stable."
    with pytest.raises(ValueError, match="revision conflict"):
        note(restored, "contract", "stale update", "requirement", [original])
    with pytest.raises(ValueError, match="bootstrap budget"):
        lesson["bootstrap"](restored, max_chars=10)


def test_foreign_sources_and_note_records_are_rejected(lesson, tmp_path):
    session = open_lesson(lesson, tmp_path)
    sid = lesson["archive_source"](session, "private evidence", "user", {"id": "one"})
    note = lesson["write_note"](
        session, "private", "private evidence", "requirement", [sid]
    )
    for task, user in [("other", "alice"), ("task", "bob")]:
        other = open_lesson(lesson, tmp_path, task, user)
        with pytest.raises(ValueError, match="Unknown source"):
            lesson["read_source"](other, sid)
        other.state["sources"].append(sid)
        with pytest.raises(ValueError, match="scope"):
            lesson["read_source"](other, sid)
        other.state["notes"]["private"] = [note["note_id"]]
        with pytest.raises(ValueError, match="scope"):
            lesson["read_notes"](other)


def test_archive_retry_slices_and_bounded_search(lesson, tmp_path):
    session = open_lesson(lesson, tmp_path)
    text = "setup\n" * 200 + "test_clock_skew expected=7000 actual=9000" + "\nend" * 300
    origin = {"id": "test-run"}
    sid = lesson["archive_source"](session, text, "tool", origin)
    index_ids = list(session.state["index_ids"])
    assert lesson["archive_source"](session, text, "tool", origin) == sid
    assert session.state["index_ids"] == index_ids
    with pytest.raises(ValueError, match="immutable"):
        lesson["archive_source"](session, "changed", "tool", origin)
    excerpt = lesson["read_source"](session, sid, 1190, 100)
    assert excerpt["content"] == text[1190:1290] and excerpt["has_more"]
    hits = lesson["search_history"](session, "test_clock_skew", limit=1, max_chars=1000)
    assert len(hits) == 1 and hits[0]["source_id"] == sid
    assert len(lesson["canonical"](hits)) <= 1000


def test_partial_index_write_can_be_retried_after_reopen(lesson, tmp_path, monkeypatch):
    session = open_lesson(lesson, tmp_path)
    store, writes = session.provider.store, 0

    def fail_second_chunk(data, memory_store_type=None, **kwargs):
        nonlocal writes
        if memory_store_type == MemoryType.KNOWLEDGE_BASE:
            writes += 1
            if writes == 2:
                raise RuntimeError("interrupted index write")
        return store(data, memory_store_type, **kwargs)

    monkeypatch.setattr(session.provider, "store", fail_second_chunk)
    text = "repeatable archive " * 150
    with pytest.raises(RuntimeError, match="interrupted"):
        lesson["archive_source"](session, text, "tool", {"id": "retry"})
    restored = open_lesson(lesson, tmp_path)
    sid = lesson["archive_source"](restored, text, "tool", {"id": "retry"})
    assert restored.state["sources"] == [sid]
    assert lesson["read_source"](restored, sid, max_chars=8000)["content"] == text
    assert len(restored.state["index_ids"]) == len(
        restored.provider.list_all(MemoryType.KNOWLEDGE_BASE)
    )


def test_cleanup_preserves_another_users_records(lesson, tmp_path):
    session = open_lesson(lesson, tmp_path)
    sid = lesson["archive_source"](session, "this task", "user", {"id": "one"})
    lesson["write_note"](session, "one", "this task", "requirement", [sid])
    other = open_lesson(lesson, tmp_path, "task", "bob")
    other_sid = lesson["archive_source"](other, "another user", "user", {"id": "two"})
    session = open_lesson(lesson, tmp_path)
    lesson["cleanup"](session)
    assert (
        session.provider.retrieve_by_id(session.state_id, MemoryType.SHARED_MEMORY)
        is None
    )
    other = open_lesson(lesson, tmp_path, "task", "bob")
    assert lesson["read_source"](other, other_sid)["content"] == "another user"
    assert other.provider.retrieve_by_id(sid, MemoryType.SHARED_MEMORY) is None
