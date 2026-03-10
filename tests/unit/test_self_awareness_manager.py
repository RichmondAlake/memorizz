"""Unit tests for the self-awareness manager."""

import subprocess

import pytest

from memorizz.memagent.managers import SelfAwarenessManager


@pytest.mark.unit
def test_root_normalization_and_confinement(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    inside_file = nested / "inside.txt"
    inside_file.write_text("inside", encoding="utf-8")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside", encoding="utf-8")

    manager = SelfAwarenessManager(
        config={"root_paths": [str(root / "nested" / "..")]},
        cwd=str(root),
    )

    cfg = manager.get_config()
    assert cfg["root_paths"] == [str(root.resolve())]

    read_result = manager.read_file("nested/inside.txt")
    assert read_result["content"] == "inside"

    with pytest.raises(ValueError):
        manager.read_file(str(outside_file))


@pytest.mark.unit
def test_run_command_blocks_forbidden_syntax(tmp_path):
    manager = SelfAwarenessManager(
        config={"root_paths": [str(tmp_path)]}, cwd=str(tmp_path)
    )

    with pytest.raises(ValueError):
        manager.run_command("ls; pwd")

    with pytest.raises(PermissionError):
        manager.run_command("python -V")


@pytest.mark.unit
def test_read_command_success_in_allowed_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    data_file = root / "data.txt"
    data_file.write_text("hello world", encoding="utf-8")

    manager = SelfAwarenessManager(config={"root_paths": [str(root)]}, cwd=str(root))
    result = manager.run_command("cat data.txt", cwd=".")

    assert result["exit_code"] == 0
    assert "hello world" in result["stdout"]


@pytest.mark.unit
def test_write_is_blocked_when_writes_disabled(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    manager = SelfAwarenessManager(config={"root_paths": [str(root)]}, cwd=str(root))

    with pytest.raises(PermissionError):
        manager.write_file("note.txt", "text", mode="overwrite")


@pytest.mark.unit
def test_delete_is_blocked_when_deletes_disabled(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "delete-me.txt"
    target.write_text("delete", encoding="utf-8")
    manager = SelfAwarenessManager(
        config={
            "root_paths": [str(root)],
            "allow_writes": True,
            "allow_deletes": False,
        },
        cwd=str(root),
    )

    with pytest.raises(PermissionError):
        manager.delete_path("delete-me.txt")


@pytest.mark.unit
def test_delete_guardrails_enforced(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    folder = root / "folder"
    folder.mkdir()
    child = folder / "child.txt"
    child.write_text("x", encoding="utf-8")

    manager = SelfAwarenessManager(
        config={
            "root_paths": [str(root)],
            "allow_writes": True,
            "allow_deletes": True,
        },
        cwd=str(root),
    )

    with pytest.raises(ValueError):
        manager.delete_path(str(root))

    with pytest.raises(ValueError):
        manager.delete_path("folder/*")

    with pytest.raises(ValueError):
        manager.delete_path("folder", recursive=True, force=False)

    with pytest.raises(ValueError):
        manager.run_command("rm -rf .")

    result = manager.delete_path("folder", recursive=True, force=True)
    assert result["deleted"] is True
    assert folder.exists() is False


@pytest.mark.unit
def test_timeout_and_output_limits_enforced(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    large = root / "large.txt"
    large.write_text("x" * 6000, encoding="utf-8")
    stream = root / "stream.txt"
    stream.write_text("line\n", encoding="utf-8")

    manager = SelfAwarenessManager(
        config={
            "root_paths": [str(root)],
            "max_output_chars": 20,
            "timeout_seconds": 1,
        },
        cwd=str(root),
    )

    truncated = manager.run_command("cat large.txt")
    assert truncated["output_truncated"] is True
    assert len(truncated["stdout"]) == manager.get_config()["max_output_chars"]

    with pytest.raises(subprocess.TimeoutExpired):
        manager.run_command("tail -f stream.txt")
