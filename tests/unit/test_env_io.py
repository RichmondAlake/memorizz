"""Environment persistence must survive quoting, failed writes and races."""

import io
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from dotenv import dotenv_values

from memorizz import _env_io as env


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MEMORIZZ_ENV_FILE", raising=False)
    monkeypatch.setattr(env, "_ENV_SOURCES", {})
    return tmp_path


@pytest.mark.parametrize(
    "value",
    [
        "",
        "simple",
        "a b",
        "a#b",
        'a"b',
        "a'b",
        "C:\\a\\b",
        'slash\\"quote',
        "a\nb\r\tc",
        "😎 café",
        "back\bform\fvertical\valarm\a",
        "$ordinary",
        "=assignment",
        "'\"\\end",
    ],
)
def test_literal_values_round_trip(value):
    assert (
        dotenv_values(stream=io.StringIO("VALUE=" + env.format_env_value(value)))[
            "VALUE"
        ]
        == value
    )


def test_update_preserves_multiline_comments_and_collapses_target_duplicates(
    isolated_env,
):
    path = isolated_env / "settings.env"
    path.write_text(
        '# keep\nMULTI="first\nsecond"\n\nexport TOKEN="old\nvalue"\nOTHER=keep\nTOKEN=duplicate\n'
    )
    env.update_env_file(path, {"TOKEN": 'quote"\\#secret', "ADDED": "new"})
    result = path.read_text()
    assert '# keep\nMULTI="first\nsecond"\n' in result
    assert "OTHER=keep\n" in result
    assert result.count("TOKEN=") == 1
    assert dotenv_values(path) == {
        "MULTI": "first\nsecond",
        "TOKEN": 'quote"\\#secret',
        "OTHER": "keep",
        "ADDED": "new",
    }


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_private_permissions_for_new_and_existing_files(isolated_env):
    path = isolated_env / "new" / ".env"
    env.update_env_file(path, {"TOKEN": "private"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.with_name(".env.lock").stat().st_mode) == 0o600
    path.chmod(0o644)
    env.update_env_file(path, {"TOKEN": "updated"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "updates",
    [
        {"KEY=secret": "x"},
        {"1BAD": "x"},
        {"BAD\nINJECTED": "x"},
        {"KEY": "a\0secret"},
        {"KEY": "${DO_NOT_EXPAND}"},
    ],
)
def test_invalid_input_is_not_echoed_and_does_not_mutate(isolated_env, updates):
    path = isolated_env / ".env"
    with pytest.raises(ValueError) as caught:
        env.update_env_file(path, updates)
    assert not path.exists()
    assert "secret" not in str(caught.value)
    assert "DO_NOT_EXPAND" not in str(caught.value)


def test_malformed_existing_file_is_left_intact(isolated_env):
    path = isolated_env / ".env"
    original = 'SECRET="unfinished\n'
    path.write_text(original)
    with pytest.raises(ValueError, match="invalid syntax"):
        env.update_env_file(path, {"OTHER": "new"})
    assert path.read_text() == original


def test_replace_failure_keeps_original_and_cleans_temporary(isolated_env, monkeypatch):
    path = isolated_env / ".env"
    path.write_text("OLD=retained\n")

    def fail(*args):
        raise PermissionError("private-secret")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(PermissionError):
        env.update_env_file(path, {"NEW": "value"})
    assert path.read_text() == "OLD=retained\n"
    assert not list(isolated_env.glob("*.tmp"))


def test_symlink_is_rejected(isolated_env):
    target = isolated_env / "actual"
    target.write_text("KEEP=yes\n")
    link = isolated_env / ".env"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        env.update_env_file(link, {"OTHER": "value"})
    assert target.read_text() == "KEEP=yes\n"


def test_concurrent_threads_keep_all_updates(isolated_env):
    path = isolated_env / ".env"
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: env.update_env_file(path, {f"KEY_{index}": str(index)}),
                range(24),
            )
        )
    assert dotenv_values(path) == {f"KEY_{index}": str(index) for index in range(24)}


def test_concurrent_processes_keep_all_updates(isolated_env):
    path = isolated_env / ".env"
    code = "from pathlib import Path; import sys; from memorizz._env_io import update_env_file; update_env_file(Path(sys.argv[1]), {sys.argv[2]: 'saved'})"
    # Run from a clean cwd, but retain the installed/source package import path.
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(path), f"KEY_{index}"],
            env=dict(os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(6)
    ]
    for child in children:
        _, error = child.communicate(timeout=30)
        assert child.returncode == 0, error.decode()
    assert len(dotenv_values(path)) == 6


def test_layering_sources_and_override_warnings_are_secret_free(
    isolated_env, monkeypatch
):
    project = isolated_env / ".env"
    env.update_env_file(
        project, {"TEST_LAYER": "project-secret", "TEST_PROJECT": "project-secret"}
    )
    env.update_env_file(
        env.resolve_env_file(),
        {"TEST_LAYER": "home-secret", "TEST_HOME": "home-secret"},
    )
    monkeypatch.setenv("TEST_LAYER", "export-secret")
    monkeypatch.delenv("TEST_PROJECT", raising=False)
    monkeypatch.delenv("TEST_HOME", raising=False)
    env.load_layered_env()
    assert os.environ["TEST_LAYER"] == "export-secret"
    assert env.environment_source("TEST_LAYER")["kind"] == "process environment"
    assert env.environment_source("TEST_PROJECT")["path"] == str(project)
    assert env.environment_source("TEST_HOME")["path"] == str(env.resolve_env_file())
    warnings = env.env_override_warnings(
        {"TEST_LAYER": "new-secret", "TEST_PROJECT": "new-secret"}
    )
    assert len(warnings) == 3
    assert "secret" not in " ".join(warnings)
    assert "project .env" in " ".join(warnings)


def test_apply_validation_failure_does_not_change_session(isolated_env, monkeypatch):
    monkeypatch.setenv("TEST_VALUE", "original")
    error = env.apply_env_updates({"TEST_VALUE": "bad\0private"})
    assert error and "private" not in error
    assert os.environ["TEST_VALUE"] == "original"


def test_apply_write_failure_keeps_legacy_session_behavior_without_secret_error(
    isolated_env, monkeypatch
):
    monkeypatch.delenv("TEST_VALUE", raising=False)

    def fail(*args):
        raise OSError("private-password")

    monkeypatch.setattr(env, "update_env_file", fail)
    error = env.apply_env_updates({"TEST_VALUE": "private-password"})
    assert "private-password" not in error
    assert os.environ["TEST_VALUE"] == "private-password"
    assert env.environment_source("TEST_VALUE")["kind"] == "session"
