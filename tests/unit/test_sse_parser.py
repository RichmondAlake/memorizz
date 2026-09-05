import shutil
import subprocess
from pathlib import Path

import pytest


def test_browser_sse_parser_fragmented_unicode_and_cleanup():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for browser parser tests")
    result = subprocess.run(
        [node, "tests/browser/sse_parser.cjs"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_npm_launcher_forwards_streaming_flags_and_stdio():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for launcher smoke tests")
    result = subprocess.run(
        [node, "tests/browser/npm_streaming.cjs"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
