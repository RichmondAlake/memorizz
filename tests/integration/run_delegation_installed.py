"""Run delegation contracts against an installed wheel, without a source fallback.

Requires uv and test dependencies in the invoking Python environment. Installs
only the supplied local wheel into a disposable target; no dependency downloads,
network services, host databases or system package changes.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(wheel):
    wheel = Path(wheel).resolve(strict=True)
    root = Path(__file__).resolve().parents[2]
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required for the isolated wheel installation")
    with tempfile.TemporaryDirectory(
        prefix="memorizz-delegation-installed-"
    ) as temporary:
        directory = Path(temporary)
        target = directory / "installed"
        env = {
            **os.environ,
            "UV_CACHE_DIR": str(directory / "uv-cache"),
            "PYTHONPATH": os.pathsep.join([str(target), str(directory)]),
            "MEMORIZZ_HOME": str(directory / "memorizz-home"),
            "MEMORIZZ_DELEGATION_LIVE": "0",
        }
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--offline",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
            env=env,
            cwd=directory,
        )
        tests = directory / "tests"
        (tests / "integration").mkdir(parents=True)
        shutil.copytree(
            root / "tests/mocks",
            tests / "mocks",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        for relative in [
            "__init__.py",
            "conftest.py",
            "integration/__init__.py",
            "integration/test_delegation_contract.py",
            "integration/test_delegation_e2e.py",
            "integration/test_shared_memory_atomic.py",
        ]:
            shutil.copy2(root / "tests" / relative, tests / relative)
        code = """
import pathlib, sys
import memorizz
import pytest
installed = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(memorizz.__file__).resolve().is_relative_to(installed), memorizz.__file__
print('Verified wheel import:', memorizz.__file__, flush=True)
raise SystemExit(pytest.main(['tests/integration', '-q', '--disable-warnings']))
"""
        return subprocess.run(
            [sys.executable, "-c", code, str(target)],
            cwd=directory,
            env=env,
            timeout=120,
            check=False,
        ).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    raise SystemExit(run(parser.parse_args().wheel))
