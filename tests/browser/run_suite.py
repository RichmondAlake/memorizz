"""Run synthetic UI acceptance tests with owned, temporary loopback servers.

Requires Node 18+ and Playwright's Chromium. No production data or paid APIs.
The same runner is used locally and by the reusable browser release gate.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    (
        "observability",
        "MEMORIZZ_BROWSER_TEST_PORT",
        "memorizz-browser-fixture-token",
        "/traces",
    ),
    (
        "usage",
        "MEMORIZZ_BROWSER_TEST_PORT",
        "memorizz-browser-fixture-token",
        "/traces/usage",
    ),
    (
        "persona",
        "MEMORIZZ_PERSONA_TEST_PORT",
        "persona-browser-fixture-token",
        "/persona-evolution",
    ),
)


def _port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _ready(process, url, token):
    deadline = time.monotonic() + 30
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Synthetic fixture exited before it became ready")
        try:
            with urllib.request.urlopen(request, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.05)
    raise TimeoutError("Synthetic fixture did not become ready within 30 seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--node", default=shutil.which("node"))
    args = parser.parse_args()
    if not args.node:
        parser.error("Node is required; specify --node /path/to/node")
    output = args.output_dir or Path(
        tempfile.mkdtemp(prefix="memorizz-browser-results-")
    )
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    for name, port_variable, token, route in SUITES:
        port = _port()
        base = f"http://127.0.0.1:{port}"
        env = {
            **os.environ,
            port_variable: str(port),
            "MEMORIZZ_NO_UPDATE_CHECK": "1",
            "PYTHONPATH": str(ROOT / "src"),
            "MEMORIZZ_BROWSER_TEST_URL": base + "/traces"
            if name == "observability"
            else base,
            "MEMORIZZ_PERSONA_TEST_URL": base,
            "MEMORIZZ_BROWSER_SCREENSHOT": str(output / "observability-mobile.png"),
            "MEMORIZZ_USAGE_SCREENSHOT": str(output / "usage-desktop.png"),
            "MEMORIZZ_USAGE_MOBILE_SCREENSHOT": str(output / "usage-mobile.png"),
            "PERSONA_SCREENSHOT_DIR": str(output),
        }
        with (output / f"{name}-server.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / f"tests/browser/{name}_app.py")],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                _ready(process, base + route, token)
                subprocess.run(
                    [args.node, str(ROOT / f"tests/browser/{name}.cjs")],
                    cwd=ROOT,
                    env=env,
                    check=True,
                    timeout=120,
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    print(f"All synthetic browser suites passed. Screenshots/logs: {output}")


if __name__ == "__main__":
    main()
