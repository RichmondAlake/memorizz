"""Run the live MongoDB gate with an explicitly supplied standalone binary.

No downloads, service discovery, Docker, system installation or shared storage.
Only the child process started here is stopped. All data is synthetic/temporary.
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


def verify_process(client, process, directory):
    """Refuse a port collision before the test can create/drop any database."""
    status = client.admin.command("serverStatus")
    options = client.admin.command("getCmdLineOpts")
    actual_path = options.get("parsed", {}).get("storage", {}).get("dbPath", "")
    if status.get("pid") != process.pid or Path(actual_path).resolve() != directory:
        raise RuntimeError("Refusing a MongoDB instance not owned by this launcher")
    return status["version"]


def run(binary):
    binary = Path(binary).expanduser().resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("--mongod must name an executable standalone mongod binary")
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="memorizz-obs-live-") as temporary:
        directory = Path(temporary).resolve()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        process = subprocess.Popen(
            [
                str(binary),
                "--bind_ip",
                "127.0.0.1",
                "--port",
                str(port),
                "--dbpath",
                str(directory),
                "--logpath",
                str(directory / "mongod.log"),
                "--nounixsocket",
                "--wiredTigerCacheSizeGB",
                "0.25",
                "--setParameter",
                "diagnosticDataCollectionEnabled=false",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        uri = f"mongodb://127.0.0.1:{port}/?directConnection=true"
        client = None
        try:
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=500,
                connectTimeoutMS=500,
                socketTimeoutMS=2000,
            )
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Standalone mongod exited with {process.returncode}"
                    )
                try:
                    version = verify_process(client, process, directory)
                    break
                except ConnectionFailure:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Standalone MongoDB did not become ready in 30s"
                        ) from None
                    time.sleep(0.1)
            print(
                f"Verified isolated MongoDB {version}; running synthetic live gates",
                flush=True,
            )
            env = {
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "MEMORIZZ_OBSERVABILITY_LIVE": "1",
                "MEMORIZZ_OBS_TEST_MONGODB_URI": uri,
            }
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/integration/test_observability_live.py",
                    "-k",
                    "mongodb",
                    "-q",
                ],
                cwd=root,
                env=env,
                timeout=300,
                check=False,
            ).returncode
        finally:
            if client is not None:
                client.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            print(
                "Standalone MongoDB stopped; temporary synthetic data removed on exit",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mongod", required=True, help="Path to a trusted mongod executable"
    )
    raise SystemExit(run(parser.parse_args().mongod))
