# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Helpers to manage the local Oracle Free Docker container for the UI.

The UI offers a one-click "Start container" / "Create container" recovery
when a connection attempt fails with connection-refused. This module wraps
the docker CLI so the endpoints in app.py stay thin.
"""

import logging
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Default name used when we create a fresh container on the user's behalf.
# Existing containers are discovered by image, so any pre-existing name works.
CONTAINER_NAME = "memorizz_oracle"
IMAGE = "gvenzl/oracle-free:23-slim-faststart"
# Substring match for discovering compatible pre-existing containers.
IMAGE_PREFIX = "gvenzl/oracle-free"
VOLUME_NAME = "memorizz_oracle_data"
INTERNAL_PORT = 1521
DEFAULT_SERVICE = "FREEPDB1"

# Readiness budgets. `start` is generous for a warm container; `create` covers
# the first-boot initialization of gvenzl/oracle-free which builds FREEPDB1.
READINESS_TIMEOUT_START = 120
READINESS_TIMEOUT_CREATE = 600
PULL_TIMEOUT = 600
# Time we'll wait after launching Docker.app / OrbStack.app for the daemon
# socket to start responding. Cold-start of Docker Desktop is ~30s typical.
RUNTIME_START_TIMEOUT = 90

# Apps we know how to launch via `open -a`. Order matters: prefer OrbStack
# when both are present (faster cold-start, lighter on Apple Silicon).
_RUNTIME_APPS = (
    ("OrbStack", "/Applications/OrbStack.app"),
    ("Docker", "/Applications/Docker.app"),
)
DOCKER_DESKTOP_DOWNLOAD_URL = "https://www.docker.com/products/docker-desktop/"


def _run(args: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """Run a docker subprocess and return (returncode, stdout, stderr).

    Uses an explicit argv list (no shell) to avoid any injection surface.
    Returns rc=-1 when docker itself is missing or the command times out.
    """
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return -1, "", "docker CLI not installed"
    except subprocess.TimeoutExpired:
        return -1, "", f"docker command timed out after {timeout}s"


def docker_available() -> bool:
    """True when the docker CLI is installed and the daemon responds."""
    rc, _, _ = _run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=5)
    return rc == 0


def get_container_state(name: str = CONTAINER_NAME) -> str:
    """Return one of 'absent', 'stopped', 'running' for the named container."""
    rc, stdout, _ = _run(
        ["docker", "inspect", "--format", "{{.State.Status}}", name],
        timeout=5,
    )
    if rc != 0:
        return "absent"
    status = stdout.strip().lower()
    if status == "running":
        return "running"
    return "stopped"


def list_oracle_containers() -> List[Dict[str, str]]:
    """Discover all containers built from a gvenzl/oracle-free image.

    Returns a list of dicts like
        {"name": "oracle-memorizz", "image": "gvenzl/oracle-free:23-slim",
         "state": "running" | "stopped", "ports": "0.0.0.0:1521->1521/tcp"}
    Sorted with the canonical `memorizz_oracle` first, then by name.
    """
    rc, stdout, _ = _run(
        [
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.Names}}\t{{.Image}}\t{{.State}}\t{{.Ports}}",
        ],
        timeout=10,
    )
    if rc != 0:
        return []
    results: List[Dict[str, str]] = []
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, image = parts[0], parts[1]
        state = parts[2].lower()
        ports = parts[3] if len(parts) > 3 else ""
        if IMAGE_PREFIX not in image:
            continue
        normalized_state = "running" if state == "running" else "stopped"
        results.append(
            {
                "name": name,
                "image": image,
                "state": normalized_state,
                "ports": ports,
            }
        )
    results.sort(key=lambda r: (r["name"] != CONTAINER_NAME, r["name"]))
    return results


def parse_port_from_dsn(dsn: str, default: int = INTERNAL_PORT) -> int:
    """Extract the TCP port from an Oracle DSN like host:port/service."""
    if not dsn:
        return default
    match = re.search(r":(\d+)(?:/|$)", dsn)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return default
    return default


def _tcp_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _wait_for_healthy(name: str, timeout: int) -> Tuple[bool, str]:
    """Poll the container's healthcheck until 'healthy' or timeout.

    gvenzl/oracle-free ships with a healthcheck that turns healthy only after
    the DB is accepting connections, so this is the most reliable signal.
    Falls back to a TCP probe on 127.0.0.1:<mapped_port> if no healthcheck is
    defined.
    """
    deadline = time.time() + timeout
    last_status = "unknown"
    while time.time() < deadline:
        rc, stdout, _ = _run(
            [
                "docker",
                "inspect",
                "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                name,
            ],
            timeout=5,
        )
        if rc == 0:
            last_status = stdout.strip() or "unknown"
            if last_status == "healthy":
                return True, "healthy"
            if last_status == "none":
                mapped = _get_mapped_port(name)
                if mapped and _tcp_open("127.0.0.1", mapped):
                    return True, "tcp-open"
        time.sleep(3)
    return False, f"last healthcheck status: {last_status}"


def _get_mapped_port(name: str) -> int:
    """Return the host port mapped to 1521 inside the named container, or 0."""
    rc, stdout, _ = _run(
        [
            "docker",
            "inspect",
            "--format",
            '{{(index (index .NetworkSettings.Ports "1521/tcp") 0).HostPort}}',
            name,
        ],
        timeout=5,
    )
    if rc != 0 or not stdout.strip():
        return 0
    try:
        return int(stdout.strip())
    except ValueError:
        return 0


def start_container(name: str = CONTAINER_NAME) -> Tuple[bool, str]:
    """`docker start` an existing stopped container, then wait for health."""
    rc, _, stderr = _run(["docker", "start", name], timeout=15)
    if rc != 0:
        return False, f"docker start failed: {stderr or 'unknown error'}"
    logger.info("Started container '%s', waiting for health...", name)
    ok, detail = _wait_for_healthy(name, READINESS_TIMEOUT_START)
    if not ok:
        return (
            False,
            f"Container '{name}' started but did not become healthy within "
            f"{READINESS_TIMEOUT_START}s ({detail})",
        )
    return True, f"Container '{name}' running and healthy"


def stream_container_logs(name: str, tail: int = 200):
    """Yield log lines from ``docker logs -f`` for the named container.

    Used by the UI's create-modal to show live Oracle startup output
    while ``create_container`` is still waiting for health. Blocks on
    each read; the caller is responsible for consuming in a generator
    context (e.g. FastAPI StreamingResponse) so that the subprocess is
    cleaned up when the client disconnects.

    If the container doesn't exist yet (common: user clicked Create
    seconds ago, ``docker run`` is still starting) we poll for up to
    ``_CONTAINER_WAIT_SEC`` seconds before giving up.
    """
    _CONTAINER_WAIT_SEC = 15
    deadline = time.time() + _CONTAINER_WAIT_SEC
    while get_container_state(name) == "absent":
        if time.time() >= deadline:
            yield f"[log-stream] container '{name}' did not appear within {_CONTAINER_WAIT_SEC}s"
            return
        time.sleep(1)

    try:
        proc = subprocess.Popen(
            ["docker", "logs", "-f", f"--tail={tail}", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge so errors flow to the same stream
            text=True,
            bufsize=1,  # line-buffered
        )
    except FileNotFoundError:
        yield "[log-stream] docker CLI not installed"
        return

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if line:
                yield line
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def create_container(username: str, password: str, host_port: int) -> Tuple[bool, str]:
    """`docker run` a new gvenzl/oracle-free container and wait for health.

    Uses the form-supplied username/password for the APP_USER, and reuses the
    password for SYS (ORACLE_PASSWORD) to keep the setup simple for local dev.
    Binds the requested host port to the container's 1521.
    """
    if get_container_state() != "absent":
        return False, f"Container '{CONTAINER_NAME}' already exists"

    logger.info("Pulling %s (may take a while on first run)...", IMAGE)
    rc, _, stderr = _run(["docker", "pull", IMAGE], timeout=PULL_TIMEOUT)
    if rc != 0:
        return False, f"docker pull failed: {stderr or 'unknown error'}"

    rc, _, stderr = _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{host_port}:{INTERNAL_PORT}",
            "-e",
            f"ORACLE_PASSWORD={password}",
            "-e",
            f"APP_USER={username}",
            "-e",
            f"APP_USER_PASSWORD={password}",
            "-v",
            f"{VOLUME_NAME}:/opt/oracle/oradata",
            IMAGE,
        ],
        timeout=60,
    )
    if rc != 0:
        return False, f"docker run failed: {stderr or 'unknown error'}"

    logger.info("Created Oracle container, waiting for first-boot health...")
    ok, detail = _wait_for_healthy(CONTAINER_NAME, READINESS_TIMEOUT_CREATE)
    if not ok:
        return (
            False,
            "Container created but did not become healthy within "
            f"{READINESS_TIMEOUT_CREATE}s ({detail}). "
            "First boot initializes the database and can take several minutes — "
            "you can retry the connection manually in a moment.",
        )
    return True, "Container created and healthy"


def _installed_runtime_app() -> Tuple[str, str] | None:
    """Return the (label, app_path) of the first runtime GUI app we find."""
    for label, path in _RUNTIME_APPS:
        if Path(path).is_dir():
            return label, path
    return None


def detect_runtime() -> Dict[str, object]:
    """Snapshot the host's container-runtime situation for the UI.

    Distinguishes the three states the recovery panel cares about:
      - daemon already running (UI proceeds straight to container flow)
      - GUI app installed but daemon down (UI offers one-click launch)
      - nothing installed (UI offers a download link, plus tells the front
        end whether Homebrew is available for a future brew-install path)
    """
    if docker_available():
        return {"daemon_running": True, "platform": sys.platform}

    installed = _installed_runtime_app()
    return {
        "daemon_running": False,
        "installed_app": installed[0] if installed else None,
        "installed_app_path": installed[1] if installed else None,
        "homebrew_available": shutil.which("brew") is not None,
        "download_url": DOCKER_DESKTOP_DOWNLOAD_URL,
        "platform": sys.platform,
    }


def start_runtime(timeout: int = RUNTIME_START_TIMEOUT) -> Tuple[bool, str]:
    """Launch an installed container runtime and poll for daemon readiness.

    Idempotent: if the daemon is already up we return immediately. We rely
    on macOS `open -a` (no admin prompt) and the runtime's own headless
    autostart machinery — no LaunchAgent munging here.
    """
    if docker_available():
        return True, "Docker daemon already running"

    if sys.platform != "darwin":
        return (
            False,
            "Auto-start is only supported on macOS — start your container "
            "runtime manually and retry.",
        )

    installed = _installed_runtime_app()
    if not installed:
        return (
            False,
            "No container runtime is installed. Install Docker Desktop or "
            "OrbStack and try again.",
        )

    label, _ = installed
    rc, _, stderr = _run(["open", "-a", label], timeout=10)
    if rc != 0:
        return False, f"Could not launch {label}: {stderr or 'unknown error'}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        if docker_available():
            return True, f"{label} started — Docker daemon is responding"
        time.sleep(2)

    return (
        False,
        f"{label} was launched but the Docker daemon did not respond within "
        f"{timeout}s. Open the app manually, finish any first-run setup, "
        "and retry.",
    )
