# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Operational commands for Oracle setup, the local UI, and automations.

Relocated verbatim from the original ``memorizz/cli.py`` so existing behavior is
preserved. The only change is the package-relative script anchors: this module
now lives one directory deeper (``memorizz/cli/legacy.py`` vs ``memorizz/cli.py``),
so paths are resolved from ``_PKG_DIR`` (``src/memorizz``) to keep the candidate
locations byte-identical to before.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

# Anchor that points at the installed package dir (``src/memorizz``), matching the
# original ``Path(__file__).parent`` when this code lived in ``memorizz/cli.py``.
_PKG_DIR = Path(__file__).resolve().parent.parent


def install_oracle(image: Optional[str] = None):
    """Install Oracle database using install_oracle.sh script."""
    possible_paths = [
        Path("install_oracle.sh"),
        _PKG_DIR / "scripts" / "install_oracle.sh",
        _PKG_DIR.parent.parent / "install_oracle.sh",
        _PKG_DIR.parent.parent.parent / "install_oracle.sh",
    ]

    script_path = None
    for path in possible_paths:
        if path.exists() and path.is_file():
            script_path = path
            break

    if not script_path:
        print("✗ install_oracle.sh script not found")
        print("\nThe install_oracle.sh script is only available when:")
        print("  1. You've cloned the repository, or")
        print("  2. You're running from the repository directory")
        print("\nAlternative: Install Oracle manually with Docker:")
        print("  docker run -d --name oracle-memorizz -p 1521:1521 \\")
        print("    -e ORACLE_PWD=MyPassword123! \\")
        print("    container-registry.oracle.com/database/free:latest-lite")
        print("\nOr use the script directly if you have it:")
        print("  ./install_oracle.sh")
        return False

    os.chmod(script_path, 0o755)

    env = os.environ.copy()
    if image:
        image_map = {"lite": "1", "full": "2", "community": "3"}
        choice = image_map.get(image.lower(), image)
        if choice not in ("1", "2", "3"):
            print(f"✗ Invalid image: {image}")
            print("  Valid options: lite, full, community (or 1, 2, 3)")
            return False
        env["ORACLE_IMAGE_CHOICE"] = choice

    try:
        result = subprocess.run(
            ["bash", str(script_path.resolve())],
            check=False,
            capture_output=False,
            env=env,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Failed to execute install_oracle.sh: {e}")
        return False


def setup_oracle():
    """Run Oracle database setup."""
    try:
        from memorizz.memory_provider.oracle import setup_oracle_user

        return setup_oracle_user()
    except ImportError as e:
        print(f"✗ Failed to import setup module: {e}")
        print("\nPlease ensure memorizz[oracle] is installed:")
        print("  pip install memorizz[oracle]")
        return False


def setup_oracle_schema():
    """Apply Oracle schema updates in-place (safe, no user drop)."""
    try:
        from memorizz.memory_provider.oracle.setup import apply_schema_updates

        return apply_schema_updates()
    except ImportError as e:
        print(f"✗ Failed to import setup module: {e}")
        print("\nPlease ensure memorizz[oracle] is installed:")
        print("  pip install memorizz[oracle]")
        return False


def teardown_oracle(mode: Optional[str] = None, force: bool = False):
    """Teardown Oracle database installation."""
    possible_paths = [
        _PKG_DIR / "scripts" / "teardown_oracle.sh",
        Path("teardown_oracle.sh"),
        _PKG_DIR.parent.parent / "teardown_oracle.sh",
    ]

    script_path = None
    for path in possible_paths:
        if path.exists() and path.is_file():
            script_path = path
            break

    if not script_path:
        print("✗ teardown_oracle.sh script not found")
        print("\nThe teardown_oracle.sh script is only available when:")
        print("  1. You've cloned the repository, or")
        print("  2. You're running from the repository directory")
        print("\nAlternative: Teardown Oracle manually:")
        print("  # Stop and remove container:")
        print("  docker stop oracle-memorizz && docker rm oracle-memorizz")
        print("\n  # Remove volume (PERMANENT):")
        print("  docker volume rm oracle-memorizz-data")
        return False

    os.chmod(script_path, 0o755)

    cmd = ["bash", str(script_path.resolve())]

    if mode:
        mode_flag_map = {
            "drop-user": "--drop-user",
            "remove-container": "--remove-container",
            "full": "--full",
        }
        if mode in mode_flag_map:
            cmd.append(mode_flag_map[mode])
        else:
            print(f"✗ Invalid mode: {mode}")
            print("  Valid modes: drop-user, remove-container, full")
            return False

    if force:
        cmd.append("--force")

    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Failed to execute teardown_oracle.sh: {e}")
        return False


def run_local(host: str = "127.0.0.1", port: int = 8765):
    """Run the Memorizz local web UI."""
    try:
        from memorizz.ui import run_server

        run_server(host=host, port=port)
        return True
    except ImportError as e:
        print(f"✗ Failed to import UI module: {e}")
        print("\nPlease ensure memorizz[ui] is installed:")
        print("  pip install memorizz[ui]")
        return False
    except Exception as e:
        print(f"✗ Failed to start UI server: {e}")
        return False


def run_automations(
    poll_interval: int = 5, lease_seconds: int = 120, concurrency: int = 2
):
    """Run the always-on automations worker for the configured backend."""
    try:
        from memorizz.automation.store.factory import get_automation_store
        from memorizz.automation.worker import run_worker
        from memorizz.cli import agent_factory
        from memorizz.cli import config as cli_config
    except ImportError as e:
        print(f"✗ Failed to import automations dependencies: {e}")
        print("\nInstall the optional dependencies for your memory backend.")
        return False

    cli_config.load_layered_env()
    backend = str(os.environ.get("MEMORIZZ_BACKEND", "")).strip().lower()
    if not backend:
        has_oracle = all(
            os.environ.get(name)
            for name in ("ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_DSN")
        )
        if has_oracle:
            backend = "oracle"
        elif os.environ.get("MONGODB_URI"):
            backend = "mongodb"
        else:
            backend = "filesystem"
        os.environ["MEMORIZZ_BACKEND"] = backend
    if backend not in {"filesystem", "mongodb", "oracle"}:
        print(f"✗ Unsupported MEMORIZZ_BACKEND: {backend}")
        print("  Supported: filesystem, mongodb, oracle")
        return False

    if backend == "oracle":
        required = ("ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_DSN")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            print(
                "✗ Missing Oracle env vars for automations worker: "
                + ", ".join(missing)
            )
            return False
    elif backend == "mongodb" and not os.environ.get("MONGODB_URI"):
        print("✗ Missing MONGODB_URI for automations worker")
        return False

    warnings = []
    try:
        provider = agent_factory.detect_memory_provider({}, warnings)
    except Exception as e:
        print(f"✗ Failed to configure {backend} memory provider: {e}")
        return False
    for warning in warnings:
        print(f"⚠ {warning}")

    store = get_automation_store(provider)
    if store is None:
        print("✗ Automations store unavailable for configured provider")
        try:
            provider.close()
        except Exception:
            pass
        return False

    try:
        run_worker(
            store=store,
            memory_provider=provider,
            poll_interval_s=int(poll_interval),
            lease_seconds=int(lease_seconds),
            max_concurrency=int(concurrency),
        )
    except KeyboardInterrupt:
        return True
    except Exception as e:
        print(f"✗ Automations worker failed: {e}")
        return False
    finally:
        try:
            provider.close()
        except Exception:
            pass

    return True
