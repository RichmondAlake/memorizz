# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Operational CLI commands: Oracle setup and teardown, the local UI, automations.

Script locations are resolved from ``_PKG_DIR`` (``src/memorizz``).
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
        print("  export ORACLE_ADMIN_PASSWORD='<generate-a-strong-password>'")
        print("  docker run -d --name oracle-memorizz -p 1521:1521 \\")
        print('    -e ORACLE_PWD="$ORACLE_ADMIN_PASSWORD" \\')
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


def ui_startup_from_flags(
    backend: Optional[str] = None,
    *,
    user: Optional[str] = None,
    dsn: Optional[str] = None,
    schema: Optional[str] = None,
    password_env: str = "ORACLE_PASSWORD",
    uri_env: str = "MONGODB_URI",
    db: Optional[str] = None,
    path: Optional[str] = None,
    data_source_id: Optional[str] = None,
    environ=None,
    prompt=None,
) -> Optional[dict]:
    """What ``memorizz ui`` connects to at startup from its flags.

    The backend is ``--backend``, or the one the other flags imply. Secrets
    never come from a flag: the Oracle password from the variable
    ``--password-env`` names (ORACLE_PASSWORD), the MongoDB URI from
    ``--uri-env`` (MONGODB_URI), the Notion token from NOTION_TOKEN; when
    the variable is unset and this is a terminal, they are asked for without
    echo. Returns None when no flag names a store.
    """
    env = os.environ if environ is None else environ
    backend = (backend or "").strip().lower() or None
    if backend is None:
        if user or dsn or schema:
            backend = "oracle"
        elif db:
            backend = "mongodb"
        elif path:
            backend = "filesystem"
        elif data_source_id:
            backend = "notion"
        else:
            return None
    if backend not in {"oracle", "mongodb", "filesystem", "notion"}:
        raise ValueError(
            f"Unknown --backend {backend!r}: use oracle, mongodb, filesystem or notion."
        )

    def secret(variable: str, label: str) -> str:
        value = str(env.get(variable, "") or "").strip()
        if value:
            return value
        import sys

        if prompt is not None:
            return str(prompt(label) or "")
        if sys.stdin is not None and sys.stdin.isatty():
            import getpass

            return getpass.getpass(f"{label}: ")
        return ""

    params: dict = {}
    if backend == "oracle":
        user = user or str(env.get("ORACLE_USER", "") or "").strip()
        dsn = dsn or str(env.get("ORACLE_DSN", "") or "").strip()
        params = {
            "oracle_user": user,
            "oracle_dsn": dsn,
            "oracle_schema": schema or str(env.get("ORACLE_SCHEMA", "") or "").strip(),
            "oracle_password": secret(
                password_env, f"Oracle password for {user or 'the user'}"
            ),
        }
    elif backend == "mongodb":
        params = {
            "mongodb_uri": secret(uri_env, "MongoDB connection URI"),
            "mongodb_db_name": db or str(env.get("MONGODB_DB_NAME", "") or "").strip(),
        }
    elif backend == "filesystem":
        from .._env_io import memory_root

        root = path or str(env.get("MEMORIZZ_MEMORY_ROOT", "") or "").strip()
        params = {"filesystem_path": root or str(memory_root())}
    elif backend == "notion":
        params = {
            "notion_data_source_id": data_source_id
            or str(env.get("MEMORIZZ_NOTION_DATA_SOURCE_ID", "") or "").strip(),
            "notion_token": secret("NOTION_TOKEN", "Notion integration token"),
        }
    return {
        "provider_type": backend,
        "source": "command line",
        "params": {key: value for key, value in params.items() if value},
    }


def run_local(
    host: str = "127.0.0.1",
    port: int = 8765,
    startup_connect="env",
    read_only: bool = False,
):
    """Run the Memorizz local web UI.

    ``startup_connect`` is ``"env"`` (connect to the store the environment
    names), a dict from ``ui_startup_from_flags``, or None (start on the
    connect page). ``read_only`` refuses every change, as
    MEMORIZZ_UI_READ_ONLY=true does.
    """
    if read_only:
        os.environ["MEMORIZZ_UI_READ_ONLY"] = "true"
    try:
        from memorizz.ui import run_server
    except ImportError as e:
        from ..extras import missing_extra_message

        print(f"✗ Failed to import UI module: {e}")
        print("\n" + (missing_extra_message(e) or "The UI needs the ui extra:"))
        if not missing_extra_message(e):
            print("  pip install 'memorizz[ui]'")
        return False
    try:
        run_server(host=host, port=port, startup_connect=startup_connect)
        return True
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
