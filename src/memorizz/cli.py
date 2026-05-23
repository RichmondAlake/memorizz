# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""CLI commands for Memorizz."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def install_oracle(image: Optional[str] = None):
    """Install Oracle database using install_oracle.sh script.

    Args:
        image: Oracle image to use without prompting. Accepted values:
               'lite' (default Lite Edition), 'full' (Full Edition),
               'community' (gvenzl community image), or '1'/'2'/'3'.
    """
    # Try to find install_oracle.sh script
    # Check multiple possible locations
    possible_paths = [
        # Current directory (for local development)
        Path("install_oracle.sh"),
        # Package scripts directory (when installed from PyPI)
        Path(__file__).parent / "scripts" / "install_oracle.sh",
        # Repository root (if installed in editable mode or running from repo)
        Path(__file__).parent.parent.parent / "install_oracle.sh",
        # Alternative repository root path
        Path(__file__).parent.parent.parent.parent / "install_oracle.sh",
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

    # Make script executable
    os.chmod(script_path, 0o755)

    # Set up environment for non-interactive mode if image is specified
    env = os.environ.copy()
    if image:
        image_map = {"lite": "1", "full": "2", "community": "3"}
        choice = image_map.get(image.lower(), image)
        if choice not in ("1", "2", "3"):
            print(f"✗ Invalid image: {image}")
            print("  Valid options: lite, full, community (or 1, 2, 3)")
            return False
        env["ORACLE_IMAGE_CHOICE"] = choice

    # Execute the script
    try:
        result = subprocess.run(
            ["bash", str(script_path.resolve())],
            check=False,  # Don't raise exception on non-zero exit
            capture_output=False,  # Show output in real-time
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
    """
    Teardown Oracle database installation.

    Args:
        mode: Teardown mode - 'drop-user', 'remove-container', or 'full'
        force: Skip confirmation prompts
    """
    # Try to find teardown_oracle.sh script
    possible_paths = [
        # Package scripts directory (when installed from PyPI)
        Path(__file__).parent / "scripts" / "teardown_oracle.sh",
        # Current directory (for local development)
        Path("teardown_oracle.sh"),
        # Repository root (if installed in editable mode or running from repo)
        Path(__file__).parent.parent.parent / "teardown_oracle.sh",
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

    # Make script executable
    os.chmod(script_path, 0o755)

    # Build command arguments
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

    # Execute the script
    try:
        result = subprocess.run(
            cmd,
            check=False,  # Don't raise exception on non-zero exit
            capture_output=False,  # Show output in real-time
        )
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
    """Run the always-on automations worker (Oracle-backed)."""
    try:
        from memorizz.automation.store.factory import get_automation_store
        from memorizz.automation.worker import run_worker
        from memorizz.memory_provider.oracle.provider import (
            OracleConfig,
            OracleProvider,
        )
    except ImportError as e:
        print(f"✗ Failed to import automations dependencies: {e}")
        print("\nPlease ensure memorizz[oracle] is installed:")
        print("  pip install memorizz[oracle]")
        return False

    user = str(os.environ.get("ORACLE_USER", "")).strip()
    password = str(os.environ.get("ORACLE_PASSWORD", "")).strip()
    dsn = str(os.environ.get("ORACLE_DSN", "")).strip()
    schema = str(os.environ.get("ORACLE_SCHEMA", "")).strip() or None
    if not user or not password or not dsn:
        print("✗ Missing Oracle env vars for automations worker")
        print("  Required: ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN")
        print("  Optional: ORACLE_SCHEMA")
        return False

    provider = OracleProvider(
        OracleConfig(user=user, password=password, dsn=dsn, schema=schema)
    )
    store = get_automation_store(provider)
    if store is None:
        print("✗ Automations store unavailable for configured provider")
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

    return True


def _load_dotenv():
    """Load .env file from current directory if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ImportError:
        pass


def main():
    """Main CLI entry point."""
    _load_dotenv()

    if len(sys.argv) < 2:
        print("Memorizz CLI")
        print("\nAvailable commands:")
        print("  run local          Start local web UI")
        print("  run automations    Run automations worker (Oracle)")
        print("  install-oracle     Install Oracle database container")
        print("  setup-oracle       Set up Oracle database schema")
        print("  setup-oracle-schema  Apply Oracle schema updates (no user drop)")
        print("  teardown-oracle    Teardown Oracle database installation")
        print("\nUsage:")
        print("  memorizz run local [--port PORT] [--host HOST]")
        print(
            "  memorizz run automations [--poll-interval N] [--lease-seconds N] [--concurrency N]"
        )
        print("  memorizz install-oracle [--image lite|full|community]")
        print("  memorizz setup-oracle")
        print("  memorizz setup-oracle-schema")
        print("  memorizz teardown-oracle [--mode MODE] [--force]")
        print("    Modes: drop-user | remove-container | full")
        print("    Examples:")
        print("      memorizz teardown-oracle                    # Interactive mode")
        print("      memorizz teardown-oracle --mode drop-user   # Drop user only")
        print(
            "      memorizz teardown-oracle --mode full --force  # Full teardown, no confirm"
        )
        print("  python -m memorizz.cli <command>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "run" and len(sys.argv) > 2 and sys.argv[2] == "local":
        # Parse optional --port and --host arguments
        host = "127.0.0.1"
        port = 8765
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--port" and i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                except ValueError:
                    print(f"✗ Invalid port: {sys.argv[i + 1]}")
                    sys.exit(1)
                i += 2
            elif sys.argv[i] == "--host" and i + 1 < len(sys.argv):
                host = sys.argv[i + 1]
                i += 2
            else:
                print(f"✗ Unknown option: {sys.argv[i]}")
                sys.exit(1)
        success = run_local(host=host, port=port)
        sys.exit(0 if success else 1)
    elif command == "run" and len(sys.argv) > 2 and sys.argv[2] == "automations":
        poll_interval = 5
        lease_seconds = 120
        concurrency = 2
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--poll-interval" and i + 1 < len(sys.argv):
                try:
                    poll_interval = int(sys.argv[i + 1])
                except ValueError:
                    print(f"✗ Invalid poll interval: {sys.argv[i + 1]}")
                    sys.exit(1)
                i += 2
            elif sys.argv[i] == "--lease-seconds" and i + 1 < len(sys.argv):
                try:
                    lease_seconds = int(sys.argv[i + 1])
                except ValueError:
                    print(f"✗ Invalid lease seconds: {sys.argv[i + 1]}")
                    sys.exit(1)
                i += 2
            elif sys.argv[i] == "--concurrency" and i + 1 < len(sys.argv):
                try:
                    concurrency = int(sys.argv[i + 1])
                except ValueError:
                    print(f"✗ Invalid concurrency: {sys.argv[i + 1]}")
                    sys.exit(1)
                i += 2
            else:
                print(f"✗ Unknown option: {sys.argv[i]}")
                sys.exit(1)

        success = run_automations(
            poll_interval=poll_interval,
            lease_seconds=lease_seconds,
            concurrency=concurrency,
        )
        sys.exit(0 if success else 1)
    elif command == "install-oracle":
        image = None
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] in ("--image", "-i") and i + 1 < len(sys.argv):
                image = sys.argv[i + 1]
                i += 2
            else:
                print(f"✗ Unknown option: {sys.argv[i]}")
                sys.exit(1)
        success = install_oracle(image=image)
        sys.exit(0 if success else 1)
    elif command == "setup-oracle":
        success = setup_oracle()
        sys.exit(0 if success else 1)
    elif command == "setup-oracle-schema":
        success = setup_oracle_schema()
        sys.exit(0 if success else 1)
    elif command == "teardown-oracle":
        # Parse optional --mode and --force arguments
        mode = None
        force = False
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
                mode = sys.argv[i + 1]
                if mode not in ["drop-user", "remove-container", "full"]:
                    print(f"✗ Invalid mode: {mode}")
                    print("  Valid modes: drop-user, remove-container, full")
                    sys.exit(1)
                i += 2
            elif sys.argv[i] == "--force":
                force = True
                i += 1
            else:
                print(f"✗ Unknown option: {sys.argv[i]}")
                sys.exit(1)
        success = teardown_oracle(mode=mode, force=force)
        sys.exit(0 if success else 1)
    else:
        print(f"✗ Unknown command: {command}")
        print("Run 'memorizz' or 'python -m memorizz.cli' for help")
        sys.exit(1)


if __name__ == "__main__":
    main()
