"""CLI commands for Memorizz."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def install_oracle():
    """Install Oracle database using install_oracle.sh script."""
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

    # Execute the script
    try:
        result = subprocess.run(
            [str(script_path)],
            check=False,  # Don't raise exception on non-zero exit
            capture_output=False,  # Show output in real-time
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


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Memorizz CLI")
        print("\nAvailable commands:")
        print("  run local       Start local web UI")
        print("  install-oracle  Install Oracle database container")
        print("  setup-oracle    Set up Oracle database schema")
        print("\nUsage:")
        print("  memorizz run local [--port PORT] [--host HOST]")
        print("  memorizz install-oracle")
        print("  memorizz setup-oracle")
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
    elif command == "install-oracle":
        success = install_oracle()
        sys.exit(0 if success else 1)
    elif command == "setup-oracle":
        success = setup_oracle()
        sys.exit(0 if success else 1)
    else:
        print(f"✗ Unknown command: {command}")
        print("Run 'memorizz' or 'python -m memorizz.cli' for help")
        sys.exit(1)


if __name__ == "__main__":
    main()
