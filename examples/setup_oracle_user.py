"""
Oracle Database Setup Script (Convenience Wrapper)

⚠️  RECOMMENDED: For most users, use the CLI command:
    memorizz oracle setup

This script is provided as a convenience wrapper for:
- Users who cloned the repository and prefer running Python scripts directly
- Development and testing scenarios

Setup Methods (in order of recommendation):
1. CLI Command (Best for pip-installed users):
   memorizz oracle setup
   # or
   python -m memorizz oracle setup

2. This Script (Good for repo-cloned users):
   python examples/setup_oracle_user.py

3. Direct Import (For programmatic use):
   from memorizz.memory_provider.oracle import setup_oracle_user
   setup_oracle_user()

The setup automatically detects your database configuration:
- Admin mode: Full setup with user creation (local/self-hosted databases)
- User-only mode: Uses existing schema (hosted databases like FreeSQL.com)
"""

import sys

# Import from package (works for both pip-installed and repo-cloned users)
try:
    from memorizz.memory_provider.oracle import setup_oracle_user
except ImportError:
    print("✗ Failed to import setup function from memorizz package.")
    print("\nPlease ensure memorizz[oracle] is installed:")
    print("  pip install memorizz[oracle]")
    print("\nThen use the CLI command (recommended):")
    print("  memorizz oracle setup")
    print("\nOr use the Python module:")
    print("  python -m memorizz oracle setup")
    sys.exit(1)


if __name__ == "__main__":
    try:
        success = setup_oracle_user()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
