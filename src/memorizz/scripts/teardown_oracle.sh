#!/usr/bin/env bash
# Oracle Database Teardown Script for MemoRizz
#
# This script provides options to teardown Oracle Database installations:
# - Drop database user and all tables (keeps container running)
# - Stop and remove container (keeps volume for data persistence)
# - Full teardown (remove container + volume, permanently delete all data)
#
# Usage:
#   ./teardown_oracle.sh                    # Interactive mode
#   ./teardown_oracle.sh --drop-user        # Drop memorizz user and tables only
#   ./teardown_oracle.sh --remove-container # Stop and remove container
#   ./teardown_oracle.sh --full            # Full teardown (container + volume)
#   ./teardown_oracle.sh --force           # Skip confirmation prompts
#
# Environment Variables:
#   ORACLE_ADMIN_USER      - Admin username (default: system)
#   ORACLE_ADMIN_PASSWORD  - Admin password (default: MyPassword123!)
#   ORACLE_USER            - User to drop (default: memorizz_user)
#   ORACLE_DSN             - Database DSN (default: localhost:1521/FREEPDB1)
#
# See SETUP.md for complete setup instructions.

set -e  # Exit immediately on error

CONTAINER_NAME="oracle-memorizz"
VOLUME_NAME="oracle-memorizz-data"

# Default credentials
ADMIN_USER="${ORACLE_ADMIN_USER:-system}"
ADMIN_PASSWORD="${ORACLE_ADMIN_PASSWORD:-MyPassword123!}"
MEMORIZZ_USER="${ORACLE_USER:-memorizz_user}"
DSN="${ORACLE_DSN:-localhost:1521/FREEPDB1}"

# Parse command line arguments
TEARDOWN_MODE=""
FORCE_MODE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --drop-user)
      TEARDOWN_MODE="drop-user"
      shift
      ;;
    --remove-container)
      TEARDOWN_MODE="remove-container"
      shift
      ;;
    --full)
      TEARDOWN_MODE="full"
      shift
      ;;
    --force)
      FORCE_MODE=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --drop-user         Drop memorizz user and all tables (keeps container)"
      echo "  --remove-container  Stop and remove container (keeps volume)"
      echo "  --full              Full teardown (remove container + volume)"
      echo "  --force             Skip confirmation prompts"
      echo "  -h, --help          Show this help message"
      echo ""
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# --- Helper functions ---
function log() {
  echo -e "\033[1;36m$1\033[0m" >&2
}

function error() {
  echo -e "\033[1;31m$1\033[0m" >&2
}

function success() {
  echo -e "\033[1;32m$1\033[0m" >&2
}

function warning() {
  echo -e "\033[1;33m$1\033[0m" >&2
}

function prompt() {
  echo -e "\033[1;33m$1\033[0m" >&2
}

# --- Confirmation function ---
function confirm() {
  local message="$1"

  if [ "$FORCE_MODE" = true ]; then
    return 0
  fi

  warning "$message"
  prompt "Are you sure? Type 'yes' to confirm: "
  read -r response

  if [ "$response" != "yes" ]; then
    log "Operation cancelled"
    exit 0
  fi
}

# --- Interactive mode selection ---
function select_teardown_mode() {
  echo "" >&2
  prompt "╔═══════════════════════════════════════════════════════════════════════╗"
  prompt "║           Oracle Database Teardown Options                           ║"
  prompt "╚═══════════════════════════════════════════════════════════════════════╝"
  echo "" >&2
  echo "  1) Drop User and Tables Only (Recommended for schema reset)" >&2
  echo "     • Drops memorizz_user and all associated tables" >&2
  echo "     • Keeps Docker container running" >&2
  echo "     • Keeps data volume (can recreate user quickly)" >&2
  echo "     • Use this when you want to reset the schema" >&2
  echo "" >&2
  echo "  2) Remove Container (keeps data volume)" >&2
  echo "     • Stops and removes the Docker container" >&2
  echo "     • Preserves data volume for future use" >&2
  echo "     • Data will be restored when container is recreated" >&2
  echo "     • Use this to free up resources temporarily" >&2
  echo "" >&2
  echo "  3) Full Teardown (PERMANENT - removes everything)" >&2
  echo "     • Removes Docker container AND data volume" >&2
  echo "     • ⚠️  ALL DATA WILL BE PERMANENTLY LOST" >&2
  echo "     • Use this for complete cleanup" >&2
  echo "" >&2
  echo "  4) Cancel" >&2
  echo "" >&2
  prompt "Enter your choice (1-4) [4]: "
  read -r choice
  choice="${choice:-4}"

  case $choice in
    1)
      TEARDOWN_MODE="drop-user"
      ;;
    2)
      TEARDOWN_MODE="remove-container"
      ;;
    3)
      TEARDOWN_MODE="full"
      ;;
    4)
      log "Teardown cancelled"
      exit 0
      ;;
    *)
      error "Invalid choice: $choice"
      exit 1
      ;;
  esac
}

# --- Drop user and tables ---
function drop_user() {
  log "📋 Dropping user '$MEMORIZZ_USER' and all tables..."

  # Check if oracledb Python package is available
  if ! python3 -c "import oracledb" 2>/dev/null; then
    error "❌ Python oracledb package not found"
    error "   Install it with: pip install oracledb"
    exit 1
  fi

  # Create a temporary Python script to drop the user
  cat > /tmp/drop_oracle_user.py << 'EOF'
import sys
import os

try:
    import oracledb
except ImportError:
    print("❌ oracledb package not installed")
    sys.exit(1)

admin_user = os.environ.get("ADMIN_USER", "system")
admin_password = os.environ.get("ADMIN_PASSWORD", "MyPassword123!")
dsn = os.environ.get("DSN", "localhost:1521/FREEPDB1")
memorizz_user = os.environ.get("MEMORIZZ_USER", "memorizz_user")

try:
    # Connect as admin
    print(f"🔌 Connecting as {admin_user}...")
    conn = oracledb.connect(user=admin_user, password=admin_password, dsn=dsn)
    cursor = conn.cursor()

    # Kill active sessions for the user
    print(f"🔍 Checking for active sessions for {memorizz_user}...")
    cursor.execute(f"""
        SELECT sid, serial# FROM v$session
        WHERE username = UPPER('{memorizz_user}')
    """)
    sessions = cursor.fetchall()

    if sessions:
        print(f"  Found {len(sessions)} active session(s), terminating...")
        for sid, serial in sessions:
            try:
                cursor.execute(f"ALTER SYSTEM KILL SESSION '{sid},{serial}' IMMEDIATE")
                print(f"  ✓ Killed session {sid},{serial}")
            except Exception as e:
                print(f"  ⚠ Could not kill session {sid},{serial}: {e}")
    else:
        print("  No active sessions found")

    # Drop the user
    print(f"🗑️  Dropping user {memorizz_user}...")
    try:
        cursor.execute(f"DROP USER {memorizz_user} CASCADE")
        print(f"  ✅ User {memorizz_user} dropped successfully")
    except Exception as e:
        error_str = str(e)
        if "ORA-01918" in error_str:
            print(f"  ℹ User {memorizz_user} does not exist")
        else:
            print(f"  ❌ Error dropping user: {e}")
            sys.exit(1)

    conn.commit()
    conn.close()
    print("✅ Teardown complete!")

except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

  # Run the Python script
  export ADMIN_USER="$ADMIN_USER"
  export ADMIN_PASSWORD="$ADMIN_PASSWORD"
  export DSN="$DSN"
  export MEMORIZZ_USER="$MEMORIZZ_USER"

  python3 /tmp/drop_oracle_user.py

  # Clean up
  rm -f /tmp/drop_oracle_user.py

  success "✅ User and tables dropped successfully!"
  echo "" >&2
  echo "To recreate the schema, run:" >&2
  echo "  python -m memorizz.memory_provider.oracle.setup" >&2
}

# --- Remove container ---
function remove_container() {
  log "🐳 Checking Docker container status..."

  # Check if container exists
  if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    # Stop container if running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
      log "⏹️  Stopping container '$CONTAINER_NAME'..."
      docker stop $CONTAINER_NAME
      success "✅ Container stopped"
    fi

    # Remove container
    log "🗑️  Removing container '$CONTAINER_NAME'..."
    docker rm $CONTAINER_NAME
    success "✅ Container removed"
  else
    warning "ℹ Container '$CONTAINER_NAME' does not exist"
  fi

  echo "" >&2
  echo "Container removed successfully!" >&2
  echo "Data volume '$VOLUME_NAME' is preserved." >&2
  echo "" >&2
  echo "To recreate the container, run:" >&2
  echo "  memorizz oracle install --image lite" >&2
}

# --- Full teardown ---
function full_teardown() {
  log "🗑️  Performing full teardown..."

  # Remove container first
  if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
      log "⏹️  Stopping container '$CONTAINER_NAME'..."
      docker stop $CONTAINER_NAME
    fi

    log "🗑️  Removing container '$CONTAINER_NAME'..."
    docker rm $CONTAINER_NAME
    success "✅ Container removed"
  else
    warning "ℹ Container '$CONTAINER_NAME' does not exist"
  fi

  # Remove volume
  if [ "$(docker volume ls -q -f name=$VOLUME_NAME)" ]; then
    log "🗑️  Removing volume '$VOLUME_NAME'..."
    docker volume rm $VOLUME_NAME
    success "✅ Volume removed (all data permanently deleted)"
  else
    warning "ℹ Volume '$VOLUME_NAME' does not exist"
  fi

  success "✅ Full teardown complete!"
  echo "" >&2
  echo "All Oracle data has been permanently deleted." >&2
  echo "" >&2
  echo "To start fresh, run:" >&2
  echo "  memorizz oracle install --image lite" >&2
}

# --- Main execution ---
echo "" >&2
log "╔═══════════════════════════════════════════════════════════════════════╗"
log "║           MemoRizz Oracle Database Teardown                          ║"
log "╚═══════════════════════════════════════════════════════════════════════╝"
echo "" >&2

# Select mode if not specified
if [ -z "$TEARDOWN_MODE" ]; then
  select_teardown_mode
fi

# Execute based on mode
case $TEARDOWN_MODE in
  drop-user)
    confirm "⚠️  This will DROP the user '$MEMORIZZ_USER' and ALL associated tables!"
    drop_user
    ;;
  remove-container)
    confirm "⚠️  This will REMOVE the Docker container (data volume will be preserved)!"
    remove_container
    ;;
  full)
    confirm "⚠️  This will PERMANENTLY DELETE the container AND data volume! ALL DATA WILL BE LOST!"
    full_teardown
    ;;
  *)
    error "Invalid teardown mode: $TEARDOWN_MODE"
    exit 1
    ;;
esac

echo "" >&2
success "🎉 Teardown completed successfully!"
