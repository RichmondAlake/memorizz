#!/usr/bin/env bash
# Oracle Database Installation Script for MemoRizz
#
# This script installs and starts Oracle Database 23ai Free in a Docker container.
# It checks if a container already exists and starts it, or creates a new one.
#
# Usage:
#   ./install_oracle.sh
#
#   Or with custom password:
#   export ORACLE_ADMIN_PASSWORD="YourSecurePassword123!"
#   ./install_oracle.sh
#
#   For Apple Silicon (M1/M2/M3):
#   export PLATFORM_FLAG="--platform linux/amd64"
#   ./install_oracle.sh
#
#   For non-interactive mode (skip image selection):
#   export ORACLE_IMAGE_CHOICE="1"  # Official Oracle 23ai Free Lite
#   export ORACLE_IMAGE_CHOICE="2"  # Official Oracle 23ai Free Full
#   export ORACLE_IMAGE_CHOICE="3"  # Community gvenzl/oracle-free
#   ./install_oracle.sh
#
# Environment Variables:
#   ORACLE_ADMIN_PASSWORD  - Admin password (default: MyPassword123!)
#   ORACLE_IMAGE_CHOICE    - Image selection (1, 2, or 3; default: interactive)
#   ORACLE_IMAGE_TAG       - Oracle image tag (legacy, use ORACLE_IMAGE_CHOICE instead)
#   ORACLE_VECTOR_MEMORY_SIZE - Vector memory pool size (default: 512M)
#                              Set higher (e.g., 1G) for large embedding workloads
#   PLATFORM_FLAG          - Docker platform flag (default: empty, auto-detect)
#                            Use "--platform linux/amd64" for Apple Silicon
#
# Features:
#   - Interactive Docker image selection
#   - Persistent data storage via Docker volume (oracle-memorizz-data)
#   - Idempotent: safe to run multiple times
#   - Waits for database readiness before completing
#   - Cross-platform support (Intel, AMD, Apple Silicon)
#
# See SETUP.md for complete setup instructions.

set -e  # Exit immediately on error

CONTAINER_NAME="oracle-memorizz"
VOLUME_NAME="oracle-memorizz-data"

# Use environment variable if set, otherwise use default
# Set ORACLE_ADMIN_PASSWORD to customize the admin password
PASSWORD="${ORACLE_ADMIN_PASSWORD:-MyPassword123!}"

# Platform flag for Apple Silicon compatibility
# Oracle Free may require emulation on ARM64
PLATFORM_FLAG="${PLATFORM_FLAG:-}"

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

function prompt() {
  echo -e "\033[1;33m$1\033[0m" >&2
}

# --- Interactive Image Selection ---
function select_oracle_image() {
  # If ORACLE_IMAGE_CHOICE is already set, use it (non-interactive mode)
  if [ -n "${ORACLE_IMAGE_CHOICE}" ]; then
    choice="${ORACLE_IMAGE_CHOICE}"
  else
    # Interactive mode
    echo "" >&2
    prompt "╔═══════════════════════════════════════════════════════════════════════╗"
    prompt "║           Select Oracle Database Docker Image                        ║"
    prompt "╚═══════════════════════════════════════════════════════════════════════╝"
    echo "" >&2
    echo "  1) Official Oracle 23ai Free - Lite Edition (Recommended)" >&2
    echo "     Image: container-registry.oracle.com/database/free:latest-lite" >&2
    echo "     Size: ~1.78GB" >&2
    echo "     Features: AI Vector Search, Full Oracle 23ai capabilities" >&2
    echo "" >&2
    echo "  2) Official Oracle 23ai Free - Full Edition" >&2
    echo "     Image: container-registry.oracle.com/database/free:latest" >&2
    echo "     Size: ~9.93GB" >&2
    echo "     Features: AI Vector Search, All Oracle 23ai features + extras" >&2
    echo "" >&2
    echo "  3) Community gvenzl/oracle-free (Faster startup)" >&2
    echo "     Image: gvenzl/oracle-free:latest" >&2
    echo "     Size: ~3GB" >&2
    echo "     Features: Oracle 23ai Free, Optimized for development" >&2
    echo "     Note: Community-maintained, faster initialization" >&2
    echo "" >&2
    prompt "Enter your choice (1-3) [1]: "
    read -r choice
    choice="${choice:-1}"  # Default to 1 if empty
  fi

  # Set IMAGE_NAME and IMAGE_TAG based on choice
  case $choice in
    1)
      IMAGE_NAME="container-registry.oracle.com/database/free:latest-lite"
      IMAGE_DISPLAY_NAME="Official Oracle 23ai Free Lite"
      IMAGE_SIZE="~1.78GB"
      ;;
    2)
      IMAGE_NAME="container-registry.oracle.com/database/free:latest"
      IMAGE_DISPLAY_NAME="Official Oracle 23ai Free Full"
      IMAGE_SIZE="~9.93GB"
      ;;
    3)
      IMAGE_NAME="gvenzl/oracle-free:latest"
      IMAGE_DISPLAY_NAME="Community gvenzl/oracle-free"
      IMAGE_SIZE="~3GB"
      ;;
    *)
      error "Invalid choice: $choice (must be 1, 2, or 3)"
      exit 1
      ;;
  esac

  success "✓ Selected: $IMAGE_DISPLAY_NAME ($IMAGE_SIZE)"
  export SELECTED_IMAGE_NAME="$IMAGE_NAME"
  export SELECTED_IMAGE_DISPLAY_NAME="$IMAGE_DISPLAY_NAME"
}

# Check if Docker is running
log "🔍 Checking if Docker is running..."
if ! docker info >/dev/null 2>&1; then
  error ""
  error "❌ Docker is not running!"
  error ""
  error "Please start Docker Desktop (or Docker daemon) and try again."
  error ""
  error "To verify Docker is running:"
  error "  docker ps"
  error ""
  exit 1
fi

log "✅ Docker is running"

# Select Oracle image (interactive or from environment variable)
select_oracle_image
IMAGE_NAME="$SELECTED_IMAGE_NAME"

log "🔍 Checking if Oracle container '$CONTAINER_NAME' exists..."

# Check if container exists
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
  # Check if container is already running
  if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    log "✅ Container '$CONTAINER_NAME' is already running!"
  else
    log "▶️ Starting existing container '$CONTAINER_NAME'..."
    docker start $CONTAINER_NAME
  fi
else
  # Check if image already exists locally
  log "🔍 Checking if Oracle image exists locally..."
  IMAGE_EXISTS=$(docker images -q $IMAGE_NAME 2>/dev/null)

  if [ -n "$IMAGE_EXISTS" ]; then
    log "✅ Oracle image found locally, skipping download"
  else
    log "🐳 Pulling $SELECTED_IMAGE_DISPLAY_NAME..."
    if [ -n "$PLATFORM_FLAG" ]; then
      log "   Using platform flag: $PLATFORM_FLAG (for Apple Silicon compatibility)"
      docker pull $PLATFORM_FLAG $IMAGE_NAME
    else
      docker pull $IMAGE_NAME
    fi
  fi

  log "📦 Creating persistent volume '$VOLUME_NAME' (if not exists)..."
  docker volume create $VOLUME_NAME 2>/dev/null || log "   Volume already exists (reusing)"

  log "🚀 Creating and starting new container '$CONTAINER_NAME'..."
  log "   Using persistent volume: $VOLUME_NAME"
  # ORACLE_PWD for official images, ORACLE_PASSWORD for gvenzl community image
  if [ -n "$PLATFORM_FLAG" ]; then
    docker run -d \
      $PLATFORM_FLAG \
      --name $CONTAINER_NAME \
      -p 1521:1521 \
      -e ORACLE_PWD=$PASSWORD \
      -e ORACLE_PASSWORD=$PASSWORD \
      -v $VOLUME_NAME:/opt/oracle/oradata \
      $IMAGE_NAME
  else
    docker run -d \
      --name $CONTAINER_NAME \
      -p 1521:1521 \
      -e ORACLE_PWD=$PASSWORD \
      -e ORACLE_PASSWORD=$PASSWORD \
      -v $VOLUME_NAME:/opt/oracle/oradata \
      $IMAGE_NAME
  fi
fi

log "⏳ Waiting for Oracle to be ready (this may take 2–3 minutes)..."
log "   Monitoring container logs for readiness..."
# Wait until the container logs show "DATABASE IS READY TO USE!"
# Using 5-second intervals for more responsive feedback
until docker logs $CONTAINER_NAME 2>&1 | grep -q "DATABASE IS READY TO USE!"; do
  sleep 5
  echo -n "." >&2
done

echo "" >&2
log "✅ Oracle Database is ready!"

# --- Configure Vector Memory Pool ---
# Strategy:
#   1. Try SCOPE=BOTH (works on official Oracle images that have an SPFILE)
#   2. If that fails, create SPFILE from PFILE, set via SCOPE=SPFILE, and restart
VECTOR_MEMORY_SIZE="${ORACLE_VECTOR_MEMORY_SIZE:-512M}"
log "🧠 Configuring vector memory pool (${VECTOR_MEMORY_SIZE})..."

# Fast path: SCOPE=BOTH (works on official Oracle images)
FAST_RESULT=$(docker exec $CONTAINER_NAME bash -c '
  echo "ALTER SYSTEM SET vector_memory_size='"${VECTOR_MEMORY_SIZE}"' SCOPE=BOTH;" | \
  sqlplus -s sys/'"${PASSWORD}"'@localhost:1521/FREE as sysdba
' 2>&1)

if echo "$FAST_RESULT" | grep -q "System altered"; then
  success "✓ Vector memory pool set to ${VECTOR_MEMORY_SIZE}"
else
  # Slow path: create SPFILE, set parameter, restart (needed for gvenzl/community images)
  log "   Setting via SPFILE (requires restart)..."
  docker exec $CONTAINER_NAME bash -c '
    sqlplus -s sys/'"${PASSWORD}"'@localhost:1521/FREE as sysdba <<EOSQL
CREATE SPFILE FROM PFILE;
ALTER SYSTEM SET vector_memory_size='"${VECTOR_MEMORY_SIZE}"' SCOPE=SPFILE;
EOSQL
  ' >/dev/null 2>&1

  log "   Restarting container to apply vector memory setting..."
  docker restart $CONTAINER_NAME >/dev/null 2>&1

  # Wait for database readiness after restart
  until docker logs $CONTAINER_NAME 2>&1 | tail -20 | grep -q "DATABASE IS READY TO USE!"; do
    sleep 5
    echo -n "." >&2
  done
  echo "" >&2

  # Verify
  ACTUAL_SIZE=$(docker exec $CONTAINER_NAME bash -c '
    echo "SELECT value FROM v\$parameter WHERE name='"'"'vector_memory_size'"'"';" | \
    sqlplus -s sys/'"${PASSWORD}"'@localhost:1521/FREE as sysdba
  ' 2>/dev/null | grep -oE '[0-9]+' | head -1)

  if [ -n "$ACTUAL_SIZE" ] && [ "$ACTUAL_SIZE" != "0" ]; then
    success "✓ Vector memory pool set to ${VECTOR_MEMORY_SIZE}"
  else
    error "⚠ Could not set vector memory pool (non-fatal, HNSW indexes may fail)"
    error "  See: https://docs.oracle.com/error-help/db/ora-51962/"
  fi
fi

echo "" >&2
echo "Connection details:" >&2
echo "  Host: localhost" >&2
echo "  Port: 1521" >&2
echo "  Service Name: FREEPDB1" >&2
echo "  Admin User: system" >&2
echo "  Admin Password: $PASSWORD" >&2
echo "" >&2
echo "📝 Environment Variables:" >&2
echo "  To use these credentials in your shell, run:" >&2
echo "    eval \$(./install_oracle.sh)" >&2
echo "" >&2
echo "  Or source the script:" >&2
echo "    source ./install_oracle.sh" >&2
echo "" >&2
echo "  Or set manually:" >&2
echo "    export ORACLE_ADMIN_PASSWORD=\"$PASSWORD\"" >&2
echo "    export ORACLE_USER=\"memorizz_user\"" >&2
echo "    export ORACLE_PASSWORD=\"SecurePass123!\"" >&2
echo "    export ORACLE_DSN=\"localhost:1521/FREEPDB1\"" >&2
echo "" >&2
# Export variables (useful if script is sourced)
export ORACLE_ADMIN_PASSWORD="$PASSWORD"
export ORACLE_USER="memorizz_user"
export ORACLE_PASSWORD="SecurePass123!"
export ORACLE_DSN="localhost:1521/FREEPDB1"
# Output export commands to stdout (for eval) - these must be clean, no colors
echo "export ORACLE_ADMIN_PASSWORD=\"$PASSWORD\""
echo "export ORACLE_USER=\"memorizz_user\""
echo "export ORACLE_PASSWORD=\"SecurePass123!\""
echo "export ORACLE_DSN=\"localhost:1521/FREEPDB1\""
