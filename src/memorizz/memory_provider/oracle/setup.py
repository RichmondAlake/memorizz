# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""
Oracle Database Setup Module for Memorizz

This module provides the setup_oracle_user function that can be imported
and used programmatically or via CLI.

Supports two modes:
- Admin mode: Full setup with user creation and privilege grants (for local/self-hosted databases)
- User-only mode: Uses existing schema, skips user creation (for hosted databases like FreeSQL.com)
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import oracledb
except ImportError:
    oracledb = None


def _safe_close_connection(conn):
    """
    Safely close a database connection, handling cases where it's already closed.

    Args:
        conn: Oracle database connection object (or None)
    """
    if conn is None:
        return

    try:
        conn.close()
    except Exception:
        # Connection is already closed or doesn't exist, ignore the error
        # This handles oracledb.exceptions.InterfaceError and any other exceptions
        pass


def _get_embedding_dimension() -> int:
    """
    Get the embedding dimension to use for VECTOR columns.

    Reads from ORACLE_EMBEDDING_DIM environment variable.
    Falls back to 384, the output size of the default in-database ONNX model,
    if not set or invalid.

    Returns:
        int: Embedding dimension (validated to be positive)
    """
    default_dim = 384
    env_dim = os.environ.get("ORACLE_EMBEDDING_DIM")

    if env_dim is None:
        return default_dim

    try:
        dim = int(env_dim)
        if dim <= 0:
            print(
                f"  ⚠ Invalid ORACLE_EMBEDDING_DIM={env_dim} (must be positive), using default {default_dim}"
            )
            return default_dim
        if dim > 65535:
            print(
                f"  ⚠ ORACLE_EMBEDDING_DIM={dim} is very large, using anyway (max 65535)"
            )
            return min(dim, 65535)
        return dim
    except ValueError:
        print(
            f"  ⚠ Invalid ORACLE_EMBEDDING_DIM='{env_dim}' (not a number), using default {default_dim}"
        )
        return default_dim


def parse_sql_file(filepath, embedding_dim: Optional[int] = None):
    """
    Parse SQL file into individual executable statements.

    Args:
        filepath: Path to the SQL file
        embedding_dim: Optional embedding dimension to use for VECTOR columns.
                      If provided, replaces VECTOR(256, FLOAT32) with VECTOR({embedding_dim}, FLOAT32)

    Returns:
        List of SQL statements
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Apply embedding dimension replacement if specified
    if embedding_dim is not None:
        # Replace VECTOR(256, FLOAT32) with VECTOR({embedding_dim}, FLOAT32)
        # This allows flexible embedding dimensions via environment variable
        import re

        content = re.sub(
            r"VECTOR\(\s*\d+\s*,\s*FLOAT32\s*\)",
            f"VECTOR({embedding_dim}, FLOAT32)",
            content,
        )

    # Remove single-line comments
    lines = []
    for line in content.split("\n"):
        if "--" in line:
            line = line[: line.index("--")]
        lines.append(line)
    content = "\n".join(lines)

    # Remove multi-line comments
    while "/*" in content:
        start = content.index("/*")
        end = content.index("*/", start) + 2
        content = content[:start] + content[end:]

    # Split by semicolons
    statements = [s.strip() for s in content.split(";") if s.strip()]

    # Filter out COMMENT statements
    statements = [s for s in statements if not s.upper().startswith("COMMENT")]

    return statements


def _escape_sql_literal(value: str) -> str:
    """Escape single quotes in SQL string literals."""
    return value.replace("'", "''") if value is not None else value


def _join_server_path(directory: str, filename: str) -> str:
    """
    Join paths using the separator that matches the server directory style.

    Oracle may run on Linux (/) or Windows (\\). We inspect the directory string
    and use the same separator to avoid generating invalid paths.
    """
    if not directory:
        return filename

    directory = directory.rstrip("/\\")
    if "\\" in directory and "/" not in directory:
        separator = "\\"
    else:
        separator = "/"
    return f"{directory}{separator}{filename}"


def _extract_directory_from_path(file_path: str) -> Optional[str]:
    """Return the directory portion of an Oracle datafile path."""
    if not file_path:
        return None
    for sep in ("/", "\\"):
        if sep in file_path:
            return file_path.rsplit(sep, 1)[0]
    return None


def _determine_datafile_path(
    admin_cursor, tablespace_name: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine a datafile path for a new tablespace.

    Returns (path, source_description). Path may be None if Oracle Managed Files
    should handle file placement.
    """
    env_file = os.environ.get("ORACLE_TABLESPACE_DATAFILE")
    if env_file:
        return env_file, "environment datafile override"

    env_dir = os.environ.get("ORACLE_DATAFILE_DIR")
    if env_dir:
        path = _join_server_path(env_dir, f"{tablespace_name.lower()}_01.dbf")
        return path, "environment datafile directory"

    # Try Oracle Managed Files destination (db_create_file_dest)
    try:
        admin_cursor.execute(
            "SELECT value FROM v$parameter WHERE name = 'db_create_file_dest'"
        )
        row = admin_cursor.fetchone()
        if row and row[0]:
            base_dir = row[0].strip()
            if base_dir:
                path = _join_server_path(base_dir, f"{tablespace_name.lower()}_01.dbf")
                return path, "db_create_file_dest parameter"
    except Exception:
        pass

    # Fall back to reusing the directory of an existing user tablespace
    try:
        admin_cursor.execute(
            """
            SELECT file_name FROM (
                SELECT file_name
                FROM dba_data_files
                WHERE tablespace_name NOT IN ('SYSTEM', 'SYSAUX')
                ORDER BY file_id
            )
            WHERE ROWNUM = 1
        """
        )
        row = admin_cursor.fetchone()
        if row and row[0]:
            directory = _extract_directory_from_path(row[0])
            if directory:
                path = _join_server_path(directory, f"{tablespace_name.lower()}_01.dbf")
                return path, "existing datafile directory"
    except Exception:
        pass

    return None, None


def _can_create_users(conn) -> bool:
    """
    Check if the current connection has CREATE USER privilege.

    Args:
        conn: Oracle database connection

    Returns:
        True if user can create other users, False otherwise
    """
    cursor = conn.cursor()
    try:
        # Check for CREATE USER system privilege
        cursor.execute(
            """
            SELECT COUNT(*) FROM user_sys_privs
            WHERE privilege = 'CREATE USER'
        """
        )
        result = cursor.fetchone()
        return result[0] > 0 if result else False
    except Exception:
        # If we can't query privileges, assume we can't create users
        return False
    finally:
        cursor.close()


def _check_user_privileges(conn) -> Dict[str, bool]:
    """
    Check what privileges the current user has.

    Args:
        conn: Oracle database connection

    Returns:
        Dictionary mapping privilege names to availability
    """
    cursor = conn.cursor()
    privileges = {
        "CREATE_TABLE": False,
        "CREATE_VIEW": False,
        "CREATE_SEQUENCE": False,
        "CREATE_TRIGGER": False,
        "CREATE_MINING_MODEL": False,
        "DBMS_VECTOR": False,
        "SODA_APP": False,
    }

    try:
        # Check system privileges
        # Note: CREATE INDEX is not a separate privilege - it's included with CREATE TABLE
        cursor.execute(
            """
            SELECT privilege FROM user_sys_privs
            WHERE privilege IN ('CREATE TABLE', 'CREATE VIEW',
                               'CREATE SEQUENCE', 'CREATE TRIGGER',
                               'CREATE MINING MODEL')
        """
        )
        sys_privs = {row[0] for row in cursor.fetchall()}

        privileges["CREATE_TABLE"] = "CREATE TABLE" in sys_privs
        privileges["CREATE_VIEW"] = "CREATE VIEW" in sys_privs
        privileges["CREATE_SEQUENCE"] = "CREATE SEQUENCE" in sys_privs
        privileges["CREATE_TRIGGER"] = "CREATE TRIGGER" in sys_privs
        privileges["CREATE_MINING_MODEL"] = "CREATE MINING MODEL" in sys_privs

        # Check role privileges
        cursor.execute(
            """
            SELECT granted_role FROM user_role_privs
            WHERE granted_role = 'SODA_APP'
        """
        )
        roles = {row[0] for row in cursor.fetchall()}
        privileges["SODA_APP"] = "SODA_APP" in roles

        # Check if DBMS_VECTOR is accessible (try to describe it)
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM all_objects WHERE object_name = 'DBMS_VECTOR' AND owner = 'SYS'"
            )
            result = cursor.fetchone()
            privileges["DBMS_VECTOR"] = result[0] > 0 if result else False
        except Exception:
            privileges["DBMS_VECTOR"] = False

    except Exception:
        # If we can't check, assume minimal privileges
        pass
    finally:
        cursor.close()

    return privileges


def _is_connection_refused_error(error: Exception) -> bool:
    """
    Check if an error is a "Connection refused" error.

    Args:
        error: Exception object to check

    Returns:
        True if this is a connection refused error, False otherwise
    """
    error_str = str(error)
    error_repr = repr(error)

    # Check for common connection refused indicators
    connection_refused_indicators = [
        "Connection refused",
        "Errno 61",
        "[Errno 61]",
        "cannot connect to database",
        "DPY-6005",  # Oracle Python driver error code for connection issues
    ]

    return any(
        indicator in error_str or indicator in error_repr
        for indicator in connection_refused_indicators
    )


def _print_connection_refused_help(dsn: str):
    """
    Print helpful guidance when connection is refused.

    Args:
        dsn: Database connection string (to check if it's localhost)
    """
    is_localhost = "localhost" in dsn or "127.0.0.1" in dsn

    print("\n🔍 Connection Refused - Database Not Reachable")
    print("-" * 70)

    if is_localhost:
        print("The database at localhost is not reachable. This usually means:")
        print()
        print("  1. Docker is not running")
        print("     → Check: docker ps")
        print("     → Start Docker Desktop if needed")
        print()
        print("  2. Oracle container is not running")
        print("     → Check: docker ps -a | grep oracle")
        print("     → Start container: docker start oracle-memorizz")
        print("     → Or create new: memorizz oracle install --image lite")
        print()
        print("  3. Database is still starting up")
        print("     → Check logs: docker logs -f oracle-memorizz")
        print("     → Wait for: 'DATABASE IS READY TO USE!'")
        print()
        print("Quick fix:")
        print("  memorizz oracle install --image lite")
        print("  # Then wait for database to be ready before running setup again")
    else:
        print("The database at the specified DSN is not reachable. Please check:")
        print()
        print("  1. Database host is correct and accessible")
        print("  2. Network connectivity to the database server")
        print("  3. Firewall rules allow connections on port 1521")
        print("  4. Database service is running on the remote server")
        print()
        print(f"  DSN: {dsn}")


def _check_admin_capabilities(
    admin_user: str, admin_password: str, dsn: str, mode=None
) -> Tuple[bool, Optional[object], Dict[str, bool], Optional[str]]:
    """
    Check if admin connection is possible and what capabilities are available.

    Args:
        admin_user: Admin username
        admin_password: Admin password
        dsn: Database connection string
        mode: Connection mode (e.g., oracledb.SYSDBA for SYS as SYSDBA)

    Returns:
        Tuple of (can_connect, connection_object, capabilities_dict, error_message)
        If connection fails, returns (False, None, {}, error_message)
    """
    try:
        if mode is not None:
            admin_conn = oracledb.connect(
                user=admin_user, password=admin_password, dsn=dsn, mode=mode
            )
        else:
            admin_conn = oracledb.connect(
                user=admin_user, password=admin_password, dsn=dsn
            )
        can_create = _can_create_users(admin_conn)
        capabilities = {
            "can_create_users": can_create,
            "can_grant_privileges": can_create,  # Assumed if can create users
        }
        return True, admin_conn, capabilities, None
    except Exception as e:
        error_msg = str(e)
        return False, None, {}, error_msg


def _create_user_and_grant_privileges(
    admin_conn,
    memorizz_user: str,
    memorizz_password: str,
    dsn: str,
    admin_user: str = None,
) -> bool:
    """
    Create user and grant all required privileges (Admin mode only).

    Args:
        admin_conn: Admin database connection
        memorizz_user: Username to create
        memorizz_password: Password for new user
        dsn: Database connection string

    Returns:
        True if successful, False otherwise
    """
    admin_cursor = admin_conn.cursor()

    try:
        # Drop existing user if exists
        print(f"\nDropping existing {memorizz_user} user (if exists)...")
        try:
            # First, kill any active sessions for this user
            print("  Checking for active sessions...")
            admin_cursor.execute(
                f"""
                SELECT sid, serial# FROM v$session WHERE username = UPPER('{memorizz_user}')
            """
            )
            sessions = admin_cursor.fetchall()

            if sessions:
                print(f"  Found {len(sessions)} active session(s), terminating...")
                for sid, serial in sessions:
                    try:
                        admin_cursor.execute(
                            f"ALTER SYSTEM KILL SESSION '{sid},{serial}' IMMEDIATE"
                        )
                        print(f"    ✓ Killed session {sid},{serial}")
                    except Exception as e:
                        error_msg = str(e)
                        if "ORA-00031" in error_msg:
                            print(f"    ✓ Session {sid},{serial} marked for kill")
                        else:
                            print(f"    ⚠ Failed to kill session {sid},{serial}: {e}")

                # Reconnect if our connection was lost
                try:
                    admin_cursor.execute("SELECT 1 FROM DUAL")
                except Exception:
                    print("  Reconnecting to database...")
                    admin_cursor.close()
                    _safe_close_connection(admin_conn)
                    # Reconnect with proper admin user (could be sys as sysdba)
                    reconnect_admin_user = admin_user or os.environ.get(
                        "ORACLE_ADMIN_USER", "system"
                    )
                    admin_password = os.environ.get("ORACLE_ADMIN_PASSWORD")
                    if not admin_password:
                        raise RuntimeError(
                            "ORACLE_ADMIN_PASSWORD is required to reconnect as admin"
                        )
                    # Try to reconnect as sys if we were using sys
                    if reconnect_admin_user.lower() == "sys":
                        admin_conn = oracledb.connect(
                            user="sys",
                            password=admin_password,
                            dsn=dsn,
                            mode=oracledb.SYSDBA,
                        )
                    else:
                        admin_conn = oracledb.connect(
                            user=reconnect_admin_user,
                            password=admin_password,
                            dsn=dsn,
                        )
                    admin_cursor = admin_conn.cursor()
                    print("  ✓ Reconnected successfully")

                # Wait a moment for sessions to fully terminate
                print("  Waiting for sessions to terminate...")
                time.sleep(2)

                # Check again for remaining sessions
                admin_cursor.execute(
                    f"""
                    SELECT sid, serial# FROM v$session WHERE username = UPPER('{memorizz_user}')
                """
                )
                remaining_sessions = admin_cursor.fetchall()
                if remaining_sessions:
                    print(
                        f"  ⚠ {len(remaining_sessions)} session(s) still active, forcing termination..."
                    )
                    for sid, serial in remaining_sessions:
                        try:
                            admin_cursor.execute(
                                f"ALTER SYSTEM DISCONNECT SESSION '{sid},{serial}' IMMEDIATE"
                            )
                            print(f"    ✓ Disconnected session {sid},{serial}")
                        except Exception as e:
                            print(
                                f"    ⚠ Could not disconnect session {sid},{serial}: {e}"
                            )
                    time.sleep(1)
            else:
                print("  No active sessions found")

            # Now drop the user (with retry)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    admin_cursor.execute(f"DROP USER {memorizz_user} CASCADE")
                    print(f"  ✓ Dropped existing {memorizz_user} user")
                    break
                except Exception as e:
                    error_str = str(e)
                    if "ORA-01918" in error_str:
                        print("  ℹ User doesn't exist yet (this is fine)")
                        break
                    elif "ORA-01940" in error_str and attempt < max_retries - 1:
                        print(
                            f"  ⚠ User still has active connections, waiting and retrying ({attempt + 1}/{max_retries})..."
                        )
                        time.sleep(2)
                        # Try to disconnect any remaining sessions
                        try:
                            admin_cursor.execute(
                                f"""
                                SELECT sid, serial# FROM v$session WHERE username = UPPER('{memorizz_user}')
                            """
                            )
                            remaining = admin_cursor.fetchall()
                            for sid, serial in remaining:
                                try:
                                    admin_cursor.execute(
                                        f"ALTER SYSTEM DISCONNECT SESSION '{sid},{serial}' IMMEDIATE"
                                    )
                                except Exception:
                                    pass
                            time.sleep(1)
                        except Exception:
                            pass
                    else:
                        print(f"  ⚠ Could not drop user: {e}")
                        # Check if user exists - if it does and we can't drop it, we'll try to continue
                        try:
                            admin_cursor.execute(
                                f"SELECT username FROM all_users WHERE username = UPPER('{memorizz_user}')"
                            )
                            user_exists = admin_cursor.fetchone()
                            if user_exists:
                                print(
                                    f"  ℹ User {memorizz_user} exists but couldn't be dropped - will try to continue"
                                )
                        except Exception:
                            pass
                        break
        except Exception as e:
            if "ORA-01918" in str(e):
                print("  ℹ User doesn't exist yet (this is fine)")
            else:
                print(f"  ⚠ Error during user drop: {e}")

        # Create user (check if it already exists first)
        print(f"\nCreating {memorizz_user} user...")
        try:
            # Check if user already exists
            admin_cursor.execute(
                f"SELECT username FROM all_users WHERE username = UPPER('{memorizz_user}')"
            )
            user_exists = admin_cursor.fetchone()

            if user_exists:
                print(f"  ℹ User {memorizz_user} already exists")
                # Try to alter password in case it changed
                try:
                    admin_cursor.execute(
                        f'ALTER USER {memorizz_user} IDENTIFIED BY "{memorizz_password}"'
                    )
                    print(f"  ✓ Updated password for existing user {memorizz_user}")
                except Exception as e:
                    print(f"  ⚠ Could not update password: {e}")
            else:
                admin_cursor.execute(
                    f'CREATE USER {memorizz_user} IDENTIFIED BY "{memorizz_password}"'
                )
                print(f"  ✓ User {memorizz_user} created")
        except Exception as e:
            error_str = str(e)
            if "ORA-01920" in error_str:
                print(
                    f"  ℹ User {memorizz_user} already exists (name conflict resolved)"
                )
                # Try to alter password
                try:
                    admin_cursor.execute(
                        f'ALTER USER {memorizz_user} IDENTIFIED BY "{memorizz_password}"'
                    )
                    print(f"  ✓ Updated password for existing user {memorizz_user}")
                except Exception as alter_e:
                    print(f"  ⚠ Could not update password: {alter_e}")
            else:
                print(f"  ✗ Failed to create user: {e}")
                raise

        # Grant basic privileges (least-privilege principle)
        print("\nGranting basic privileges (least-privilege)...")
        admin_cursor.execute(f"GRANT CREATE SESSION TO {memorizz_user}")
        print("  ✓ CREATE SESSION (required for database connections)")

        admin_cursor.execute(f"GRANT CREATE TABLE TO {memorizz_user}")
        print("  ✓ CREATE TABLE (required for memory storage tables and indexes)")

        admin_cursor.execute(f"GRANT CREATE VIEW TO {memorizz_user}")
        print("  ✓ CREATE VIEW (required for Memorizz views)")

        admin_cursor.execute(f"GRANT CREATE SEQUENCE TO {memorizz_user}")
        print("  ✓ CREATE SEQUENCE (required for ID generation)")

        admin_cursor.execute(f"GRANT CREATE TRIGGER TO {memorizz_user}")
        print("  ✓ CREATE TRIGGER (required for view triggers)")

        admin_cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {memorizz_user}")
        print("  ✓ UNLIMITED TABLESPACE (required for data storage)")

        # Set default tablespace for VECTOR support (required for Oracle 23ai VECTOR types)
        # VECTOR types require automatic segment space management tablespace
        print("\nSetting default tablespace for VECTOR support...")
        try:
            env_tablespace = os.environ.get("ORACLE_TABLESPACE_NAME")
            preferred_tablespaces = []
            if env_tablespace:
                preferred_tablespaces.append(env_tablespace.upper())
            if "USERS" not in preferred_tablespaces:
                preferred_tablespaces.append("USERS")

            def _set_default_tablespace(ts_name: str) -> bool:
                admin_cursor.execute(
                    f"ALTER USER {memorizz_user} DEFAULT TABLESPACE {ts_name}"
                )
                admin_cursor.execute(
                    f"""
                    SELECT default_tablespace FROM dba_users
                    WHERE username = UPPER('{memorizz_user}')
                """
                )
                result = admin_cursor.fetchone()
                return bool(result and result[0] == ts_name.upper())

            default_tablespace_set = False

            for candidate in preferred_tablespaces:
                try:
                    if _set_default_tablespace(candidate):
                        print(
                            f"  ✓ Set default tablespace to {candidate} (supports VECTOR types)"
                        )
                        default_tablespace_set = True
                        break
                except Exception as candidate_error:
                    print(
                        f"  ⚠ Could not set {candidate} tablespace: {candidate_error}"
                    )

            if not default_tablespace_set:
                # Find any AUTO tablespace
                admin_cursor.execute(
                    """
                    SELECT tablespace_name FROM dba_tablespaces
                    WHERE segment_space_management = 'AUTO'
                    AND tablespace_name NOT IN ('SYSTEM', 'SYSAUX')
                    AND status = 'ONLINE'
                    ORDER BY tablespace_name
                """
                )
                auto_tablespace = admin_cursor.fetchone()
                if auto_tablespace:
                    ts_name = auto_tablespace[0]
                    try:
                        if _set_default_tablespace(ts_name):
                            print(
                                f"  ✓ Set default tablespace to {ts_name} (supports VECTOR types)"
                            )
                            default_tablespace_set = True
                    except Exception as auto_error:
                        print(
                            f"  ⚠ Could not set auto tablespace {ts_name}: {auto_error}"
                        )

            if not default_tablespace_set:
                # No suitable tablespace found, create one with explicit datafile path
                tablespace_name = (
                    env_tablespace.upper()
                    if env_tablespace
                    else f"{memorizz_user.upper()}_TS"
                )

                size_env = os.environ.get("ORACLE_TABLESPACE_SIZE_MB")
                autoextend_env = os.environ.get("ORACLE_TABLESPACE_AUTOEXTEND_MB")
                try:
                    tablespace_size_mb = (
                        max(int(size_env), 1) if size_env is not None else 100
                    )
                except ValueError:
                    print(
                        f"  ⚠ Invalid ORACLE_TABLESPACE_SIZE_MB='{size_env}', defaulting to 100"
                    )
                    tablespace_size_mb = 100

                try:
                    autoextend_mb = (
                        max(int(autoextend_env), 1)
                        if autoextend_env is not None
                        else 10
                    )
                except ValueError:
                    print(
                        f"  ⚠ Invalid ORACLE_TABLESPACE_AUTOEXTEND_MB='{autoextend_env}', defaulting to 10"
                    )
                    autoextend_mb = 10

                datafile_path, datafile_source = _determine_datafile_path(
                    admin_cursor, tablespace_name
                )
                if datafile_path:
                    escaped_path = _escape_sql_literal(datafile_path)
                    datafile_clause = (
                        f"DATAFILE '{escaped_path}' SIZE {tablespace_size_mb}M"
                    )
                    print(
                        f"  ℹ Using datafile path '{datafile_path}' ({datafile_source})"
                    )
                else:
                    datafile_clause = f"DATAFILE SIZE {tablespace_size_mb}M"
                    print(
                        "  ℹ No datafile directory detected; relying on Oracle Managed Files configuration"
                    )

                try:
                    create_sql = f"""
                        CREATE TABLESPACE {tablespace_name}
                        {datafile_clause}
                        AUTOEXTEND ON NEXT {autoextend_mb}M MAXSIZE UNLIMITED
                        SEGMENT SPACE MANAGEMENT AUTO
                    """
                    admin_cursor.execute(create_sql)
                    if _set_default_tablespace(tablespace_name):
                        print(
                            f"  ✓ Created and set default tablespace {tablespace_name} (supports VECTOR types)"
                        )
                        default_tablespace_set = True
                except Exception as create_error:
                    error_str2 = str(create_error)
                    if (
                        "ORA-01543" in error_str2
                        or "already exists" in error_str2.lower()
                    ):
                        try:
                            if _set_default_tablespace(tablespace_name):
                                print(
                                    f"  ✓ Set default tablespace to existing {tablespace_name} (supports VECTOR types)"
                                )
                                default_tablespace_set = True
                        except Exception as alter_error:
                            print(
                                f"  ⚠ Tablespace {tablespace_name} exists but could not be assigned: {alter_error}"
                            )
                    else:
                        print(f"  ⚠ Could not create tablespace: {create_error}")
                        print(
                            "    VECTOR types may not work - continuing anyway. See SETUP.md for manual tablespace steps."
                        )

            if not default_tablespace_set:
                print("  ⚠ Default tablespace could not be configured automatically")
                print(
                    "    VECTOR types may not work - user may need manual tablespace setup"
                )
        except Exception as e:
            print(f"  ⚠ Could not configure tablespace: {e}")
            print(
                "    VECTOR types may not work - user may need manual tablespace setup"
            )

        # Grant AI Vector Search privileges (Oracle 23ai+)
        print("\nGranting AI Vector Search privileges...")
        try:
            admin_cursor.execute(f"GRANT CREATE MINING MODEL TO {memorizz_user}")
            print("  ✓ CREATE MINING MODEL (required to install ONNX models)")

            admin_cursor.execute(f"GRANT EXECUTE ON DBMS_VECTOR TO {memorizz_user}")
            print("  ✓ EXECUTE ON DBMS_VECTOR (required for vector operations)")

            admin_cursor.execute(
                f"GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO {memorizz_user}"
            )
            print("  ✓ EXECUTE ON DBMS_VECTOR_CHAIN (required for vector chains)")
        except Exception as e:
            print(f"  ⚠ Vector privileges not available: {e}")
            print("    This is OK if not using Oracle 23ai+")

        # Grant JSON view privileges
        print("\nGranting JSON view privileges...")
        try:
            admin_cursor.execute(f"GRANT SODA_APP TO {memorizz_user}")
            print("  ✓ SODA_APP (required for Memorizz views)")
            print(
                "  ℹ SELECT ANY TABLE removed (not required, follows least-privilege)"
            )
        except Exception as e:
            print(f"  ⚠ Failed to grant view privileges: {e}")
            print("    Some features may not be available")

        # The production preflight reads only two dynamic performance views.
        # Grant those objects explicitly instead of broad catalog access so it
        # can report PDB state and VECTOR_MEMORY_SIZE under least privilege.
        print("\nGranting production preflight diagnostics...")
        for view_name in ("V_$PDBS", "V_$PARAMETER"):
            try:
                admin_cursor.execute(
                    f"GRANT SELECT ON SYS.{view_name} TO {memorizz_user}"
                )
                print(f"  ✓ SELECT ON SYS.{view_name}")
            except Exception as e:
                print(f"  ⚠ Could not grant SELECT ON SYS.{view_name}: {e}")
        print("  ℹ No broad SELECT_CATALOG_ROLE grant is required")

        admin_conn.commit()
        print("\n✓ User creation and privilege grants complete!")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        admin_conn.rollback()
        return False
    finally:
        admin_cursor.close()


def _create_schema(user_conn, schema_file: Path) -> Tuple[int, int, int]:
    """
    Create relational schema tables.

    Args:
        user_conn: User database connection
        schema_file: Path to schema SQL file

    Returns:
        Tuple of (success_count, skip_count, fail_count)
    """
    print("\nExecuting schema SQL...")

    # Get embedding dimension from environment variable
    embedding_dim = _get_embedding_dimension()
    if embedding_dim != 384:
        print(
            f"  ℹ Using custom embedding dimension: {embedding_dim} (from ORACLE_EMBEDDING_DIM)"
        )
    else:
        print(f"  ℹ Using default embedding dimension: {embedding_dim}")

    statements = parse_sql_file(schema_file, embedding_dim=embedding_dim)
    print(f"Found {len(statements)} SQL statements")

    user_cursor = user_conn.cursor()
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, stmt in enumerate(statements, 1):
        try:
            user_cursor.execute(stmt)
            success_count += 1
            if i % 10 == 0 or i == len(statements):
                print(f"  ✓ Executed {i}/{len(statements)} statements...")
        except Exception as e:
            error_str = str(e)
            if "ORA-00955" in error_str or "ORA-01408" in error_str:
                skip_count += 1
            else:
                fail_count += 1
                if fail_count <= 3:  # Only show first 3 errors
                    print(f"  ✗ Statement {i}: {e}")

    user_conn.commit()
    user_cursor.close()

    print(
        f"\n📊 Schema Summary: ✓ {success_count} success, ⚠ {skip_count} skipped, ✗ {fail_count} failed"
    )

    return success_count, skip_count, fail_count


def _verify_setup(user_conn) -> Tuple[list, list, list]:
    """
    Verify that setup completed successfully.

    Args:
        user_conn: User database connection

    Returns:
        Tuple of (tables_list, views_list, indexes_list)
    """
    user_cursor = user_conn.cursor()

    # Check tables
    print("\n📊 Relational Tables:")
    user_cursor.execute(
        """
        SELECT table_name FROM user_tables
        WHERE table_name IN ('AGENTS', 'AGENT_LLM_CONFIGS', 'AGENT_MEMORIES', 'PERSONAS',
                             'TOOLBOX', 'SKILLBOX', 'CONVERSATION_MEMORY', 'KNOWLEDGE_BASE',
                             'SHORT_TERM_MEMORY', 'WORKFLOW_MEMORY', 'SHARED_MEMORY',
                             'SUMMARIES', 'SEMANTIC_CACHE', 'ENTITY_MEMORY',
                             'AUTOMATION_JOBS', 'AUTOMATION_RUNS', 'AUTOMATION_DELIVERIES')
        ORDER BY table_name
    """
    )
    tables = [row[0] for row in user_cursor.fetchall()]
    for table in tables:
        print(f"  ✓ {table}")
    print(f"  Total: {len(tables)} tables")

    # Check views
    print("\n📄 Memorizz views:")
    user_cursor.execute(
        "SELECT view_name FROM user_views WHERE view_name LIKE '%_DV' ORDER BY view_name"
    )
    views = [row[0] for row in user_cursor.fetchall()]
    for view in views:
        print(f"  ✓ {view}")
    print(f"  Total: {len(views)} views")

    # Check vector indexes
    print("\n🔍 Vector Indexes:")
    user_cursor.execute(
        "SELECT index_name FROM user_indexes WHERE index_name LIKE 'IDX_%_VEC' ORDER BY index_name"
    )
    indexes = [row[0] for row in user_cursor.fetchall()]
    print(f"  Total: {len(indexes)} vector indexes")

    user_cursor.close()

    return tables, views, indexes


def setup_oracle_user():
    """
    Create and configure memorizz user in Oracle database.

    Supports two modes:
    - Admin mode: Full setup with user creation (for local/self-hosted databases)
    - User-only mode: Uses existing schema (for hosted databases like FreeSQL.com)

    The function automatically detects which mode to use based on available capabilities.
    """
    if oracledb is None:
        print("✗ oracledb package not found. Install with: pip install oracledb")
        return False

    # Configuration - can be overridden via environment variables
    ADMIN_USER = os.environ.get("ORACLE_ADMIN_USER", "system")
    ADMIN_PASSWORD = str(os.environ.get("ORACLE_ADMIN_PASSWORD", "") or "").strip()
    DSN = os.environ.get("ORACLE_DSN", "localhost:1521/FREEPDB1")

    # User to create/use - can be overridden via environment variables
    MEMORIZZ_USER = os.environ.get("ORACLE_USER", "memorizz_user")
    MEMORIZZ_PASSWORD = str(os.environ.get("ORACLE_PASSWORD", "") or "").strip()

    if not MEMORIZZ_PASSWORD:
        print("✗ ORACLE_PASSWORD is required; insecure default passwords are not used")
        return False

    # SQL files - resolve paths relative to this package
    PACKAGE_DIR = Path(__file__).parent
    SCHEMA_FILE = PACKAGE_DIR / "schema_relational.sql"

    print("=" * 70)
    print("Oracle Database Complete Setup for Memorizz")
    print("=" * 70)
    print()

    # Verify SQL files exist
    if not SCHEMA_FILE.exists():
        print(f"✗ Schema file not found: {SCHEMA_FILE}")
        print("  Please verify the package installation is correct")
        return False

    print(f"✓ Found schema file: {SCHEMA_FILE.name}")
    print()

    # ========== DETECT SETUP MODE ==========
    print("Detecting setup mode...")
    print("-" * 70)

    # Try to connect as admin first (SYSTEM)
    if ADMIN_PASSWORD:
        (
            can_connect_admin,
            admin_conn,
            admin_capabilities,
            admin_error,
        ) = _check_admin_capabilities(ADMIN_USER, ADMIN_PASSWORD, DSN)
    else:
        can_connect_admin = False
        admin_conn = None
        admin_capabilities = {}
        admin_error = None

    # Track which admin user we're using (for display purposes)
    active_admin_user = ADMIN_USER

    # If SYSTEM can connect but can't create users, try SYS as SYSDBA
    if can_connect_admin and not admin_capabilities.get("can_create_users", False):
        print(f"  Admin user '{ADMIN_USER}' cannot create users")
        print("  Trying SYS as SYSDBA (has full privileges)...")

        # Try SYS as SYSDBA (SYS password is same as SYSTEM in Oracle Free)
        try:
            # Check if SYSDBA mode is available
            if not hasattr(oracledb, "SYSDBA"):
                print(
                    "  ⚠ oracledb.SYSDBA not available (may need newer oracledb version)"
                )
                raise AttributeError("SYSDBA not available")

            sys_mode = oracledb.SYSDBA
            (
                can_connect_sys,
                sys_conn,
                sys_capabilities,
                sys_error,
            ) = _check_admin_capabilities("sys", ADMIN_PASSWORD, DSN, mode=sys_mode)

            if can_connect_sys and sys_capabilities.get("can_create_users", False):
                # Close SYSTEM connection and use SYS
                _safe_close_connection(admin_conn)
                admin_conn = sys_conn
                admin_capabilities = sys_capabilities
                active_admin_user = "sys"  # Update for display
                print("  ✓ Connected as SYS as SYSDBA (has CREATE USER privilege)")
            else:
                print("  ⚠ SYS as SYSDBA connection failed or cannot create users")
        except Exception as e:
            print(f"  ⚠ Could not try SYS as SYSDBA: {e}")

    setup_mode = None
    user_conn = None

    if can_connect_admin and admin_capabilities.get("can_create_users", False):
        # Admin mode: Full setup possible
        setup_mode = "admin"
        print("✓ Admin mode detected: Full setup with user creation")
        print(f"  Connected as: {active_admin_user}")
        print("  Can create users: Yes")
    else:
        # User-only mode: Use existing schema
        setup_mode = "user_only"
        print("ℹ User-only mode detected: Using existing schema")
        if not can_connect_admin:
            print("  Admin connection failed (this is OK for hosted databases)")
            if admin_error:
                # Check if it's a credential error (likely local setup issue)
                if (
                    "ORA-01017" in admin_error
                    or "invalid credential" in admin_error.lower()
                ):
                    print(f"\n  ⚠ Admin credential error: {admin_error}")
                    print("  This suggests the admin password may be incorrect.")
                    print("  For local Docker setup:")
                    print(
                        "    1. If you used `memorizz oracle install`, rerun that command"
                    )
                    print("       to display the connection environment variables")
                    print(
                        "    2. Set ORACLE_ADMIN_PASSWORD to the container admin password"
                    )
                    print(
                        "    3. Ensure the password matches the Oracle container password"
                    )
                    print("  MemoRizz never supplies a default database password")
                elif _is_connection_refused_error(Exception(admin_error)):
                    # Connection refused - already handled by _print_connection_refused_help
                    pass
                else:
                    print(f"  Error details: {admin_error}")
        else:
            print(
                f"  Admin user '{active_admin_user}' cannot create users (tried SYSTEM and SYS)"
            )
        print(f"  Will use existing user: {MEMORIZZ_USER}")

        # Try to connect as the regular user
        try:
            user_conn = oracledb.connect(
                user=MEMORIZZ_USER, password=MEMORIZZ_PASSWORD, dsn=DSN
            )
            print(f"  ✓ Connected as {MEMORIZZ_USER}")

            # Check user privileges
            user_privs = _check_user_privileges(user_conn)
            print("\n  User privileges:")
            for priv, available in user_privs.items():
                status = "✓" if available else "✗"
                print(f"    {status} {priv}")

            # Warn about missing privileges
            missing = [k for k, v in user_privs.items() if not v]
            if missing:
                print(f"\n  ⚠ Missing privileges: {', '.join(missing)}")
                print("    Some features may not be available")
        except Exception as e:
            print(f"  ✗ Failed to connect as {MEMORIZZ_USER}: {e}")

            # Check if this is a connection refused error
            if _is_connection_refused_error(e):
                _print_connection_refused_help(DSN)
            else:
                # For other errors, show credential/configuration guidance
                print("\nPlease check:")
                print("  1. ORACLE_USER environment variable is set correctly")
                print("  2. ORACLE_PASSWORD environment variable is set correctly")
                print("  3. ORACLE_DSN environment variable is set correctly")
                print("  4. User credentials are valid for the database")
                print("  5. Database service is running and accessible")

            _safe_close_connection(admin_conn)
            return False

    print()

    # ========== STEP 1: Create User (Admin Mode Only) ==========
    if setup_mode == "admin":
        print("STEP 1: Creating User and Granting Privileges")
        print("-" * 70)

        success = _create_user_and_grant_privileges(
            admin_conn, MEMORIZZ_USER, MEMORIZZ_PASSWORD, DSN, active_admin_user
        )

        if not success:
            _safe_close_connection(admin_conn)
            return False

        _safe_close_connection(admin_conn)
        print()

    # ========== STEP 2: Create Schema ==========
    print("STEP 2: Creating Relational Schema")
    print("-" * 70)

    # Connect as the user (if not already connected)
    if user_conn is None:
        print(f"Connecting as {MEMORIZZ_USER}...")
        try:
            user_conn = oracledb.connect(
                user=MEMORIZZ_USER, password=MEMORIZZ_PASSWORD, dsn=DSN
            )
            print("✓ Connected!")
        except Exception as e:
            print(f"✗ Connection failed: {e}")

            # Check if this is a connection refused error
            if _is_connection_refused_error(e):
                _print_connection_refused_help(DSN)
            else:
                # For other errors, show credential/configuration guidance
                print("\nPlease check:")
                print("  1. ORACLE_USER environment variable is set correctly")
                print("  2. ORACLE_PASSWORD environment variable is set correctly")
                print("  3. ORACLE_DSN environment variable is set correctly")
                print("  4. User credentials are valid for the database")
                print("  5. Database service is running and accessible")

            return False
    else:
        print(f"Using existing connection as {MEMORIZZ_USER}...")

    print(f"\nExecuting {SCHEMA_FILE.name}...")
    success_count, skip_count, fail_count = _create_schema(user_conn, SCHEMA_FILE)
    print()

    # ========== STEP 3: Verify Setup ==========
    print("STEP 3: Verifying Setup")
    print("-" * 70)

    tables, views, indexes = _verify_setup(user_conn)
    print()

    # ========== SUMMARY ==========
    print("=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    print()
    print("Connection details for your Memorizz application:")
    print(f"  User:     {MEMORIZZ_USER}")
    print("  Password: <configured in ORACLE_PASSWORD; redacted>")
    print(f"  DSN:      {DSN}")
    print()

    if setup_mode == "user_only":
        print("ℹ Setup Mode: User-only (existing schema)")
        print("  Some admin-granted privileges may not be available.")
        print("  Contact your database administrator if you need:")
        print("    - CREATE MINING MODEL (for in-database ONNX embeddings)")
        print("    - DBMS_VECTOR execute privileges (for vector search)")
        print("    - SODA_APP role (for Memorizz views)")
    else:
        print("ℹ Setup Mode: Admin (full setup)")

    print()
    print("Example usage in Python:")
    print()
    print("  import os")
    print("  from memorizz.memory_provider.oracle import OracleProvider, OracleConfig")
    print("  from memorizz.memagent.builders import MemAgentBuilder")
    print("  ")
    print("  # Create Oracle provider")
    print("  oracle_config = OracleConfig(")
    print(f'      user="{MEMORIZZ_USER}",')
    print('      password=os.environ["ORACLE_PASSWORD"],')
    print(f'      dsn="{DSN}",')
    print("      in_database_embedding=True,")
    print('      embedding_config={"model": "ALL_MINILM_L12_V2"}')
    print("  )")
    print("  oracle_provider = OracleProvider(oracle_config)")
    print("  ")
    print("  # Build your agent")
    print("  agent = (MemAgentBuilder()")
    print('      .with_instruction("You are a helpful assistant.")')
    print("      .with_memory_provider(oracle_provider)")
    print("      .with_llm_config({")
    print('          "provider": "openai",')
    print('          "model": "gpt-4o-mini",')
    print('          "api_key": "your-openai-key"')
    print("      })")
    print("      .build()")
    print("  )")
    print()
    print("Summary counts:")
    print(f"  - Tables: {len(tables)}/12 expected")
    print(f"  - views: {len(views)}/10 expected")
    print(f"  - Vector Indexes: {len(indexes)}/10 expected")

    if len(tables) >= 12 and len(views) >= 10:
        print("\n✅ All components created successfully!")
    else:
        print("\n⚠️  Some components may be missing. Check the logs above.")

    user_conn.close()
    print()

    return True


def apply_schema_updates() -> bool:
    """
    Apply schema changes in-place for an existing Oracle user/schema.

    This is the safe/non-destructive counterpart to setup_oracle_user(). It:
    - Connects as ORACLE_USER (no admin / no user drop)
    - Executes schema_relational.sql (skipping existing objects)
    - Verifies the resulting objects
    """
    if oracledb is None:
        print("✗ oracledb package not found. Install with: pip install oracledb")
        return False

    user = str(os.environ.get("ORACLE_USER", "memorizz_user") or "").strip()
    password = str(os.environ.get("ORACLE_PASSWORD", "") or "").strip()
    dsn = str(os.environ.get("ORACLE_DSN", "localhost:1521/FREEPDB1") or "").strip()
    if not user or not password or not dsn:
        print("✗ Missing Oracle env vars")
        print("  Required: ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN")
        return False

    package_dir = Path(__file__).parent
    schema_file = package_dir / "schema_relational.sql"

    if not schema_file.exists():
        print(f"✗ Schema file not found: {schema_file}")
        return False

    print("=" * 70)
    print("Oracle Schema Update (In-Place)")
    print("=" * 70)
    print(f"User: {user}")
    print(f"DSN:  {dsn}")
    print()

    try:
        conn = oracledb.connect(user=user, password=password, dsn=dsn)
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        if _is_connection_refused_error(e):
            _print_connection_refused_help(dsn)
        return False

    try:
        print("STEP 1: Applying relational schema updates")
        print("-" * 70)
        _create_schema(conn, schema_file)
        print()

        print("STEP 2: Verifying")
        print("-" * 70)
        _verify_setup(conn)
        print()
        print("✅ Schema update complete.")
        return True
    finally:
        _safe_close_connection(conn)
