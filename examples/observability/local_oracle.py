"""Create/check/test a dedicated local Oracle instance without existing stores.

Uses an explicitly selected, already-local image. Keeps the container and volume
for developer use. Credentials are generated in a private file, never printed.
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

NAME = "memorizz-observability-oracle"
VOLUME = NAME + "-data"
LABEL = "io.memorizz.purpose=isolated-observability-development"
STATE = (
    Path(__file__).resolve().parents[3] / "memorizz-oracle-local" / "connection.json"
)


def docker(*args, **kwargs):
    return subprocess.run(
        ["docker", *args], check=True, text=True, capture_output=True, **kwargs
    ).stdout.strip()


def verify(state):
    """Fail closed even under python -O before operating on any resource."""
    item = json.loads(docker("inspect", state["container_id"]))[0]
    expected_data = str(STATE.parent / "oradata")
    storage_matches = any(
        (
            m.get("Source") == expected_data
            if state.get("data_directory")
            else m.get("Name") == VOLUME
        )
        and m["Destination"] == "/opt/oracle/oradata"
        for m in item["Mounts"]
    )
    checks = [
        item["Name"] == "/" + NAME,
        item["Config"]["Labels"].get("io.memorizz.purpose")
        == "isolated-observability-development",
        item["Image"] == state["image_id"],
        item["HostConfig"]["PortBindings"]["1521/tcp"]
        == [{"HostIp": "127.0.0.1", "HostPort": "1529"}],
        state["dsn"] == "127.0.0.1:1529/FREEPDB1",
        state["dev_user"] == "MEMORIZZ_DEV",
        state["test_user"] == "MEMORIZZ_OBS_TEST_LOCAL",
        not state.get("data_directory") or state["data_directory"] == expected_data,
        storage_matches,
    ]
    if not all(checks):
        raise RuntimeError(
            "Oracle ownership, connection or storage verification failed; operation refused"
        )
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["create", "relocate", "repair-storage", "clean-test", "check", "test"],
    )
    parser.add_argument("--image", help="Existing local image ID/tag; never pulled")
    args = parser.parse_args()
    if args.action == "repair-storage":
        state = json.loads(STATE.read_text())
        verify(state)
        if not state.get("data_directory"):
            parser.error("Host-data relocation must be completed first")
        docker("stop", "--time=60", state["container_id"])
        mounts = ["-v", state["data_directory"] + ":/opt/oracle/oradata"]
        for name in ("diag", "admin"):
            target = STATE.parent / name
            target.mkdir(mode=0o777)
            target.chmod(0o777)
            docker(
                "cp", state["container_id"] + ":/opt/oracle/" + name + "/.", str(target)
            )
            mounts += ["-v", str(target) + ":/opt/oracle/" + name]
        docker("rm", state["container_id"])
        state["container_id"] = docker(
            "run",
            "-d",
            "--pull=never",
            "--name",
            NAME,
            "--label",
            LABEL,
            "--restart=no",
            "--memory=3g",
            "--cpus=2",
            "--shm-size=1g",
            "--log-opt=max-size=10m",
            "--log-opt=max-file=2",
            "-p",
            "127.0.0.1:1529:1521",
            *mounts,
            "--health-cmd=healthcheck.sh",
            "--health-interval=10s",
            "--health-timeout=5s",
            "--health-retries=30",
            state["image_id"],
        )
        state["host_diagnostics"] = True
        with STATE.open("w") as stream:
            json.dump(state, stream, indent=2)
        print(
            "Recreated only this Oracle container; database, audit and diagnostic files retained on private host storage.",
            flush=True,
        )
    elif args.action == "relocate":
        state = json.loads(STATE.read_text())
        item = verify(state)
        if (
            item["State"]["Running"]
            or item["State"].get("ExitCode") != 2
            or state.get("data_directory")
        ):
            parser.error("Relocation is only for the verified failed initial unpack")
        failed = subprocess.run(
            ["docker", "logs", state["container_id"]],
            check=True,
            text=True,
            capture_output=True,
        )
        logs = failed.stdout + failed.stderr
        if "No space left on device" not in logs:
            parser.error("Container did not fail with the expected disk-capacity error")
        docker("rm", state["container_id"])
        docker("volume", "rm", VOLUME)
        data_directory = STATE.parent / "oradata"
        data_directory.mkdir(mode=0o777)
        data_directory.chmod(
            0o777
        )  # Container UID; parent directory remains private (0700).
        state["data_directory"] = str(data_directory)
        environment = {
            **os.environ,
            "ORACLE_PASSWORD": state["system_password"],
            "APP_USER": state["dev_user"],
            "APP_USER_PASSWORD": state["dev_password"],
        }
        state["container_id"] = docker(
            "run",
            "-d",
            "--pull=never",
            "--name",
            NAME,
            "--label",
            LABEL,
            "--restart=no",
            "--memory=3g",
            "--cpus=2",
            "--shm-size=1g",
            "-p",
            "127.0.0.1:1529:1521",
            "-v",
            str(data_directory) + ":/opt/oracle/oradata",
            "-e",
            "ORACLE_PASSWORD",
            "-e",
            "APP_USER",
            "-e",
            "APP_USER_PASSWORD",
            "--health-cmd=healthcheck.sh",
            "--health-interval=10s",
            "--health-timeout=5s",
            "--health-retries=30",
            state["image_id"],
            env=environment,
        )
        with STATE.open("w") as stream:
            json.dump(state, stream, indent=2)
        print(
            "Removed only the failed setup's partial container/volume; recreated on private host storage.",
            flush=True,
        )
    elif args.action == "create":
        if not args.image:
            parser.error("create requires --image")
        if STATE.exists():
            parser.error(
                "State already exists; use check/test. No existing database is overwritten."
            )
        for kind, name in (("container", NAME), ("volume", VOLUME)):
            if (
                subprocess.run(
                    ["docker", kind, "inspect", name], capture_output=True
                ).returncode
                == 0
            ):
                parser.error(
                    f"{kind} {name} already exists; no existing resource is reused"
                )
        image = json.loads(docker("image", "inspect", args.image))[0]
        if image["Architecture"] != "arm64":
            parser.error("Select an existing native arm64 Oracle image")
        state = {
            "image_id": image["Id"],
            "container_name": NAME,
            "volume": VOLUME,
            "dsn": "127.0.0.1:1529/FREEPDB1",
            "dev_user": "MEMORIZZ_DEV",
            "test_user": "MEMORIZZ_OBS_TEST_LOCAL",
            "system_password": "M" + secrets.token_hex(20),
            "dev_password": "M" + secrets.token_hex(20),
            "test_password": "M" + secrets.token_hex(20),
        }
        STATE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(STATE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, indent=2)
        docker("volume", "create", "--label", LABEL, VOLUME)
        environment = {
            **os.environ,
            "ORACLE_PASSWORD": state["system_password"],
            "APP_USER": state["dev_user"],
            "APP_USER_PASSWORD": state["dev_password"],
        }
        state["container_id"] = docker(
            "run",
            "-d",
            "--pull=never",
            "--name",
            NAME,
            "--label",
            LABEL,
            "--restart=no",
            "--memory=3g",
            "--cpus=2",
            "--shm-size=1g",
            "-p",
            "127.0.0.1:1529:1521",
            "-v",
            VOLUME + ":/opt/oracle/oradata",
            "-e",
            "ORACLE_PASSWORD",
            "-e",
            "APP_USER",
            "-e",
            "APP_USER_PASSWORD",
            "--health-cmd=healthcheck.sh",
            "--health-interval=10s",
            "--health-timeout=5s",
            "--health-retries=30",
            state["image_id"],
            env=environment,
        )
        with STATE.open("w") as stream:
            json.dump(state, stream, indent=2)
        print(
            f"Created {NAME}; waiting for Oracle startup. Credentials: {STATE}",
            flush=True,
        )
    else:
        state = json.loads(STATE.read_text())
    item = verify(state)
    if args.action in {"create", "relocate", "repair-storage"}:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            item = verify(state)
            if not item["State"]["Running"]:
                raise RuntimeError("Isolated Oracle container stopped during startup")
            if item["State"].get("Health", {}).get("Status") == "healthy":
                break
            time.sleep(5)
        else:
            raise TimeoutError(
                "Oracle startup exceeded 10 minutes; container retained for inspection"
            )
    import oracledb

    with oracledb.connect(
        user="system", password=state["system_password"], dsn=state["dsn"]
    ) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "select sys_context('USERENV','CON_NAME'), banner_full from v$version"
        )
        pdb, version = cursor.fetchone()
        assert pdb == "FREEPDB1"
        if args.action in {"create", "relocate"}:
            cursor.execute(
                f'CREATE USER {state["test_user"]} IDENTIFIED BY "{state["test_password"]}" DEFAULT TABLESPACE USERS QUOTA 1G ON USERS'
            )
            cursor.execute(
                f'GRANT CREATE SESSION, CREATE TABLE, CREATE SEQUENCE TO {state["test_user"]}'
            )
        print(version.splitlines()[0])
    with oracledb.connect(
        user=state["dev_user"], password=state["dev_password"], dsn=state["dsn"]
    ) as conn:
        assert conn.cursor().execute("select 1 from dual").fetchone()[0] == 1
    print(
        f"Verified {NAME}: {state['dsn']}; development user {state['dev_user']}; isolated test schema {state['test_user']}",
        flush=True,
    )
    if args.action == "clean-test":
        from memorizz.observability.sql_index import TABLES

        with oracledb.connect(
            user=state["test_user"], password=state["test_password"], dsn=state["dsn"]
        ) as conn:
            cursor = conn.cursor()
            names = {
                row[0] for row in cursor.execute("select table_name from user_tables")
            }
            if not names <= {name.upper() for name in TABLES}:
                raise RuntimeError("Unexpected tables; cleanup refused")
            for table in sorted(names):
                cursor.execute(f"DROP TABLE {table} PURGE")
            print(
                f"Removed {len(names)} disposable test tables left by the interrupted test; development schema unchanged."
            )
    if args.action == "test":
        env = {
            **os.environ,
            "MEMORIZZ_OBSERVABILITY_LIVE": "1",
            "MEMORIZZ_OBS_TEST_ORACLE_DSN": state["dsn"],
            "MEMORIZZ_OBS_TEST_ORACLE_USER": state["test_user"],
            "MEMORIZZ_OBS_TEST_ORACLE_PASSWORD": state["test_password"],
        }
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/integration/test_observability_live.py",
                    "-k",
                    "oracle",
                    "-q",
                ],
                env=env,
            )
        )


if __name__ == "__main__":
    main()
