"""Package-owned local Oracle runtime bootstrap helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LocalOracleRuntime:
    """Describe and safely provision MemoRizz's local Oracle Free runtime."""

    user: str
    password: str
    dsn: str
    container_name: str = "memorizz_oracle"
    provision_if_missing: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        provision_if_missing: bool | None = None,
        container_name: str | None = None,
    ):
        user = os.getenv("ORACLE_USER", "").strip()
        password = os.getenv("ORACLE_PASSWORD", "").strip()
        dsn = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1").strip()
        if not user or not password or not dsn:
            raise ValueError(
                "ORACLE_USER, ORACLE_PASSWORD, and ORACLE_DSN are required"
            )
        provision = (
            _env_bool("MEMORIZZ_ORACLE_PROVISION_IF_MISSING")
            if provision_if_missing is None
            else bool(provision_if_missing)
        )
        return cls(
            user=user,
            password=password,
            dsn=dsn,
            container_name=(
                str(container_name).strip()
                if container_name is not None
                else os.getenv("MEMORIZZ_ORACLE_CONTAINER", "memorizz_oracle").strip()
            )
            or "memorizz_oracle",
            provision_if_missing=provision,
        )

    def ensure_ready(self) -> Dict[str, Any]:
        """Start/create the package's Docker runtime and wait for readiness."""
        # Reuse the same injection-safe argv based implementation as the UI,
        # keeping one container contract across SDK, CLI, and browser paths.
        from ...ui import docker_oracle

        if not docker_oracle.docker_available():
            raise RuntimeError(
                "Docker is not installed or its daemon is unavailable; start a "
                "container runtime or provision Oracle externally"
            )
        state = docker_oracle.get_container_state(self.container_name)
        action = "none"
        detail = "already running"
        if state == "stopped":
            action = "started"
            ok, detail = docker_oracle.start_container(self.container_name)
            if not ok:
                raise RuntimeError(detail)
        elif state == "absent":
            if not self.provision_if_missing:
                raise RuntimeError(
                    f"Oracle container '{self.container_name}' is absent; set "
                    "provision_if_missing=True to create it"
                )
            action = "created"
            port = docker_oracle.parse_port_from_dsn(self.dsn)
            ok, detail = docker_oracle.create_container(
                self.user,
                self.password,
                port,
                name=self.container_name,
            )
            if not ok:
                raise RuntimeError(detail)
        else:
            ok, detail = docker_oracle._wait_for_healthy(
                self.container_name, docker_oracle.READINESS_TIMEOUT_START
            )
            if not ok:
                raise RuntimeError(detail)
        return {
            "ok": True,
            "container": self.container_name,
            "state": "running",
            "action": action,
            "dsn": self.dsn,
            "detail": detail,
        }
