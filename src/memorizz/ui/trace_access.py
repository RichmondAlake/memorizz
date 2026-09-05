"""Trace permissions and request-local tenant scope, independent of UI state."""

from contextvars import ContextVar
from dataclasses import dataclass

from fastapi import HTTPException

PERMISSIONS = {
    "viewer": frozenset({"trace.read"}),
    "analyst": frozenset({"trace.read", "artifact.lookup", "replay.create"}),
    "operator": frozenset(
        {"trace.read", "artifact.lookup", "account.resolve", "replay.create"}
    ),
    "admin": frozenset(
        {
            "trace.read",
            "trace.reveal",
            "artifact.lookup",
            "account.resolve",
            "replay.create",
            "trace.write",
        }
    ),
}


@dataclass(frozen=True)
class TracePrincipal:
    principal_id: str = "local"
    role: str = "admin"
    application_id: str | None = None
    user_id: str | None = None
    user_bound: bool = False

    def allows(self, permission):
        return permission in PERMISSIONS.get(self.role, frozenset())

    @property
    def restricted(self):
        return (
            self.role != "admin" or self.application_id is not None or self.user_bound
        )

    def filters(self):
        result = {}
        if self.application_id is not None:
            result["application_id"] = self.application_id
        if self.user_bound:
            result["user_id"] = self.user_id
        return result


current_principal = ContextVar("memorizz_trace_principal", default=TracePrincipal())


def require_trace_permission(permission):
    principal = current_principal.get()
    if not principal.allows(permission):
        raise HTTPException(status_code=403, detail="Trace permission denied")
    return principal


def scoped_trace_filters(filters=None):
    result = dict(filters or {})
    for key, value in current_principal.get().filters().items():
        if key in result and result[key] != value:
            raise HTTPException(
                status_code=403, detail="Requested trace scope is not authorized"
            )
        result[key] = value
    return result
