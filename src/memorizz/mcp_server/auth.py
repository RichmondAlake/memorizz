# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Static bearer verification and request-principal extraction."""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from typing import FrozenSet, Optional

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken

from .config import ALL_SCOPES, StaticAPIKeyGrant


@dataclass(frozen=True)
class RequestIdentity:
    principal: Optional[str]
    scopes: FrozenSet[str]
    authenticated: bool


class StaticAPIKeyVerifier:
    """Constant-time verifier for operator-provided static bearer tokens."""

    def __init__(self, grants: list[StaticAPIKeyGrant], resource: str):
        self._grants = tuple(grants)
        self._resource = resource

    async def verify_token(self, token: str) -> AccessToken | None:
        for grant in self._grants:
            if hmac.compare_digest(token, grant.token):
                return AccessToken(
                    token=token,
                    client_id=f"memorizz:{grant.principal}",
                    subject=grant.principal,
                    scopes=list(grant.scopes),
                    resource=self._resource,
                )
        return None


def current_identity() -> RequestIdentity:
    token = get_access_token()
    if token is None:
        # A run-scoped stdio server may be launched by the trusted MemoRizz
        # meta-harness. Bind its otherwise anonymous local connection to the
        # exact tenant selected by the host; remote transports never use this
        # fallback because authenticated requests carry an access token.
        local_principal = str(
            os.getenv("MEMORIZZ_MCP_SERVER_LOCAL_PRINCIPAL", "")
        ).strip()
        return RequestIdentity(
            principal=local_principal or None,
            scopes=frozenset(ALL_SCOPES),
            authenticated=bool(local_principal),
        )
    return RequestIdentity(
        principal=token.subject or token.client_id,
        scopes=frozenset(token.scopes),
        authenticated=True,
    )
