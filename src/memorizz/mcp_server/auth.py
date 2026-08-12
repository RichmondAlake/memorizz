# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Static bearer verification and request-principal extraction."""

from __future__ import annotations

import hmac
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
        return RequestIdentity(
            principal=None,
            scopes=frozenset(ALL_SCOPES),
            authenticated=False,
        )
    return RequestIdentity(
        principal=token.subject or token.client_id,
        scopes=frozenset(token.scopes),
        authenticated=True,
    )
