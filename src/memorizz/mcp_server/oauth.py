# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""OAuth sign-in for the remote MCP server, for cloud apps.

Grok Bot, ChatGPT and its dots reach an MCP server only through a public HTTPS
URL and sign in with OAuth. They identify themselves with a client metadata
document (an HTTPS ``client_id``) or dynamic client registration; the owner
approves each one on an approval page with a pairing code from
``memorizz mcp apps pair`` or the local UI. Each approved app acts as the
principal the code names, so apps paired to one principal share one memory.

ChatGPT signs its token requests with a key (``private_key_jwt``); other
metadata-document clients may use none. PKCE (S256) is always required.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import logging
import secrets
import socket
import time
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse, urlsplit

from mcp.server.auth.handlers.authorize import AuthorizationHandler
from mcp.server.auth.handlers.metadata import (
    MetadataHandler,
    ProtectedResourceMetadataHandler,
)
from mcp.server.auth.handlers.register import RegistrationHandler
from mcp.server.auth.handlers.revoke import RevocationHandler
from mcp.server.auth.handlers.token import TokenHandler
from mcp.server.auth.middleware.client_auth import (
    AuthenticationError,
    ClientAuthenticator,
)
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.routes import build_metadata, cors_middleware
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthToken,
    ProtectedResourceMetadata,
)
from pydantic import AnyUrl, ValidationError
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from .config import LOCAL_AGENTS_PRINCIPAL, READ_SCOPE, WRITE_SCOPE
from .oauth_store import MAX_PAIRING_ATTEMPTS, OAuthStore

logger = logging.getLogger(__name__)

OAUTH_SCOPES = (READ_SCOPE, WRITE_SCOPE)
ACCESS_TOKEN_SECONDS = 3600
REFRESH_TOKEN_SECONDS = 90 * 86400
CODE_SECONDS = 300
APPROVE_PATH = "/oauth/approve"
JWT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
_SIGNING_ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "ES256", "ES384", "EdDSA"]
_DOCUMENT_MAX_BYTES = 64 * 1024
# A cached client metadata document is read again after a day.
_DOCUMENT_SECONDS = 86400
_ASSERTION_MAX_SECONDS = 600
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _same_loopback_redirect(requested: Any, registered: Any) -> bool:
    """True when two local redirects differ at most in their port."""
    asked, known = urlsplit(str(requested)), urlsplit(str(registered))
    return (
        asked.scheme == known.scheme == "http"
        and asked.hostname in _LOOPBACK_HOSTS
        and asked.hostname == known.hostname
        and asked.username is None
        and asked.password is None
        and (asked.path, asked.query) == (known.path, known.query)
    )


class MemorizzClient(OAuthClientInformationFull):
    """An app's OAuth record. A native app such as Claude Code listens for its
    sign-in on whichever local port is free, and may publish its redirects with
    no port at all, so a loopback redirect matches on any port (RFC 8252 7.3)."""

    def validate_redirect_uri(self, redirect_uri: Optional[AnyUrl]) -> AnyUrl:
        if redirect_uri is not None and any(
            _same_loopback_redirect(redirect_uri, known)
            for known in self.redirect_uris or []
        ):
            return redirect_uri
        return super().validate_redirect_uri(redirect_uri)


class MemorizzAuthorizationCode(AuthorizationCode):
    grant_id: str


class MemorizzRefreshToken(RefreshToken):
    grant_id: str
    resource: Optional[str] = None


def _origin(url: str) -> str:
    parsed = urlparse(str(url))
    return f"{parsed.scheme}://{parsed.netloc}".lower()


# Grok Bot and Grok sign in as "Cursor" (Cursor's shared client document, or
# a registration under that name); the callback says which app is connecting.
_GROK_CALLBACK_HOSTS = {"grok.com", "x.com", "chat.x.com", "lite.x.com"}


def _cursor_app(redirect_uri: str) -> Optional[str]:
    parsed = urlparse(redirect_uri)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "grokbot":
        return "Grok Bot"
    if host == "www.cursor.com" and parsed.path.startswith("/bot/"):
        return "Grok Bot"
    if parsed.scheme == "http" and host in ("localhost", "127.0.0.1"):
        return "Grok Bot"
    if parsed.scheme == "https" and host in _GROK_CALLBACK_HOSTS:
        return "Grok"
    return None


def app_label(
    client: OAuthClientInformationFull, redirect_uri: Optional[str] = None
) -> str:
    """What the approval page and the apps list call a client."""
    name = str(client.client_name or "").strip()
    if name == "Cursor" and redirect_uri:
        name = _cursor_app(str(redirect_uri)) or name
    if name:
        return name[:80]
    for url in (client.client_uri, client.client_id, *(client.redirect_uris or [])):
        host = urlparse(str(url or "")).hostname
        if host:
            return host
    return "An app"


def _is_metadata_url(client_id: str) -> bool:
    parsed = urlparse(str(client_id or ""))
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.path not in ("", "/")
    )


async def _public_host(host: str) -> bool:
    """True when every address the host resolves to is on the public internet."""
    try:
        infos = await asyncio.get_running_loop().run_in_executor(
            None, lambda: socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        )
    except OSError:
        return False
    addresses = {info[4][0] for info in infos}
    return bool(addresses) and all(
        ipaddress.ip_address(address.split("%")[0]).is_global for address in addresses
    )


async def fetch_json_document(url: str) -> Optional[Dict[str, Any]]:
    """GET a small public JSON document: HTTPS only, no redirects, no private hosts."""
    import httpx

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    if not await _public_host(parsed.hostname):
        logger.warning("Refusing a client document on a private host: %s", url)
        return None
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            async with client.stream(
                "GET", url, headers={"Accept": "application/json"}
            ) as response:
                if response.status_code != 200:
                    return None
                body = b""
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > _DOCUMENT_MAX_BYTES:
                        return None
    except httpx.HTTPError as exc:
        logger.warning("Could not read %s (%s)", url, type(exc).__name__)
        return None
    try:
        value = json.loads(body)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class MemorizzOAuthProvider:
    """The SDK's authorization-server provider, backed by :class:`OAuthStore`."""

    def __init__(
        self,
        store: OAuthStore,
        *,
        issuer: str,
        resource: str,
        fetch_json=None,
        allow_local_agents: bool = True,
    ):
        self.store = store
        # An app sharing the local agents' memory acts as the local owner, so
        # it may only sign in where just the memory tools are listed.
        self.allow_local_agents = allow_local_agents
        self.issuer = issuer.rstrip("/")
        self.resource = resource.rstrip("/")
        self.token_endpoint = self.issuer + "/token"
        self._fetch_json = fetch_json or fetch_json_document
        self._jwks: Dict[str, Dict[str, Any]] = {}

    # Clients

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        saved = self.store.client(client_id)
        if saved and saved["kind"] == "metadata_document":
            if time.time() - saved["updated_at"] > _DOCUMENT_SECONDS:
                fresh = await self._read_client_document(client_id)
                if fresh is not None:
                    return fresh
        if saved:
            try:
                return MemorizzClient.model_validate(saved["info"])
            except ValidationError:
                return None
        if _is_metadata_url(client_id):
            return await self._read_client_document(client_id)
        return None

    async def _read_client_document(
        self, client_id: str
    ) -> Optional[OAuthClientInformationFull]:
        document = await self._fetch_json(client_id)
        if not document or document.get("client_id") != client_id:
            return None
        supported = document.get("token_endpoint_auth_methods_supported")
        if not isinstance(supported, list):
            supported = [document.get("token_endpoint_auth_method") or "none"]
        usable = [m for m in ("private_key_jwt", "none") if m in supported]
        if not usable:
            logger.warning(
                "%s signs in with a method MemoRizz does not accept", client_id
            )
            return None
        preferred = document.get("token_endpoint_auth_method")
        method = preferred if preferred in usable else usable[0]
        jwks_uri = document.get("jwks_uri")
        if method == "private_key_jwt":
            # Keys must come from the same site as the document itself.
            if not document.get("jwks") and (
                not jwks_uri or _origin(jwks_uri) != _origin(client_id)
            ):
                return None
        try:
            client = MemorizzClient(
                client_id=client_id,
                redirect_uris=document.get("redirect_uris"),
                grant_types=document.get("grant_types")
                or ["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method=method,
                scope=" ".join(OAUTH_SCOPES),
                client_name=document.get("client_name"),
                client_uri=document.get("client_uri"),
                logo_uri=document.get("logo_uri"),
                jwks_uri=jwks_uri,
                jwks=document.get("jwks"),
            )
        except ValidationError:
            logger.warning("%s is not a valid client metadata document", client_id)
            return None
        if not client.redirect_uris:
            return None
        self.store.save_client(
            client_id, client.model_dump(mode="json"), "metadata_document"
        )
        return client

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.store.save_client(
            client_info.client_id, client_info.model_dump(mode="json"), "registered"
        )

    async def verify_client_assertion(
        self, client: OAuthClientInformationFull, assertion: str
    ) -> None:
        """Check a ``private_key_jwt`` assertion; raise AuthenticationError if not."""
        import jwt

        try:
            header = jwt.get_unverified_header(assertion)
        except jwt.PyJWTError as exc:
            raise AuthenticationError("The client assertion is not a JWT") from exc
        key = await self._signing_key(client, header.get("kid"))
        if key is None:
            raise AuthenticationError(
                "No key in the client's JWKS matches the assertion"
            )
        try:
            claims = jwt.decode(
                assertion,
                key,
                algorithms=_SIGNING_ALGORITHMS,
                audience=[self.token_endpoint, self.issuer, self.issuer + "/"],
                issuer=client.client_id,
                options={"require": ["exp", "iss", "sub", "aud"]},
                leeway=30,
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError(f"Invalid client assertion: {exc}") from exc
        if claims.get("sub") != client.client_id:
            raise AuthenticationError("The client assertion names another client")
        expires_at = float(claims["exp"])
        if expires_at - time.time() > _ASSERTION_MAX_SECONDS:
            raise AuthenticationError("The client assertion lives too long")
        jti = claims.get("jti")
        if jti and not self.store.use_assertion_id(
            client.client_id, str(jti), expires_at
        ):
            raise AuthenticationError("The client assertion was already used")

    async def _signing_key(
        self, client: OAuthClientInformationFull, kid: Optional[str]
    ):
        import jwt

        for refresh in (False, True):
            keys = await self._client_keys(client, refresh=refresh)
            matching = [k for k in keys if not kid or k.get("kid") == kid]
            if matching:
                try:
                    return jwt.PyJWK(matching[0]).key
                except jwt.PyJWTError:
                    return None
        return None

    async def _client_keys(
        self, client: OAuthClientInformationFull, *, refresh: bool
    ) -> List[Dict[str, Any]]:
        if isinstance(client.jwks, dict):
            return [k for k in client.jwks.get("keys") or [] if isinstance(k, dict)]
        uri = str(client.jwks_uri or "")
        if not uri or _origin(uri) != _origin(client.client_id):
            return []
        cached = self._jwks.get(uri)
        if refresh or not cached or time.time() - cached["at"] > 3600:
            document = await self._fetch_json(uri) or {}
            keys = [k for k in document.get("keys") or [] if isinstance(k, dict)]
            cached = {"keys": keys, "at": time.time()}
            self._jwks[uri] = cached
        return cached["keys"]

    # Authorization: hand the browser to the approval page.

    def _check_resource(self, resource: Optional[str]) -> str:
        """The MCP endpoint a token is for; some clients name just the site."""
        if resource and resource.rstrip("/") not in (self.resource, self.issuer):
            raise AuthorizeError(
                error="invalid_target",
                error_description=f"This server only issues tokens for {self.resource}",
            )
        return self.resource

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        self._check_resource(params.resource)
        request_id = self.store.create_sign_in_request(
            client.client_id,
            {
                "state": params.state,
                "scopes": params.scopes,
                "code_challenge": params.code_challenge,
                "redirect_uri": str(params.redirect_uri),
                "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
                "resource": params.resource,
            },
        )
        return f"{self.issuer}{APPROVE_PATH}?request={request_id}"

    def approve(
        self, sign_in: Dict[str, Any], pairing: Dict[str, Any], app_name: str
    ) -> str:
        """Record the approved app; return the client's redirect with a code."""
        params = sign_in["params"]
        grant_id = self.store.create_grant(
            sign_in["client_id"], app_name, pairing["principal"], pairing["scopes"]
        )
        code = secrets.token_urlsafe(32)
        self.store.save_code(
            code,
            grant_id,
            sign_in["client_id"],
            {
                "scopes": pairing["scopes"],
                "code_challenge": params["code_challenge"],
                "redirect_uri": params["redirect_uri"],
                "redirect_uri_provided_explicitly": params[
                    "redirect_uri_provided_explicitly"
                ],
                "resource": params.get("resource"),
                "principal": pairing["principal"],
            },
            time.time() + CODE_SECONDS,
        )
        self.store.finish_sign_in_request(sign_in["request_id"])
        return construct_redirect_uri(
            params["redirect_uri"], code=code, state=params.get("state")
        )

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[MemorizzAuthorizationCode]:
        row = self.store.code(authorization_code)
        if row is None or row["client_id"] != client.client_id:
            return None
        payload = row["payload"]
        return MemorizzAuthorizationCode(
            code=authorization_code,
            scopes=payload["scopes"],
            expires_at=row["expires_at"],
            client_id=row["client_id"],
            code_challenge=payload["code_challenge"],
            redirect_uri=payload["redirect_uri"],
            redirect_uri_provided_explicitly=payload[
                "redirect_uri_provided_explicitly"
            ],
            resource=payload.get("resource"),
            subject=payload["principal"],
            grant_id=row["grant_id"],
        )

    def _issue(
        self, grant_id: str, client_id: str, scopes: Sequence[str], resource
    ) -> OAuthToken:
        access = "mza_" + secrets.token_urlsafe(32)
        refresh = "mzr_" + secrets.token_urlsafe(32)
        now = time.time()
        resource = (resource or self.resource).rstrip("/")
        self.store.save_token(
            access,
            "access",
            grant_id,
            client_id,
            scopes,
            resource,
            now + ACCESS_TOKEN_SECONDS,
        )
        self.store.save_token(
            refresh,
            "refresh",
            grant_id,
            client_id,
            scopes,
            resource,
            now + REFRESH_TOKEN_SECONDS,
        )
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_SECONDS,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: MemorizzAuthorizationCode,
    ) -> OAuthToken:
        if not self.store.take_code(authorization_code.code):
            raise TokenError(
                error="invalid_grant", error_description="The code was already used"
            )
        try:
            resource = self._check_resource(authorization_code.resource)
        except AuthorizeError as exc:
            raise TokenError(
                error="invalid_target", error_description=exc.error_description
            )
        return self._issue(
            authorization_code.grant_id,
            client.client_id,
            authorization_code.scopes,
            resource,
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> Optional[MemorizzRefreshToken]:
        row = self.store.token(refresh_token, "refresh")
        if row is None or row["client_id"] != client.client_id:
            return None
        return MemorizzRefreshToken(
            token=refresh_token,
            client_id=row["client_id"],
            scopes=row["scopes"],
            expires_at=int(row["expires_at"]) if row["expires_at"] else None,
            subject=row["principal"],
            grant_id=row["grant_id"],
            resource=row["resource"],
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: MemorizzRefreshToken,
        scopes: List[str],
    ) -> OAuthToken:
        if not self.store.delete_token(refresh_token.token):
            raise TokenError(
                error="invalid_grant",
                error_description="The refresh token was already used",
            )
        granted = [
            s for s in (scopes or refresh_token.scopes) if s in refresh_token.scopes
        ]
        return self._issue(
            refresh_token.grant_id, client.client_id, granted, refresh_token.resource
        )

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        if not token.startswith("mza_"):
            return None
        row = self.store.token(token, "access")
        if row is None or (row["resource"] and row["resource"] != self.resource):
            return None
        if row["principal"] == LOCAL_AGENTS_PRINCIPAL and not self.allow_local_agents:
            return None
        self.store.touch_grant(row["grant_id"])
        return AccessToken(
            token=token,
            client_id=row["client_id"],
            scopes=row["scopes"],
            expires_at=int(row["expires_at"]),
            resource=row["resource"],
            subject=row["principal"],
            claims={
                "iss": self.issuer,
                "app": row["app_name"],
                "grant_id": row["grant_id"],
            },
        )

    async def revoke_token(self, token) -> None:
        self.store.delete_token(token.token)


class MemorizzClientAuthenticator(ClientAuthenticator):
    """The SDK's client checks plus ``private_key_jwt`` (RFC 7523)."""

    provider: MemorizzOAuthProvider

    async def authenticate_request(
        self, request: Request
    ) -> OAuthClientInformationFull:
        form = await request.form()
        if form.get("client_assertion_type") or form.get("client_assertion"):
            if form.get("client_assertion_type") != JWT_ASSERTION_TYPE:
                raise AuthenticationError("Unsupported client_assertion_type")
            assertion = str(form.get("client_assertion") or "")
            client_id = str(form.get("client_id") or "")
            if not client_id:
                client_id = _unverified_subject(assertion)
                # The token handler reads client_id from the form too.
                request._form = FormData(
                    [*form.multi_items(), ("client_id", client_id)]
                )
            client = await self.provider.get_client(client_id)
            if client is None:
                raise AuthenticationError("Invalid client_id")
            if client.token_endpoint_auth_method != "private_key_jwt":
                raise AuthenticationError("This client does not sign its requests")
            await self.provider.verify_client_assertion(client, assertion)
            return client
        client = await super().authenticate_request(request)
        if client.token_endpoint_auth_method == "private_key_jwt":
            raise AuthenticationError("This client must sign its token requests")
        return client


def _unverified_subject(assertion: str) -> str:
    import jwt

    try:
        claims = jwt.decode(assertion, options={"verify_signature": False})
    except jwt.PyJWTError as exc:
        raise AuthenticationError("The client assertion is not a JWT") from exc
    return str(claims.get("sub") or claims.get("iss") or "")


class MemorizzTokenVerifier:
    """Static operator tokens first, then tokens issued to approved apps."""

    def __init__(self, static=None, provider: Optional[MemorizzOAuthProvider] = None):
        self._static = static
        self._provider = provider

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        if self._static is not None:
            found = await self._static.verify_token(token)
            if found is not None:
                return found
        if self._provider is not None:
            return await self._provider.load_access_token(token)
        return None


# The approval page

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connect {app} to MemoRizz</title>
<style>
:root {{ color-scheme: light dark; --bg:#f6f7f9; --card:#fff; --text:#15171c; --muted:#5d6472;
  --line:#dde1e7; --accent:#2f5bea; --danger:#b42318; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d0f13; --card:#161920; --text:#e8eaef;
  --muted:#9aa2b1; --line:#2a2f3a; --accent:#7b9bff; --danger:#ff8a80; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; min-height:100vh; display:grid; place-items:center; padding:16px;
  background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,sans-serif; }}
main {{ width:100%; max-width:420px; background:var(--card); border:1px solid var(--line);
  border-radius:14px; padding:24px; }}
h1 {{ font-size:20px; margin:0 0 8px; }}
p {{ margin:0 0 14px; color:var(--muted); }}
ul {{ margin:0 0 16px; padding-left:20px; }}
label {{ display:block; font-weight:600; margin-bottom:6px; }}
input {{ width:100%; font:600 20px/1 ui-monospace,monospace; letter-spacing:.12em; text-transform:uppercase;
  padding:12px; border:1px solid var(--line); border-radius:10px; background:transparent; color:var(--text); }}
.row {{ display:flex; gap:10px; margin-top:16px; }}
button {{ flex:1; font:600 15px system-ui,sans-serif; padding:11px; border-radius:10px; cursor:pointer;
  border:1px solid var(--line); background:transparent; color:var(--text); }}
button.allow {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
.error {{ color:var(--danger); font-weight:600; }}
code {{ font-family:ui-monospace,monospace; font-size:13px; }}
</style></head><body><main>{body}</main></body></html>"""


def _page(app: str, body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(
        _PAGE.format(app=html.escape(app), body=body),
        status_code=status,
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
            "Referrer-Policy": "no-referrer",
        },
    )


def _approval_form(app: str, site: str, request_id: str, error: str = "") -> str:
    shown = html.escape(app)
    return (
        f"<h1>Connect {shown} to your MemoRizz memory?</h1>"
        f"<p>{shown} ({html.escape(site)}) asks to:</p>"
        "<ul><li>search and read your memories</li>"
        "<li>save new memories and update existing ones</li></ul>"
        "<p>Other apps you connect share the same memory. You can disconnect it any "
        "time in MemoRizz under Settings → Connected apps.</p>"
        + (f'<p class="error">{html.escape(error)}</p>' if error else "")
        + f'<form method="post" action="{APPROVE_PATH}">'
        f'<input type="hidden" name="request" value="{html.escape(request_id)}">'
        '<label for="code">Pairing code</label>'
        '<input id="code" name="code" autocomplete="one-time-code" autocapitalize="characters" '
        'spellcheck="false" maxlength="12" placeholder="ABCD-EFGH" autofocus>'
        '<p style="margin-top:8px">Get one from MemoRizz → Settings → Connected apps, '
        "or run <code>memorizz mcp apps pair</code>.</p>"
        '<div class="row"><button type="submit" name="action" value="deny">Don\'t allow</button>'
        '<button class="allow" type="submit" name="action" value="allow">Allow</button></div>'
        "</form>"
    )


def _site(client: OAuthClientInformationFull, redirect_uri: str) -> str:
    """The web site the sign-in goes back to (app links like grokbot:// have none)."""
    urls = (redirect_uri, *(client.redirect_uris or []), client.client_uri)
    for url in (*urls, client.client_id):
        parsed = urlparse(str(url or ""))
        if parsed.scheme in ("http", "https") and parsed.hostname:
            return parsed.hostname
    return "unknown site"


def _approval_routes(provider: MemorizzOAuthProvider) -> List[Route]:
    store = provider.store

    async def deny(sign_in: Dict[str, Any], description: str) -> Response:
        store.finish_sign_in_request(sign_in["request_id"])
        params = sign_in["params"]
        return RedirectResponse(
            construct_redirect_uri(
                params["redirect_uri"],
                error="access_denied",
                error_description=description,
                state=params.get("state"),
            ),
            status_code=302,
        )

    async def approve_page(request: Request) -> Response:
        if request.method == "GET":
            request_id = request.query_params.get("request", "")
            form = None
        else:
            form = await request.form()
            request_id = str(form.get("request") or "")
        sign_in = store.sign_in_request(request_id)
        client = await provider.get_client(sign_in["client_id"]) if sign_in else None
        if sign_in is None or client is None:
            return _page(
                "an app",
                "<h1>This sign-in link has expired</h1>"
                "<p>Go back to the app and connect MemoRizz again.</p>",
                status=400,
            )
        redirect_uri = sign_in["params"]["redirect_uri"]
        app = app_label(client, redirect_uri)
        site = _site(client, redirect_uri)
        if form is None:
            return _page(app, _approval_form(app, site, request_id))
        if form.get("action") != "allow":
            return await deny(sign_in, "The owner did not allow access")
        if store.count_pairing_attempt(request_id) > MAX_PAIRING_ATTEMPTS:
            return await deny(sign_in, "Too many wrong pairing codes")
        if store.pairing_locked():
            return _page(
                app,
                _approval_form(
                    app,
                    site,
                    request_id,
                    "Too many wrong pairing codes were tried recently. Wait ten minutes and try again.",
                ),
                status=429,
            )
        pairing = store.redeem_pairing_code(str(form.get("code") or ""))
        if pairing is None:
            return _page(
                app,
                _approval_form(
                    app,
                    site,
                    request_id,
                    "That pairing code is wrong, used or expired. Make a new one and try again.",
                ),
                status=400,
            )
        if (
            pairing["principal"] == LOCAL_AGENTS_PRINCIPAL
            and not provider.allow_local_agents
        ):
            return _page(
                app,
                _approval_form(
                    app,
                    site,
                    request_id,
                    "This server can't share your local agents' memory. Start it "
                    "with memorizz mcp apps serve, then make a new code.",
                ),
                status=400,
            )
        logger.info("Connected %s to MemoRizz as %s", app, pairing["principal"])
        return RedirectResponse(
            provider.approve(sign_in, pairing, app), status_code=302
        )

    return [Route(APPROVE_PATH, endpoint=approve_page, methods=["GET", "POST"])]


def oauth_routes(
    provider: MemorizzOAuthProvider,
    *,
    resource_path: str,
    documentation_url: Optional[str] = None,
) -> List[Route]:
    """Authorization-server routes, metadata, and the approval page."""
    # Plain strings, so the models keep the issuer exactly (no trailing "/");
    # clients compare it character for character.
    issuer = provider.issuer
    documentation = documentation_url or None
    registration = ClientRegistrationOptions(
        enabled=True,
        valid_scopes=list(OAUTH_SCOPES),
        default_scopes=list(OAUTH_SCOPES),
    )
    metadata = build_metadata(
        issuer,
        documentation,
        registration,
        RevocationOptions(enabled=True),
    )
    metadata.scopes_supported = list(OAUTH_SCOPES)
    metadata.client_id_metadata_document_supported = True
    metadata.token_endpoint_auth_methods_supported = [
        "none",
        "private_key_jwt",
        "client_secret_post",
        "client_secret_basic",
    ]
    metadata.token_endpoint_auth_signing_alg_values_supported = list(
        _SIGNING_ALGORITHMS
    )
    resource_metadata = ProtectedResourceMetadata(
        resource=provider.resource,
        authorization_servers=[issuer],
        scopes_supported=list(OAUTH_SCOPES),
        resource_name="MemoRizz memory",
        resource_documentation=documentation,
    )
    authenticator = MemorizzClientAuthenticator(provider)
    server_metadata = cors_middleware(
        MetadataHandler(metadata).handle, ["GET", "OPTIONS"]
    )
    protected = cors_middleware(
        ProtectedResourceMetadataHandler(resource_metadata).handle, ["GET", "OPTIONS"]
    )
    token = TokenHandler(provider, authenticator)
    return [
        Route(
            "/.well-known/oauth-authorization-server",
            endpoint=server_metadata,
            methods=["GET", "OPTIONS"],
        ),
        Route(
            "/.well-known/openid-configuration",
            endpoint=server_metadata,
            methods=["GET", "OPTIONS"],
        ),
        Route(
            "/.well-known/oauth-protected-resource" + resource_path,
            endpoint=protected,
            methods=["GET", "OPTIONS"],
        ),
        Route(
            "/.well-known/oauth-protected-resource",
            endpoint=protected,
            methods=["GET", "OPTIONS"],
        ),
        Route(
            "/authorize",
            endpoint=AuthorizationHandler(provider).handle,
            methods=["GET", "POST"],
        ),
        Route(
            "/token",
            endpoint=cors_middleware(token.handle, ["POST", "OPTIONS"]),
            methods=["POST", "OPTIONS"],
        ),
        Route(
            "/register",
            endpoint=cors_middleware(
                RegistrationHandler(provider, options=registration).handle,
                ["POST", "OPTIONS"],
            ),
            methods=["POST", "OPTIONS"],
        ),
        Route(
            "/revoke",
            endpoint=cors_middleware(
                RevocationHandler(provider, authenticator).handle, ["POST", "OPTIONS"]
            ),
            methods=["POST", "OPTIONS"],
        ),
        *_approval_routes(provider),
    ]
