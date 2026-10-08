"""Content-free observability validation and scoped display pseudonyms."""

import hashlib
import hmac
import os
import re
import secrets

from ..redaction import EMAIL_ADDRESS, SECRET_TOKEN

_PROCESS_KEY = secrets.token_bytes(32)
# An opaque identifier must not carry an address, any bearer token, a vendor
# API key or a URI.
_SENSITIVE = re.compile(
    rf"(?:{EMAIL_ADDRESS.pattern}|\bBearer\s+\S+|{SECRET_TOKEN.pattern}|\w+://)",
    re.I,
)


def validate_opaque(value):
    if isinstance(value, str) and _SENSITIVE.search(value):
        raise ValueError(
            "telemetry identifiers and attributes must be content-free; use opaque IDs"
        )
    return value


def pseudonym(value, *, scope="user"):
    configured = os.getenv("MEMORIZZ_UI_PSEUDONYM_KEY", "")
    key = configured.encode() if configured else _PROCESS_KEY
    audit_scope = os.getenv("MEMORIZZ_UI_AUDIT_SCOPE", "local")
    digest = hmac.new(
        key, f"{audit_scope}:{scope}:{value}".encode(), hashlib.sha256
    ).hexdigest()
    return f"{scope}:{digest[:16]}"
