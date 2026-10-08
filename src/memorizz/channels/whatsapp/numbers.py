# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""E.164 normalisation shared by every WhatsApp address handler."""

from __future__ import annotations

import re
from typing import Any

_FORMATTING = re.compile(r"[\s\-()]")


def normalize_whatsapp_number(
    value: Any, *, strict: bool = True, field_name: str = "WhatsApp number"
) -> str:
    """The E.164 number (``+15551234567``) in a WhatsApp address.

    Accepts an optional ``whatsapp:`` prefix, surrounding whitespace and
    common formatting characters (spaces, hyphens, parentheses); a bare run of
    digits gains the leading ``+``. With ``strict`` (the default) an empty or
    non-E.164 value raises ``ValueError`` naming ``field_name``. Without it
    the cleaned text comes back with a leading ``+`` whatever it contains.
    """
    raw = str(value or "").strip()
    text = raw
    if text.lower().startswith("whatsapp:"):
        text = text.split(":", 1)[1].strip()
    number = _FORMATTING.sub("", text)
    if number and not number.startswith("+") and number.isdigit():
        number = f"+{number}"
    if number.startswith("+") and number[1:].isdigit():
        return number
    if not strict:
        return number if number.startswith("+") else f"+{number}"
    if not raw:
        raise ValueError(f"'{field_name}' is required")
    raise ValueError(
        f"{field_name} must be an E.164 number (e.g. +15551234567) optionally "
        f"prefixed with 'whatsapp:'. Got: {raw!r}"
    )


__all__ = ["normalize_whatsapp_number"]
