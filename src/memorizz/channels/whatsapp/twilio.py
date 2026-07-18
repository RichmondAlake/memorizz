# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Twilio WhatsApp sender.

Uses Twilio Messages API via HTTPS (urllib), no extra dependencies.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict


def _normalize_whatsapp_address(value: str, *, field_name: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"'{field_name}' is required")

    text = raw
    if text.lower().startswith("whatsapp:"):
        text = text.split(":", 1)[1].strip()

    # Remove common formatting characters; Twilio expects E.164.
    number = re.sub(r"[\s\-()]", "", text)
    if number and not number.startswith("+") and number.isdigit():
        number = f"+{number}"

    if not (number.startswith("+") and number[1:].isdigit()):
        raise ValueError(
            f"{field_name} must be an E.164 number (e.g. +15551234567) optionally prefixed with 'whatsapp:'. Got: {raw!r}"
        )

    return f"whatsapp:{number}"


class TwilioWhatsAppSender:
    def __init__(self, *, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = _normalize_whatsapp_address(
            from_number, field_name="TWILIO_WHATSAPP_FROM"
        )

    @classmethod
    def from_env(cls) -> "TwilioWhatsAppSender":
        account_sid = str(os.environ.get("TWILIO_ACCOUNT_SID", "")).strip()
        auth_token = str(os.environ.get("TWILIO_AUTH_TOKEN", "")).strip()
        from_number = str(os.environ.get("TWILIO_WHATSAPP_FROM", "")).strip()
        if not account_sid or not auth_token or not from_number:
            raise RuntimeError(
                "Missing Twilio WhatsApp env vars: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM"
            )
        return cls(
            account_sid=account_sid, auth_token=auth_token, from_number=from_number
        )

    def send(self, *, to: str, body: str) -> Dict[str, Any]:
        to_value = _normalize_whatsapp_address(to, field_name="to")
        body_value = str(body or "")

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        form = urllib.parse.urlencode(
            {"From": self.from_number, "To": to_value, "Body": body_value}
        ).encode("utf-8")

        auth = f"{self.account_sid}:{self.auth_token}".encode("utf-8")
        auth_header = base64.b64encode(auth).decode("ascii")

        req = urllib.request.Request(url=url, data=form, method="POST")
        req.add_header("Authorization", f"Basic {auth_header}")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(payload) if payload else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"Twilio HTTP {exc.code}: {raw or exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Twilio send failed: {exc}") from exc

        provider_message_id = parsed.get("sid") or parsed.get("message_sid")
        return {
            "provider_message_id": provider_message_id,
            "response": parsed,
        }
