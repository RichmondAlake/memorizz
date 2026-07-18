# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Twilio webhook signature validation.

Follows Twilio's webhook validation spec:
https://www.twilio.com/docs/usage/security#validating-requests
"""

import base64
import hashlib
import hmac
from typing import Dict


def validate_twilio_signature(
    auth_token: str, url: str, post_params: Dict[str, str], signature: str
) -> bool:
    """
    Validate Twilio webhook signature.

    Args:
        auth_token: Twilio auth token
        url: Full webhook URL (including protocol and query string)
        post_params: POST parameters from the webhook request
        signature: X-Twilio-Signature header value

    Returns:
        True if signature is valid, False otherwise
    """
    if not auth_token or not signature:
        return False

    try:
        # 1. Start with the full URL
        data_string = url

        # 2. Sort parameters alphabetically and append to URL
        # Twilio includes all POST parameters in the signature calculation
        sorted_params = sorted(post_params.items())
        for key, value in sorted_params:
            data_string += key + value

        # 3. Compute HMAC-SHA1 with auth_token as the key
        computed_signature = base64.b64encode(
            hmac.new(
                auth_token.encode("utf-8"), data_string.encode("utf-8"), hashlib.sha1
            ).digest()
        ).decode("utf-8")

        # 4. Compare signatures (constant-time comparison to prevent timing attacks)
        return hmac.compare_digest(computed_signature, signature)

    except Exception:
        return False
