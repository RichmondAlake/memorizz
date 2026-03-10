# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Channel sender protocol."""

from __future__ import annotations

from typing import Any, Dict, Protocol


class ChannelSender(Protocol):
    def send(self, to: str, body: str) -> Dict[str, Any]:
        ...
