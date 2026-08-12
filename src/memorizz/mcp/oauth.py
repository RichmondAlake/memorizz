# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Cross-request OAuth coordination for the local Memorizz UI."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PendingOAuthFlow:
    owner_id: str
    server_name: str
    created_at: float = field(default_factory=time.time)
    authorization_url: Optional[str] = None
    state: Optional[str] = None
    code: Optional[str] = None
    callback_state: Optional[str] = None
    issuer: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    url_ready: threading.Event = field(default_factory=threading.Event)
    callback_ready: threading.Event = field(default_factory=threading.Event)
    completed: threading.Event = field(default_factory=threading.Event)


class OAuthFlowRegistry:
    """Process-local, short-lived OAuth state for UI authorization callbacks."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_state: Dict[str, PendingOAuthFlow] = {}

    def add(self, flow: PendingOAuthFlow) -> None:
        if not flow.state:
            raise ValueError("OAuth flow state is not available")
        with self._lock:
            self._purge_unlocked()
            self._by_state[flow.state] = flow

    def get(self, state: str) -> Optional[PendingOAuthFlow]:
        with self._lock:
            self._purge_unlocked()
            return self._by_state.get(state)

    def complete_callback(
        self,
        *,
        state: str,
        code: Optional[str],
        issuer: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[PendingOAuthFlow]:
        with self._lock:
            self._purge_unlocked()
            flow = self._by_state.pop(state, None)
        if not flow:
            return None
        flow.callback_state = state
        flow.code = code
        flow.issuer = issuer
        flow.error = error
        flow.callback_ready.set()
        return flow

    def _purge_unlocked(self) -> None:
        cutoff = time.time() - 900
        stale = [
            key for key, flow in self._by_state.items() if flow.created_at < cutoff
        ]
        for key in stale:
            self._by_state.pop(key, None)


oauth_flows = OAuthFlowRegistry()
