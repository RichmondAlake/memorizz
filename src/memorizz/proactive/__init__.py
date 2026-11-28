# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Proactive agents: an agent that offers a next step, and may take it.

``MemAgent(proactive=...)`` turns it on. See :class:`ProactivePolicy` for the
levels, :class:`ProactiveEngine` for how a suggestion is decided, and
:mod:`memorizz.proactive.guard` for what an agent may do without being asked.
"""

from .engine import ProactiveEngine, reconcile_store, run_worker
from .guard import CAREFUL, OPEN, acting, active_guard, guarded_policy
from .policy import LEVELS, ProactivePolicy
from .store import ProactiveStore
from .suggestions import Suggestion

__all__ = [
    "CAREFUL",
    "LEVELS",
    "OPEN",
    "ProactiveEngine",
    "ProactivePolicy",
    "ProactiveStore",
    "Suggestion",
    "acting",
    "active_guard",
    "guarded_policy",
    "reconcile_store",
    "run_worker",
]
