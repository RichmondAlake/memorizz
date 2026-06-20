# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Shared UI runtime state and the Jinja2 template engine.

This module imports nothing from ``ui.app`` so that both ``create_app()`` and the
extracted routers in ``ui/routers/`` can share the single connected-provider
state and the template engine without a circular import.

``_state`` is mutated in place (its keys are reassigned, but the dict itself is
never rebound), so everyone importing it observes the same object.
"""

from pathlib import Path
from typing import Any, Dict

from fastapi.templating import Jinja2Templates

# Paths
UI_DIR = Path(__file__).parent
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"

# Global state for the connected memory provider.
_state: Dict[str, Any] = {
    "provider": None,
    "provider_type": None,
    "connection_info": {},
    "provider_secrets": {},
    "whatsapp_worker_thread": None,
    "whatsapp_worker_stop_event": None,
}

# Shared Jinja2 template engine for all HTML routes.
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
