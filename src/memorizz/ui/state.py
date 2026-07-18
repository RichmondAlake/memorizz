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
from typing import Any, Dict, Optional

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


class CompatibleJinja2Templates(Jinja2Templates):
    """Render templates across both legacy and current Starlette signatures."""

    def TemplateResponse(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        media_type: Optional[str] = None,
        background: Any = None,
    ):
        context = context or {}
        request = context.get("request")
        try:
            return super().TemplateResponse(
                request=request,
                name=name,
                context=context,
                status_code=status_code,
                headers=headers,
                media_type=media_type,
                background=background,
            )
        except TypeError as exc:
            # Starlette before the request-first API does not accept a
            # ``request=`` keyword. Preserve Memorizz's documented FastAPI
            # minimum by falling back only for that signature mismatch.
            if "unexpected keyword argument 'request'" not in str(exc):
                raise
            return super().TemplateResponse(
                name,
                context,
                status_code=status_code,
                headers=headers,
                media_type=media_type,
                background=background,
            )


# Shared Jinja2 template engine for all HTML routes.
templates = CompatibleJinja2Templates(directory=str(TEMPLATES_DIR))
