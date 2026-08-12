# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Normalized browser-control results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BrowserControlResult:
    """Provider-neutral outcome from one bounded browser task."""

    success: bool
    task: str
    output: Any = None
    urls: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    steps: int = 0
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe result dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the result for a model tool response."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)
