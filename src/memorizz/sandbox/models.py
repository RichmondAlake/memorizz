# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Data structures for standardized sandbox execution responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    """Normalized representation of a code execution result from any sandbox provider."""

    stdout: List[str] = field(default_factory=list)
    stderr: List[str] = field(default_factory=list)
    error: Optional[str] = None
    exit_code: Optional[int] = None
    results: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return True if execution completed without errors."""
        return self.error is None and self.exit_code in (None, 0)

    @property
    def output(self) -> str:
        """Return combined stdout as a single string."""
        return "\n".join(self.stdout) if self.stdout else ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict for tool / LLM consumption."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "exit_code": self.exit_code,
            "results": self.results,
            "success": self.success,
        }

    def to_json(self) -> str:
        """Return a JSON string for tool / LLM consumption."""
        return json.dumps(self.to_dict())
