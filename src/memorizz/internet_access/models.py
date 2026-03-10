# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Data structures for standardized internet access responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class InternetSearchResult:
    """Normalized representation of a single search result."""

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict for tool / LLM consumption."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
            "metadata": self.metadata or None,
            "raw": self.raw or None,
        }


@dataclass
class InternetPageContent:
    """Normalized representation of page content scraped from the web."""

    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict for tool / LLM consumption."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata or None,
            "raw": self.raw or None,
        }
