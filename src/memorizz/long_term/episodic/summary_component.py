# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""
Summary Component for MemAgent

Provides a structured approach to working with memory summaries that compress
multiple memory components into emotionally and situationally relevant content.
"""

import time
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SummaryComponent(BaseModel):
    """
    A structured representation of a memory summary.

    Summaries compress multiple memory components from a time period into
    emotionally and situationally relevant content using an LLM.
    """

    memory_id: str
    agent_id: str
    summary_content: str
    period_start: float
    period_end: float
    memory_units_count: int
    created_at: float
    embedding: Optional[List[float]] = None
    user_id: Optional[str] = None

    # Optional metadata
    summary_type: str = "automatic"  # automatic, manual, scheduled
    compression_ratio: Optional[float] = None  # original_count / summarized_count
    emotional_tags: Optional[List[str]] = None  # emotional themes identified
    situational_tags: Optional[List[str]] = None  # situational contexts
    importance_score: Optional[float] = None  # 0.0 to 1.0 relevance score

    def __init__(self, **data):
        """Initialize summary component with current timestamp if not provided."""
        if "created_at" not in data:
            data["created_at"] = time.time()
        super().__init__(**data)

    @property
    def period_start_datetime(self) -> datetime:
        """Get period start as a datetime object."""
        return datetime.fromtimestamp(self.period_start)

    @property
    def period_end_datetime(self) -> datetime:
        """Get period end as a datetime object."""
        return datetime.fromtimestamp(self.period_end)

    @property
    def created_datetime(self) -> datetime:
        """Get creation time as a datetime object."""
        return datetime.fromtimestamp(self.created_at)

    @property
    def period_duration_hours(self) -> float:
        """Get the duration of the summarized period in hours."""
        return (self.period_end - self.period_start) / 3600

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "SummaryComponent":
        """Create from dictionary loaded from storage."""
        return cls(**data)

    def get_short_preview(self, max_length: int = 100) -> str:
        """Get a short preview of the summary content."""
        if len(self.summary_content) <= max_length:
            return self.summary_content
        return self.summary_content[:max_length] + "..."

    def add_emotional_tag(self, tag: str):
        """Add an emotional tag to the summary."""
        if self.emotional_tags is None:
            self.emotional_tags = []
        if tag not in self.emotional_tags:
            self.emotional_tags.append(tag)

    def add_situational_tag(self, tag: str):
        """Add a situational tag to the summary."""
        if self.situational_tags is None:
            self.situational_tags = []
        if tag not in self.situational_tags:
            self.situational_tags.append(tag)

    def calculate_compression_ratio(self, original_memory_count: int):
        """Calculate and set the compression ratio."""
        if original_memory_count > 0:
            self.compression_ratio = (
                original_memory_count / 1
            )  # Summary is 1 compressed item

    def __str__(self) -> str:
        """String representation of the summary."""
        return f"Summary({self.memory_id}, {self.period_start_datetime.strftime('%Y-%m-%d')} to {self.period_end_datetime.strftime('%Y-%m-%d')}, {self.memory_units_count} memories)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"SummaryComponent(memory_id='{self.memory_id}', agent_id='{self.agent_id}', period='{self.period_start_datetime}' to '{self.period_end_datetime}', memories={self.memory_units_count})"


class SummaryMetrics(BaseModel):
    """
    Metrics and analytics for summary generation and usage.
    """

    total_summaries: int = 0
    total_memories_compressed: int = 0
    average_compression_ratio: float = 0.0
    most_common_emotional_tags: List[str] = []
    most_common_situational_tags: List[str] = []
    persona_updates_triggered: int = 0

    def add_summary(self, summary: SummaryComponent):
        """Add a summary to the metrics."""
        self.total_summaries += 1
        self.total_memories_compressed += summary.memory_units_count

        if summary.compression_ratio:
            current_total = self.average_compression_ratio * (self.total_summaries - 1)
            self.average_compression_ratio = (
                current_total + summary.compression_ratio
            ) / self.total_summaries

    def get_compression_efficiency(self) -> float:
        """Get overall compression efficiency."""
        if self.total_summaries == 0:
            return 0.0
        return self.total_memories_compressed / self.total_summaries
