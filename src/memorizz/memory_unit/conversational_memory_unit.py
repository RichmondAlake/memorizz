# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ConversationMemoryUnit(BaseModel):
    role: str
    content: str
    timestamp: str
    memory_id: str
    thread_id: str
    embedding: list[float]
    user_id: Optional[str] = None
    recall_recency: Optional[float] = None
    associated_thread_ids: Optional[list[str]] = None
