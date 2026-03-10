# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""
MemAgent module.

This module provides a maintainable structure while maintaining
100% backward compatibility with existing code.
"""

# Import core components
from .core import MemAgent

# Optional: Import managers for advanced users
from .managers import (
    CacheManager,
    MemoryManager,
    PersonaManager,
    ToolManager,
    WorkflowManager,
)
from .models import MemAgentConfig, MemAgentModel

# Export all public APIs
__all__ = [
    "MemAgent",
    "MemAgentModel",
    "MemAgentConfig",
    "MemoryManager",
    "ToolManager",
    "CacheManager",
    "PersonaManager",
    "WorkflowManager",
]
