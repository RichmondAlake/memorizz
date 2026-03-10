# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Sandbox code execution provider interfaces and implementations.

Supported providers:
- **e2b** (default): Cloud sandbox using Firecracker microVMs.
- **daytona**: Cloud dev environments with unlimited runtime and GPU support.
- **graalpy**: Local sandbox using GraalVM's Python runtime (no cloud dependency).
"""

import logging

from .base import (
    SandboxProvider,
    create_sandbox_provider,
    get_provider_class,
    register_provider,
)
from .models import ExecutionResult
from .providers.daytona_provider import DaytonaSandboxProvider
from .providers.e2b_provider import E2BSandboxProvider
from .providers.graalpy_provider import GraalPySandboxProvider

logger = logging.getLogger(__name__)

__all__ = [
    "SandboxProvider",
    "ExecutionResult",
    "E2BSandboxProvider",
    "DaytonaSandboxProvider",
    "GraalPySandboxProvider",
    "create_sandbox_provider",
    "register_provider",
    "get_provider_class",
]
