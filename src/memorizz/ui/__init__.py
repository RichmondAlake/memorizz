# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Memorizz Local UI - Web interface for exploring memory providers."""

from .app import create_app, run_server

__all__ = ["create_app", "run_server"]
