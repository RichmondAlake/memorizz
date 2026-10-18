# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Open, read and operate the apps on this Mac.

``memorizz desktop serve`` exposes :class:`MacDesktop` as an MCP server, and
``memorizz desktop connect`` attaches it to the chat agent.
"""

from .macos import DesktopError, MacDesktop, parse_keys

__all__ = ["DesktopError", "MacDesktop", "parse_keys"]
