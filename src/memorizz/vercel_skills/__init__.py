# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""Vercel Agent Skills integration for Memorizz.

Provides tools for searching the Vercel skills directory (skills.sh)
and fetching SKILL.md instruction files from GitHub repositories.
"""

from .provider import VercelSkillsProvider

__all__ = ["VercelSkillsProvider"]
