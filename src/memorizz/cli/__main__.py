# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Enable ``python -m memorizz.cli`` (back-compat with the old module path)."""

from .app import main

main()
