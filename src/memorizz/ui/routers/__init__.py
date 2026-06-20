# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""FastAPI routers extracted from the monolithic ui/app.py.

Each module here exposes an ``APIRouter`` that ``create_app()`` mounts via
``app.include_router(...)``. This lets the UI grow domain-by-domain instead of
one ever-larger create_app body, and gives each group its own focused tests.
"""
