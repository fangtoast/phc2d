"""Coordinator for family-scoped built-in model catalog entries.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Callable

from phc2d.models.registry import ModelFactory, ModelSpec, is_registered, register_model

CatalogProvider = Callable[[], tuple[tuple[str, ModelFactory, ModelSpec], ...]]


def _catalog_providers() -> tuple[CatalogProvider, ...]:
    from phc2d.models.catalog.field import model_specs as field_specs
    from phc2d.models.catalog.forward import model_specs as forward_specs
    from phc2d.models.catalog.full_cell import model_specs as full_cell_specs
    from phc2d.models.catalog.query import model_specs as query_specs

    return field_specs, forward_specs, query_specs, full_cell_specs


def register_builtin_models() -> None:
    """Register built-ins without importing optional torch implementations."""

    for provider in _catalog_providers():
        for name, factory, spec in provider():
            if not is_registered(name):
                register_model(name, factory, spec=spec)
