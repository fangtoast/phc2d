"""Shared lazy-dependency and shape helpers for model builders.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from math import prod
from typing import Any


def _require_torch(model_name: str) -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            f"Building model '{model_name}' requires optional dependency 'torch'. "
            "Install phc2d with the train extra or skip torch-only smoke tests."
        ) from exc
    return torch, nn


def _shape_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{name} is required")
    if isinstance(value, int):
        shape = (value,)
    else:
        shape = tuple(int(dim) for dim in value)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"{name} must contain positive dimensions, got {value!r}")
    return shape


def _numel(shape: tuple[int, ...]) -> int:
    return int(prod(shape))
