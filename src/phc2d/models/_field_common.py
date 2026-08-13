"""Shared lazy torch and normalization helpers for field models.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Field models require optional torch.") from exc
    return torch, nn


def _group_count(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if int(channels) % candidate == 0:
            return candidate
    return 1
