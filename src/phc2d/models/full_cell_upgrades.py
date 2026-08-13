"""Compatibility exports for split project-native Full-Wh upgrades.

The stable import path is retained for existing training and evaluation code.
Concrete implementations now live in one family-focused module each.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.models.full_cell_convnext import build_convnext_band_v1
from phc2d.models.full_cell_mixer import build_mlp_mixer_band_v1
from phc2d.models.full_cell_operators import (
    build_dual_lane_band_operator_v1,
    build_tfno2d_band_v3,
)
from phc2d.models.full_cell_resmlp import build_res_mlp_gated_v2

__all__ = [
    "build_convnext_band_v1",
    "build_dual_lane_band_operator_v1",
    "build_mlp_mixer_band_v1",
    "build_res_mlp_gated_v2",
    "build_tfno2d_band_v3",
]
