"""Compatibility exports for split TE Bloch eigenfield model builders.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.models.field_baselines import build_fno_field, build_unet_field
from phc2d.models.field_conditioned import build_fno_field_v2, build_unet_field_v2

__all__ = [
    "build_fno_field",
    "build_fno_field_v2",
    "build_unet_field",
    "build_unet_field_v2",
]
