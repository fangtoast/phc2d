"""Compatibility exports for the split baseline model builders.

New code should import the family modules directly.  This module remains stable
for historical imports and registry adapters.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.models.band_cnn import build_lifted_cnn
from phc2d.models.band_mlp import build_mlp, build_pod_res_mlp, build_res_mlp
from phc2d.models.band_operators import build_fno2d_band, build_fno_lite
from phc2d.models.query import build_deep_onet_query, build_fno2d_query, build_mlp_query

__all__ = [
    "build_deep_onet_query",
    "build_fno2d_band",
    "build_fno2d_query",
    "build_fno_lite",
    "build_lifted_cnn",
    "build_mlp",
    "build_mlp_query",
    "build_pod_res_mlp",
    "build_res_mlp",
]
