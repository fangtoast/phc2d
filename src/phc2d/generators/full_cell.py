"""Compatibility facade for full-cell structure generators.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.generators.full_cell_binary import (
    generate_binary_polar_holes,
    generate_binary_spectral_threshold,
    generate_unified_fourier_binary,
)
from phc2d.generators.full_cell_graded import (
    generate_graded_rbf_fourier,
    generate_unified_fourier_graded,
)
from phc2d.generators.full_cell_stats import full_cell_structure_stats
from phc2d.generators.full_cell_legacy import build_full_cell_structure_spec


__all__ = [
    "build_full_cell_structure_spec",
    "full_cell_structure_stats",
    "generate_binary_polar_holes",
    "generate_binary_spectral_threshold",
    "generate_graded_rbf_fourier",
    "generate_unified_fourier_binary",
    "generate_unified_fourier_graded",
]
