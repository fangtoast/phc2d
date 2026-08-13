"""Structure generator registry for 2D photonic-crystal datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.generators.registry import (
    StructureSample,
    build_structure_generator,
    collect_structure_metadata,
)
from phc2d.generators.full_cell import (
    full_cell_structure_stats,
    generate_binary_polar_holes,
    generate_binary_spectral_threshold,
    generate_graded_rbf_fourier,
)

__all__ = [
    "StructureSample",
    "build_structure_generator",
    "collect_structure_metadata",
    "full_cell_structure_stats",
    "generate_binary_polar_holes",
    "generate_binary_spectral_threshold",
    "generate_graded_rbf_fourier",
]
