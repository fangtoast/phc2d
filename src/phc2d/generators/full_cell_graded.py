"""Graded full-cell generator interfaces, including preserved v1 logic.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.generators.full_cell_latent import generate_unified_fourier_graded
from phc2d.generators.full_cell_legacy import generate_graded_rbf_fourier


__all__ = ["generate_graded_rbf_fourier", "generate_unified_fourier_graded"]
