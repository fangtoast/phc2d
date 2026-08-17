"""Binary full-cell generator interfaces, including preserved v1 logic.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.generators.full_cell_legacy import (
    generate_binary_polar_holes,
    generate_binary_spectral_threshold,
)


__all__ = [
    "generate_binary_polar_holes",
    "generate_binary_spectral_threshold",
    "generate_unified_fourier_binary",
]
