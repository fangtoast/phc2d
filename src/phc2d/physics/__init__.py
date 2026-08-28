"""Physics operators shared by field-eigen experiments.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.physics.eigen_residual import (
    canonicalize_complex_efield,
    derive_epsilon_on_field_grid,
    evaluate_te_bloch_eigen_residual,
    make_deterministic_random_efield,
)
from phc2d.physics.effective_medium import (
    FoldedSquareLatticeFrequencies,
    effective_refractive_index,
    folded_square_lattice_frequencies,
)
from phc2d.physics.torch_eigen_residual import (
    derive_epsilon_on_field_grid_torch,
    evaluate_te_bloch_eigen_residual_torch,
    pino_te_physics_loss,
)

__all__ = [
    "FoldedSquareLatticeFrequencies",
    "canonicalize_complex_efield",
    "derive_epsilon_on_field_grid",
    "evaluate_te_bloch_eigen_residual",
    "make_deterministic_random_efield",
    "effective_refractive_index",
    "folded_square_lattice_frequencies",
    "derive_epsilon_on_field_grid_torch",
    "evaluate_te_bloch_eigen_residual_torch",
    "pino_te_physics_loss",
]
