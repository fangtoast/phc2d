"""Real solver evaluation for analytic inverse-design candidates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.robust_gap import common_gap
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig, solve_bands_for_full_cell_mpb


def evaluate_sampled_candidate(
    structure: FullCellStructureSpec, *, k_points: np.ndarray,
    config: FullCellMpbBandSolverConfig, lower_band: int, target_frequency: float,
    solver: Callable[..., tuple[np.ndarray, dict[str, Any]]] = solve_bands_for_full_cell_mpb,
) -> dict[str, Any]:
    """Evaluate a single nominal candidate at exactly the supplied k points.

    The result is sampled solver evidence only: it has no convergence or
    manufacturing-robustness gate. Caller must record raw bands and metadata.
    """
    points = np.asarray(k_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("k_points must be a nonempty finite [k,2] array")
    if config.epsilon_air != structure.epsilon_air or config.epsilon_material != structure.epsilon_material:
        raise ValueError("solver and structure material identity mismatch")
    bands, metadata = solver(structure, k_points=points, config=config)
    bands = np.asarray(bands)
    if bands.shape != (len(points), config.num_bands):
        raise ValueError("solver output does not match k and band contract")
    if metadata.get("geometry_hash") != structure.geometry_hash:
        raise ValueError("solver geometry identity mismatch")
    gap = common_gap(bands[None, ...], lower_band=lower_band, target_frequency=target_frequency)
    return {"status": "sampled_solver_only", "parameter_hash": structure.parameter_hash,
            "geometry_hash": structure.geometry_hash, "k_points": points.tolist(),
            "bands": bands.tolist(), "gap": asdict(gap), "solver_metadata": dict(metadata)}
