"""Solver adapter namespace for FEM, MPB, and future physics backends.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.solvers.mpb import (
    MpbBandSolverConfig,
    MpbEigenfieldResult,
    build_block_geometry_from_rho,
    solve_bands_for_rho_mpb,
    solve_eigenfield_bundle_for_rho_mpb,
    solve_eigenfield_multiband_bundle_for_rho_mpb,
    solve_eigenfield_for_rho_mpb,
)
from phc2d.solvers.mpb_parallel import (
    MPBParallelConfig,
    MPBWorkerJob,
    MPBWorkerResult,
    run_mpb_worker_jobs,
)
from phc2d.solvers.full_cell import (
    FullCellMpbBandSolverConfig,
    build_full_cell_material_function,
    solve_bands_for_full_cell_mpb,
    solve_circular_air_hole_mpb,
    solve_eigenfield_for_full_cell_mpb,
    solve_uniform_medium_mpb,
)

__all__ = [
    "MPBParallelConfig",
    "FullCellMpbBandSolverConfig",
    "MpbBandSolverConfig",
    "MpbEigenfieldResult",
    "MPBWorkerJob",
    "MPBWorkerResult",
    "build_block_geometry_from_rho",
    "build_full_cell_material_function",
    "run_mpb_worker_jobs",
    "solve_bands_for_rho_mpb",
    "solve_bands_for_full_cell_mpb",
    "solve_circular_air_hole_mpb",
    "solve_eigenfield_for_full_cell_mpb",
    "solve_eigenfield_bundle_for_rho_mpb",
    "solve_eigenfield_multiband_bundle_for_rho_mpb",
    "solve_eigenfield_for_rho_mpb",
    "solve_uniform_medium_mpb",
]
