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

__all__ = [
    "MPBParallelConfig",
    "MpbBandSolverConfig",
    "MpbEigenfieldResult",
    "MPBWorkerJob",
    "MPBWorkerResult",
    "build_block_geometry_from_rho",
    "run_mpb_worker_jobs",
    "solve_bands_for_rho_mpb",
    "solve_eigenfield_bundle_for_rho_mpb",
    "solve_eigenfield_multiband_bundle_for_rho_mpb",
    "solve_eigenfield_for_rho_mpb",
]
