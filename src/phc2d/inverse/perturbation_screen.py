"""Shared periodic manufacturing perturbations for proxy and MPB screening.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any, Callable

import numpy as np

from phc2d.data.full_cell_raster import block_mean_downsample
from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.morphology import periodic_binary_offset
from phc2d.inverse.robust_gap import common_gap
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig, solve_bands_for_periodic_binary_grid_mpb


def periodic_perturbation_grids(structure: FullCellStructureSpec, *, resolution: int,
                                offsets: tuple[float, ...]) -> tuple[np.ndarray, ...]:
    """Rasterize once and apply periodic offsets, in supplied order."""
    if structure.material_kind != 'binary' or resolution <= 0:
        raise ValueError('binary structure and positive resolution required')
    if not offsets or not np.isfinite(offsets).all() or len(set(offsets)) != len(offsets) or 0.0 not in offsets:
        raise ValueError('offsets must be finite, unique, nonempty and include nominal zero')
    base = structure.rasterize((resolution, resolution))
    if not np.isin(base, (0, 1)).all():
        raise ValueError('structure did not produce a binary master raster')
    return tuple(periodic_binary_offset(base, offset=float(offset)) for offset in offsets)


def predict_perturbed_gap(grids: tuple[np.ndarray, ...], predictor: Any, *,
                          input_resolution: int, lower_band: int, target_frequency: float) -> dict[str, Any]:
    """Compute a surrogate-only common interval using area-averaged inputs.

    The caller must verify the checkpoint uses this material-area representation.
    """
    if not grids:
        raise ValueError('at least one perturbation grid required')
    import torch

    inputs = np.stack([block_mean_downsample(g, input_resolution) for g in grids]).astype(np.float32)
    with torch.no_grad():
        bands = predictor.predict_tensor(torch.as_tensor(inputs)).detach().cpu().numpy()
    if bands.ndim != 3 or bands.shape[0] != len(grids) or not np.isfinite(bands).all():
        raise ValueError('predictor output must be finite [perturbation,k,band]')
    if not 1 <= lower_band < bands.shape[2] or np.any(np.diff(bands, axis=2) < 0):
        raise ValueError('invalid target band or predicted band order')
    pair = bands[:, :, lower_band-1:lower_band+1]
    result = common_gap(pair, lower_band=1, target_frequency=target_frequency)
    return {'status': 'surrogate_only', 'gap': asdict(result), 'bands': bands.tolist(),
            'grid_sha256': [hashlib.sha256(np.ascontiguousarray(g, dtype=np.uint8).tobytes()).hexdigest() for g in grids]}


def solve_perturbed_gap(grids: tuple[np.ndarray, ...], *, k_points: np.ndarray,
                        config: FullCellMpbBandSolverConfig, lower_band: int,
                        target_frequency: float,
                        solver: Callable[..., tuple[np.ndarray, dict[str, Any]]] = solve_bands_for_periodic_binary_grid_mpb) -> dict[str, Any]:
    """Compute sampled MPB common gap; no refinement or convergence implied."""
    if not grids:
        raise ValueError('at least one perturbation grid required')
    points = np.asarray(k_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError('k_points must be nonempty finite [k,2]')
    bands, metadata = [], []
    for grid in grids:
        values, meta = solver(grid, k_points=points, config=config)
        values = np.asarray(values)
        expected = hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest()
        if values.shape != (len(points), config.num_bands) or meta.get('grid_sha256') != expected:
            raise ValueError('solver output shape or grid identity mismatch')
        bands.append(values)
        metadata.append(meta)
    spectra = np.stack(bands)
    result = common_gap(spectra, lower_band=lower_band, target_frequency=target_frequency)
    return {'status': 'sampled_solver_only', 'gap': asdict(result), 'bands': spectra.tolist(),
            'k_points': points.tolist(), 'solver_metadata': metadata}
