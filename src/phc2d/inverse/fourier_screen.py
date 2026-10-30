"""Screen analytic Fourier candidates with a frozen band predictor.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from phc2d.data.full_cell_raster import build_full_cell_raster_pyramid
from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.morphology import periodic_geometry_gate
from phc2d.inverse.robust_gap import common_gap


@dataclass(frozen=True)
class CandidateScreen:
    """One candidate's surrogate-only result and screening status."""

    sample_id: str
    parameter_hash: str
    material_fraction: float
    status: str
    predicted_gap: dict[str, Any] | None
    band_order_violations: int | None
    geometry_gate: dict[str, Any] | None = None


def screen_fourier_candidate(
    structure: FullCellStructureSpec, predictor: Any, *,
    raster_method: str, source_resolution: int, input_resolution: int,
    material_fraction_bounds: tuple[float, float], lower_band: int,
    target_frequency: float, geometry_constraints: dict[str, Any] | None = None,
) -> CandidateScreen:
    """Predict a nominal structure on the fixed model k grid.

    Optional geometry constraints use periodic raster diagnostics. Manufacturing
    perturbations and MPB validity require separate gates.
    The predictor must accept an already contract-matched material raster.
    """
    if structure.generator_family != "unified_fourier_binary" or structure.material_kind != "binary":
        raise ValueError("expected unified Fourier binary structure")
    low, high = material_fraction_bounds
    if not (0 <= low <= high <= 1):
        raise ValueError("invalid material fraction bounds")
    raster = build_full_cell_raster_pyramid(
        structure, resolutions=[input_resolution], source_resolution=source_resolution,
        method=raster_method,
    ).grids[input_resolution]
    fraction = float(np.mean(raster, dtype=np.float64))
    identity = dict(sample_id=structure.sample_id, parameter_hash=structure.parameter_hash,
                    material_fraction=fraction)
    if not low <= fraction <= high:
        return CandidateScreen(**identity, status="material_fraction_rejected", predicted_gap=None,
                               band_order_violations=None)
    geometry = None
    if geometry_constraints is not None:
        master = structure.rasterize((source_resolution, source_resolution))
        geometry = periodic_geometry_gate(master, **geometry_constraints)
        if geometry['status'] != 'passed':
            return CandidateScreen(**identity, status='geometry_rejected', predicted_gap=None,
                                   band_order_violations=None, geometry_gate=geometry)
    import torch

    with torch.no_grad():
        prediction = predictor.predict_tensor(torch.as_tensor(raster[None, ...], dtype=torch.float32))
    bands = prediction.detach().cpu().numpy()
    if bands.ndim != 3 or bands.shape[0] != 1 or any(size == 0 for size in bands.shape):
        raise ValueError("predictor must return [1,k,band] frequencies")
    if not np.isfinite(bands).all():
        return CandidateScreen(**identity, status="nonfinite_prediction", predicted_gap=None,
                               band_order_violations=None, geometry_gate=geometry)
    violations = int(np.count_nonzero(np.diff(bands, axis=2) < 0))
    if violations:
        return CandidateScreen(**identity, status="band_order_violation", predicted_gap=None,
                               band_order_violations=violations, geometry_gate=geometry)
    if isinstance(lower_band, bool) or not isinstance(lower_band, (int, np.integer)) or not 1 <= lower_band < bands.shape[2]:
        raise ValueError("lower_band must select an adjacent band pair")
    pair = bands[:, :, lower_band - 1:lower_band + 1]
    if np.any(pair < 0):
        return CandidateScreen(**identity, status="negative_frequency", predicted_gap=None,
                               band_order_violations=0, geometry_gate=geometry)
    result = common_gap(pair, lower_band=1, target_frequency=target_frequency)
    return CandidateScreen(**identity, status="surrogate_only", predicted_gap=asdict(result),
                           band_order_violations=0, geometry_gate=geometry)
