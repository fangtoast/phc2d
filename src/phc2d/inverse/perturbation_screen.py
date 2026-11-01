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


def compare_predictor_devices(grids: tuple[np.ndarray, ...], predictor: Any, reference: Any, *,
                              input_resolution: int, absolute_tolerance: float) -> dict[str, Any]:
    """Compare full spectra on identical inputs; this is not a physical accuracy test."""
    import torch

    if (not grids or isinstance(absolute_tolerance, bool)
            or not isinstance(absolute_tolerance, (int, float))
            or not np.isfinite(absolute_tolerance) or absolute_tolerance < 0):
        raise ValueError('grids and finite nonnegative absolute tolerance required')
    if predictor.input_shape != reference.input_shape or predictor.output_shape != reference.output_shape:
        raise ValueError('predictor input or output contract differs')
    if (predictor.checkpoint_sha256 != reference.checkpoint_sha256
            or predictor.k_points_sha256 != reference.k_points_sha256):
        raise ValueError('predictor checkpoint or k-point identity differs')
    inputs = np.stack([block_mean_downsample(g, input_resolution) for g in grids]).astype(np.float32)
    with torch.no_grad():
        actual = predictor.predict_tensor(torch.as_tensor(inputs)).detach().cpu().numpy()
        expected = reference.predict_tensor(torch.as_tensor(inputs)).detach().cpu().numpy()
    if actual.shape != expected.shape or actual.shape != (len(grids), *predictor.output_shape):
        raise ValueError('device prediction shape differs from checkpoint')
    actual_finite, expected_finite = np.isfinite(actual), np.isfinite(expected)
    finite = bool(actual_finite.all() and expected_finite.all())
    maximum = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if finite else None
    return {'passed': bool(finite and maximum <= absolute_tolerance),
            'absolute_tolerance': float(absolute_tolerance), 'max_absolute_difference': maximum,
            'prediction_nonfinite_count': int((~actual_finite).sum()),
            'reference_nonfinite_count': int((~expected_finite).sum())}


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
    if bands.ndim != 3 or bands.shape[0] != len(grids) or any(n == 0 for n in bands.shape):
        raise ValueError('predictor output must be nonempty [perturbation,k,band]')
    if not 1 <= lower_band < bands.shape[2]:
        raise ValueError('invalid target band')
    hashes = [hashlib.sha256(np.ascontiguousarray(g, dtype=np.uint8).tobytes()).hexdigest() for g in grids]
    if not np.isfinite(bands).all():
        def encode(value: Any) -> float | str:
            if np.isnan(value):
                return 'nan'
            if np.isposinf(value):
                return '+inf'
            if np.isneginf(value):
                return '-inf'
            return float(value)
        spectrum = [[[encode(value) for value in point] for point in perturbation] for perturbation in bands]
        return {'status': 'nonfinite_prediction', 'gap': None, 'grid_sha256': hashes,
                'nonfinite_spectrum': spectrum}
    if np.any(np.diff(bands, axis=2) < 0):
        return {'status': 'band_order_violation', 'gap': None, 'bands': bands.tolist(), 'grid_sha256': hashes}
    pair = bands[:, :, lower_band-1:lower_band+1]
    if np.any(pair < 0):
        return {'status': 'negative_frequency', 'gap': None, 'bands': bands.tolist(), 'grid_sha256': hashes}
    result = common_gap(pair, lower_band=1, target_frequency=target_frequency)
    return {'status': 'surrogate_only', 'gap': asdict(result), 'bands': bands.tolist(),
            'grid_sha256': hashes}


def project_perturbed_band_order(prediction: dict[str, Any], *, lower_band: int,
                                 target_frequency: float) -> dict[str, Any]:
    """Score an explicitly transformed spectrum without changing its raw status.

    Sorting is a model variant, not evidence that the original prediction was
    valid or physically accurate. Negative non-target bands are retained and
    counted; a negative projected target band or nonfinite spectrum is rejected.
    """
    raw = np.asarray(prediction.get('bands'), dtype=np.float64)
    if raw.ndim != 3 or any(size == 0 for size in raw.shape) or not np.isfinite(raw).all():
        raise ValueError('finite raw [perturbation,k,band] spectrum required')
    if isinstance(lower_band, bool) or not isinstance(lower_band, (int, np.integer)) or not 1 <= lower_band < raw.shape[2]:
        raise ValueError('lower_band must select an adjacent band pair')
    hashes = prediction.get('grid_sha256')
    if not isinstance(hashes, list) or len(hashes) != raw.shape[0] or not all(isinstance(x, str) and x for x in hashes):
        raise ValueError('one grid identity per perturbation is required')
    violations = int(np.count_nonzero(np.diff(raw, axis=2) < 0))
    raw_pair = raw[:, :, lower_band-1:lower_band+1]
    expected = ('band_order_violation' if violations else
                'negative_frequency' if np.any(raw_pair < 0) else 'surrogate_only')
    if prediction.get('status') != expected:
        raise ValueError('raw prediction status disagrees with spectrum')
    projected = np.sort(raw, axis=2)
    pair = projected[:, :, lower_band-1:lower_band+1]
    if np.any(pair < 0):
        raise ValueError('projected target bands contain negative frequency')
    gap = common_gap(pair, lower_band=1, target_frequency=target_frequency)
    return {'status': 'projected_surrogate_only', 'original_status': expected,
            'transformation': 'ascending_per_k_and_perturbation',
            'raw_band_order_violations': violations,
            'raw_negative_frequency_count': int(np.count_nonzero(raw < 0)),
            'projection_max_abs_change': float(np.max(np.abs(projected - raw))),
            'grid_sha256': list(hashes), 'raw_bands': raw.tolist(),
            'projected_bands': projected.tolist(), 'gap': asdict(gap)}


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
