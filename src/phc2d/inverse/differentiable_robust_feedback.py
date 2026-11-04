"""Bind and fit per-scenario physical feedback for robust design.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.inverse.composite_feedback import composite_edge_observation
from phc2d.inverse.local_edge_correction import fit_composite_edge_correction


def bind_scenario_feedback(prediction: Mapping[str, Any], result: Mapping[str, Any], *,
                           expected_case_index: int, expected_parameter_hash: str,
                           offsets: Sequence[float], manifest_sha256: str, protocol_sha256: str,
                           surrogate_contract_sha256: str, k_points: Any,
                           lower_band: int, beta: float) -> list[dict[str, Any]]:
    """Bind frozen hard/soft predictions to gated physical bands for each offset."""
    values = [float(value) for value in offsets]
    if (len(values) != 3 or len(set(values)) != 3 or 0.0 not in values or not np.isfinite(values).all()
            or prediction.get('status') != 'computed' or result.get('case_index') != expected_case_index
            or result.get('parameter_hash') != expected_parameter_hash
            or result.get('manifest_sha256') != manifest_sha256 or result.get('protocol_sha256') != protocol_sha256
            or result.get('aggregate', {}).get('status') != 'numerically_stable'
            or not surrogate_contract_sha256):
        raise ValueError('robust feedback identity or numerical status mismatch')
    traces, hard = result['observations'], prediction['hard']
    if (len(traces) != len(values) or [float(trace['offset']) for trace in traces] != values
            or hard.get('status') != 'surrogate_only' or len(hard['grid_sha256']) != len(values)):
        raise ValueError('robust feedback scenario roster mismatch')
    fields = ('hard_edges', 'soft_exact_edges', 'soft_smooth_edges')
    arrays = {field: np.asarray(prediction[field], dtype=np.float64) for field in fields}
    if any(array.shape != (len(values), 2) or not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError('invalid frozen robust scenario edges')
    points = np.asarray(k_points, dtype=float)
    hard_bands, soft_bands = np.asarray(hard['bands'], dtype=float), np.asarray(prediction['soft_bands'], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or hard_bands.shape != soft_bands.shape or hard_bands.ndim != 3
            or hard_bands.shape[:2] != (len(values), len(points)) or lower_band < 1
            or hard_bands.shape[2] <= lower_band or not np.isfinite(hard_bands).all()
            or not np.isfinite(soft_bands).all() or not np.isfinite(beta) or beta <= 0):
        raise ValueError('invalid robust feedback spectra or k-point contract')
    expected_hard = np.stack((hard_bands[:, :, lower_band-1].max(axis=1),
                              hard_bands[:, :, lower_band].min(axis=1)), axis=1)
    low, high = soft_bands[:, :, lower_band-1], soft_bands[:, :, lower_band]
    low_max, high_min = low.max(axis=1), high.min(axis=1)
    exact = np.stack((low_max, high_min), axis=1)
    smooth = np.stack((low_max + np.log(np.exp(beta * (low-low_max[:, None])).sum(axis=1))/beta,
                       high_min - np.log(np.exp(-beta * (high-high_min[:, None])).sum(axis=1))/beta), axis=1)
    if (not np.allclose(expected_hard, arrays['hard_edges'], rtol=0, atol=1e-6)
            or not np.allclose(exact, arrays['soft_exact_edges'], rtol=0, atol=1e-6)
            or not np.allclose(smooth, arrays['soft_smooth_edges'], rtol=0, atol=1e-6)):
        raise ValueError('frozen robust edges do not reproduce from spectra')
    rows = []
    for i, trace in enumerate(traces):
        provenance, row = trace['provenance'], trace['row']
        if (provenance.get('evaluated_grid_sha256') != hard['grid_sha256'][i]
                or provenance.get('parameter_hash') != expected_parameter_hash
                or not provenance.get('comparison_contract_sha256')
                or not all(row.get(key) is True for key in
                           ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged'))):
            raise ValueError('robust feedback physical grid or gate mismatch')
        actual_points = np.asarray(trace['arrays']['k_points'], dtype=float)
        actual_bands = np.asarray(trace['arrays']['bands'], dtype=float)
        if (actual_points.ndim != 2 or len(actual_points) < len(points)
                or not np.allclose(actual_points[:len(points)], points, rtol=0, atol=1e-7)
                or actual_bands.ndim != 2 or actual_bands.shape[0] < len(points)
                or actual_bands.shape[1] != hard_bands.shape[2] or not np.isfinite(actual_bands[:len(points)]).all()):
            raise ValueError('robust feedback physical spectra or k-points mismatch')
        contract = {'refinement': provenance['comparison_contract_sha256'], 'offset': values[i],
                    'physical_protocol_sha256': protocol_sha256}
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                            allow_nan=False).encode()).hexdigest()
        observation = composite_edge_observation(parameter_hash=prediction['parameter_hash'],
            parameters=prediction['point'], physical_edges=[row['refined_lower_edge_frequency'],
                                                          row['refined_upper_edge_frequency']],
            hard_edges=arrays['hard_edges'][i], soft_exact_edges=arrays['soft_exact_edges'][i],
            soft_smooth_edges=arrays['soft_smooth_edges'][i], comparison_contract_sha256=digest,
            surrogate_contract_sha256=surrogate_contract_sha256, numerical_gate_passed=True)
        rows.append({'offset': values[i], 'physical_grid_sha256': hard['grid_sha256'][i],
                     'source_case_index': expected_case_index, 'observation': observation})
    return rows


def fit_scenario_corrections(observations: Sequence[Mapping[str, Any]], *, offsets: Sequence[float],
                             center: Any, radius: float, ridge: float) -> list[Any]:
    """Fit each manufacturing scenario using only its own physical observations."""
    values = [float(value) for value in offsets]
    if len(values) != len(set(values)) or not values or not np.isfinite(values).all():
        raise ValueError('invalid correction offsets')
    result = []
    for offset in values:
        local = [row['observation'] for row in observations if row['offset'] == offset
                 and np.linalg.norm(np.asarray(row['observation']['parameters'], dtype=float) - np.asarray(center, dtype=float))
                 <= radius + 1e-12]
        if not local:
            raise ValueError('no local physical feedback for one manufacturing scenario')
        result.append(fit_composite_edge_correction(local, center=center, radius=radius, ridge=ridge))
    return result
