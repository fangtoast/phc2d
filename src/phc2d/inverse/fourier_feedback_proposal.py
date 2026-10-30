"""Construct a budgeted Fourier proposal inside a feedback trust region.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.feedback_state import FeedbackState
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.fourier_screen import screen_fourier_candidate
from phc2d.inverse.fourier_search import search_fourier
from phc2d.inverse.local_edge_correction import LocalEdgeCorrection
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids, predict_perturbed_gap


def propose_fourier_candidate(state: FeedbackState, source: FullCellStructureSpec, predictor: Any, *,
                              coefficient_keys: tuple[tuple[int, str], ...],
                              global_bounds: Sequence[tuple[float, float]], origin: Sequence[float],
                              scales: Sequence[float], max_mode_radius: float, seed: int,
                              population_size: int, surrogate_budget: int, raster_method: str,
                              source_resolution: int, input_resolution: int,
                              material_fraction_bounds: tuple[float, float], lower_band: int,
                              surrogate_contract_sha256: str,
                              perturbation_offsets: tuple[float, ...] | None = None,
                              geometry_constraints: dict[str, Any] | None = None,
                              edge_correction: LocalEdgeCorrection | None = None) -> dict[str, Any]:
    """Search a conservative box inside the normalized Euclidean trust region.

    Returned proposals remain surrogate-only. The caller persists the trace and
    uses the exact returned structure for a solver-gated feedback evaluation.
    """
    if (source.parameter_hash != state.incumbent_hash or state.pending is not None
            or state.remaining_evaluations <= 0 or not surrogate_contract_sha256):
        raise ValueError('source, budget, pending state or surrogate identity is invalid')
    d = len(coefficient_keys) + 1
    bounds = np.asarray(global_bounds, dtype=np.float64)
    origin_array, scale_array = np.asarray(origin, dtype=np.float64), np.asarray(scales, dtype=np.float64)
    if (bounds.shape != (d, 2) or origin_array.shape != (d,) or scale_array.shape != (d,)
            or not np.isfinite(bounds).all() or not np.isfinite(origin_array).all()
            or not np.isfinite(scale_array).all() or np.any(scale_array <= 0)
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError('invalid global bounds or fixed normalization')
    modes = source.parameters['modes']
    try:
        center = np.asarray([float(modes[i][name]) for i, name in coefficient_keys]
                            + [float(source.parameters['threshold'])], dtype=np.float64)
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError('invalid Fourier coefficient keys') from exc
    normalized = (center - origin_array) / scale_array
    if not np.allclose(normalized, state.incumbent_parameters, rtol=0, atol=1e-10):
        raise ValueError('source parameters do not match feedback coordinates')
    if edge_correction is not None:
        if (edge_correction.surrogate_contract_sha256 != surrogate_contract_sha256
                or edge_correction.comparison_contract_sha256 != state.incumbent_observation['provenance']['comparison_contract_sha256']
                or edge_correction.center.shape != normalized.shape
                or not np.allclose(edge_correction.center, normalized, rtol=0, atol=1e-10)
                or edge_correction.radius + 1e-12 < state.radius):
            raise ValueError('edge correction contract, center or radius mismatch')
    half_width = state.radius * scale_array / np.sqrt(d)
    lower = np.maximum(bounds[:, 0], center - half_width)
    upper = np.minimum(bounds[:, 1], center + half_width)
    if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError('empty trust-region search box')
    anchor = screen_fourier_candidate(source, predictor, raster_method=raster_method,
        source_resolution=source_resolution, input_resolution=input_resolution,
        material_fraction_bounds=material_fraction_bounds, lower_band=lower_band,
        target_frequency=state.target_frequency, geometry_constraints=geometry_constraints)
    if anchor.status != 'surrogate_only':
        raise ValueError('incumbent failed surrogate screening')
    anchor_gap = anchor.predicted_gap
    if perturbation_offsets is not None:
        if raster_method != 'master_block_mean_v1':
            raise ValueError('perturbations require the master block-mean raster contract')
        grids = periodic_perturbation_grids(source, resolution=source_resolution, offsets=perturbation_offsets)
        robust = predict_perturbed_gap(grids, predictor, input_resolution=input_resolution,
            lower_band=lower_band, target_frequency=state.target_frequency)
        if robust['status'] != 'surrogate_only':
            raise ValueError('incumbent failed robust surrogate screening')
        anchor_gap = robust['gap']
    def adjusted_margin(vector: np.ndarray, gap: dict[str, Any]) -> float:
        edges = np.asarray([gap['lower_edge'], gap['upper_edge']], dtype=np.float64)
        if edge_correction is not None:
            edges = edge_correction.correct((vector - origin_array) / scale_array, edges)
        return float(2 * min(state.target_frequency - edges[0], edges[1] - state.target_frequency)
                     / state.target_frequency)

    result = search_fourier(source, predictor, coefficient_keys=coefficient_keys,
        bounds=tuple(zip(lower.tolist(), upper.tolist())), max_mode_radius=max_mode_radius,
        seed=seed, population_size=population_size, evaluation_budget=surrogate_budget,
        raster_method=raster_method, source_resolution=source_resolution, input_resolution=input_resolution,
        material_fraction_bounds=material_fraction_bounds, lower_band=lower_band,
        target_frequency=state.target_frequency, perturbation_offsets=perturbation_offsets,
        geometry_constraints=geometry_constraints,
        score_adjuster=adjusted_margin if edge_correction is not None else None)
    eligible = [row for row in result.evaluations if row['status'] == 'surrogate_only'
                and row['parameter_hash'] not in state.seen_hashes]
    eligible.sort(key=lambda row: (-float(row['selection_score']), row['candidate_index']))
    if not eligible:
        return {'status': 'no_unseen_candidate', 'proposal': None, 'trace': list(result.evaluations)}
    best = eligible[0]
    anchor_score = adjusted_margin(center, anchor_gap)
    if float(best['selection_score']) <= anchor_score + 1e-12:
        return {'status': 'no_predicted_improvement', 'proposal': None, 'trace': list(result.evaluations)}
    vector = np.asarray(best['parameters'], dtype=np.float64)
    candidate = update_fourier_binary(source, sample_id=best['sample_id'],
        coefficients={key: float(vector[i]) for i, key in enumerate(coefficient_keys)},
        threshold=float(vector[-1]), max_mode_radius=max_mode_radius)
    if candidate.parameter_hash != best['parameter_hash']:
        raise ValueError('candidate reconstruction identity mismatch')
    candidate_normalized = (vector - origin_array) / scale_array
    proposal = {'parameter_hash': candidate.parameter_hash, 'parameters': candidate_normalized.tolist(),
                'predicted_center': anchor_score,
                'predicted_candidate': float(best['selection_score']),
                'surrogate_contract_sha256': surrogate_contract_sha256}
    return {'status': 'surrogate_proposal_only', 'proposal': proposal,
            'structure': candidate.to_mapping(), 'anchor_screen': asdict(anchor),
            'edge_correction_applied': edge_correction is not None,
            'trace': list(result.evaluations)}
