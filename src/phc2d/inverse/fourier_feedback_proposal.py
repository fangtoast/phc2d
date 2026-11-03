"""Construct a budgeted Fourier proposal inside a feedback trust region.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.feedback_state import FeedbackState
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.fourier_screen import screen_fourier_candidate
from phc2d.inverse.fourier_search import search_fourier
from phc2d.inverse.local_edge_correction import LocalEdgeCorrection, correct_projected_with_anchor
from phc2d.inverse.perturbation_screen import (periodic_perturbation_grids, predict_perturbed_gap,
                                               project_perturbed_band_order)


def run_budgeted_nominal_feedback_proposal(state: FeedbackState, source: FullCellStructureSpec, predictor: Any, *,
                                            output_root: Path, provenance: dict[str, Any],
                                            **options: Any) -> dict[str, Any]:
    """Save one strictly budgeted nominal proposal and its cost, including failures.

    The caller binds the state to verified physics, freezes all options and
    authenticates execution. This local operation does not submit a job or
    advance feedback state.
    """
    from phc2d.inverse.frozen_band_adapter import SingleSampleBudgetedPredictor

    budget = options.get('surrogate_budget')
    if (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
            or tuple(options.get('perturbation_offsets') or ()) != (0.0,)
            or options.get('raster_method') != 'master_block_mean_v1'
            or options.get('edge_correction') is not None or options.get('projected_anchor_calibration', False)
            or not isinstance(provenance, dict) or not provenance):
        raise ValueError('budgeted nominal feedback requires a zero-offset uncorrected contract')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    request = {'scope': 'nominal_feedback_request_not_execution_lock', 'state': asdict(state),
               'source': source.to_mapping(), 'options': options, 'provenance': provenance}
    with (root/'request.json').open('x', encoding='utf8') as stream:
        json.dump(request, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    wrapped = SingleSampleBudgetedPredictor(predictor, max_inferences=budget, cache_capacity=budget,
                                            evidence_directory=root/'inferences')
    status, error = 'failed', None
    try:
        result = propose_fourier_candidate(state, source, wrapped, **options)
        payload = {'scope': 'budgeted_nominal_proposal_not_physical_validation', 'provenance': provenance,
                   'request_sha256': hashlib.sha256((root/'request.json').read_bytes()).hexdigest(), 'result': result}
        with (root/'proposal.json').open('x', encoding='utf8') as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
        status = result['status']
        return payload
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        archive_error = None
        try:
            wrapped.save_prediction_archive(root/'predictions.npz')
        except Exception as exc:
            archive_error = {'type': type(exc).__name__, 'message': str(exc)}
        cost = {'status': status, 'error': error, 'max_inferences': budget,
                'requests': wrapped.requests, 'inferences': wrapped.inferences, 'cache_hits': wrapped.cache_hits,
                'successful_inferences': wrapped.successful_inferences, 'archive_error': archive_error}
        with (root/'cost.json').open('x', encoding='utf8') as stream:
            json.dump(cost, stream, sort_keys=True, allow_nan=False)
        if archive_error is not None and error is None:
            raise RuntimeError('prediction archive failed; see cost record')


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
                              edge_correction: LocalEdgeCorrection | None = None,
                              projected_anchor_calibration: bool = False) -> dict[str, Any]:
    """Search a conservative box inside the normalized Euclidean trust region.

    Returned proposals remain surrogate-only. The caller persists the trace and
    uses the exact returned structure for a solver-gated feedback evaluation.
    """
    if any(entry.get('status') == 'worker_result_unknown' for entry in state.history):
        raise ValueError('unknown worker result must be reconciled before further feedback')
    if (source.parameter_hash != state.incumbent_hash or state.pending is not None
            or state.remaining_evaluations <= 0 or not surrogate_contract_sha256):
        raise ValueError('source, budget, pending state or surrogate identity is invalid')
    if (not isinstance(projected_anchor_calibration, bool) or
            (projected_anchor_calibration and (edge_correction is not None or perturbation_offsets is None))):
        raise ValueError('projected anchor calibration requires perturbations and no external correction')
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
    if anchor.status != 'surrogate_only' and not (
            projected_anchor_calibration and anchor.status == 'band_order_violation'):
        raise ValueError('incumbent failed surrogate screening')
    anchor_gap = anchor.predicted_gap
    anchor_projection = None
    if perturbation_offsets is not None:
        if raster_method != 'master_block_mean_v1':
            raise ValueError('perturbations require the master block-mean raster contract')
        grids = periodic_perturbation_grids(source, resolution=source_resolution, offsets=perturbation_offsets)
        robust = predict_perturbed_gap(grids, predictor, input_resolution=input_resolution,
            lower_band=lower_band, target_frequency=state.target_frequency)
        if projected_anchor_calibration and robust['status'] in ('surrogate_only', 'band_order_violation'):
            anchor_projection = project_perturbed_band_order(robust, lower_band=lower_band,
                target_frequency=state.target_frequency, surrogate_contract_sha256=surrogate_contract_sha256)
            calibrated = correct_projected_with_anchor(anchor_projection, anchor_projection,
                anchor_structure=source, candidate_structure=source,
                verified_anchor=state.incumbent_observation, raster_resolution=source_resolution)
            coefficients = np.zeros((d+1, 2))
            coefficients[0] = calibrated['anchor_edge_residual']
            edge_correction = LocalEdgeCorrection(normalized.copy(), state.radius, coefficients,
                state.incumbent_observation['provenance']['comparison_contract_sha256'],
                surrogate_contract_sha256, 1)
            anchor_gap = anchor_projection['gap']
        elif robust['status'] != 'surrogate_only':
            raise ValueError('incumbent failed robust surrogate screening')
        else:
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
        score_adjuster=adjusted_margin if edge_correction is not None else None,
        projection_contract_sha256=surrogate_contract_sha256 if projected_anchor_calibration else None)
    eligible_status = 'projected_surrogate_only' if projected_anchor_calibration else 'surrogate_only'
    eligible = [row for row in result.evaluations if row['status'] == eligible_status
                and row['parameter_hash'] not in state.seen_hashes]
    eligible.sort(key=lambda row: (-float(row['selection_score']), row['candidate_index']))
    evidence = {'anchor_screen': asdict(anchor), 'anchor_projection': anchor_projection,
                'edge_correction_applied': edge_correction is not None,
                'anchor_edge_residual': (None if not projected_anchor_calibration else
                                         edge_correction.coefficients[0].tolist())}
    if not eligible:
        return {'status': 'no_unseen_candidate', 'proposal': None, 'trace': list(result.evaluations), **evidence}
    best = eligible[0]
    anchor_score = adjusted_margin(center, anchor_gap)
    if float(best['selection_score']) <= anchor_score + 1e-12:
        return {'status': 'no_predicted_improvement', 'proposal': None, 'trace': list(result.evaluations), **evidence}
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
            'structure': candidate.to_mapping(), 'trace': list(result.evaluations), **evidence}
