"""Budgeted low-dimensional Fourier search using only physical observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.feedback_loop import SolverAttemptFailed
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.fourier_screen import screen_fourier_geometry
from phc2d.inverse.fourier_search import initial_fourier_population


@dataclass(frozen=True)
class DirectFourierSearchResult:
    """Finite search trace; solver observations are not continuous-BZ certification."""

    status: str
    best_structure: dict[str, Any] | None
    best_margin: float | None
    solver_attempts: int
    evaluations: tuple[dict[str, Any], ...]


class _SolverBudgetExhausted(Exception):
    pass


def search_fourier_physically(
    source: FullCellStructureSpec, evaluator: Callable[[FullCellStructureSpec], Mapping[str, Any]], *,
    before_solve: Callable[[Mapping[str, Any]], None],
    after_solve: Callable[[Mapping[str, Any], Mapping[str, Any]], None],
    coefficient_keys: tuple[tuple[int, str], ...], bounds: tuple[tuple[float, float], ...],
    max_mode_radius: float, seed: int, population_size: int, proposal_budget: int,
    solver_budget: int, raster_method: str, source_resolution: int, input_resolution: int,
    material_fraction_bounds: tuple[float, float], lower_band: int, target_frequency: float,
    comparison_contract_sha256: str, geometry_constraints: dict[str, Any] | None = None,
) -> DirectFourierSearchResult:
    """Maximize solver-verified margin without a surrogate.

    The geometry contract and initial population are shared with the surrogate
    search. ``proposal_budget`` bounds all proposed structures; ``solver_budget``
    bounds actual attempts, including recorded failures. An evaluator returns the
    observation and status from a real solver adapter, and must persist raw
    records before returning. The caller independently verifies solver identity,
    resource reservations, trace provenance and final physical acceptance.

    ``before_solve`` must durably record the pending attempt before dispatch;
    ``after_solve`` must record its outcome before another attempt. Unexpected
    exceptions propagate and an interrupted attempt requires reconciliation,
    never an automatic retry. This synchronous baseline does not dispatch jobs
    or implement crash-safe resume on its own.
    """
    from scipy.optimize import differential_evolution

    if (not callable(evaluator) or not callable(before_solve) or not callable(after_solve)
            or isinstance(solver_budget, bool) or not isinstance(solver_budget, int) or solver_budget < 1
            or isinstance(lower_band, bool) or not isinstance(lower_band, int) or lower_band < 1
            or isinstance(target_frequency, bool) or not math.isfinite(target_frequency) or target_frequency <= 0
            or not isinstance(comparison_contract_sha256, str) or not comparison_contract_sha256):
        raise ValueError('valid solver, persistence hooks, target, contract and budget are required')
    population = initial_fourier_population(source, coefficient_keys=coefficient_keys, bounds=bounds,
        seed=seed, population_size=population_size, evaluation_budget=proposal_budget)
    history: list[dict[str, Any]] = []
    cache: dict[str, float] = {}
    attempts = 0
    best: FullCellStructureSpec | None = None
    best_margin: float | None = None

    def objective(vector: np.ndarray) -> float:
        nonlocal attempts, best, best_margin
        index = len(history)
        candidate = update_fourier_binary(source, sample_id=f'{source.sample_id}-search-{seed}-{index}',
            coefficients={key: float(vector[i]) for i, key in enumerate(coefficient_keys)},
            threshold=float(vector[-1]), max_mode_radius=max_mode_radius)
        geometry = screen_fourier_geometry(candidate, raster_method=raster_method,
            source_resolution=source_resolution, input_resolution=input_resolution,
            material_fraction_bounds=material_fraction_bounds, geometry_constraints=geometry_constraints)
        event = {'candidate_index': index, 'parameter_hash': candidate.parameter_hash,
                 'sample_id': candidate.sample_id, 'parameters': vector.tolist(),
                 'material_fraction': geometry.material_fraction, 'geometry_gate': geometry.geometry_gate,
                 'comparison_contract_sha256': comparison_contract_sha256}
        if geometry.status != 'eligible':
            history.append({**event, 'status': geometry.status})
            return float('inf')
        if candidate.parameter_hash in cache:
            score = cache[candidate.parameter_hash]
            history.append({**event, 'status': 'duplicate_cached',
                            'physical_margin': None if math.isinf(score) else -score})
            return score
        if attempts >= solver_budget:
            raise _SolverBudgetExhausted
        before_solve(dict(event))
        attempts += 1
        try:
            evaluated = evaluator(candidate)
        except SolverAttemptFailed as exc:
            reason = str(exc).strip()
            if not reason:
                raise ValueError('solver failure must have a reason') from exc
            outcome = {'status': 'solver_failure', 'reason': reason}
            after_solve(dict(event), dict(outcome))
            history.append({**event, **outcome})
            cache[candidate.parameter_hash] = float('inf')
            return float('inf')
        if not isinstance(evaluated, Mapping) or not isinstance(evaluated.get('observation'), Mapping):
            raise ValueError('evaluator must return a gateable observation')
        observation = evaluated['observation']
        row, provenance = observation.get('row'), observation.get('provenance')
        if not isinstance(row, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError('physical row and provenance are required')
        if (provenance.get('parameter_hash') != candidate.parameter_hash
                or provenance.get('comparison_contract_sha256') != comparison_contract_sha256
                or isinstance(row.get('lower_band_one_based'), bool)
                or row.get('lower_band_one_based') != lower_band):
            raise ValueError('candidate, physical contract or band identity differs')
        keys = ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged')
        if not all(isinstance(row.get(key), bool) for key in keys):
            raise ValueError('physical numerical gates must be explicit')
        stable = all(row[key] for key in keys)
        if evaluated.get('status') != ('numerically_stable' if stable else 'numerical_gate_failed'):
            raise ValueError('physical status and numerical gates disagree')
        lower, upper = float(row['refined_lower_edge_frequency']), float(row['refined_upper_edge_frequency'])
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError('physical edges must be finite')
        margin = 2 * min(target_frequency - lower, upper - target_frequency) / target_frequency
        outcome = {'status': evaluated['status'], 'physical_margin': margin,
                   'comparison_contract_sha256': comparison_contract_sha256}
        after_solve(dict(event), dict(outcome))
        history.append({**event, **outcome})
        if not stable:
            cache[candidate.parameter_hash] = float('inf')
            return float('inf')
        if best_margin is None or margin > best_margin:
            best, best_margin = candidate, margin
        cache[candidate.parameter_hash] = -margin
        return -margin

    status = 'search_complete'
    try:
        differential_evolution(objective, bounds=bounds, init=population, seed=seed,
            maxiter=proposal_budget // population_size - 1, polish=False,
            updating='immediate', workers=1, tol=0, atol=0)
    except _SolverBudgetExhausted:
        status = 'solver_budget_exhausted'
    except RuntimeError as exc:
        # SciPy wraps objective ValueError/TypeError in a generic mapper error;
        # preserve the actual contract failure for reconciliation and diagnosis.
        if isinstance(exc.__cause__, (ValueError, TypeError)):
            raise exc.__cause__ from exc
        raise
    if best is None and status == 'search_complete':
        status = 'no_numerically_valid_candidates'
    return DirectFourierSearchResult(status, best.to_mapping() if best is not None else None,
                                     best_margin, attempts, tuple(history))
