"""Budgeted differential evolution over analytic Fourier structures.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.fourier_screen import screen_fourier_candidate
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids, predict_perturbed_gap


@dataclass(frozen=True)
class FourierSearchResult:
    """Candidate trace and best surrogate result; no physical success implied."""

    status: str
    best_structure: dict[str, Any] | None
    best_margin: float | None
    evaluations: tuple[dict[str, Any], ...]


def initial_fourier_population(
    source: FullCellStructureSpec, *, coefficient_keys: tuple[tuple[int, str], ...],
    bounds: tuple[tuple[float, float], ...], seed: int, population_size: int,
    evaluation_budget: int,
) -> np.ndarray:
    """Build a reproducible bounded population with the source as its first member."""
    if len(set(coefficient_keys)) != len(coefficient_keys) or len(bounds) != len(coefficient_keys) + 1:
        raise ValueError('bounds must cover unique coefficients and one threshold')
    limits = np.asarray(bounds, dtype=np.float64)
    if limits.shape != (len(bounds), 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('bounds must contain finite increasing pairs')
    if (isinstance(population_size, bool) or not isinstance(population_size, (int, np.integer)) or population_size < 5
            or isinstance(evaluation_budget, bool) or not isinstance(evaluation_budget, (int, np.integer))
            or evaluation_budget < population_size):
        raise ValueError('budget must cover at least five population members')
    modes = source.parameters['modes']
    try:
        initial = np.asarray([float(modes[i][name]) for i, name in coefficient_keys]
                             + [float(source.parameters['threshold'])])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('invalid source coefficient identity') from exc
    if not np.isfinite(initial).all() or np.any(initial < limits[:, 0]) or np.any(initial > limits[:, 1]):
        raise ValueError('source parameters lie outside configured bounds')
    rng = np.random.default_rng(seed)
    population = rng.uniform(limits[:, 0], limits[:, 1], size=(population_size, len(bounds)))
    population[0] = initial
    return population


def search_fourier(
    source: FullCellStructureSpec, predictor: Any, *,
    coefficient_keys: tuple[tuple[int, str], ...], bounds: tuple[tuple[float, float], ...],
    max_mode_radius: float, seed: int, population_size: int, evaluation_budget: int,
    raster_method: str, source_resolution: int, input_resolution: int,
    material_fraction_bounds: tuple[float, float], lower_band: int, target_frequency: float,
    perturbation_offsets: tuple[float, ...] | None = None,
    geometry_constraints: dict[str, Any] | None = None,
    score_adjuster: Callable[[np.ndarray, dict[str, Any]], float] | None = None,
) -> FourierSearchResult:
    """Maximize sampled surrogate margin; last parameter is the binary threshold.

    The evaluation budget caps the number of whole populations, including
    initialization; convergence may stop the search earlier. Comparisons must
    report actual evaluations and resource use, not assume the cap was spent.
    The unedited source is always included. Traces include all
    rejected proposals; each invocation is one seeded optimization trajectory.
    If offsets are supplied, optimize their common sampled gap rather than the
    nominal gap; the caller must verify the area-averaged checkpoint contract.
    A caller-supplied score adjuster may calibrate the sampled edges; the raw
    predicted gap remains in the trace and no physical success is implied.
    """
    from scipy.optimize import differential_evolution

    population = initial_fourier_population(source, coefficient_keys=coefficient_keys, bounds=bounds,
        seed=seed, population_size=population_size, evaluation_budget=evaluation_budget)
    if perturbation_offsets is not None and raster_method != 'master_block_mean_v1':
        raise ValueError('perturbed search requires master_block_mean_v1')
    history: list[dict[str, Any]] = []
    best: FullCellStructureSpec | None = None
    best_margin: float | None = None

    def objective(vector: np.ndarray) -> float:
        nonlocal best, best_margin
        index = len(history)
        candidate = update_fourier_binary(
            source, sample_id=f"{source.sample_id}-search-{seed}-{index}",
            coefficients={key: float(vector[i]) for i, key in enumerate(coefficient_keys)},
            threshold=float(vector[-1]), max_mode_radius=max_mode_radius,
        )
        result = screen_fourier_candidate(
            candidate, predictor, raster_method=raster_method, source_resolution=source_resolution,
            input_resolution=input_resolution, material_fraction_bounds=material_fraction_bounds,
            lower_band=lower_band, target_frequency=target_frequency, geometry_constraints=geometry_constraints,
        )
        row = {"candidate_index": index, "parameters": vector.tolist(), **asdict(result)}
        if result.status == 'surrogate_only' and perturbation_offsets is not None:
            grids = periodic_perturbation_grids(candidate, resolution=source_resolution,
                                                offsets=perturbation_offsets)
            robust = predict_perturbed_gap(grids, predictor, input_resolution=input_resolution,
                                          lower_band=lower_band, target_frequency=target_frequency)
            row.update(status=robust['status'], predicted_gap=robust['gap'],
                       perturbation_offsets=list(perturbation_offsets),
                       perturbation_grid_sha256=robust['grid_sha256'])
        if row['status'] != "surrogate_only":
            history.append(row)
            return float('inf')
        score = float(row['predicted_gap']["target_margin"] if score_adjuster is None
                      else score_adjuster(vector.copy(), dict(row['predicted_gap'])))
        if not np.isfinite(score):
            raise ValueError('adjusted surrogate score must be finite')
        row['selection_score'] = score
        history.append(row)
        if best_margin is None or score > best_margin:
            best, best_margin = candidate, score
        return -score

    differential_evolution(objective, bounds=bounds, init=population, seed=seed,
                           maxiter=evaluation_budget // population_size - 1,
                           polish=False, updating="immediate", workers=1, tol=0, atol=0)
    return FourierSearchResult(status="surrogate_only" if best is not None else "no_valid_candidates",
                               best_structure=best.to_mapping() if best is not None else None,
                               best_margin=best_margin, evaluations=tuple(history))
