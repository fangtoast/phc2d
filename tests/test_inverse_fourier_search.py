"""Tests for budgeted non-generative Fourier search.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.fourier_search import search_fourier


class ToyPredictor:
    def predict_tensor(self, raster):
        mean = raster.mean(dim=(1, 2))
        return torch.stack((0.2 + 0.1 * mean, 0.8 - 0.1 * mean), dim=1).unsqueeze(1)


def run(seed, bounds=(0, 1)):
    s = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    t = s.parameters['threshold']
    return search_fourier(s, ToyPredictor(), coefficient_keys=(), bounds=((t-0.2,t+0.2),),
                          max_mode_radius=1, seed=seed, population_size=5, evaluation_budget=10,
                          raster_method='master_block_mean_v1', source_resolution=16,
                          input_resolution=8, material_fraction_bounds=bounds,
                          lower_band=1, target_frequency=0.5)


def test_search_is_seeded_budgeted_and_selects_recorded_best():
    a, b = run(42), run(42)
    assert a.status == 'surrogate_only'
    assert len(a.evaluations) <= 10
    assert a.evaluations == b.evaluations
    assert a.best_margin == max(x['predicted_gap']['target_margin'] for x in a.evaluations)


def test_all_invalid_proposals_are_retained():
    result = run(1, (0, 0))
    assert result.status == 'no_valid_candidates'
    assert result.best_structure is None
    assert all(row['status'] == 'material_fraction_rejected' for row in result.evaluations)


def test_search_can_optimize_perturbation_common_gap():
    s = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    t = s.parameters['threshold']
    result = search_fourier(s, ToyPredictor(), coefficient_keys=(), bounds=((t-.2, t+.2),),
        max_mode_radius=1, seed=42, population_size=5, evaluation_budget=5,
        raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
        material_fraction_bounds=(0, 1), lower_band=1, target_frequency=.5,
        perturbation_offsets=(-1/16, 0, 1/16))
    assert result.status == 'surrogate_only'
    assert all(len(row['perturbation_grid_sha256']) == 3 for row in result.evaluations)
    assert result.best_margin == max(row['predicted_gap']['target_margin'] for row in result.evaluations)


def test_search_reports_actual_evaluations_when_constant_scores_converge_early():
    class ConstantPredictor:
        def predict_tensor(self, raster):
            return torch.tensor([.2, .8], dtype=raster.dtype, device=raster.device).expand(raster.shape[0], 1, 2)

    s = generate_unified_fourier_binary('constant-source', master_seed=1, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    t = s.parameters['threshold']
    result = search_fourier(s, ConstantPredictor(), coefficient_keys=(), bounds=((t-.2, t+.2),),
        max_mode_radius=1, seed=42, population_size=5, evaluation_budget=20,
        raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
        material_fraction_bounds=(0, 1), lower_band=1, target_frequency=.5)
    assert result.status == 'surrogate_only'
    assert len(result.evaluations) < 20
