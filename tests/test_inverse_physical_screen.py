"""Contract tests for sampled candidate solver evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np
import pytest
import hashlib

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.physical_screen import evaluate_sampled_candidate
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig
from phc2d.replay.gap_refinement import refine_inverse_candidate


def fixtures():
    s = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    c = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
                                   epsilon_air=s.epsilon_air, epsilon_material=s.epsilon_material,
                                   eigensolver_tolerance=1e-7)
    return s, c


def test_sampled_solver_result_preserves_geometry_and_evidence_boundary():
    s, c = fixtures()
    def solver(structure, *, k_points, config):
        return np.array([[0.2, 0.7], [0.3, 0.8]]), {'geometry_hash': structure.geometry_hash}
    result = evaluate_sampled_candidate(s, k_points=np.array([[0, 0], [0.1, 0]]),
                                        config=c, lower_band=1, target_frequency=0.5, solver=solver)
    assert result['status'] == 'sampled_solver_only'
    assert result['gap']['lower_edge'] == pytest.approx(0.3)
    assert result['parameter_hash'] == s.parameter_hash


def test_mismatched_solver_identity_is_rejected():
    s, c = fixtures()
    def solver(structure, *, k_points, config):
        return np.array([[0.2, 0.7]]), {'geometry_hash': 'wrong'}
    with pytest.raises(ValueError, match='geometry identity'):
        evaluate_sampled_candidate(s, k_points=np.array([[0, 0]]), config=c,
                                   lower_band=1, target_frequency=0.5, solver=solver)


def test_inverse_refinement_uses_existing_edge_algorithm_and_checks_resolution():
    s, c = fixtures()
    def solver(structure, *, k_points, config):
        r = np.sum(np.asarray(k_points) ** 2, axis=1)
        return np.column_stack((0.2 + .01*r, 0.7 - .01*r)), {'geometry_hash': structure.geometry_hash}
    rows, arrays, history, provenance = refine_inverse_candidate(
        s, k_points=np.array([[0, 0], [0.25, 0]]), config=c, verification_resolution=24,
        lower_band=1, initial_spacing=0.25, min_levels=2, max_levels=2,
        required_stable_levels=2, edge_tolerance=1, resolution_tolerance=1e-6, solver=solver)
    assert len(rows) == 1 and len(history) == 2
    assert rows[0]['resolution_stable'] and rows[0]['adaptive_refinement_converged']
    assert len(arrays['k_points']) > 2
    assert provenance['evidence_scope'] == 'finite_local_refinement_only'


def test_inverse_refinement_rejects_incompatible_resolution():
    s, c = fixtures()
    with pytest.raises(ValueError, match='verification resolution'):
        refine_inverse_candidate(s, k_points=np.array([[0, 0]]), config=c,
            verification_resolution=16, lower_band=1, initial_spacing=0.25,
            min_levels=1, max_levels=1, required_stable_levels=1,
            edge_tolerance=1e-3, resolution_tolerance=1e-3)


def test_inverse_refinement_checks_grid_identity():
    s, c = fixtures()
    grid = np.eye(16, dtype=np.uint8)
    def solver(actual_grid, *, k_points, config):
        assert np.array_equal(actual_grid, grid)
        digest = hashlib.sha256(np.ascontiguousarray(actual_grid).tobytes()).hexdigest()
        return np.tile([.2, .7], (len(k_points), 1)), {'grid_sha256': digest}
    rows, arrays, history, provenance = refine_inverse_candidate(
        s, k_points=np.array([[0, 0], [.25, 0]]), config=c, verification_resolution=24,
        lower_band=1, initial_spacing=.25, min_levels=1, max_levels=1,
        required_stable_levels=1, edge_tolerance=1e-3, resolution_tolerance=1e-3,
        material_grid=grid, solver=solver)
    assert provenance['evaluated_grid_sha256'] == hashlib.sha256(grid.tobytes()).hexdigest()
    assert provenance['evaluated_geometry'] == 'periodic_binary_pixel_cells_v1'
    assert rows[0]['resolution_stable']
