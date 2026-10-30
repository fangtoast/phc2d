"""Solver-gated feedback physical evaluator contracts."""

import hashlib

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def fixture():
    structure = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=structure.epsilon_air, epsilon_material=structure.epsilon_material,
        eigensolver_tolerance=1e-7)
    options = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
               'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
               'resolution_tolerance': 1e-6}
    return structure, config, options


def test_nominal_feedback_exposes_gate_and_raw_spectra():
    structure, config, options = fixture()
    def solver(item, *, k_points, config):
        return np.tile([.2, .7], (len(k_points), 1)), {'geometry_hash': item.geometry_hash}
    result = evaluate_feedback_candidate(structure, k_points=np.array([[0, 0], [.25, 0]]),
        config=config, lower_band=1, refinement=options, solver=solver)
    assert result['status'] == 'numerically_stable'
    assert result['observation']['provenance']['parameter_hash'] == structure.parameter_hash
    assert len(result['traces'][0]['arrays']['k_points']) > 2
    assert result['solver_work'][0]['primary_k_points'] == len(result['traces'][0]['arrays']['k_points'])
    assert result['solver_work'][0]['verification_resolution'] == 24
    assert result['solver_work'][0]['solver_calls'] >= 2


def test_robust_feedback_aggregates_same_grid_family_and_preserves_numerical_failure():
    structure, config, options = fixture()
    def solver(grid, *, k_points, config):
        digest = hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest()
        upper = .7 if config.resolution == 16 else .69
        return np.tile([.2, upper], (len(k_points), 1)), {'grid_sha256': digest}
    result = evaluate_feedback_candidate(structure, k_points=np.array([[0, 0], [.25, 0]]),
        config=config, lower_band=1, refinement=options, offsets=(-.0625, 0., .0625),
        raster_resolution=16, solver=solver)
    assert result['status'] == 'numerical_gate_failed'
    assert len(result['traces']) == 3
    assert len(result['solver_work']) == 3
    assert result['observation']['row']['refined_lower_edge_frequency'] == pytest.approx(.2)
    assert result['observation']['provenance']['offsets'] == [-.0625, 0., .0625]
