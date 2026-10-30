"""Contract tests for sampled candidate solver evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.physical_screen import evaluate_sampled_candidate
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


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
