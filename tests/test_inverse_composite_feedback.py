"""Check composite physical residual observations and fitting.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import numpy as np
import pytest
from phc2d.inverse.composite_feedback import composite_edge_observation
from phc2d.inverse.local_edge_correction import fit_composite_edge_correction


def test_physical_soft_edge_residual_decomposes_and_fits():
    def observe(index, point):
        return composite_edge_observation(parameter_hash=str(index), parameters=point,
            physical_edges=[0.4, 0.7], hard_edges=[0.39, 0.71],
            soft_exact_edges=[0.38, 0.72], soft_smooth_edges=[0.37, 0.73],
            comparison_contract_sha256='physical', surrogate_contract_sha256='surrogate',
            numerical_gate_passed=True)
    rows = [observe(0, [0., 0.]), observe(1, [0.1, 0.]), observe(2, [0., 0.1])]
    row = rows[0]
    parts = sum(np.array(row[key]) for key in ('residual_physical_minus_hard',
                'residual_hard_minus_soft_exact', 'residual_soft_exact_minus_smooth'))
    assert parts == pytest.approx(row['composite_residual'])
    model = fit_composite_edge_correction(rows, center=[0., 0.], radius=1., ridge=1e-6)
    assert model.correct([0., 0.], [0.37, 0.73]) == pytest.approx([0.4, 0.7])
    with pytest.raises(ValueError, match='ungated'):
        composite_edge_observation(parameter_hash='x', parameters=[0.], physical_edges=[0.4, 0.7],
            hard_edges=[0.4, 0.7], soft_exact_edges=[0.4, 0.7], soft_smooth_edges=[0.4, 0.7],
            comparison_contract_sha256='physical', surrogate_contract_sha256='surrogate',
            numerical_gate_passed=False)
