"""Tests for solver-gated local band-edge correction."""

import numpy as np
import pytest

from phc2d.inverse.local_edge_correction import fit_local_edge_correction


def observation(identity, point, residual, *, stable=True, contract='c'):
    return {'parameter_hash': identity, 'parameters': [point], 'predicted_edges': [.4, .7],
            'physical_edges': [.4 + residual, .7 - residual],
            'numerical_gate_passed': stable, 'comparison_contract_sha256': contract,
            'surrogate_contract_sha256': 'surrogate'}


def test_local_residual_fit_recovers_affine_edge_bias_and_rejects_extrapolation():
    records = [observation('a', -.5, -.04), observation('b', 0., .01), observation('c', .5, .06)]
    model = fit_local_edge_correction(records, center=[0.], radius=1., ridge=1e-10)
    assert model.correct([.25], [.4, .7]) == pytest.approx([.435, .665], abs=1e-8)
    with pytest.raises(ValueError, match='outside'):
        model.correct([1.1], [.4, .7])


def test_local_correction_rejects_unstable_and_mixed_protocols():
    with pytest.raises(ValueError, match='numerical gates'):
        fit_local_edge_correction([observation('a', 0, .1, stable=False)], center=[0], radius=1, ridge=.01)
    with pytest.raises(ValueError, match='comparison contract'):
        fit_local_edge_correction([observation('a', 0, .1), observation('b', .1, .2, contract='d')],
                                  center=[0], radius=1, ridge=.01)
    other = observation('b', .1, .2)
    other['surrogate_contract_sha256'] = 'different'
    with pytest.raises(ValueError, match='surrogate contract'):
        fit_local_edge_correction([observation('a', 0, .1), other], center=[0], radius=1, ridge=.01)


def test_single_observation_gives_constant_correction():
    model = fit_local_edge_correction([observation('a', 0, .02)], center=[0], radius=1, ridge=.01)
    assert np.allclose(model.correct([.5], [.3, .8]), [.32, .78])
