"""Check differentiable offsets against the periodic binary geometry contract.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from phc2d.inverse.differentiable_morphology import periodic_soft_offset
from phc2d.inverse.morphology import periodic_binary_offset
from phc2d.inverse.differentiable_objective import fourier_surrogate_margin, robust_fourier_margin, optimize_robust_fourier_adam
from phc2d.inverse.differentiable_selection import rank_robust_trajectory
from phc2d.inverse.differentiable_robust import run_robust_shard
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary


@pytest.mark.parametrize('offset', [-2/16, -1/16, 0., 1/32, 1/16, 2/16])
def test_binary_limit_matches_periodic_offset(offset):
    grid = np.zeros((16, 16), dtype=np.float64)
    grid[0:4, 0:3] = 1
    grid[8:12, 13:16] = 1
    actual = periodic_soft_offset(torch.tensor(grid), offset=offset).numpy()
    np.testing.assert_array_equal(actual, periodic_binary_offset(grid, offset=offset))


def test_soft_offset_preserves_gradient_and_wraps():
    grid = torch.full((8, 8), 0.1, dtype=torch.float64)
    grid[0, 0] = 0.8
    grid.requires_grad_()
    result = periodic_soft_offset(grid, offset=1/8)
    assert result[0, -1].item() == pytest.approx(0.8)
    gradient, = torch.autograd.grad(result[0, -1], grid)
    assert gradient[0, 0] == 1


def test_robust_margin_uses_shared_target_across_scenarios():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1*value, 0.7 + 0.1*value), dim=1).unsqueeze(1)
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    params = torch.tensor([source.parameters['modes'][0]['cos'], source.parameters['threshold']],
                          dtype=torch.float64, requires_grad=True)
    args = dict(coefficient_keys=[(0, 'cos')], master_resolution=64, input_resolution=16,
                temperature=0.2, lower_band=1, target_frequency=0.5, beta=100)
    nominal = fourier_surrogate_margin(source, Predictor(), params, **args)
    one = robust_fourier_margin(source, Predictor(), params, offsets=[0.], **args)
    assert one.item() == pytest.approx(nominal.item())
    robust = robust_fourier_margin(source, Predictor(), params, offsets=[-1/64, 0., 1/64], **args)
    assert robust.item() <= nominal.item()
    gradient, = torch.autograd.grad(robust, params)
    assert torch.isfinite(gradient).all()
    history = optimize_robust_fourier_adam(source, Predictor(), scales=[0.2, 0.1],
        lower=[-0.5, -0.5], upper=[0.5, 0.5], offsets=[-1/64, 0., 1/64],
        learning_rate=0.05, steps=2, **args)
    assert len(history) == 3
    assert all(len(row['fourier_parameters']) == 2 for row in history)
    paired = optimize_robust_fourier_adam(source, Predictor(), scales=[0.2, 0.1],
        lower=[-0.5, -0.5], upper=[0.5, 0.5], offsets=[-1/64, 0., 1/64],
        learning_rate=0.05, steps=0, initial=[.15, 0.], **args)
    assert paired[0]['parameters'] == pytest.approx([.15, 0.])
    ranked = rank_robust_trajectory(source, Predictor(), history, offsets=[-1/64, 0., 1/64],
        coefficient_keys=[(0, 'cos')], max_mode_radius=1, raster_method='master_block_mean_v1',
        source_resolution=64, input_resolution=16, material_fraction_bounds=(0., 1.),
        lower_band=1, target_frequency=.5)
    assert ranked['status'] == 'selected'
    candidate = ranked['candidates'][ranked['selected_index']]
    assert len(candidate['robust_prediction']['grid_sha256']) == 3
    assert len(candidate['robust_prediction']['bands']) == 3
    assert candidate['robust_prediction']['gap']['target_margin'] <= candidate['predicted_gap']['target_margin'] + 1e-8
    config = {'offsets': [-1/64, 0., 1/64], 'coefficient_keys': [(0, 'cos')], 'scales': [.2, .1],
        'lower': [-.5, -.5], 'upper': [.5, .5], 'source_resolution': 64, 'input_resolution': 16,
        'temperature': .2, 'lower_band': 1, 'target_frequency': .5, 'beta': 100, 'learning_rate': .05,
        'evaluations': 2, 'initial': [.15, 0.], 'max_mode_radius': 1, 'material_fraction_bounds': [0., 1.]}
    nominal_result = run_robust_shard(config, source, Predictor(), mode='nominal')
    robust_result = run_robust_shard(config, source, Predictor(), mode='worst')
    assert nominal_result['history'][0]['parameters'] == pytest.approx([.15, 0.])
    assert robust_result['history'][0]['parameters'] == pytest.approx([.15, 0.])
    assert nominal_result['selection']['status'] == 'selected' and nominal_result['robust_diagnostic'] is not None
    assert robust_result['selection']['score'] == 'hard_common_target_margin'
    assert robust_result['objective_evaluations'] == 2 and robust_result['scenario_inputs_per_objective'] == 3
