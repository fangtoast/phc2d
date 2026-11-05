"""Exercise the smooth margin and projected surrogate optimization.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

torch = pytest.importorskip('torch')
from phc2d.inverse.differentiable_objective import optimize_adam, optimize_black_box, optimize_fourier_adam, optimize_fourier_black_box, smooth_target_margin
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary


def test_smooth_margin_is_conservative_and_differentiable():
    bands = torch.tensor([[[0.2, 0.8], [0.4, 0.7]]], dtype=torch.float64, requires_grad=True)
    score = smooth_target_margin(bands, lower_band=1, target_frequency=0.5, beta=100)
    assert 0 < score.item() < 0.4
    assert torch.autograd.gradcheck(lambda x: smooth_target_margin(x, lower_band=1,
                                    target_frequency=0.5, beta=100), (bands,))


def test_projected_adam_maximizes_and_records_final_point():
    history = optimize_adam(lambda x: -(x[0] - 2).square(), torch.tensor([0.0]),
                            lower=[-1], upper=[1], learning_rate=0.2, steps=20)
    assert len(history) == 21
    assert history[-1]['parameters'][0] == pytest.approx(1)
    assert history[-1]['objective'] > history[0]['objective']


@pytest.mark.parametrize('method', ['random', 'de'])
def test_black_box_respects_budget_and_is_reproducible(method):
    def run():
        return optimize_black_box(lambda x: -(x - 0.3).square().sum(), torch.zeros(2),
                                  lower=[-0.5, -0.5], upper=[0.5, 0.5],
                                  evaluations=20, seed=3, method=method)
    first = run()
    assert first == run()
    assert len(first) == 20
    assert all(-0.5 <= value <= 0.5 for row in first for value in row['parameters'])


def test_fourier_adam_uses_geometry_and_preserves_input_gradients():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1 * value, 0.7 + 0.1 * value), dim=1).unsqueeze(1)

    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    history = optimize_fourier_adam(source, Predictor(), coefficient_keys=[(0, 'cos')],
                                   scales=[0.2, 0.1], lower=[-0.5, -0.5], upper=[0.5, 0.5],
                                   master_resolution=64, input_resolution=16, temperature=0.2,
                                   lower_band=1, target_frequency=0.5, beta=100,
                                   learning_rate=0.1, steps=3)
    assert len(history) == 4
    assert history[-1]['objective'] > history[0]['objective']
    assert len(history[-1]['fourier_parameters']) == 2


def test_fourier_black_box_shares_adam_initial_objective():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1 * value, 0.7 + 0.1 * value), dim=1).unsqueeze(1)

    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    args = dict(coefficient_keys=[(0, 'cos')], scales=[0.2, 0.1], lower=[-0.5, -0.5],
                upper=[0.5, 0.5], master_resolution=64, input_resolution=16, temperature=0.2,
                lower_band=1, target_frequency=0.5, beta=100)
    adam = optimize_fourier_adam(source, Predictor(), **args, learning_rate=0.1, steps=0)
    for method in ('de', 'random'):
        search = optimize_fourier_black_box(source, Predictor(), **args,
                                             evaluations=5, seed=1, method=method)
        assert search[0]['objective'] == pytest.approx(adam[0]['objective'])
        assert search[0]['fourier_parameters'] == pytest.approx(adam[0]['fourier_parameters'])


def test_fourier_algorithms_share_nonzero_initial_and_budget():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1 * value, 0.7 + 0.1 * value), dim=1).unsqueeze(1)

    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    args = dict(coefficient_keys=[(0, 'cos')], scales=[0.2, 0.1], lower=[-0.5, -0.5],
                upper=[0.5, 0.5], master_resolution=64, input_resolution=16, temperature=0.2,
                lower_band=1, target_frequency=0.5, beta=100, initial=[0.15, 0.0])
    adam = optimize_fourier_adam(source, Predictor(), **args, learning_rate=0.1, steps=5)
    for method in ('de', 'random'):
        result = optimize_fourier_black_box(source, Predictor(), **args,
                                             evaluations=6, seed=1, method=method)
        assert len(result) == len(adam) == 6
        assert result[0]['parameters'] == pytest.approx([0.15, 0.0])
        assert result[0]['fourier_parameters'] == pytest.approx(adam[0]['fourier_parameters'])
        assert result[0]['objective'] == pytest.approx(adam[0]['objective'])


def test_four_parameter_fourier_algorithms_use_shared_origin_and_budget():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1 * value, 0.7 + 0.1 * value), dim=1).unsqueeze(1)

    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 5, 'audit_resolution': 32})
    args = dict(coefficient_keys=[(0, 'cos'), (0, 'sin'), (9, 'cos')], scales=[0.2, 0.2, 0.2, 0.1],
                lower=[-0.5]*4, upper=[0.5]*4, master_resolution=64, input_resolution=16,
                temperature=0.2, lower_band=1, target_frequency=0.5, beta=100, initial=[0.0]*4)
    adam = optimize_fourier_adam(source, Predictor(), **args, learning_rate=0.1, steps=4)
    for method in ('de', 'random'):
        result = optimize_fourier_black_box(source, Predictor(), **args, evaluations=5, seed=1, method=method)
        assert len(result) == len(adam) == 5
        assert result[0]['parameters'] == pytest.approx([0.0]*4)
        assert result[0]['objective'] == pytest.approx(adam[0]['objective'])
        assert all(len(row['fourier_parameters']) == 4 for row in result)
