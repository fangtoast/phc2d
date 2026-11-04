"""Exercise the smooth margin and projected surrogate optimization.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import pytest

torch = pytest.importorskip('torch')
from phc2d.inverse.differentiable_objective import optimize_adam, optimize_fourier_adam, smooth_target_margin
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
