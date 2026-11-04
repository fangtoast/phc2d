"""Check local edge corrections preserve the design gradient.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from phc2d.inverse.local_edge_correction import LocalEdgeCorrection
from phc2d.inverse.differentiable_objective import corrected_fourier_margin
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary


def test_tensor_edge_correction_matches_numpy_and_derivative():
    correction = LocalEdgeCorrection(np.array([0., 0.]), 1.,
        np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]), 'physical', 'model', 3)
    point = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([0.4, 0.7], dtype=torch.float64)
    result = correction.correct_tensor(point, edges)
    np.testing.assert_allclose(result.detach().numpy(), correction.correct(point.detach().numpy(), edges.numpy()))
    assert torch.autograd.gradcheck(lambda x: correction.correct_tensor(x, edges), (point,))
    with pytest.raises(ValueError, match='radius'):
        correction.correct_tensor(torch.tensor([2., 0.]), edges)


def test_corrected_geometry_objective_keeps_physical_residual_derivative():
    class Predictor:
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((0.2 + 0.1 * value, 0.7 + 0.1 * value), dim=1).unsqueeze(1)
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    base = torch.tensor([source.parameters['modes'][0]['cos'], source.parameters['threshold']], dtype=torch.float64)
    scales = torch.tensor([0.2, 0.1], dtype=torch.float64)
    correction = LocalEdgeCorrection(np.array([0., 0.]), 1.,
        np.array([[0.01, -0.02], [0.03, 0.04], [-0.05, 0.06]]), 'physical', 'model', 3)
    point = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    def objective(z):
        return corrected_fourier_margin(source, Predictor(), base + scales*z, z, correction,
            coefficient_keys=[(0, 'cos')], master_resolution=64, input_resolution=16,
            temperature=0.2, lower_band=1, target_frequency=0.5, beta=100)
    assert torch.autograd.gradcheck(objective, (point,), eps=1e-5, atol=1e-5, rtol=1e-4)
