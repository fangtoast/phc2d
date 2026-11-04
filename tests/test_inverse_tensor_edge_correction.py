"""Check local edge corrections preserve the design gradient.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from phc2d.inverse.local_edge_correction import LocalEdgeCorrection


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
