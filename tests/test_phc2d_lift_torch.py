"""Optional torch lifting tests."""

from __future__ import annotations

import numpy as np
import pytest

from phc2d.data.lift import lift_rho_to_grid, wedge_indices

torch = pytest.importorskip("torch")


def test_lift_rho_to_grid_torch_matches_numpy_and_keeps_grad():
    from phc2d.data.lift_torch import lift_rho_to_grid_torch

    grid_shape = (8, 8)
    n_wedge = len(wedge_indices(grid_shape))
    rho_np = np.zeros((1, n_wedge), dtype=np.float32)
    rho_np[0, [0, n_wedge // 3, n_wedge - 1]] = 1.0
    rho = torch.tensor(rho_np, requires_grad=True)

    lifted = lift_rho_to_grid_torch(rho, grid_shape=grid_shape)
    expected = lift_rho_to_grid(rho_np[0], grid_shape=grid_shape).astype(np.float32)

    assert tuple(lifted.shape) == (1, 1, *grid_shape)
    assert np.allclose(lifted.detach().numpy()[0, 0], expected)

    lifted.sum().backward()
    assert rho.grad is not None
    assert float(rho.grad.abs().sum()) > 0.0
