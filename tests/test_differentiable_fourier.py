"""Check differentiable Fourier geometry against the existing hard contract.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.data.full_cell_raster import build_full_cell_raster_pyramid
from phc2d.data.differentiable_fourier import fourier_material_weight


def test_hard_geometry_matches_existing_edited_structure():
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    value = source.parameters['modes'][0]['cos'] + 0.13
    threshold = source.parameters['threshold'] + 0.03
    edited = update_fourier_binary(source, sample_id='edited', coefficients={(0, 'cos'): value},
                                  threshold=threshold, max_mode_radius=1)
    expected = build_full_cell_raster_pyramid(edited, resolutions=[16], source_resolution=64).grids[16]
    actual = fourier_material_weight(source, torch.tensor([value, threshold], dtype=torch.float64),
                                     coefficient_keys=[(0, 'cos')], master_resolution=64,
                                     output_resolution=16, temperature=None)
    np.testing.assert_allclose(actual.numpy(), expected, atol=0, rtol=0)


def test_parameter_gradient_matches_finite_difference():
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    params = torch.tensor([source.parameters['modes'][0]['cos'], source.parameters['threshold']],
                          dtype=torch.float64, requires_grad=True)
    def objective(x):
        grid = fourier_material_weight(source, x, coefficient_keys=[(0, 'cos')],
                                       master_resolution=64, output_resolution=16, temperature=0.2)
        weights = torch.arange(grid.numel(), dtype=grid.dtype).reshape(grid.shape)
        return (grid * weights).mean()
    assert torch.autograd.gradcheck(objective, (params,), eps=1e-5, atol=1e-5, rtol=1e-4)
