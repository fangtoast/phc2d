"""Tests for paired perturbation grids and common gap calculation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.perturbation_screen import (periodic_perturbation_grids, predict_perturbed_gap,
                                               project_perturbed_band_order, solve_perturbed_gap)
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def setup():
    s = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    return s, periodic_perturbation_grids(s, resolution=16, offsets=(-1/16, 0, 1/16))


def test_proxy_and_solver_share_grid_identity():
    s, grids = setup()
    class Predictor:
        def predict_tensor(self, x):
            return torch.tensor([[[0.2, 0.8]]] * len(x))
    proxy = predict_perturbed_gap(grids, Predictor(), input_resolution=8, lower_band=1, target_frequency=.5)
    def solver(grid, *, k_points, config):
        digest = hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest()
        return np.array([[.2, .8]] * len(k_points)), {'grid_sha256': digest}
    c = FullCellMpbBandSolverConfig(num_bands=2, resolution=8, polarization='TE',
                                    epsilon_air=s.epsilon_air, epsilon_material=s.epsilon_material)
    physical = solve_perturbed_gap(grids, k_points=np.array([[0, 0]]), config=c,
                                   lower_band=1, target_frequency=.5, solver=solver)
    assert proxy['gap']['target_margin'] == pytest.approx(physical['gap']['target_margin'])
    assert proxy['grid_sha256'] == [m['grid_sha256'] for m in physical['solver_metadata']]
    assert physical['status'] == 'sampled_solver_only'


def test_offsets_require_nominal_and_unique_values():
    s, _ = setup()
    for offsets in ((.1,), (0, 0)):
        with pytest.raises(ValueError, match='offsets'):
            periodic_perturbation_grids(s, resolution=16, offsets=offsets)


def test_invalid_order_retains_spectra_without_a_score():
    _, grids = setup()
    class Predictor:
        def predict_tensor(self, x):
            return torch.tensor([[[.2, .8, .7]]] * len(x))
    result = predict_perturbed_gap(grids, Predictor(), input_resolution=8, lower_band=1, target_frequency=.5)
    assert result['status'] == 'band_order_violation' and result['gap'] is None
    assert np.asarray(result['bands']).shape == (3, 1, 3)
    assert result['bands'][0][0][2] == pytest.approx(.7)


def test_nonfinite_prediction_preserves_locations_without_nonstandard_json():
    import json
    _, grids = setup()
    class Predictor:
        def predict_tensor(self, x):
            return torch.tensor([[[float('nan'), .8]], [[.2, float('inf')]], [[float('-inf'), .8]]])
    result = predict_perturbed_gap(grids, Predictor(), input_resolution=8, lower_band=1, target_frequency=.5)
    assert result['status'] == 'nonfinite_prediction' and result['gap'] is None
    assert result['nonfinite_spectrum'][0][0][0] == 'nan'
    assert result['nonfinite_spectrum'][1][0][1] == '+inf'
    assert result['nonfinite_spectrum'][2][0][0] == '-inf'
    json.dumps(result, allow_nan=False)


def test_projection_retains_raw_invalid_spectrum_and_scores_new_output():
    import copy
    _, grids = setup()
    class Predictor:
        def predict_tensor(self, x):
            return torch.tensor([[[.2, .8, .7]]] * len(x))
    original=predict_perturbed_gap(grids, Predictor(), input_resolution=8, lower_band=1, target_frequency=.5)
    snapshot=copy.deepcopy(original)
    projected=project_perturbed_band_order(original,lower_band=1,target_frequency=.5)
    assert original==snapshot and original['status']=='band_order_violation'
    assert projected['status']=='projected_surrogate_only'
    assert projected['original_status']=='band_order_violation'
    assert projected['raw_band_order_violations']==3
    assert projected['projection_max_abs_change']==pytest.approx(.1)
    assert projected['raw_bands'][0][0]==pytest.approx([.2,.8,.7])
    assert projected['projected_bands'][0][0]==pytest.approx([.2,.7,.8])
    assert projected['gap']['upper_edge']==pytest.approx(.7)


def test_projection_refuses_nonfinite_negative_target_and_forged_status():
    with pytest.raises(ValueError,match='finite raw'):
        project_perturbed_band_order({'status':'nonfinite_prediction'},lower_band=1,target_frequency=.5)
    raw={'status':'negative_frequency','bands':[[[-.2,.1,.8]]],'grid_sha256':['x']}
    with pytest.raises(ValueError,match='negative frequency'):
        project_perturbed_band_order(raw,lower_band=1,target_frequency=.5)
    with pytest.raises(ValueError,match='status disagrees'):
        project_perturbed_band_order(dict(raw,status='surrogate_only'),lower_band=1,target_frequency=.5)


def test_projection_preserves_count_of_negative_non_target_frequencies():
    raw={'status':'surrogate_only','bands':[[[-.01,.3,.7]]],'grid_sha256':['x']}
    result=project_perturbed_band_order(raw,lower_band=2,target_frequency=.5)
    assert result['raw_negative_frequency_count']==1
    assert result['projection_max_abs_change']==0
    assert result['gap']['target_margin']==pytest.approx(.8)
