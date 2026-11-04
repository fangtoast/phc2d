"""Check cross-platform preflight comparison.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import copy
import pytest
from phc2d.inverse.differentiable_preflight import compare_preflights


def test_preflight_requires_matching_geometry_model_and_spectrum():
    first = {'role': 'preflight', 'config_sha256': 'c', 'source_sha256': 's',
             'source_parameter_hash': 'p', 'checkpoint_sha256': 'm', 'k_points_sha256': 'k',
             'code_commit': 'code', 'device_parity_tolerance': 1e-5,
             'diagnostics': {'fd_passed': True}, 'center': {'status': 'surrogate_only',
             'parameter_hash': 'p', 'master_grid_sha256': 'g', 'input_sha256': 'i', 'bands': [[0.4, 0.6]]}}
    second = copy.deepcopy(first)
    second['center']['bands'][0][0] += 1e-6
    assert compare_preflights([first, second])['status'] == 'passed'
    second['center']['bands'][0][0] += 1e-3
    with pytest.raises(ValueError, match='parity'):
        compare_preflights([first, second])
    second = copy.deepcopy(first)
    second['center']['master_grid_sha256'] = 'different'
    with pytest.raises(ValueError, match='geometry'):
        compare_preflights([first, second])
