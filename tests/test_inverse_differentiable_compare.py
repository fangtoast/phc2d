"""Check same-k and refined physical comparisons against locked identities.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import copy
import pytest
from phc2d.inverse.differentiable_compare import compare_physical_results


def test_physical_comparison_uses_original_k_and_retains_missing_roles():
    candidate = {'master_grid_sha256': 'grid', 'bands': [[0.3, 0.7], [0.4, 0.8]]}
    lock = {'records': [{'label': 'center', 'candidate': candidate},
                        {'label': 'missing', 'candidate': None}]}
    manifest = {'selection_lock_sha256': 'lock', 'protocol_sha256': 'protocol',
                'k_points': [[0, 0], [0.5, 0]], 'lower_band': 1, 'target_frequency': 0.5,
                'rows': [{'parameter_hash': 'parameter', 'master_grid_sha256': 'grid'}],
                'role_coverage': [{'label': 'center', 'case_index': 0},
                                  {'label': 'missing', 'case_index': None, 'status': 'no_valid_candidate'}]}
    trace = {'offset': 0., 'provenance': {'evaluated_grid_sha256': 'grid'},
             'arrays': {'k_points': [[0, 0], [0.5, 0], [0.25, 0]],
                        'bands': [[0.31, 0.71], [0.41, 0.81], [0.42, 0.69]]},
             'row': {'refined_lower_edge_frequency': 0.42, 'refined_upper_edge_frequency': 0.69}}
    result = {'case_index': 0, 'manifest_sha256': 'manifest', 'protocol_sha256': 'protocol',
              'parameter_hash': 'parameter', 'observations': [trace], 'elapsed_s': 3,
              'aggregate': {'status': 'numerically_stable'}}
    args = dict(lock_sha256='lock', manifest_sha256='manifest', protocol_sha256='protocol')
    report = compare_physical_results(lock, manifest, {0: result}, **args)
    assert report['roles'][0]['same_k_spectrum_mae'] == pytest.approx(0.01)
    assert report['roles'][0]['physical_refined_target_margin'] == pytest.approx(0.32)
    assert report['roles'][1]['status'] == 'no_valid_candidate'
    altered = copy.deepcopy(result)
    altered['observations'][0]['provenance']['evaluated_grid_sha256'] = 'other'
    with pytest.raises(ValueError, match='grid'):
        compare_physical_results(lock, manifest, {0: altered}, **args)
    assert compare_physical_results(lock, manifest, {}, **args)['roles'][0]['status'] == 'physical_result_missing'
