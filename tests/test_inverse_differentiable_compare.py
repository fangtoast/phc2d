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
             'row': {'refined_lower_edge_frequency': 0.42, 'refined_upper_edge_frequency': 0.69,
                     'last_refinement_stable': True, 'resolution_stable': True,
                     'adaptive_refinement_converged': True}}
    result = {'case_index': 0, 'manifest_sha256': 'manifest', 'protocol_sha256': 'protocol',
              'parameter_hash': 'parameter', 'observations': [trace], 'elapsed_s': 3,
              'aggregate': {'status': 'numerically_stable'}}
    args = dict(lock_sha256='lock', manifest_sha256='manifest', protocol_sha256='protocol')
    report = compare_physical_results(lock, manifest, {0: result}, **args)
    assert report['roles'][0]['same_k_spectrum_mae'] == pytest.approx(0.01)
    assert report['roles'][0]['same_k_spectrum_rmse'] == pytest.approx(0.01)
    assert report['roles'][0]['same_k_spectrum_mre_pct'] == pytest.approx(
        100 * sum(0.01 / x for x in (0.31, 0.71, 0.41, 0.81)) / 4)
    assert report['roles'][0]['same_k_edge_error'] == pytest.approx([-0.01, -0.01])
    assert report['roles'][0]['same_k_target_margin_error'] == pytest.approx(0.04)
    assert report['roles'][0]['physical_refined_target_margin'] == pytest.approx(0.32)
    assert report['roles'][0]['physical_refined_contains_target'] is True
    assert report['roles'][0]['physical_refined_margin_change_from_center'] == 0
    assert report['roles'][1]['status'] == 'no_valid_candidate'
    altered = copy.deepcopy(result)
    altered['observations'][0]['provenance']['evaluated_grid_sha256'] = 'other'
    with pytest.raises(ValueError, match='grid'):
        compare_physical_results(lock, manifest, {0: altered}, **args)
    altered = copy.deepcopy(result)
    altered['observations'][0]['row']['resolution_stable'] = False
    with pytest.raises(ValueError, match='numerical gates'):
        compare_physical_results(lock, manifest, {0: altered}, **args)
    assert compare_physical_results(lock, manifest, {}, **args)['roles'][0]['status'] == 'physical_result_missing'
    unresolved = copy.deepcopy(result)
    unresolved['aggregate']['status'] = 'numerical_gate_failed'
    assert compare_physical_results(lock, manifest, {0: unresolved}, **args)['roles'][0]['physical_refined_contains_target'] is None
    zero = copy.deepcopy(result)
    zero['observations'][0]['arrays']['bands'][0][0] = 0.0
    zero_report = compare_physical_results(lock, manifest, {0: zero}, **args)['roles'][0]
    assert zero_report['same_k_floor_count'] == 1
    assert zero_report['same_k_spectrum_mre_pct'] > zero_report['same_k_spectrum_mre_bands_2_plus_pct']
    duplicate_lock = copy.deepcopy(lock)
    duplicate_manifest = copy.deepcopy(manifest)
    duplicate_lock['records'][1] = {'label': 'other', 'candidate': {
        'master_grid_sha256': 'grid', 'bands': [[0.32, 0.72], [0.42, 0.82]]}}
    duplicate_manifest['role_coverage'][1] = {'label': 'other', 'case_index': 0}
    duplicate_report = compare_physical_results(duplicate_lock, duplicate_manifest, {0: result}, **args)
    assert duplicate_report['unique_cases_with_results'] == 1
    assert duplicate_report['physical_case_elapsed_s_sum'] == 3
    assert duplicate_report['roles'][0]['same_k_edge_error'] == pytest.approx([-0.01, -0.01])
    assert duplicate_report['roles'][1]['same_k_edge_error'] == pytest.approx([0.01, 0.01])
    failure = {'case_index': 0, 'manifest_sha256': 'manifest', 'protocol_sha256': 'protocol',
               'error_type': 'TimeoutError', 'elapsed_s': 4.5}
    failed_report = compare_physical_results(duplicate_lock, duplicate_manifest, {}, failures={0: failure}, **args)
    assert [row['status'] for row in failed_report['roles']] == ['physical_solver_failed'] * 2
    assert failed_report['unique_cases_failed'] == 1 and failed_report['physical_case_elapsed_s_sum'] == 4.5
    with pytest.raises(ValueError, match='both'):
        compare_physical_results(lock, manifest, {0: result}, failures={0: failure}, **args)
