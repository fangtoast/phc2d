"""Test same-grid physical observation reuse and numerical gate preservation."""
import copy
import pytest

from phc2d.inverse.differentiable_feedback_observations import bind_seed_observations


def test_seed_observations_reuse_only_identical_grids():
    edges = {'hard_edges': [.4, .6], 'soft_exact_edges': [.39, .61], 'soft_smooth_edges': [.41, .59]}
    records = [{'label': f'trial/offset_{i}', 'trial_id': 'trial', 'role': f'offset_{i}',
                'point': [i * .1, 0.], 'candidate': {'parameter_hash': x, 'master_grid_sha256': 'grid'},
                'surrogate_edges': edges} for i, x in enumerate(('a', 'b'))]
    lock = {'purpose': 'feedback_initial_stencil', 'records': records}
    manifest = {'selection_lock_sha256': 'lock', 'protocol_sha256': 'protocol', 'k_points': [[0., 0.]],
                'rows': [{'parameter_hash': 'a', 'master_grid_sha256': 'grid'}],
                'role_coverage': [{'label': r['label'], 'case_index': 0} for r in records]}
    stable = {'lower_band_one_based': 1, 'refined_lower_edge_frequency': .38, 'refined_upper_edge_frequency': .62,
              'last_refinement_stable': True, 'resolution_stable': True, 'adaptive_refinement_converged': True}
    result = {'case_index': 0, 'manifest_sha256': 'manifest', 'protocol_sha256': 'protocol', 'parameter_hash': 'a',
              'aggregate': {'status': 'numerically_stable', 'row': stable,
                            'provenance': {'parameter_hash': 'a', 'comparison_contract_sha256': 'physical'}},
              'observations': [{'offset': 0.0, 'provenance': {'evaluated_grid_sha256': 'grid'},
                                'arrays': {'k_points': [[0., 0.]], 'bands': [[.38, .62]]}}]}
    base = {'checkpoint_sha256': 'model', 'k_points_sha256': 'k', 'input_resolution': 16,
            'source_resolution': 64, 'temperature': .2, 'beta': 100, 'lower_band': 1,
            'coefficient_keys': [[0, 'cos']], 'scales': [.2, .1]}
    kwargs = dict(seed_lock=lock, manifest=manifest, base=base, lock_sha256='lock', manifest_sha256='manifest')
    output = bind_seed_observations(results={0: result}, **kwargs)
    assert [row['status'] for row in output['records']] == ['observed', 'observed']
    assert output['records'][1]['composite']['parameter_hash'] == 'b'
    assert output['records'][1]['physical']['provenance']['reused_from_parameter_hash'] == 'a'
    bad = copy.deepcopy(result); bad['observations'][0]['provenance']['evaluated_grid_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='raster'):
        bind_seed_observations(results={0: bad}, **kwargs)
    bad = copy.deepcopy(result); bad['aggregate']['status'] = 'numerical_gate_failed'
    assert all(row['status'] == 'physical_numerical_gate_failed' for row in bind_seed_observations(results={0: bad}, **kwargs)['records'])
