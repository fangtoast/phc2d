"""Exercise cross-environment comparison, identity and missing cases."""
import copy
import pytest
from phc2d.inverse.differentiable_crosscheck import compare_crosscheck


def test_crosscheck_rejects_mismatch_and_reports_missing():
    manifest = {'protocol_sha256': 'protocol', 'k_points': [[0, 0], [0.5, 0]],
                'rows': [{'parameter_hash': 'params', 'master_grid_sha256': 'grid'}],
                'role_coverage': [{'label': 'center', 'case_index': 0}]}
    result = {'case_index': 0, 'parameter_hash': 'params', 'manifest_sha256': 'manifest',
              'protocol_sha256': 'protocol', 'elapsed_s': 1,
              'aggregate': {'status': 'numerically_stable'},
              'observations': [{'provenance': {'evaluated_grid_sha256': 'grid'},
                                'arrays': {'k_points': [[0, 0], [0.5, 0]],
                                           'bands': [[0.1, 0.2], [0.3, 0.4]]}}]}
    args = dict(manifest=manifest, roles=['center'], reference={0: result},
                tolerance=1e-6, manifest_sha256='manifest')
    assert compare_crosscheck(**args, replay={0: result})['all_passed']
    assert compare_crosscheck(**args, replay={})['roles'][0]['status'] == 'missing'
    other = copy.deepcopy(result)
    other['observations'][0]['arrays']['bands'][0][0] += 2e-6
    assert not compare_crosscheck(**args, replay={0: other})['all_passed']
    other['manifest_sha256'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        compare_crosscheck(**args, replay={0: other})
