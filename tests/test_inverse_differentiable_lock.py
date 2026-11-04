"""Check origin-sub-batch assembly and cross-platform identity checks.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import copy
import pytest

from phc2d.inverse.differentiable_lock import combine_shards


def test_lock_checks_identity_roster_and_hard_ranking():
    labels = ['center', 'gradient_0.05', 'negative_gradient_0.05', 'random_0.05',
              'negative_random_0.05', 'gradient_0.1', 'negative_gradient_0.1',
              'random_0.1', 'negative_random_0.1']
    identity = dict(config_sha256='config', source_sha256='source', source_parameter_hash='p',
                    checkpoint_sha256='model', k_points_sha256='k', code_commit='commit', k_points=[[0, 0]],
                    device_parity_tolerance=1e-5)
    def candidate(i):
        return {'index': i, 'status': 'surrogate_only', 'master_grid_sha256': str(i),
                'predicted_gap': {'target_margin': float(i)}, 'bands': [[0.4, 0.6]]}
    shards = {'directions': dict(identity, role='directions', cluster_id='paracloud_n40', history=[{'label': x} for x in labels],
              diagnostics={'fd_passed': True}, selection={'candidates': [candidate(i) for i in range(9)]})}
    for role in ('adam', 'de', 'random'):
        shards[role] = dict(identity, role=role, cluster_id='paracloud_n40' if role == 'adam' else 'paracloud_n56', history=[{}]*20,
                            selection={'selected_index': 19, 'candidates': [candidate(i) for i in range(20)]})
    lock = combine_shards(shards)
    assert len(lock['records']) == 12
    assert lock['records'][-1]['duplicate_of'] == 9
    altered = copy.deepcopy(shards)
    altered['de']['checkpoint_sha256'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        combine_shards(altered)
    altered = copy.deepcopy(shards)
    altered['adam']['selection']['selected_index'] = 0
    with pytest.raises(ValueError, match='ranking'):
        combine_shards(altered)
    altered = copy.deepcopy(shards)
    altered['directions']['diagnostics']['fd_passed'] = False
    with pytest.raises(ValueError, match='finite-difference'):
        combine_shards(altered)
    altered = copy.deepcopy(shards)
    altered['de']['selection']['candidates'][0]['bands'][0][0] += 0.01
    with pytest.raises(ValueError, match='parity'):
        combine_shards(altered)
