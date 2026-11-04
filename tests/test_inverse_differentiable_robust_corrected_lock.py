"""Verify corrected robust selection is frozen before physical replay."""
import copy
import hashlib

import numpy as np
import pytest

from phc2d.inverse.differentiable_robust_corrected_lock import lock_corrected_robust_campaign


def test_corrected_lock_uses_soft_objective_without_reselecting_by_physics():
    points = [[0., 0.]]
    k_hash = hashlib.sha256(np.asarray(points, dtype=np.float32).tobytes()).hexdigest()
    runtime = {'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False}
    base = {'initial': [0., 0.], 'offsets': [-.01, 0., .01], 'source_sha256': 'source',
            'checkpoint_sha256': 'checkpoint', 'k_points_sha256': k_hash, 'runtime': runtime}
    cfg = {**base, 'center': [0., 0.], 'radius': .2, 'ridge': 1e-6,
           'surrogate_contract_sha256': 'model', 'scenario_reduction': 'worst'}
    spec = {'trial_id': 'trial', 'source_id': 8, 'initial_index': 0, 'mode': 'worst',
            'status': 'ready', 'config': base}
    manifest = {'scope': 'preregistered_robust_design', 'offsets': base['offsets'], 'evaluations': 2,
        'feedback': {'radius': .2, 'ridge': 1e-6}, 'sources': [{'source_id': 8, 'source_parameter_hash': 'p'}],
        'trials': [spec]}
    candidates = [{'index': 0, 'status': 'surrogate_only', 'master_grid_sha256': 'a'},
                  {'index': 1, 'status': 'surrogate_only', 'master_grid_sha256': 'b'}]
    diagnostic = [{'index': 0, 'status': 'surrogate_only', 'master_grid_sha256': 'a', 'robust_prediction': {'id': 'a'}},
                  {'index': 1, 'status': 'surrogate_only', 'master_grid_sha256': 'b', 'robust_prediction': {'id': 'b'}}]
    item = {'trial_id': 'trial_corrected', 'manifest_sha256': 'manifest', 'feedback_sha256': 'feedback',
        'code_commit': 'commit', 'cluster_id': 'n56', 'status': 'completed', 'mode': 'worst', 'source_id': 8,
        'source_parameter_hash': 'p', 'source_sha256': 'source', 'initial_index': 0, 'checkpoint_sha256': 'checkpoint',
        'k_points_sha256': k_hash, 'k_points': points, 'runtime': {'contract': runtime,
            'matmul_tf32_enabled': False, 'cudnn_tf32_enabled': False}, 'config': cfg,
        'history': [{'parameters': [0., 0.], 'objective': .1}, {'parameters': [.1, 0.], 'objective': .2}],
        'hard_screening': {'candidates': candidates}, 'selected_index': 1,
        'prefit_prediction': {'point': [.1, 0.]}, 'full_scenarios_diagnostic': {'candidates': diagnostic}}
    handoff, lock = lock_corrected_robust_campaign(manifest, {'trial_corrected': item},
        manifest_sha256='manifest', feedback_sha256='feedback')
    assert handoff['trials'][0]['trial_id'] == 'trial_corrected'
    assert lock['records'][1]['candidate']['robust_prediction']['id'] == 'b'
    bad = copy.deepcopy(item); bad['selected_index'] = 0
    with pytest.raises(ValueError, match='selection'):
        lock_corrected_robust_campaign(manifest, {'trial_corrected': bad},
            manifest_sha256='manifest', feedback_sha256='feedback')
