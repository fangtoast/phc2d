"""Unseen manufacturing replays preserve locked candidates and role registration."""
import copy

import pytest

from phc2d.inverse.differentiable_robust_unseen import build_unseen_replay


def test_unseen_replay_keeps_top1_without_reusing_design_spectrum():
    manifest = {'scope': 'preregistered_robust_design', 'experiment_id': 'EXP-INV-015',
                'offsets': [-1/512, 0., 1/512], 'physical_protocol': {'offsets': [-1/512, 0., 1/512]}}
    lock = {'purpose': 'robust_design', 'manifest_sha256': 'parent', 'records': [
        {'label': 'a/top1', 'role': 'top1', 'candidate': {'master_grid_sha256': 'grid', 'robust_prediction': {}}},
        {'label': 'b/top1', 'role': 'top1', 'candidate': {'master_grid_sha256': 'other'}}]}
    registration = {'experiment_id': 'EXP-INV-015', 'offsets': [-2/512, 0., 2/512], 'roles': ['a/top1']}
    original = copy.deepcopy(lock)
    new_manifest, new_lock = build_unseen_replay(lock, manifest, registration, lock_sha256='lock',
        manifest_sha256='parent', registration_sha256='registration')
    assert new_manifest['physical_protocol']['offsets'] == registration['offsets']
    assert new_lock['records'] == [{'label': 'a/top1', 'role': 'top1', 'candidate': {'master_grid_sha256': 'grid'}}]
    assert new_lock['parent_selection_lock_sha256'] == 'lock'
    assert lock == original
    registration['offsets'] = manifest['offsets']
    with pytest.raises(ValueError, match='unseen offsets'):
        build_unseen_replay(lock, manifest, registration, lock_sha256='lock', manifest_sha256='parent',
                            registration_sha256='registration')
