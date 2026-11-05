"""Ensure high-resolution replays preserve preregistered physical candidates."""
import copy

import pytest

from phc2d.inverse.independent_high_accuracy import build_high_accuracy_replay


def test_high_accuracy_replay_tightens_physics_without_reselecting():
    base = {'target_frequency': .57, 'solver_config': {'resolution': 192}, 'physical_protocol': {
        'offsets': [0.0], 'refinement': {'initial_spacing': 1/9, 'edge_tolerance': .0005,
        'resolution_tolerance': .0005, 'verification_resolution': 224}}}
    manifest = {'experiment_id': 'EXP-INV-016', 'base_config_sha256': 'base', 'trials': [{'trial_id': 'a'}]}
    lock = {'scope': 'locked_paired_surrogate_campaign', 'manifest_sha256': 'manifest', 'records': [
        {'label': 'a/top1', 'role': 'top1', 'candidate': {'master_grid_sha256': 'grid'}},
        {'label': 'a/evaluation_20', 'role': 'evaluation_20', 'candidate': {'master_grid_sha256': 'other'}}]}
    registration = {'experiment_id':'EXP-INV-016', 'target_frequency':.57, 'roles':['a/top1'], 'offsets':[0.0],
                    'resolution':224, 'verification_resolution':256, 'initial_spacing':1/18,
                    'edge_tolerance':.00025, 'resolution_tolerance':.00025}
    original = copy.deepcopy(base)
    new_base, new_manifest, new_lock = build_high_accuracy_replay(lock, manifest, base, registration,
        lock_sha256='lock', manifest_sha256='manifest', base_sha256='base', registration_sha256='registered')
    assert base == original
    assert new_base['solver_config']['resolution'] == 224
    assert new_base['physical_protocol']['refinement']['verification_resolution'] == 256
    assert new_lock['records'] == [lock['records'][0]]
    assert new_manifest['parent_nominal_manifest_sha256'] == 'manifest'
    registration['roles'] = ['a/evaluation_20']
    with pytest.raises(ValueError, match='selected endpoints'):
        build_high_accuracy_replay(lock, manifest, base, registration,
            lock_sha256='lock', manifest_sha256='manifest', base_sha256='base', registration_sha256='registered')
