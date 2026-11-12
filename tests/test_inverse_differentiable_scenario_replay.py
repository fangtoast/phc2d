"""Preserve nominal endpoint selections when registering manufacturing replays."""
import copy

import pytest

from phc2d.inverse.differentiable_robust_unseen import build_registered_scenario_replay


def test_scenario_replay_uses_only_preregistered_endpoint_roles():
    manifest = {'scope': 'preregistered_surrogate_design', 'experiment_id': 'EXP-INV-016',
                'base_config_sha256': 'base', 'trials': [{'trial_id': 'a'}]}
    lock = {'scope': 'locked_paired_surrogate_campaign', 'manifest_sha256': 'manifest', 'records': [
        {'label': 'a/top1', 'role': 'top1', 'candidate': {'master_grid_sha256': 'grid'}},
        {'label': 'a/eval20', 'role': 'eval20', 'candidate': {'master_grid_sha256': 'other'}}]}
    base = {'physical_protocol': {'offsets': [0.0], 'raster_resolution': 512}}
    registration = {'experiment_id': 'EXP-INV-016', 'offsets': [-1/512, 0., 1/512], 'roles': ['a/top1']}
    original = copy.deepcopy(lock)
    scenario, selection = build_registered_scenario_replay(lock, manifest, base, registration,
        lock_sha256='lock', manifest_sha256='manifest', base_sha256='base', registration_sha256='registered')
    assert scenario['physical_protocol']['offsets'] == registration['offsets']
    assert selection['records'] == [lock['records'][0]]
    assert selection['parent_selection_lock_sha256'] == 'lock'
    assert lock == original
    registration['roles'] = ['a/eval20']
    with pytest.raises(ValueError, match='selected endpoints'):
        build_registered_scenario_replay(lock, manifest, base, registration,
            lock_sha256='lock', manifest_sha256='manifest', base_sha256='base', registration_sha256='registered')
    lock['records'][0]['label'], lock['records'][0]['role'] = 'a/final', 'final'
    registration['roles'] = ['a/final']
    _, selection = build_registered_scenario_replay(lock, manifest, base, registration,
        lock_sha256='lock', manifest_sha256='manifest', base_sha256='base', registration_sha256='registered')
    assert selection['records'][0]['label'] == 'a/final'
