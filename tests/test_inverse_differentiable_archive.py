"""Check run archive binds its configuration, lock and all shard bytes.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import hashlib
import json
import pytest
from phc2d.inverse.differentiable_archive import archive_design
from phc2d.inverse.differentiable_lock import combine_shards


def test_archive_requires_exact_shards_and_keeps_design_scope(tmp_path):
    roles = ('directions', 'adam', 'de', 'random')
    config = json.dumps({'experiment_id': 'EXP-INV-012', 'batch': 'origin'}).encode()
    labels = ['center', 'gradient_0.05', 'negative_gradient_0.05', 'random_0.05',
              'negative_random_0.05', 'gradient_0.1', 'negative_gradient_0.1',
              'random_0.1', 'negative_random_0.1']
    identity = dict(config_sha256=hashlib.sha256(config).hexdigest(), source_sha256='source',
        source_parameter_hash='p', checkpoint_sha256='model', k_points_sha256='points',
        code_commit='commit', k_points=[[0, 0]], device_parity_tolerance=1e-5,
        precision_contract={'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False})
    def candidate(index):
        return {'index': index, 'status': 'surrogate_only', 'master_grid_sha256': str(index),
                'predicted_gap': {'target_margin': float(index)}, 'bands': [[0.4, 0.6]]}
    parsed = {}
    for role in roles:
        count = 9 if role == 'directions' else 20
        history = [{'step': i, 'parameters': [0.0], 'fourier_parameters': [0.1],
                    **({'label': labels[i]} if role == 'directions' else {})} for i in range(count)]
        parsed[role] = dict(identity, role=role, cluster_id='paracloud_n40' if role in ('directions', 'adam') else 'paracloud_n56', elapsed_s=1.0, history=history,
                            selection={'selected_index': count-1, 'candidates': [candidate(i) for i in range(count)]})
    parsed['directions']['diagnostics'] = {'fd_passed': True, 'gradient': [0.1],
                                            'finite_differences': [{'step': 0.01}]}
    shards = {role: json.dumps(parsed[role]).encode() for role in roles}
    data = combine_shards(parsed)
    data['shard_sha256'] = {role: hashlib.sha256(raw).hexdigest() for role, raw in shards.items()}
    lock = json.dumps(data).encode()
    archive_design(config_raw=config, lock_raw=lock, shard_raw=shards, output=tmp_path/'run')
    result = json.loads((tmp_path/'run/results.json').read_text())
    assert result['status'] == 'design_locked' and result['physical_validation'] == 'not_run'
    metrics = json.loads((tmp_path/'run/metrics.json').read_text())
    assert metrics['inference_counts_derived_from_completed_shards']['adam']['backward'] == 20
    assert metrics['inference_counts_derived_from_completed_shards']['directions']['forward_without_grad'] == 11
    assert (tmp_path/'run/shards/adam.json').read_bytes() == shards['adam']
    for role in ('directions', 'adam'):
        parsed[role]['cluster_id'] = 'paracloud_n56'
    takeover_shards = {role: json.dumps(parsed[role]).encode() for role in roles}
    takeover_lock = combine_shards(parsed, n56_takeover=True)
    takeover_lock['shard_sha256'] = {role: hashlib.sha256(raw).hexdigest() for role, raw in takeover_shards.items()}
    archive_design(config_raw=config, lock_raw=json.dumps(takeover_lock).encode(),
                   shard_raw=takeover_shards, output=tmp_path/'takeover')
    assert json.loads((tmp_path/'takeover/selection.lock.json').read_text())['platforms_by_role']['adam'] == 'paracloud_n56'
    with pytest.raises(ValueError, match='hash mismatch'):
        archive_design(config_raw=config, lock_raw=lock, shard_raw={**shards, 'de': b'bad'}, output=tmp_path/'bad')
    data['records'][0]['label'] = 'tampered'
    with pytest.raises(ValueError, match='reconstructed'):
        archive_design(config_raw=config, lock_raw=json.dumps(data).encode(), shard_raw=shards, output=tmp_path/'bad2')
