"""Verify robust selections are reproducible before the MPB handoff."""
import copy
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_robust import run_robust_shard
from phc2d.inverse.differentiable_robust_lock import lock_robust_campaign


def test_lock_checks_scores_initials_and_all_three_modes():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, image):
            value = image.mean(dim=(1, 2))
            return torch.stack((.2 + .1*value, .7 + .1*value), dim=1).unsqueeze(1)
    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
        config={'fourier_cutoff': 3, 'audit_resolution': 32})
    points = [[0., 0.]]
    k_hash = hashlib.sha256(np.asarray(points, dtype=np.float32).tobytes()).hexdigest()
    runtime_contract = {'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False}
    cfg = {'offsets': [-1/64, 0., 1/64], 'coefficient_keys': [(0, 'cos')], 'scales': [.2, .1],
        'lower': [-.5, -.5], 'upper': [.5, .5], 'source_resolution': 64, 'input_resolution': 16,
        'temperature': .2, 'lower_band': 1, 'target_frequency': .5, 'beta': 100, 'learning_rate': .05,
        'evaluations': 2, 'initial': [.15, 0.], 'max_mode_radius': 1, 'material_fraction_bounds': [0., 1.],
        'source_sha256': 'source', 'checkpoint_sha256': 'checkpoint', 'k_points_sha256': k_hash,
        'device_parity_tolerance': 1e-5, 'runtime': runtime_contract}
    cfg['coefficient_keys'] = [[0, 'cos']]
    modes = ('nominal', 'mean', 'worst')
    specs = [{'trial_id': mode, 'source_id': 8, 'initial_index': 0, 'mode': mode,
              'method': mode, 'status': 'ready', 'config': cfg} for mode in modes]
    manifest = {'scope': 'preregistered_robust_design', 'sources': [{'source_id': 8,
        'source_parameter_hash': source.parameter_hash}], 'evaluations': 2, 'offsets': cfg['offsets'], 'trials': specs}
    results = {}
    for mode in modes:
        results[mode] = {**run_robust_shard(cfg, source, Predictor(), mode=mode), 'status': 'completed',
            'trial_id': mode, 'manifest_sha256': 'manifest', 'code_commit': 'commit', 'cluster_id': 'n56',
            'source_id': 8, 'initial_index': 0, 'source_parameter_hash': source.parameter_hash,
            'source_sha256': 'source', 'checkpoint_sha256': 'checkpoint', 'k_points_sha256': k_hash,
            'k_points': points, 'config': cfg, 'runtime': {'contract': runtime_contract,
                'matmul_tf32_enabled': False, 'cudnn_tf32_enabled': False}}
    locked = lock_robust_campaign(manifest, results, manifest_sha256='manifest')
    assert len(locked['records']) == 6 and sum(row['role'] == 'top1' for row in locked['records']) == 3
    bad = copy.deepcopy(results)
    idx = bad['mean']['selection']['selected_index']
    bad['mean']['selection']['candidates'][idx]['robust_prediction']['selection_score'] += .1
    with pytest.raises(ValueError, match='score mismatch'):
        lock_robust_campaign(manifest, bad, manifest_sha256='manifest')
