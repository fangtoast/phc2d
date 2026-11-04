"""Exercise complete pure-surrogate MVP shards with a small predictor.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import pytest
import json
import sys

torch = pytest.importorskip('torch')
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.differentiable_mvp import run_shard
from phc2d.inverse import differentiable_mvp


def test_mvp_roles_share_geometry_and_budget():
    class Predictor:
        device = torch.device('cpu')
        def predict_tensor(self, inputs):
            value = inputs.mean(dim=(1, 2))
            low = 0.3 + 0.1 * value
            return torch.stack((low, low + 0.3), dim=1).unsqueeze(1)

    source = generate_unified_fourier_binary('source', master_seed=12, material_fraction_bin=1,
                                             config={'fourier_cutoff': 3, 'audit_resolution': 32})
    config = {'coefficient_keys': [[0, 'cos']], 'scales': [0.2, 0.1], 'lower': [-0.5, -0.5],
              'upper': [0.5, 0.5], 'source_resolution': 64, 'input_resolution': 16,
              'temperature': 0.2, 'lower_band': 1, 'target_frequency': 0.5, 'beta': 100,
              'learning_rate': 0.1, 'evaluations': 5, 'seed': 10, 'max_mode_radius': 1,
              'material_fraction_bounds': [0, 1], 'finite_difference_steps': [0.01, 0.003],
              'direction_radii': [0.05, 0.1], 'gradient_fd_absolute_tolerance': 0.00002,
              'gradient_fd_relative_tolerance': 0.05, 'gradient_fd_minimum_steps': 2}
    results = {role: run_shard(config, source, Predictor(), role=role)
               for role in ('directions', 'adam', 'de', 'random')}
    assert len(results['directions']['history']) == 9
    assert [row['label'] for row in results['directions']['history']][0] == 'center'
    assert all(len(results[role]['history']) == 5 for role in ('adam', 'de', 'random'))
    assert all(results[role]['selection']['selected_index'] is not None for role in results)
    assert results['adam']['history'][0]['objective'] == pytest.approx(results['de']['history'][0]['objective'])
    assert len(results['directions']['diagnostics']['finite_differences']) == 2


def test_failed_preflight_keeps_diagnostic_values(tmp_path, monkeypatch):
    config = {'source_sha256': 'source', 'checkpoint_sha256': 'model', 'k_points_sha256': 'k',
              'input_resolution': 16, 'output_shape': [2, 2], 'local_kernel': 3,
              'dataset_domain': 'k5_binary', 'model_seed': 0, 'architecture_fingerprint': 'fingerprint',
              'runtime': {'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False},
              'solver_config': {'epsilon_air': 1., 'epsilon_material': 8.9}}
    cfg = tmp_path/'config.json'; cfg.write_text(json.dumps(config))
    class Source:
        epsilon_air, epsilon_material = 1., 8.9
    class Predictor:
        model_name, k_points_sha256, input_shape, output_shape = 'global_local_band_operator_v1', 'k', (16, 16), (2, 2)
    monkeypatch.setattr(differentiable_mvp, 'read_checked_json', lambda *args: ({}, 'source'))
    monkeypatch.setattr(differentiable_mvp.FullCellStructureSpec, 'from_mapping', lambda data: Source())
    monkeypatch.setattr(differentiable_mvp, 'FrozenBandAdapter', lambda *args, **kwargs: Predictor())
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'model_params': {'local_kernel': 3},
        'resolved_config': {'experiment': {'domain': 'k5_binary', 'seed': 0}, 'runtime': config['runtime']},
        'model_identity': {'architecture_fingerprint': 'fingerprint'}})
    monkeypatch.setattr(differentiable_mvp, 'run_shard', lambda *args, **kwargs: {
        'selection': {'candidates': [{'status': 'surrogate_only'}]}, 'elapsed_s': 1,
        'diagnostics': {'fd_passed': False, 'gradient': [0.123]}})
    out = tmp_path/'preflight.json'
    monkeypatch.setattr(sys, 'argv', ['mvp', '--config', str(cfg), '--source', 'source',
        '--checkpoint', 'model', '--role', 'preflight', '--output', str(out)])
    with pytest.raises(ValueError, match='gradient gate'):
        differentiable_mvp.main()
    failure = json.loads((tmp_path/'preflight.json.failure.json').read_text())
    assert failure['partial_result']['diagnostics']['gradient'] == [0.123]
