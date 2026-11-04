"""Exercise trial pairing, resume identity and explicit failure handling."""
import json
from types import SimpleNamespace
import pytest
import torch

from phc2d.inverse import differentiable_campaign_runner as runner


def test_trials_resume_without_repeating_and_reject_changed_identity(tmp_path, monkeypatch):
    calls = []
    def run(config, source, predictor, role):
        calls.append((config['initial'], role))
        return {'history': [{}] * config['evaluations'], 'selection': {'selected_index': 0}}
    monkeypatch.setattr(runner, 'run_shard', run)
    cfg = {'initial': [0.0, 0.0], 'evaluations': 60, 'checkpoint_sha256': 'checkpoint',
           'source_sha256': 'source-digest', 'k_points_sha256': 'k-digest'}
    manifest = {'trials': [{'source_id': 8, 'method': 'adam', 'initial_index': i,
                            'trial_id': f'trial_{i}', 'status': 'ready',
                            'config': {**cfg, 'initial': [i * 0.1, 0.0]}} for i in range(3)]}
    args = dict(manifest=manifest, manifest_sha256='manifest', source_id=8, method='adam',
                source=SimpleNamespace(parameter_hash='source'),
                predictor=SimpleNamespace(checkpoint_sha256='checkpoint', k_points=torch.tensor([[0., 0.]])), runtime={},
                code_commit='commit', cluster_id='paracloud_n40', output=tmp_path)
    assert len(runner.run_trials(**args)) == len(calls) == 3
    assert len(runner.run_trials(**args)) == 3 and len(calls) == 3
    result = json.loads((tmp_path/'trial_1.json').read_text())
    assert result['config']['initial'] == [0.1, 0.0]
    with pytest.raises(ValueError, match='identity'):
        runner.run_trials(**{**args, 'manifest_sha256': 'different'})


def test_trial_failure_is_not_silently_retried(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('solver error')
    monkeypatch.setattr(runner, 'run_shard', fail)
    cfg = {'initial': [0.0, 0.0], 'evaluations': 60, 'checkpoint_sha256': 'checkpoint'}
    manifest = {'trials': [{'source_id': 8, 'method': 'adam', 'initial_index': i,
                            'trial_id': f'trial_{i}', 'status': 'ready', 'config': cfg} for i in range(3)]}
    args = dict(manifest=manifest, manifest_sha256='manifest', source_id=8, method='adam',
                source=SimpleNamespace(parameter_hash='source'),
                predictor=SimpleNamespace(checkpoint_sha256='checkpoint'), runtime={},
                code_commit='commit', cluster_id='paracloud_n40', output=tmp_path)
    with pytest.raises(RuntimeError, match='solver error'):
        runner.run_trials(**args)
    assert (tmp_path/'trial_0.failure.json').exists()
    with pytest.raises(FileExistsError, match='reconciled'):
        runner.run_trials(**args)
