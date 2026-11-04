"""Check paired corrected trials use their frozen scenario feedback."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from phc2d.inverse import differentiable_robust_corrected_runner as runner


def test_corrected_trials_use_same_stencil_and_mode_specific_scenarios(tmp_path, monkeypatch):
    calls = []
    def propose(source, predictor, bank, *, config):
        calls.append((bank, config))
        return {'status': 'proposal_ready', 'history': [{'objective': 0.}], 'selected_index': 0}
    monkeypatch.setattr(runner, 'propose_corrected_robust_design', propose)
    monkeypatch.setattr(runner, 'rank_robust_trajectory', lambda *a, **k: {'status': 'selected'})
    offsets = [-.01, 0., .01]
    cfg = {'initial': [0., 0.], 'checkpoint_sha256': 'checkpoint', 'source_sha256': 'source',
        'k_points_sha256': 'k', 'coefficient_keys': [[0, 'cos']], 'max_mode_radius': 1,
        'source_resolution': 64, 'input_resolution': 16, 'material_fraction_bounds': [0, 1],
        'lower_band': 1, 'target_frequency': .5}
    specs = [{'trial_id': f'trial_{mode}', 'source_id': 8, 'initial_index': 0, 'mode': mode,
              'status': 'ready', 'config': cfg} for mode in ('nominal', 'mean', 'worst')]
    manifest = {'scope': 'preregistered_robust_design', 'offsets': offsets, 'trials': specs,
        'feedback': {'radius': .2, 'ridge': 1e-6},
        'sources': [{'source_id': 8, 'source_parameter_hash': 'source-hash', 'initials': [{'initial_index': 0}]}]}
    rows = [{'trial_id': 'source_0008_initial_0_nominal', 'role': f'offset_{index}', 'status': 'observed',
             'scenarios': [{'offset': offset, 'observation': {}} for offset in offsets]} for index in range(5)]
    feedback = {'scope': 'robust_stencil_physical_observations', 'offsets': offsets, 'campaign_manifest_sha256': 'manifest',
                'surrogate_contract_sha256': 'model', 'records': rows}
    kwargs = dict(manifest=manifest, feedback=feedback, manifest_sha256='manifest', feedback_sha256='feedback',
        source_id=8, source=SimpleNamespace(parameter_hash='source-hash'),
        predictor=SimpleNamespace(checkpoint_sha256='checkpoint', k_points=torch.tensor([[0., 0.]])),
        runtime={}, code_commit='commit', cluster_id='n56', output=tmp_path, radius=.2, ridge=1e-6)
    for mode in ('nominal', 'mean', 'worst'):
        assert len(runner.run_corrected_robust_trials(**kwargs, mode=mode)) == 1
    assert len(calls[0][0]) == 5 and calls[0][1]['offsets'] == [0.]
    assert len(calls[1][0]) == 15 and calls[1][1]['scenario_reduction'] == 'mean'
    assert len(calls[2][0]) == 15 and calls[2][1]['scenario_reduction'] == 'worst'
    assert len(runner.run_corrected_robust_trials(**kwargs, mode='nominal')) == 1 and len(calls) == 3
