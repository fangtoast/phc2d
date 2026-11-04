"""Check robust trial modes preserve the frozen paired initial identities."""
import copy

import pytest

from phc2d.inverse.differentiable_robust_campaign import build_robust_campaign


def test_robust_campaign_pairs_modes_and_rejects_identity_changes():
    source = {'source_id': 8, 'source_sha256': 'source', 'source_parameter_hash': 'p',
              'initials': [{'initial_index': 0, 'status': 'ready', 'initial': [.15, 0.]}]}
    trial = {'source_id': 8, 'initial_index': 0, 'method': 'adam', 'status': 'ready',
             'config': {'initial': [.15, 0.], 'source_sha256': 'source', 'evaluations': 60,
                        'physical_protocol': {'offsets': [0.0], 'raster_resolution': 64}}}
    design = {'scope': 'preregistered_surrogate_design', 'base_config_sha256': 'base',
              'sources': [source], 'trials': [trial]}
    settings = {'experiment_id': 'EXP-INV-015', 'modes': ['nominal', 'mean', 'worst'],
                'offsets': [-.01, 0, .01], 'evaluations': 60, 'feedback': {'radius': .2, 'ridge': 1e-6}}
    manifest = build_robust_campaign(design, settings, design_manifest_sha256='parent')
    assert [row['mode'] for row in manifest['trials']] == settings['modes']
    assert all(row['config']['initial'] == [.15, 0.] for row in manifest['trials'])
    assert all(row['config']['evaluations'] == 60 for row in manifest['trials'])
    assert manifest['physical_protocol']['offsets'] == [-.01, 0., .01]
    bad = copy.deepcopy(design); bad['trials'][0]['config']['initial'] = [0., 0.]
    with pytest.raises(ValueError, match='frozen initial'):
        build_robust_campaign(bad, settings, design_manifest_sha256='parent')
    with pytest.raises(ValueError, match='offsets'):
        build_robust_campaign(design, {**settings, 'offsets': [0., 0., .01]}, design_manifest_sha256='parent')
