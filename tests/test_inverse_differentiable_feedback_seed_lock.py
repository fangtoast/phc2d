"""Test frozen feedback seed roster and edge prediction identity."""
import copy
import hashlib
import numpy as np
import pytest

from phc2d.inverse.differentiable_feedback_seed_lock import lock_stencils


def test_feedback_seed_lock_checks_offsets_and_frozen_identity():
    points = [[0.0, 0.0]]
    k_hash = hashlib.sha256(np.asarray(points, dtype=np.float32).tobytes()).hexdigest()
    runtime = {'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False}
    base = {'k_points_sha256': k_hash, 'checkpoint_sha256': 'model', 'runtime': runtime}
    manifest = {'base_config_sha256': 'base', 'sources': [{'source_id': 8, 'source_parameter_hash': 'source',
                'initials': [{'initial_index': 0, 'status': 'ready', 'initial': [0.0, 0.0]}]}]}
    lock = {'scope': 'locked_paired_surrogate_campaign', 'manifest_sha256': 'manifest', 'k_points': points}
    offsets = [[0., 0.], [.1, 0.], [-.1, 0.], [0., .1], [0., -.1]]
    edges = {'hard_edges': [.4, .6], 'soft_exact_edges': [.4, .6], 'soft_smooth_edges': [.41, .59]}
    seed = {'source_id': 8, 'initial_index': 0, 'design_manifest_sha256': 'manifest',
            'design_lock_sha256': 'lock', 'campaign_sha256': 'campaign', 'base_config_sha256': 'base',
            'checkpoint_sha256': 'model', 'k_points_sha256': k_hash, 'code_commit': 'commit',
            'cluster_id': 'paracloud_n56', 'status': 'prepared', 'initial': [0., 0.],
            'source_parameter_hash': 'source',
            'runtime': {'contract': runtime, 'matmul_tf32_enabled': False, 'cudnn_tf32_enabled': False},
            'records': [{'label': f'offset_{i}', 'point': point, 'candidate': {'status': 'surrogate_only'},
                         'surrogate_edges': edges} for i, point in enumerate(offsets)]}
    kw = dict(design_manifest=manifest, design_lock=lock, base=base, campaign={'initial_offsets': offsets},
              manifest_sha256='manifest', design_lock_sha256='lock', campaign_sha256='campaign', base_sha256='base')
    result = lock_stencils([seed], **kw)
    assert len(result['records']) == 5 and result['records'][0]['label'] == 'source_0008_initial_0_adam/offset_0'
    bad = copy.deepcopy(seed); bad['records'][1]['point'] = [.2, 0.]
    with pytest.raises(ValueError, match='offset'):
        lock_stencils([bad], **kw)
    bad = copy.deepcopy(seed); bad['records'][0]['surrogate_edges'] = None
    with pytest.raises(ValueError, match='edge'):
        lock_stencils([bad], **kw)
