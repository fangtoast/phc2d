"""Check campaign selection and preregistered diagnostics cannot be altered."""
import copy
import hashlib
import numpy as np
import pytest

from phc2d.inverse.differentiable_campaign_lock import lock_campaign


def test_campaign_lock_checks_complete_trajectory_and_top1():
    points = [[0.0, 0.0]]
    grid_hash = hashlib.sha256(np.asarray(points, dtype=np.float32).tobytes()).hexdigest()
    cfg = {'initial': [0.15, 0.0], 'source_sha256': 'source-file',
           'checkpoint_sha256': 'checkpoint', 'k_points_sha256': grid_hash}
    spec = {'trial_id': 'trial', 'source_id': 8, 'initial_index': 1,
            'method': 'adam', 'status': 'ready', 'config': cfg}
    manifest = {'trials': [spec], 'sources': [{'source_id': 8, 'source_parameter_hash': 'source'}]}
    candidates = [{'index': i, 'status': 'surrogate_only', 'predicted_gap': {'target_margin': i / 100}}
                  for i in range(60)]
    result = {'trial_id': 'trial', 'manifest_sha256': 'manifest', 'code_commit': 'commit',
              'cluster_id': 'paracloud_n40', 'status': 'completed', 'role': 'adam', 'config': cfg,
              'source_id': 8, 'initial_index': 1, 'source_parameter_hash': 'source',
              'source_sha256': 'source-file', 'checkpoint_sha256': 'checkpoint', 'k_points_sha256': grid_hash,
              'k_points': points,
              'history': [{'parameters': [0.15, 0.0]}] + [{}] * 59,
              'selection': {'status': 'selected', 'selected_index': 59, 'candidates': candidates}}
    kw = dict(manifest=manifest, manifest_sha256='manifest', diagnostic_counts=[1, 20, 40, 60])
    locked = lock_campaign(**kw, trials={'trial': result})
    assert [row['candidate']['index'] for row in locked['records']] == [0, 19, 39, 59, 59]
    with pytest.raises(ValueError, match='roster'):
        lock_campaign(**kw, trials={})
    bad = copy.deepcopy(result)
    bad['selection']['selected_index'] = 0
    with pytest.raises(ValueError, match='ranking'):
        lock_campaign(**kw, trials={'trial': bad})
    bad = copy.deepcopy(result)
    bad['history'][0]['parameters'] = [0.0, 0.0]
    with pytest.raises(ValueError, match='pairing'):
        lock_campaign(**kw, trials={'trial': bad})
