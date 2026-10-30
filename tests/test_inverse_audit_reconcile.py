"""Tests for auditable proxy/solver evidence reconciliation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_reconcile import reconcile_audit, reconcile_sampled_solver


def data(tmp_path):
    structure = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
                                                config={'fourier_cutoff': 2, 'audit_resolution': 16})
    labels = ['anchor', 'a', 'b']
    rows = [dict(source_index=1, label=label, parameter_hash=structure.parameter_hash,
                 master_grid_sha256=label, structure=structure.to_mapping()) for label in labels]
    config = dict(resolution=16, polarization='TE', num_bands=2, mesh_size=3, band_guard_count=0)
    manifest = dict(rows=rows, k_points=[[0., 0.]], dataset_sha256='data', source_manifest_sha256='source',
                    target_frequency=.5, lower_band=1, solver_config=config)
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest), encoding='utf8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    proxy = dict(checkpoint_sha256='checkpoint', dataset_sha256='data', candidate_manifest_sha256='source',
                 target_frequency=.5, lower_band=1, k_count=1,
                 rows=[dict(source_index=1, label=l, parameter_hash=structure.parameter_hash,
                            grid_sha256=l, status='surrogate_only', predicted_margin=.5) for l in labels])
    results = [dict(case_index=i, source_index=1, label=l, parameter_hash=structure.parameter_hash,
                    geometry_hash=structure.geometry_hash, manifest_sha256=digest, k_points=[[0., 0.]],
                    solver_metadata=config, bands=[[.25, .75]], gap={'target_margin': 1.0},
                    status='sampled_solver_only') for i,l in enumerate(labels)]
    khash = hashlib.sha256(np.asarray([[0.,0.]], dtype=np.float32).tobytes()).hexdigest()
    return path, proxy, results, khash


def test_complete_identity_checked_audit(tmp_path):
    path, proxy, results, khash = data(tmp_path)
    result = reconcile_audit(path, proxy, results, expected_checkpoint_sha256='checkpoint', checkpoint_k_sha256=khash)
    assert result['candidate_count'] == 3 and result['source_count'] == 1
    assert result['status'] == 'exploratory_sampled_solver_only'


def test_mismatch_and_missing_result_fail_closed(tmp_path):
    path, proxy, results, khash = data(tmp_path)
    with pytest.raises(ValueError, match='incomplete'):
        reconcile_audit(path, proxy, results[:-1], expected_checkpoint_sha256='checkpoint', checkpoint_k_sha256=khash)
    results[0]['bands'] = [[.35,.75]]
    with pytest.raises(ValueError, match='gap does not reproduce'):
        reconcile_audit(path, proxy, results, expected_checkpoint_sha256='checkpoint', checkpoint_k_sha256=khash)


def test_solver_only_reconciliation_restores_manifest_order_and_rejects_mismatch(tmp_path):
    path, _, results, _ = data(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for item in results:
        item['solver_metadata'] = {**item['solver_metadata'], 'solver_backend': 'mock', 'solver_version': '1'}
    identity = {'solver_backend': 'mock', 'solver_version': '1'}
    result = reconcile_sampled_solver(path, list(reversed(results)), expected_manifest_sha256=digest,
                                      expected_solver_identity=identity)
    assert result['status'] == 'sampled_solver_only'
    assert result['source_indices'].tolist() == [1, 1, 1]
    assert result['spectra'].shape == (3, 1, 2)
    with pytest.raises(ValueError, match='incomplete'):
        reconcile_sampled_solver(path, results[:-1], expected_manifest_sha256=digest)
    with pytest.raises(ValueError, match='SHA-256'):
        reconcile_sampled_solver(path, results, expected_manifest_sha256='0' * 64)
    results[0]['solver_metadata']['solver_version'] = 'different'
    with pytest.raises(ValueError, match='identity'):
        reconcile_sampled_solver(path, results, expected_manifest_sha256=digest, expected_solver_identity=identity)
    results[0]['solver_metadata']['solver_version'] = '1'
    results[0]['bands'] = [[.35, .75]]
    with pytest.raises(ValueError, match='gap does not reproduce'):
        reconcile_sampled_solver(path, results, expected_manifest_sha256=digest)


@pytest.mark.parametrize('field,value', [('eigensolver_tolerance', 1e-7),
                                        ('epsilon_air', 1.0), ('epsilon_material', 8.9)])
def test_solver_contract_rejects_missing_or_changed_physical_configuration(tmp_path, field, value):
    path, _, results, _ = data(tmp_path)
    manifest = json.loads(path.read_text(encoding='utf8'))
    manifest['solver_config'][field] = value
    path.write_text(json.dumps(manifest), encoding='utf8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in results:
        row['manifest_sha256'] = digest
        row['solver_metadata'] = {**row['solver_metadata'], field: value}
    assert reconcile_sampled_solver(path, results, expected_manifest_sha256=digest)['status'] == 'sampled_solver_only'
    results[0]['solver_metadata'][field] = value * 2
    with pytest.raises(ValueError, match=field):
        reconcile_sampled_solver(path, results, expected_manifest_sha256=digest)
    del results[0]['solver_metadata'][field]
    with pytest.raises(ValueError, match=field):
        reconcile_sampled_solver(path, results, expected_manifest_sha256=digest)
