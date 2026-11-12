"""Audit case execution, immutable identity and failure tests.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_mpb import run_case


def manifest(tmp_path):
    s = generate_unified_fourier_binary('s', master_seed=1, material_fraction_bin=1,
                                        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    d = {'rows': [{'structure': s.to_mapping(), 'parameter_hash': s.parameter_hash,
                   'source_index': 3, 'label': 'anchor'}], 'k_points': [[0, 0]],
         'solver_config': {'num_bands': 2, 'resolution': 16, 'polarization': 'TE',
                           'epsilon_air': s.epsilon_air, 'epsilon_material': s.epsilon_material},
         'lower_band': 1, 'target_frequency': .5}
    p = tmp_path / 'manifest.json'
    p.write_text(json.dumps(d), encoding='utf8')
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def test_case_persists_result_and_refuses_overwrite(tmp_path):
    p, digest = manifest(tmp_path)
    def solver(s, *, k_points, config):
        return np.array([[.2, .8]]), {'geometry_hash': s.geometry_hash}
    output = run_case(p, expected_sha256=digest, case_index=0, output_root=tmp_path/'out', solver=solver)
    result = json.loads((output/'results.json').read_text())
    assert result['evidence_scope'] == 'sampled_solver_no_refinement'
    assert result['manifest_sha256'] == digest
    with pytest.raises(FileExistsError):
        run_case(p, expected_sha256=digest, case_index=0, output_root=tmp_path/'out', solver=solver)


def test_failed_case_and_wrong_manifest_remain_separate(tmp_path):
    p, digest = manifest(tmp_path)
    def solver(*args, **kwargs):
        raise RuntimeError('solver failed')
    with pytest.raises(ValueError, match='SHA-256'):
        run_case(p, expected_sha256='0'*64, case_index=0, output_root=tmp_path/'out', solver=solver)
    with pytest.raises(RuntimeError, match='solver failed'):
        run_case(p, expected_sha256=digest, case_index=0, output_root=tmp_path/'out', solver=solver)
    assert json.loads((tmp_path/'out/case_0000/failure.json').read_text())['error_type'] == 'RuntimeError'
