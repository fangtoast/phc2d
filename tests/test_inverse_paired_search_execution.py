"""Run a materialized target search with a real synthetic checkpoint.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from phc2d.inverse.paired_search_contract import build_paired_search_requests
from phc2d.inverse.paired_search_execution import execute_paired_search_request, inspect_nominal_static_request
from phc2d.inverse.paired_search_materialization import materialize_paired_search_requests
from phc2d.inverse.surrogate_design import audit_surrogate_design
from phc2d.models import build_model


def test_inspect_two_seed_nominal_draft_preserves_identity(tmp_path):
    from phc2d.generators.full_cell_latent import generate_unified_fourier_binary

    root = tmp_path/'requests'
    (root/'sources').mkdir(parents=True)
    (root/'configs').mkdir()
    source = generate_unified_fourier_binary('pilot', master_seed=20, material_fraction_bin=1,
                                            config={'fourier_cutoff': 2, 'audit_resolution': 16})
    source_raw = json.dumps(source.to_mapping()).encode()
    config = {'source': '/remote/requests/sources/source_0008.json', 'checkpoint_sha256': 'a'*64,
              'seeds': [11, 12], 'search_options': {'perturbation_offsets': [0.0]}}
    config_raw = json.dumps(config).encode()
    (root/'sources/source_0008.json').write_bytes(source_raw)
    (root/'configs/source_0008.json').write_bytes(config_raw)
    record = {'source_index': 8, 'source_file': 'sources/source_0008.json',
              'config_file': 'configs/source_0008.json', 'parameter_hash': source.parameter_hash,
              'source_sha256': hashlib.sha256(source_raw).hexdigest(),
              'config_sha256': hashlib.sha256(config_raw).hexdigest()}
    manifest = {'scope': 'static_request_materialization_not_executed', 'status': 'draft_not_execution_lock',
                'remote_request_directory': '/remote/requests', 'records': [record]}
    raw = json.dumps(manifest).encode()
    (root/'manifest.json').write_bytes(raw)
    args = {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'source_index': 8,
            'expected_seeds': [11, 12], 'checkpoint_sha256': 'a'*64}
    assert inspect_nominal_static_request(root, **args)['scope'] == 'draft_inspection_not_execution_lock'
    with pytest.raises(ValueError, match='seeds'):
        inspect_nominal_static_request(root, **dict(args, expected_seeds=[12, 11]))
    (root/'configs/source_0008.json').write_text('{}')
    with pytest.raises(ValueError, match='configuration hash'):
        inspect_nominal_static_request(root, **args)


def test_execute_one_frozen_request_and_reject_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS','1')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS','1')
    torch.manual_seed(42)
    model = build_model('mlp',input_shape=(8,8),output_shape=(2,2))
    points = torch.tensor([[0.,0.],[.25,.25]],dtype=torch.float32)
    k_sha = hashlib.sha256(np.ascontiguousarray(points.numpy()).tobytes()).hexdigest()
    checkpoint = tmp_path/'model.pt'
    torch.save({'model_name':'mlp','model_params':{},'input_shape':(8,8),'output_shape':(2,2),
        'state_dict':model.state_dict(),'target_mean':torch.tensor([[.2,.8],[.2,.8]]),
        'target_std':torch.full((2,2),.00001),'k_points':points,'k_points_sha256':k_sha},checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fixture = runpy.run_path(str(Path(__file__).with_name('test_inverse_paired_search_contract.py')))['fixture_contract']
    args = fixture()
    config = yaml.safe_load(args['config_bytes'])
    config['surrogate']['checkpoint_sha256_by_seed'] = {i:checkpoint_sha for i in range(3)}
    config_raw = yaml.safe_dump(config).encode()
    audit = json.loads(args['source_audit_bytes'])
    audit['config_sha256'] = hashlib.sha256(config_raw).hexdigest()
    audit_raw = json.dumps(audit).encode()
    request_root = tmp_path/'requests'
    args.update(config_bytes=config_raw,source_audit_bytes=audit_raw,
        expected_source_audit_sha256=hashlib.sha256(audit_raw).hexdigest(),
        checkpoint_paths={i:str(checkpoint) for i in range(3)},source_directory=str(request_root/'sources'))
    batch = build_paired_search_requests(**args)
    created = materialize_paired_search_requests(batch,request_root)
    output = tmp_path/'search'
    result = execute_paired_search_request(request_root,manifest_sha256=created['manifest_sha256'],
                                           request_index=0,output_root=output,device='cpu')
    assert result['scope'] == 'surrogate_execution_not_physical_validation'
    assert result['summary_sha256'] == hashlib.sha256((output/'summary.json').read_bytes()).hexdigest()
    audit = audit_surrogate_design(output,config_bytes=(request_root/'configs/source_0000_model_0.json').read_bytes(),
        source_bytes=(request_root/'sources/source_0000.json').read_bytes(),checkpoint_sha256=checkpoint_sha)
    assert audit['status'] == 'artifact_consistent' and audit['physical_validation'] == 'not_run'
    with pytest.raises(FileExistsError):
        execute_paired_search_request(request_root,manifest_sha256=created['manifest_sha256'],
                                      request_index=0,output_root=output,device='cpu')
    (request_root/'sources/source_0000.json').write_text('{}')
    with pytest.raises(ValueError,match='source or configuration hash'):
        execute_paired_search_request(request_root,manifest_sha256=created['manifest_sha256'],
                                      request_index=0,output_root=tmp_path/'other',device='cpu')
