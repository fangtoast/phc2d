"""Test deterministic, collision-checked inverse development source creation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json

import numpy as np
import pytest
import yaml

from phc2d.inverse.source_pool import generate_source_pool


def fixture_inputs(tmp_path):
    points=[[0.,0.],[.25,0.]]
    khash=hashlib.sha256(np.ascontiguousarray(points,dtype=np.float32).tobytes()).hexdigest()
    config={'experiment_id':'EXP-INV-test','target':{'polarization':'TE','lower_band_one_based':1,
        'frequency_a_over_lambda':.5,'checkpoint_k_points_sha256':khash},
        'geometry':{'family':'unified_fourier_binary','cutoff':2,'generation_audit_resolution':16,
        'model_input_resolution':8,'epsilon_air':1.,'epsilon_material':8.9,
        'target_material_fraction_range':[.5,.9]},
        'source_pool':{'count':3,'generator_seed':42,'selection_seed':43,'coarse_margin_threshold':.005,
        'coarse_solver':{'resolution':8,'mesh_size':3,'eigensolver_tolerance':1e-7,'num_bands':2,
        'band_guard_count':0,'band_order_policy':'frequency_ascending'}}}
    config_path=tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest()
    reference={'k_points':points,'lower_band':1,'target_frequency':.5,
        'solver_config':{'resolution':8,'mesh_size':3,'eigensolver_tolerance':1e-7,'num_bands':2,
        'band_guard_count':0,'epsilon_air':1.,'epsilon_material':8.9,'polarization':'TE'}}
    reference_path=tmp_path/'reference.json'
    reference_path.write_text(json.dumps(reference))
    excluded={'config_sha256':config_hash,'identity_sets':{'parameter_hash':[],
        'content_hash_r8':[],'content_hash_r16':[]}}
    excluded_path=tmp_path/'excluded.json'
    excluded_path.write_text(json.dumps(excluded))
    scope={'status':'frozen_before_generation','experiment_id':'EXP-INV-test','config_sha256':config_hash,
        'exclusions_sha256':hashlib.sha256(excluded_path.read_bytes()).hexdigest(),
        'reference_manifest_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'sample_id_prefix':'independent-test-'}
    scope_path=tmp_path/'scope.json'
    scope_path.write_text(json.dumps(scope))
    return dict(config=config_path,scope=scope_path,scope_sha256=hashlib.sha256(scope_path.read_bytes()).hexdigest(),
        exclusions=excluded_path,reference_manifest=reference_path,output=tmp_path/'output.json',
        workers=1,cpu_limit=2,memory_budget_mib=2000,worker_memory_mib=500,setup_reserve_mib=1000)


def test_pool_is_deterministic_across_worker_counts_and_never_overwrites(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_CPUS_PER_TASK',raising=False)
    monkeypatch.delenv('SLURM_MEM_PER_NODE',raising=False)
    args=fixture_inputs(tmp_path)
    first=generate_source_pool(**args)
    second_args=dict(args,workers=2,output=tmp_path/'second.json')
    second=generate_source_pool(**second_args)
    assert first['count']==3 and first['manifest_sha256']==second['manifest_sha256']
    assert args['output'].read_bytes()==second_args['output'].read_bytes()
    rows=json.loads(args['output'].read_bytes())['rows']
    assert [r['source_index'] for r in rows]==[0,1,2]
    with pytest.raises(FileExistsError):
        generate_source_pool(**args)


def test_pool_stops_on_collision_without_creating_partial_manifest(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_CPUS_PER_TASK',raising=False)
    monkeypatch.delenv('SLURM_MEM_PER_NODE',raising=False)
    args=fixture_inputs(tmp_path)
    generate_source_pool(**args)
    row=json.loads(args['output'].read_bytes())['rows'][0]
    excluded=json.loads(args['exclusions'].read_bytes())
    excluded['identity_sets']['content_hash_r8'].append(row['content_hash_r8'])
    args['exclusions'].write_text(json.dumps(excluded))
    scope=json.loads(args['scope'].read_bytes())
    scope['exclusions_sha256']=hashlib.sha256(args['exclusions'].read_bytes()).hexdigest()
    args['scope'].write_text(json.dumps(scope))
    args.update(scope_sha256=hashlib.sha256(args['scope'].read_bytes()).hexdigest(),output=tmp_path/'collision.json')
    with pytest.raises(ValueError,match='collision'):
        generate_source_pool(**args)
    assert not args['output'].exists()


def test_pool_rejects_capacity_and_source_contract_mismatch(tmp_path,monkeypatch):
    args=fixture_inputs(tmp_path)
    with pytest.raises(ValueError,match='budget'):
        generate_source_pool(**dict(args,workers=3))
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','1')
    with pytest.raises(ValueError,match='scheduler'):
        generate_source_pool(**args)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK')
    args['reference_manifest'].write_text('{}')
    with pytest.raises(ValueError,match='reference hash'):
        generate_source_pool(**args)
    assert not args['output'].exists()
