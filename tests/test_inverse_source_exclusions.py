"""Verify collision exclusion extension and provenance contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json

import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.source_exclusions import extend_source_exclusions


def fixture_inputs():
    prior=json.dumps({'identity_sets':{'parameter_hash':['previous'], 'content_hash_r8':['low'],
                                     'content_hash_r16':['audit']},
                      'prior_candidate_master_grid_sha256':['physical']}).encode()
    structures=[generate_unified_fourier_binary(f'exclude-{i}',master_seed=20+i,material_fraction_bin=1,
        config={'fourier_cutoff':2,'audit_resolution':16}) for i in range(2)]
    rows=[{'structure':s.to_mapping(),'parameter_hash':s.parameter_hash} for s in structures]
    raw=json.dumps({'rows':rows}).encode()
    return {'previous_bytes':prior,'previous_sha256':hashlib.sha256(prior).hexdigest(),
            'config_bytes':b'new config','additions':[{'name':'prior sources', 'bytes':raw,
                'sha256':hashlib.sha256(raw).hexdigest()}],
            'low_resolution':8,'audit_resolution':16,'physical_resolution':32,
            'workers':1,'cpu_limit':2,'memory_budget_mib':2000,'worker_memory_mib':500,'setup_reserve_mib':1000}


def test_extend_preserves_prior_and_reconstructs_added_geometry():
    args=fixture_inputs()
    one=extend_source_exclusions(**args)
    two=extend_source_exclusions(**dict(args,workers=2))
    assert one==two and one['scope']=='extended_source_exclusions_not_exhaustiveness_certification'
    assert len(one['identity_sets']['parameter_hash'])==3
    assert len(one['prior_candidate_master_grid_sha256'])==3
    assert 'previous' in one['identity_sets']['parameter_hash']
    assert one['config_sha256']==hashlib.sha256(b'new config').hexdigest()


def test_extend_rejects_changed_source_or_resource_contract():
    args=fixture_inputs()
    with pytest.raises(ValueError,match='prior exclusions hash'):
        extend_source_exclusions(**dict(args,previous_sha256='0'*64))
    with pytest.raises(ValueError,match='addition hash'):
        extend_source_exclusions(**dict(args,additions=[dict(args['additions'][0],sha256='0'*64)]))
    with pytest.raises(ValueError,match='resource budget'):
        extend_source_exclusions(**dict(args,workers=3))
    data=json.loads(args['additions'][0]['bytes'])
    data['rows'][0]['parameter_hash']='wrong'
    raw=json.dumps(data).encode()
    with pytest.raises(ValueError,match='identity or material mismatch'):
        extend_source_exclusions(**dict(args,additions=[{'name':'bad','bytes':raw,
            'sha256':hashlib.sha256(raw).hexdigest()}]))


def test_extend_respects_scheduler_allocation(monkeypatch):
    args = fixture_inputs()
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '1')
    with pytest.raises(ValueError, match='CPU limit exceeds scheduler allocation'):
        extend_source_exclusions(**args)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '1999')
    with pytest.raises(ValueError, match='memory budget exceeds scheduler allocation'):
        extend_source_exclusions(**args)


def test_extend_retains_authenticated_historical_parameter_hash():
    args = fixture_inputs()
    manifest = json.loads(args['additions'][0]['bytes'])
    manifest['rows'][0]['historical_parameter_hash'] = 'a' * 64
    raw = json.dumps(manifest).encode()
    args['additions'] = [{'name': 'historical', 'bytes': raw, 'sha256': hashlib.sha256(raw).hexdigest()}]
    result = extend_source_exclusions(**args)
    assert 'a' * 64 in result['identity_sets']['parameter_hash']
    manifest['rows'][0]['historical_parameter_hash'] = 'not-a-hash'
    raw = json.dumps(manifest).encode()
    args['additions'] = [{'name': 'historical', 'bytes': raw, 'sha256': hashlib.sha256(raw).hexdigest()}]
    with pytest.raises(ValueError, match='historical parameter hash'):
        extend_source_exclusions(**args)
