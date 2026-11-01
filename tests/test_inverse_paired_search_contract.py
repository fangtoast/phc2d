"""Validate target-driven paired search request construction.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json

import pytest
import yaml

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.paired_search_contract import build_paired_search_requests


def fixture_contract():
    config = {'experiment_id':'EXP-INV-test',
        'source_pool':{'selection_seed':7,'per_stratum':1,'coarse_margin_threshold':.005,
                       'stratum_order':['positive','near_zero','negative'], 'check_physical_grid_collisions':True},
        'geometry':{'physical_master_raster_resolution':32,'model_input_resolution':8,
                    'model_raster_method':'master_block_mean_v1','material_fraction_bounds':[.5,.9],
                    'epsilon_air':1.,'epsilon_material':8.9,'family':'unified_fourier_binary',
                    'generation_audit_resolution':16},
        'search':{'algorithm':'differential_evolution','search_seeds':[10,11,12],'coefficient_keys':[[0,'cos']],
                  'coefficient_radius':.1,'threshold_radius':.05,'max_mode_radius':1.,
                  'population_size':5,'evaluation_budget_per_source_seed':20},
        'surrogate':{'checkpoint_sha256_by_seed':{0:'a'*64,1:'b'*64,2:'c'*64},
                     'band_order_method':'ascending_projection_preserve_raw',
                     'device_parity_absolute_tolerance':.00005},
        'target':{'lower_band_one_based':1,'frequency_a_over_lambda':.5},
        'physics':{'offsets_a':[-.01,0.,.01]}}
    config_raw = yaml.safe_dump(config).encode()
    rows = []
    for i in range(3):
        s = generate_unified_fourier_binary(f'source-{i}',master_seed=i+20,material_fraction_bin=1,
                                            config={'fourier_cutoff':2,'audit_resolution':16})
        rows.append({'source_index':i,'parameter_hash':s.parameter_hash,'structure':s.to_mapping()})
    manifest = json.dumps({'experiment_id':'EXP-INV-test','rows':rows,'sample_count':3,
                           'scope':'new_source_development_not_physical'}).encode()
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    audit = json.dumps({'status':'pass','manifest_sha256':manifest_sha,
                        'config_sha256':hashlib.sha256(config_raw).hexdigest(),'reconstructed_count':3,
                        'scope':'source_identity_only_no_physical_validation','verified_resolutions':[8,16,32]}).encode()
    selection = json.dumps({'status':'ready','source_manifest_sha256':manifest_sha,'seed':7,
        'per_stratum':1,'lower_band':1,'target_frequency':.5,'margin_threshold':.005,
        'selected':[{'index':i,'stratum':s} for i,s in enumerate(config['source_pool']['stratum_order'])]}).encode()
    return {'config_bytes':config_raw,'source_manifest_bytes':manifest,'source_audit_bytes':audit,
        'selection_bytes':selection,'expected_source_audit_sha256':hashlib.sha256(audit).hexdigest(),
        'expected_selection_sha256':hashlib.sha256(selection).hexdigest(),
        'checkpoint_paths':{i:f'/checkpoint-{i}' for i in range(3)},'source_directory':'/sources'}


def test_requests_bind_target_model_source_and_search_bounds():
    args = fixture_contract()
    result = build_paired_search_requests(**args)
    assert result == build_paired_search_requests(**args)
    assert len(result['requests']) == 9 and result['scope'] == 'frozen_search_requests_not_executed'
    request = result['requests'][0]
    options = request['config']['search_options']
    source = request['source']
    assert request['config']['source'] == '/sources/source_0000.json'
    assert request['config']['checkpoint_sha256'] == 'a'*64
    assert request['config']['seeds'] == [10]
    assert options['projection_contract_sha256'] == hashlib.sha256(args['config_bytes']).hexdigest()
    assert options['bounds'][0] == pytest.approx([source['parameters']['modes'][0]['cos']-.1,
                                                   source['parameters']['modes'][0]['cos']+.1])
    assert options['bounds'][1] == pytest.approx([source['parameters']['threshold']-.05,
                                                   source['parameters']['threshold']+.05])


def test_requests_reject_wrong_identity_and_duplicate_selection():
    args = fixture_contract()
    with pytest.raises(ValueError,match='hash mismatch'):
        build_paired_search_requests(**dict(args,expected_selection_sha256='0'*64))
    selection = json.loads(args['selection_bytes'])
    selection['selected'][1]['index'] = 0
    raw = json.dumps(selection).encode()
    with pytest.raises(ValueError,match='selected source roster'):
        build_paired_search_requests(**dict(args,selection_bytes=raw,
            expected_selection_sha256=hashlib.sha256(raw).hexdigest()))
    with pytest.raises(ValueError,match='checkpoint or search seed roster'):
        build_paired_search_requests(**dict(args,checkpoint_paths={0:'/only'}))
    config = yaml.safe_load(args['config_bytes'])
    config['geometry']['epsilon_material'] = 4.0
    config_raw = yaml.safe_dump(config).encode()
    audit = json.loads(args['source_audit_bytes'])
    audit['config_sha256'] = hashlib.sha256(config_raw).hexdigest()
    audit_raw = json.dumps(audit).encode()
    with pytest.raises(ValueError,match='selected source identity mismatch'):
        build_paired_search_requests(**dict(args,config_bytes=config_raw,source_audit_bytes=audit_raw,
            expected_source_audit_sha256=hashlib.sha256(audit_raw).hexdigest()))
    audit = json.loads(args['source_audit_bytes'])
    audit['verified_resolutions'] = [8,16]
    audit_raw = json.dumps(audit).encode()
    with pytest.raises(ValueError,match='source selection or audit contract'):
        build_paired_search_requests(**dict(args,source_audit_bytes=audit_raw,
            expected_source_audit_sha256=hashlib.sha256(audit_raw).hexdigest()))
    config = yaml.safe_load(args['config_bytes'])
    config['surrogate']['band_order_method'] = 'raw'
    config_raw = yaml.safe_dump(config).encode()
    audit = json.loads(args['source_audit_bytes'])
    audit['config_sha256'] = hashlib.sha256(config_raw).hexdigest()
    audit_raw = json.dumps(audit).encode()
    with pytest.raises(ValueError,match='unsupported search, geometry or projection'):
        build_paired_search_requests(**dict(args,config_bytes=config_raw,source_audit_bytes=audit_raw,
            expected_source_audit_sha256=hashlib.sha256(audit_raw).hexdigest()))
