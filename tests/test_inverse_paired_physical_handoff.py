"""Contract tests for deduplicated paired-search physical requests.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import copy
import hashlib
import json

import numpy as np
import pytest

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_candidates import local_fourier_audit_candidates
from phc2d.inverse.paired_physical_handoff import build_paired_physical_batch
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def fixture_inputs():
    sources, locks = [], []
    points=np.asarray([[0.,0.],[.25,.25]])
    k_hash=hashlib.sha256(np.ascontiguousarray(points,dtype=np.float32).tobytes()).hexdigest()
    for sid, stratum in enumerate(('positive', 'near_zero', 'negative')):
        source = generate_unified_fourier_binary(f'source-{sid}', master_seed=100+sid,
            material_fraction_bin=1, config={'fourier_cutoff': 2, 'audit_resolution': 32})
        candidates = local_fourier_audit_candidates(source, coefficient_delta=.3, threshold_delta=.2,
                                                     max_mode_radius=1, raster_resolution=32)
        anchor = candidates[0]['master_grid_sha256']
        choices = [r for r in candidates[1:] if r['duplicate_of'] is None]
        assert len(choices) >= 2
        sources.append({'source_index':sid,'stratum':stratum,'structure':source.to_mapping(),
                        'parameter_hash':source.parameter_hash})
        for seed in (0,1):
            def choice(row, index):
                return {'candidate_index':index, 'structure':row['structure'],
                        'parameter_hash':row['parameter_hash'], 'nominal_grid_sha256':row['master_grid_sha256']}
            locks.append({'source_index':sid,'model_seed':seed,'status':'locked_surrogate_only',
                'physical_validation':'not_run','audit_status':'artifact_consistent',
                'checkpoint_sha256':str(seed)*64,'search_seed':10+seed,'random_choice_seed':19,
                'checkpoint_k_points_sha256':k_hash,'source_resolution':32,
                'perturbation_offsets':[-1/32,0,1/32], 'lower_band':1,'target_frequency':.5,
                'source_parameter_hash':source.parameter_hash,'source_grid_sha256':anchor,
                'eligible_indices':[1,2], 'surrogate_choice':choice(choices[0],1),
                'random_choice':choice(choices[1],2)})
    protocol=json.dumps({'raster_resolution':32,'offsets':[-1/32,0,1/32],
        'refinement':{'verification_resolution':48,'initial_spacing':.25,'min_levels':2,'max_levels':2,
                      'required_stable_levels':2,'edge_tolerance':1,'resolution_tolerance':1e-6}}).encode()
    kwargs={'expected_source_counts':{'positive':1,'near_zero':1,'negative':1},
            'checkpoint_sha256_by_seed':{0:'0'*64,1:'1'*64}, 'search_seed_by_model_seed':{0:10,1:11},
            'random_choice_seed':19, 'k_points':points,
            'k_points_sha256':hashlib.sha256(np.ascontiguousarray(points,dtype=np.float32).tobytes()).hexdigest(),
            'config':FullCellMpbBandSolverConfig(resolution=32,num_bands=2,eigensolver_tolerance=1e-9,
                                                 polarization='TE',epsilon_air=1.,epsilon_material=8.9),
            'lower_band':1,'target_frequency':.5,'protocol_bytes':protocol,
            'protocol_sha256':hashlib.sha256(protocol).hexdigest()}
    return sources, locks, kwargs


def test_paired_batch_deduplicates_roles_and_freezes_crosscheck():
    sources,locks,kwargs=fixture_inputs()
    result=build_paired_physical_batch(sources,locks,**kwargs)
    assert result['status']=='ready_for_physical_evaluation' and result['physical_validation']=='not_run'
    assert len(result['manifest']['rows']) == 9
    assert len(result['role_map']) == 15 and len(result['cross_environment_roles']) == 6
    for sid in range(3):
        rows=[r for r in result['role_map'] if r['source_index']==sid and r['role']=='surrogate_choice']
        assert len({r['case_index'] for r in rows})==1


def test_paired_batch_rejects_missing_roster_and_modified_identity():
    sources,locks,kwargs=fixture_inputs()
    with pytest.raises(ValueError,match='roster incomplete'):
        build_paired_physical_batch(sources,locks[:-1],**kwargs)
    damaged=copy.deepcopy(locks)
    damaged[0]['surrogate_choice']['nominal_grid_sha256']='f'*64
    with pytest.raises(ValueError,match='grid identity'):
        build_paired_physical_batch(sources,damaged,**kwargs)
    with pytest.raises(ValueError,match='k-point identity'):
        build_paired_physical_batch(sources,locks,**dict(kwargs,k_points_sha256='f'*64))


def test_paired_batch_preserves_unscoreable_without_manifest():
    sources,locks,kwargs=fixture_inputs()
    locks[0]['status']='unscoreable_insufficient_eligible'
    locks[0]['surrogate_choice']=locks[0]['random_choice']=None
    result=build_paired_physical_batch(sources,locks,**kwargs)
    assert result['status']=='unscoreable_trace_roster'
    assert result['manifest'] is None and result['unscoreable']==[
        {'source_index':0,'model_seed':0,'status':'unscoreable_insufficient_eligible'}]


def test_paired_batch_does_not_replace_duplicate_crosscheck_role():
    sources,locks,kwargs=fixture_inputs()
    other=sources[1]
    from phc2d.data.full_cell_spec import FullCellStructureSpec
    structure=FullCellStructureSpec.from_mapping(other['structure'])
    grid=np.ascontiguousarray(structure.rasterize((32,32)),dtype=np.uint8)
    locks[0]['surrogate_choice'].update(structure=other['structure'],parameter_hash=other['parameter_hash'],
        nominal_grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest())
    result=build_paired_physical_batch(sources,locks,**kwargs)
    assert result['status']=='cross_environment_roles_not_unique' and result['manifest'] is None


def test_paired_batch_drives_existing_solver_contract(tmp_path):
    from phc2d.inverse.audit_perturb_refine import run_case
    sources,locks,kwargs=fixture_inputs()
    batch=build_paired_physical_batch(sources,locks,**kwargs)
    manifest_raw=json.dumps(batch['manifest'],allow_nan=False).encode()
    manifest=tmp_path/'manifest.json'; manifest.write_bytes(manifest_raw)
    protocol=tmp_path/'protocol.json'; protocol.write_bytes(kwargs['protocol_bytes'])
    def solver(grid, *, k_points, config):
        digest=hashlib.sha256(np.ascontiguousarray(grid,dtype=np.uint8).tobytes()).hexdigest()
        return np.tile([.2,.7],(len(k_points),1)), {'grid_sha256':digest}
    selected=next(r for r in batch['role_map'] if r['role']=='surrogate_choice')
    path=run_case(manifest,expected_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        protocol_path=protocol,expected_protocol_sha256=kwargs['protocol_sha256'],
        case_index=selected['case_index'],output_root=tmp_path/'solver',solver=solver)
    result=json.loads((path/'results.json').read_text())
    assert result['parameter_hash']==batch['manifest']['rows'][selected['case_index']]['parameter_hash']
    assert result['aggregate']['status']=='numerically_stable'
