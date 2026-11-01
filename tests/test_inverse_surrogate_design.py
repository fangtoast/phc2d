"""Verify target-driven surrogate search returns reusable structure artifacts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import json
import hashlib
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.surrogate_design import audit_surrogate_design, build_surrogate_replay_request, run_surrogate_design
from phc2d.inverse.audit_perturb_refine import run_case
from phc2d.inverse.feedback_handoff import validate_robust_observation
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


class Predictor:
    def predict_tensor(self, raster):
        mean = raster.mean(dim=(1, 2))
        return torch.stack((0.2 + .1*mean, 0.8 - .1*mean), dim=1).unsqueeze(1)


class ContractPredictor(Predictor):
    input_shape=(8,8)
    output_shape=(1,2)
    checkpoint_sha256='a'*64
    k_points_sha256='b'*64

    def __init__(self, shift=0):
        self.shift=shift

    def predict_tensor(self, raster):
        return super().predict_tensor(raster)+self.shift


def inputs():
    source=generate_unified_fourier_binary('source',master_seed=1,material_fraction_bin=1,
        config={'fourier_cutoff':2,'audit_resolution':16})
    t=source.parameters['threshold']
    options=dict(coefficient_keys=(),bounds=((t-.2,t+.2),),max_mode_radius=1,
        population_size=5,evaluation_budget=5,raster_method='master_block_mean_v1',
        source_resolution=16,input_resolution=8,material_fraction_bounds=(0,1),
        lower_band=1,target_frequency=.5)
    return source,options


def test_artifact_audit_detects_tampering_and_accepts_legacy_provenance(tmp_path):
    source,options=inputs()
    source_bytes=json.dumps(source.to_mapping()).encode()
    config_bytes=json.dumps({'search_options':options,'seeds':[0],
                             'checkpoint_sha256':'a'*64}).encode()
    provenance={'config_sha256':hashlib.sha256(config_bytes).hexdigest(),
                'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
                'checkpoint_sha256':'a'*64}
    out=tmp_path/'design'
    run_surrogate_design(source,Predictor(),seeds=[0],search_options=options,
                         output_root=out,provenance=provenance)
    args={'config_bytes':config_bytes,'source_bytes':source_bytes,'checkpoint_sha256':'a'*64}
    audit=audit_surrogate_design(out,**args)
    assert audit['status']=='artifact_consistent' and not audit['checkpoint_k_identity_recorded']
    assert audit['trajectories'][0]['evaluation_count']==5
    with pytest.raises(ValueError,match='provenance'):
        audit_surrogate_design(out,**dict(args,checkpoint_sha256='b'*64))
    raster_path=out/'seed_0'/'structure.npy'
    raster=np.load(raster_path)
    raster[0,0]=1-raster[0,0]
    np.save(raster_path,raster)
    with pytest.raises(ValueError,match='raster mismatch'):
        audit_surrogate_design(out,**args)


def test_artifact_audit_rejects_altered_candidate_trace(tmp_path):
    source,options=inputs()
    source_bytes=json.dumps(source.to_mapping()).encode()
    config_bytes=json.dumps({'search_options':options,'seeds':[0],
                             'checkpoint_sha256':'a'*64}).encode()
    out=tmp_path/'design'
    run_surrogate_design(source,Predictor(),seeds=[0],search_options=options,output_root=out,
        provenance={'config_sha256':hashlib.sha256(config_bytes).hexdigest(),
        'source_sha256':hashlib.sha256(source_bytes).hexdigest(),'checkpoint_sha256':'a'*64})
    path=out/'seed_0'/'search.json'
    trace=json.loads(path.read_text())
    trace['evaluations'][0]['parameter_hash']='0'*64
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError,match='candidate identity'):
        audit_surrogate_design(out,config_bytes=config_bytes,source_bytes=source_bytes,
                               checkpoint_sha256='a'*64)


def test_artifact_audit_preserves_failed_search_status(tmp_path):
    source,options=inputs()
    options['material_fraction_bounds']=(0,0)
    source_bytes=json.dumps(source.to_mapping()).encode()
    config_bytes=json.dumps({'search_options':options,'seeds':[0],
                             'checkpoint_sha256':'a'*64}).encode()
    out=tmp_path/'design'
    run_surrogate_design(source,Predictor(),seeds=[0],search_options=options,output_root=out,
        provenance={'config_sha256':hashlib.sha256(config_bytes).hexdigest(),
        'source_sha256':hashlib.sha256(source_bytes).hexdigest(),'checkpoint_sha256':'a'*64})
    args={'config_bytes':config_bytes,'source_bytes':source_bytes,'checkpoint_sha256':'a'*64}
    result=audit_surrogate_design(out,**args)
    assert result['trajectories'][0]['status']=='no_valid_candidates'
    assert result['trajectories'][0]['eligible_count']==0
    (out/'seed_0'/'structure.json').write_text('{}')
    with pytest.raises(ValueError,match='empty search result'):
        audit_surrogate_design(out,**args)


def test_device_parity_preflight_records_pass_and_binds_audit(tmp_path):
    source,options=inputs()
    options['perturbation_offsets']=(-1/16,0,1/16)
    source_bytes=json.dumps(source.to_mapping()).encode()
    config_bytes=json.dumps({'search_options':options,'seeds':[0],
        'checkpoint_sha256':'a'*64,'device_parity_absolute_tolerance':1e-5}).encode()
    out=tmp_path/'design'
    summary=run_surrogate_design(source,ContractPredictor(),seeds=[0],search_options=options,
        output_root=out,provenance={'config_sha256':hashlib.sha256(config_bytes).hexdigest(),
        'source_sha256':hashlib.sha256(source_bytes).hexdigest(),'checkpoint_sha256':'a'*64},
        parity_reference=ContractPredictor(),parity_absolute_tolerance=1e-5)
    assert summary['device_parity']['passed'] is True
    args={'config_bytes':config_bytes,'source_bytes':source_bytes,'checkpoint_sha256':'a'*64}
    assert audit_surrogate_design(out,**args)['status']=='artifact_consistent'
    parity_path=out/'device_parity.json'
    parity=json.loads(parity_path.read_text())
    parity['max_absolute_difference']=0.1
    parity_path.write_text(json.dumps(parity))
    with pytest.raises(ValueError,match='parity evidence'):
        audit_surrogate_design(out,**args)


def test_device_parity_failure_is_recorded_before_search(tmp_path):
    source,options=inputs()
    options['perturbation_offsets']=(-1/16,0,1/16)
    out=tmp_path/'design'
    with pytest.raises(ValueError,match='device parity preflight failed'):
        run_surrogate_design(source,ContractPredictor(shift=.1),seeds=[0],search_options=options,
            output_root=out,provenance={'test':True},parity_reference=ContractPredictor(),
            parity_absolute_tolerance=1e-5)
    evidence=json.loads((out/'device_parity.json').read_text())
    assert not evidence['passed'] and evidence['max_absolute_difference']>1e-5
    assert not (out/'summary.json').exists() and not (out/'seed_0').exists()


def test_device_parity_nonfinite_prediction_is_not_accepted(tmp_path):
    source,options=inputs()
    options['perturbation_offsets']=(0,)
    out=tmp_path/'design'
    with pytest.raises(ValueError,match='device parity preflight failed'):
        run_surrogate_design(source,ContractPredictor(shift=float('nan')),seeds=[0],
            search_options=options,output_root=out,provenance={'test':True},
            parity_reference=ContractPredictor(),parity_absolute_tolerance=1e-5)
    evidence=json.loads((out/'device_parity.json').read_text())
    assert evidence['max_absolute_difference'] is None
    assert evidence['prediction_nonfinite_count'] > 0 and not evidence['passed']


def test_search_returns_reconstructible_structure_and_preserves_prediction_scope(tmp_path):
    source,options=inputs()
    result=run_surrogate_design(source,Predictor(),seeds=[0,1],search_options=options,
        output_root=tmp_path/'design',provenance={'checkpoint_sha256':'a'*64})
    assert result['status']=='surrogate_only' and result['physical_validation']=='not_run'
    for trajectory in result['trajectories']:
        folder=tmp_path/'design'/f"seed_{trajectory['seed']}"
        structure=FullCellStructureSpec.from_mapping(json.loads((folder/'structure.json').read_text()))
        assert structure.parameter_hash==trajectory['structure_parameter_hash']
        assert np.array_equal(np.load(folder/'structure.npy'),structure.rasterize((16,16)))
        trace=json.loads((folder/'search.json').read_text())
        assert trajectory['evaluation_count']==len(trace['evaluations'])
        assert trajectory['predicted_target_margin']==trace['best_margin']
        response=json.loads((folder/'predicted_response.json').read_text())
        assert response['prediction_stage']=='post_search_reprediction'
        assert response['structure_parameter_hash']==structure.parameter_hash
        assert np.asarray(response['bands']).shape==(1,1,2)
        assert response['target_margin_difference']==pytest.approx(0)
    with pytest.raises(FileExistsError):
        run_surrogate_design(source,Predictor(),seeds=[0],search_options=options,
                             output_root=tmp_path/'design',provenance={'checkpoint_sha256':'a'*64})


def test_no_valid_candidate_does_not_create_structure(tmp_path):
    source,options=inputs()
    options['material_fraction_bounds']=(0,0)
    result=run_surrogate_design(source,Predictor(),seeds=[0],search_options=options,
        output_root=tmp_path/'empty',provenance={'test':True})
    assert result['status']=='no_valid_candidates'
    assert result['trajectories'][0]['status']=='no_valid_candidates'
    assert result['trajectories'][0]['structure_parameter_hash'] is None
    assert not (tmp_path/'empty'/'seed_0'/'structure.json').exists()


def test_perturbed_response_contains_all_offsets_and_search_margin(tmp_path):
    source,options=inputs()
    options['perturbation_offsets']=(-1/16,0,1/16)
    out=tmp_path/'design'
    result=run_surrogate_design(source,Predictor(),seeds=[0],search_options=options,
        output_root=out,provenance={'test':True})
    response=json.loads((out/'seed_0'/'predicted_response.json').read_text())
    assert response['offsets']==[-1/16,0,1/16]
    assert np.asarray(response['bands']).shape==(3,1,2)
    assert len(response['grid_sha256'])==3
    assert response['target_margin_difference']==pytest.approx(0)
    assert result['trajectories'][0]['response_reprediction_status']=='surrogate_only'


def test_failed_reprediction_is_recorded_without_claiming_response_success(tmp_path):
    class ChangingPredictor(Predictor):
        def __init__(self):
            self.calls=0
        def predict_tensor(self, raster):
            self.calls+=1
            if self.calls > 5:
                return torch.full((len(raster),1,2), float('nan'))
            return super().predict_tensor(raster)
    source,options=inputs()
    out=tmp_path/'design'
    result=run_surrogate_design(source,ChangingPredictor(),seeds=[0],search_options=options,
        output_root=out,provenance={'test':True})
    response=json.loads((out/'seed_0'/'predicted_response.json').read_text())
    assert response['status']=='nonfinite_prediction'
    assert response['gap'] is None and response['target_margin_difference'] is None
    assert result['trajectories'][0]['response_reprediction_status']=='nonfinite_prediction'


def test_finite_reprediction_drift_is_exposed(tmp_path):
    class DriftingPredictor(Predictor):
        def __init__(self):
            self.calls=0
        def predict_tensor(self, raster):
            self.calls+=1
            bands=super().predict_tensor(raster)
            return bands + (0.01 if self.calls > 5 else 0)
    source,options=inputs()
    out=tmp_path/'design'
    run_surrogate_design(source,DriftingPredictor(),seeds=[0],search_options=options,
        output_root=out,provenance={'test':True})
    response=json.loads((out/'seed_0'/'predicted_response.json').read_text())
    assert response['status']=='surrogate_only'
    assert abs(response['target_margin_difference'])>0.001


def test_replay_request_binds_selected_structure_and_original_k_grid(tmp_path):
    source,options=inputs()
    options['perturbation_offsets']=(-1/16,0,1/16)
    points=np.array([[1/9,0],[-1/9,0]],dtype=np.float64)
    digest=hashlib.sha256(np.ascontiguousarray(points,dtype=np.float32).tobytes()).hexdigest()
    out=tmp_path/'design'
    run_surrogate_design(source,Predictor(),seeds=[0],search_options=options,output_root=out,
        provenance={'checkpoint_sha256':'a'*64,'k_points_sha256':digest})
    summary=(out/'summary.json').read_bytes()
    structure=(out/'seed_0'/'structure.json').read_bytes()
    protocol=json.dumps({'offsets':[-1/16,0,1/16],'raster_resolution':16}).encode()
    args=dict(summary_bytes=summary,summary_sha256=hashlib.sha256(summary).hexdigest(),
              structure_bytes=structure,seed=0,k_points=points,k_points_sha256=digest,
              solver_config=FullCellMpbBandSolverConfig(num_bands=2,resolution=16,polarization='TE',
                    epsilon_air=1.,epsilon_material=8.9),source_index=4,protocol_bytes=protocol,
              protocol_sha256=hashlib.sha256(protocol).hexdigest())
    request=build_surrogate_replay_request(**args)
    assert request['status']=='surrogate_replay_request'
    assert request['rows'][0]['parameter_hash']==json.loads(summary)['trajectories'][0]['structure_parameter_hash']
    assert request['k_points']==points.tolist()
    assert request['target_frequency']==.5 and request['lower_band']==1
    with pytest.raises(ValueError,match='summary hash'):
        build_surrogate_replay_request(**dict(args,summary_sha256='0'*64))
    with pytest.raises(ValueError,match='k-grid hash'):
        build_surrogate_replay_request(**dict(args,k_points=points+1))
    altered=json.loads(summary)
    altered['provenance']['k_points_sha256']='0'*64
    altered_bytes=json.dumps(altered).encode()
    with pytest.raises(ValueError,match='frozen checkpoint'):
        build_surrogate_replay_request(**dict(args,summary_bytes=altered_bytes,
            summary_sha256=hashlib.sha256(altered_bytes).hexdigest()))
    altered['provenance']['k_points_sha256']=digest
    altered['trajectories'][0]['structure_parameter_hash']='0'*64
    altered_bytes=json.dumps(altered).encode()
    with pytest.raises(ValueError,match='structure identity'):
        build_surrogate_replay_request(**dict(args,summary_bytes=altered_bytes,
            summary_sha256=hashlib.sha256(altered_bytes).hexdigest()))
    with pytest.raises(ValueError,match='no unique candidate'):
        build_surrogate_replay_request(**dict(args,seed=3))
    wrong=json.dumps({'offsets':[0],'raster_resolution':16}).encode()
    with pytest.raises(ValueError,match='protocol differs'):
        build_surrogate_replay_request(**dict(args,protocol_bytes=wrong,protocol_sha256=hashlib.sha256(wrong).hexdigest()))


def test_surrogate_structure_can_be_sent_to_and_checked_by_physical_runner(tmp_path):
    source,options=inputs()
    options['perturbation_offsets']=(-1/16,0,1/16)
    points=np.array([[0.,0.],[.25,0.]])
    k_hash=hashlib.sha256(np.ascontiguousarray(points,dtype=np.float32).tobytes()).hexdigest()
    out=tmp_path/'design'
    run_surrogate_design(source,Predictor(),seeds=[0],search_options=options,output_root=out,
                         provenance={'checkpoint_sha256':'a'*64,'k_points_sha256':k_hash})
    summary=(out/'summary.json').read_bytes()
    protocol={'offsets':[-1/16,0,1/16],'raster_resolution':16,
        'refinement':{'verification_resolution':24,'initial_spacing':.25,'min_levels':2,
                      'max_levels':2,'required_stable_levels':2,'edge_tolerance':1.,
                      'resolution_tolerance':1e-6}}
    protocol_bytes=json.dumps(protocol).encode()
    protocol_hash=hashlib.sha256(protocol_bytes).hexdigest()
    config=FullCellMpbBandSolverConfig(num_bands=2,resolution=16,polarization='TE',
        epsilon_air=1.,epsilon_material=8.9,eigensolver_tolerance=1e-7)
    request=build_surrogate_replay_request(summary_bytes=summary,summary_sha256=hashlib.sha256(summary).hexdigest(),
        structure_bytes=(out/'seed_0'/'structure.json').read_bytes(),seed=0,k_points=points,k_points_sha256=k_hash,
        solver_config=config,source_index=4,protocol_bytes=protocol_bytes,protocol_sha256=protocol_hash)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(request),encoding='utf8')
    protocol_path=tmp_path/'protocol.json'
    protocol_path.write_bytes(protocol_bytes)
    manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()

    def solver(grid, *, k_points, config):
        fill=float(grid.mean())
        metadata={'grid_sha256':hashlib.sha256(np.ascontiguousarray(grid,dtype=np.uint8).tobytes()).hexdigest(),
            'resolution':config.resolution,'polarization':config.polarization,'num_bands':config.num_bands,
            'mesh_size':config.mesh_size,'band_guard_count':config.band_guard_count,
            'band_order_policy':config.band_order_policy,'eigensolver_tolerance':config.eigensolver_tolerance,
            'epsilon_air':config.epsilon_air,'epsilon_material':config.epsilon_material,
            'solver_backend':'mock','solver_version':'1'}
        return np.tile([.2+.1*fill,.8-.1*fill],(len(k_points),1)),metadata

    result_dir=run_case(manifest,expected_sha256=manifest_hash,protocol_path=protocol_path,
        expected_protocol_sha256=protocol_hash,case_index=0,output_root=tmp_path/'physical',solver=solver)
    result=json.loads((result_dir/'results.json').read_text())
    observation=validate_robust_observation(manifest_bytes=manifest.read_bytes(),expected_manifest_sha256=manifest_hash,
        protocol_bytes=protocol_bytes,protocol_sha256=protocol_hash,case_index=0,result=result)
    assert observation['provenance']['parameter_hash']==request['rows'][0]['parameter_hash']
    assert observation['provenance']['solver_identity']['solver_backend']=='mock'
