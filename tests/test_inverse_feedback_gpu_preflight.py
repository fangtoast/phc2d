"""Realistic contract test for the cross-server proposal preflight.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_candidates import local_fourier_audit_candidates
from phc2d.inverse.audit_perturb_refine import run_case
from phc2d.inverse.feedback_gpu_preflight import run
from phc2d.inverse.feedback_journal import read_robust_round
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def test_preflight_persists_candidate_request_with_frozen_anchor_and_model(tmp_path, monkeypatch):
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 32})
    rows = list(local_fourier_audit_candidates(source, coefficient_delta=.1, threshold_delta=.1,
                                                max_mode_radius=1, raster_resolution=32))
    for row in rows:
        row['source_index'] = 3
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    points = [[0.,0.],[.25,0.]]
    manifest = {'rows':rows, 'k_points':points, 'lower_band':1, 'target_frequency':.5, 'dataset_sha256':'dataset',
                'solver_config':asdict(config)}
    options = {'verification_resolution':24, 'initial_spacing':.25, 'min_levels':2,
               'max_levels':2, 'required_stable_levels':2, 'edge_tolerance':1., 'resolution_tolerance':1e-6}
    protocol = {'offsets':[-1/32,0,1/32], 'raster_resolution':32, 'refinement':options}
    design = {'source_resolution':32, 'input_resolution':8, 'expected_dataset_sha256':'dataset',
              'coefficient_keys':[[rows[0]['mode_index'],'cos']], 'scales':[.5,.5], 'radius':1.,
              'min_radius':.1, 'max_radius':2., 'ridge':.01, 'max_mode_radius':1,
              'seed':3, 'population_size':8, 'surrogate_budget':16, 'material_fraction_bounds':[0,1]}
    def save(name, data):
        path=tmp_path/name
        path.write_text(json.dumps(data),encoding='utf8')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()
    mp,mh=save('manifest.json',manifest)
    pp,ph=save('protocol.json',protocol)
    dp,dh=save('design.json',design)
    def solver(grid, *, k_points, config):
        fill=float(grid.mean())
        metadata={'grid_sha256':hashlib.sha256(np.ascontiguousarray(grid,dtype=np.uint8).tobytes()).hexdigest(),
            'resolution':config.resolution,'polarization':config.polarization,'num_bands':config.num_bands,
            'mesh_size':config.mesh_size,'band_guard_count':config.band_guard_count,
            'band_order_policy':config.band_order_policy,'eigensolver_tolerance':config.eigensolver_tolerance,
            'epsilon_air':config.epsilon_air,'epsilon_material':config.epsilon_material,
            'solver_backend':'mock','solver_version':'1'}
        return np.tile([.2+.1*fill,.8-.1*fill],(len(k_points),1)),metadata
    ap=run_case(mp,expected_sha256=mh,protocol_path=pp,expected_protocol_sha256=ph,
                case_index=0,output_root=tmp_path/'anchor',solver=solver)/'results.json'
    ah=hashlib.sha256(ap.read_bytes()).hexdigest()
    class Predictor:
        input_shape=(8,8)
        k_points_sha256=hashlib.sha256(np.ascontiguousarray(points,dtype=np.float32).tobytes()).hexdigest()
        def __init__(self,*args,**kwargs):
            pass
        def predict_tensor(self,x):
            fill=x.mean(dim=(1,2))
            first=torch.stack((.2+.1*fill,.8-.1*fill),dim=1)
            return first[:,None,:].expand(-1,2,-1)
    monkeypatch.setattr('phc2d.inverse.feedback_gpu_preflight.FrozenBandAdapter',Predictor)
    output=run(manifest_path=mp,manifest_sha256=mh,protocol_path=pp,protocol_sha256=ph,
        anchor_path=ap,anchor_sha256=ah,case_index=0,design_path=dp,design_sha256=dh,
        checkpoint='unused',checkpoint_sha256='model',device='cpu',output_root=tmp_path/'proposal')
    result=json.loads((output/'results.json').read_text())
    assert result['status']=='surrogate_proposal_only'
    assert result['proposal']['edge_correction_applied']
    assert len(result['proposal']['trace']) <= 16
    pending=json.loads((output/'pending_state.json').read_text())
    solver_manifest=json.loads((output/'solver_manifest.json').read_text())
    assert pending['pending']['parameter_hash']==solver_manifest['rows'][0]['parameter_hash']
    assert hashlib.sha256((output/'solver_manifest.json').read_bytes()).hexdigest()==result['solver_manifest_sha256']
    saved, status = read_robust_round(output/'journal')
    assert status == 'pending_reconciliation'
    assert saved.pending.parameter_hash == pending['pending']['parameter_hash']
