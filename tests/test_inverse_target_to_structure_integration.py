"""Exercise target-to-structure contracts through the physical handoff.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import torch

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.paired_physical_handoff import build_paired_physical_batch
from phc2d.inverse.paired_search_contract import build_paired_search_requests
from phc2d.inverse.paired_search_materialization import materialize_paired_search_requests, audit_paired_search_requests
from phc2d.inverse.surrogate_design import run_surrogate_design
from phc2d.inverse.trace_selection import lock_paired_trace_selection
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


class ToyPredictor:
    input_shape = (8,8)
    output_shape = (2,2)

    def __init__(self, checkpoint_sha256, k_points_sha256):
        self.checkpoint_sha256 = checkpoint_sha256
        self.k_points_sha256 = k_points_sha256

    def predict_tensor(self, raster):
        mean = raster.mean(dim=(1,2))
        low, high = .2+.05*mean, .8-.05*mean
        return torch.stack((low,high),dim=1).unsqueeze(1).repeat(1,2,1)


def test_target_search_trace_to_deduplicated_physical_request(tmp_path):
    fixture = runpy.run_path(str(Path(__file__).with_name('test_inverse_paired_search_contract.py')))['fixture_contract']
    args = fixture()
    batch = build_paired_search_requests(**args)
    inputs = tmp_path/'input'
    materialize_paired_search_requests(batch,inputs)
    audit_paired_search_requests(batch,inputs)
    points = np.asarray([[0.,0.],[.25,.25]],dtype=np.float64)
    k_sha = hashlib.sha256(np.ascontiguousarray(points,dtype=np.float32).tobytes()).hexdigest()
    locks = []
    for request in batch['requests']:
        sid, model_seed, seed = request['source_index'],request['model_seed'],request['search_seed']
        config_raw = (inputs/f'configs/source_{sid:04d}_model_{model_seed}.json').read_bytes()
        source_raw = (inputs/f'sources/source_{sid:04d}.json').read_bytes()
        config = json.loads(config_raw)
        source = FullCellStructureSpec.from_mapping(json.loads(source_raw))
        options = dict(config['search_options'])
        options['coefficient_keys'] = tuple(tuple(x) for x in options['coefficient_keys'])
        options['bounds'] = tuple(tuple(x) for x in options['bounds'])
        options['material_fraction_bounds'] = tuple(options['material_fraction_bounds'])
        options['perturbation_offsets'] = tuple(options['perturbation_offsets'])
        output = tmp_path/f'run_{sid}_{model_seed}'
        predictor = ToyPredictor(config['checkpoint_sha256'],k_sha)
        run_surrogate_design(source,predictor,seeds=[seed],search_options=options,output_root=output,
            provenance={'config_sha256':hashlib.sha256(config_raw).hexdigest(),
                        'source_sha256':hashlib.sha256(source_raw).hexdigest(),
                        'checkpoint_sha256':config['checkpoint_sha256'],'k_points_sha256':k_sha,
                        'model_name':'toy'},parity_reference=ToyPredictor(config['checkpoint_sha256'],k_sha),
            parity_absolute_tolerance=config['device_parity_absolute_tolerance'])
        lock = lock_paired_trace_selection(output,config_bytes=config_raw,source_bytes=source_raw,
            checkpoint_sha256=config['checkpoint_sha256'],source_index=sid,model_seed=model_seed,
            search_seed=seed,random_choice_seed=19)
        assert lock['status'] == 'locked_surrogate_only' and lock['physical_validation'] == 'not_run'
        locks.append(lock)
    source_map = {r['source_index']:{'source_index':r['source_index'],'stratum':r['stratum'],
                  'parameter_hash':FullCellStructureSpec.from_mapping(r['source']).parameter_hash,
                  'structure':r['source']} for r in batch['requests']}
    protocol = json.dumps({'raster_resolution':32,'offsets':[-.01,0.,.01],
        'refinement':{'verification_resolution':24}}).encode()
    physical = build_paired_physical_batch(list(source_map.values()),locks,
        expected_source_counts={'positive':1,'near_zero':1,'negative':1},
        checkpoint_sha256_by_seed={0:'a'*64,1:'b'*64,2:'c'*64},
        search_seed_by_model_seed={0:10,1:11,2:12},random_choice_seed=19,
        k_points=points,k_points_sha256=k_sha,
        config=FullCellMpbBandSolverConfig(resolution=16,num_bands=2,polarization='TE',
                                          epsilon_air=1.,epsilon_material=8.9),
        lower_band=1,target_frequency=.5,protocol_bytes=protocol,
        protocol_sha256=hashlib.sha256(protocol).hexdigest())
    assert physical['status'] == 'ready_for_physical_evaluation'
    assert physical['physical_validation'] == 'not_run' and len(physical['cross_environment_roles']) == 6
