"""Verify target-driven surrogate search returns reusable structure artifacts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import json
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.surrogate_design import run_surrogate_design


class Predictor:
    def predict_tensor(self, raster):
        mean = raster.mean(dim=(1, 2))
        return torch.stack((0.2 + .1*mean, 0.8 - .1*mean), dim=1).unsqueeze(1)


def inputs():
    source=generate_unified_fourier_binary('source',master_seed=1,material_fraction_bin=1,
        config={'fourier_cutoff':2,'audit_resolution':16})
    t=source.parameters['threshold']
    options=dict(coefficient_keys=(),bounds=((t-.2,t+.2),),max_mode_radius=1,
        population_size=5,evaluation_budget=5,raster_method='master_block_mean_v1',
        source_resolution=16,input_resolution=8,material_fraction_bounds=(0,1),
        lower_band=1,target_frequency=.5)
    return source,options


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
