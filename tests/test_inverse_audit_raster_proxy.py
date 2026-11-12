"""Contract tests for paired raster surrogate diagnostics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.audit_raster_proxy import run


def test_paired_raster_predictions_keep_identity_and_full_spectra(tmp_path, monkeypatch):
    structure = generate_unified_fourier_binary('s',master_seed=1,material_fraction_bin=1,
        config={'fourier_cutoff':2,'audit_resolution':16})
    grid = np.ascontiguousarray(structure.rasterize((16,16)),dtype=np.uint8)
    points=[[0.,0.]]
    k_hash=hashlib.sha256(np.ascontiguousarray(points,dtype=np.float32).tobytes()).hexdigest()
    manifest={'rows':[{'label':'anchor','source_index':1,'structure':structure.to_mapping(),
        'parameter_hash':structure.parameter_hash,'master_grid_sha256':hashlib.sha256(grid.tobytes()).hexdigest()}],
        'k_points':points,'lower_band':1,'target_frequency':.5,'raster_resolution':16}
    a={'offsets':[-1/16,0.,1/16],'raster_resolution':16,'refinement':{'resolution':16}}
    b={**a,'raster_resolution':32}
    def save(name,value):
        path=tmp_path/name
        path.write_text(json.dumps(value),encoding='utf8')
        return path,hashlib.sha256(path.read_bytes()).hexdigest()
    mp,mh=save('manifest.json',manifest)
    ap,ah=save('a.json',a)
    bp,bh=save('b.json',b)
    class Predictor:
        input_shape=(8,8)
        k_points_sha256=k_hash
        def __init__(self,*args,**kwargs):
            pass
        def predict_tensor(self,x):
            fill=x.mean(dim=(1,2))
            return torch.stack((.2+.1*fill,.8-.1*fill),dim=1).unsqueeze(1)
    monkeypatch.setattr('phc2d.inverse.audit_raster_proxy.FrozenBandAdapter',Predictor)
    output=tmp_path/'result.json'
    run(manifest_path=mp,manifest_sha256=mh,protocol_a_path=ap,protocol_a_sha256=ah,
        protocol_b_path=bp,protocol_b_sha256=bh,checkpoint='unused',checkpoint_sha256='model',
        device='cpu',output=output)
    result=json.loads(output.read_text())
    assert result['anchor_count']==1
    assert len(result['rows'][0]['prediction_a']['bands'])==3
    assert result['rows'][0]['prediction_a']['grid_sha256'][1]==manifest['rows'][0]['master_grid_sha256']
    assert result['rows'][0]['prediction_b']['grid_sha256'][1]!=manifest['rows'][0]['master_grid_sha256']
    b['offsets']=[0.]
    bp,bh=save('b.json',b)
    with pytest.raises(ValueError,match='only in raster'):
        run(manifest_path=mp,manifest_sha256=mh,protocol_a_path=ap,protocol_a_sha256=ah,
            protocol_b_path=bp,protocol_b_sha256=bh,checkpoint='unused',checkpoint_sha256='model',
            device='cpu',output=tmp_path/'bad.json')
