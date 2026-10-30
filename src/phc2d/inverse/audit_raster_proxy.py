"""Paired surrogate predictions for two physical master-raster resolutions.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids, predict_perturbed_gap


def run(*, manifest_path: str | Path, manifest_sha256: str, protocol_a_path: str | Path,
        protocol_a_sha256: str, protocol_b_path: str | Path, protocol_b_sha256: str,
        checkpoint: str | Path, checkpoint_sha256: str, device: str, output: str | Path) -> Path:
    """Predict the same anchor and offsets after two area-averaging paths.

    This is an exploratory representation comparison; neither path upgrades the
    checkpoint's frozen training contract or proves physical convergence.
    """
    manifest, digest = read_checked_json(manifest_path, manifest_sha256)
    a, a_digest = read_checked_json(protocol_a_path, protocol_a_sha256)
    b, b_digest = read_checked_json(protocol_b_path, protocol_b_sha256)
    if ({k:v for k,v in a.items() if k != 'raster_resolution'}
            != {k:v for k,v in b.items() if k != 'raster_resolution'}
            or a['raster_resolution'] == b['raster_resolution']):
        raise ValueError('protocols must differ only in raster resolution')
    predictor = FrozenBandAdapter(checkpoint, sha256=checkpoint_sha256, device=device)
    points = np.ascontiguousarray(manifest['k_points'],dtype=np.float32)
    if hashlib.sha256(points.tobytes()).hexdigest() != predictor.k_points_sha256:
        raise ValueError('checkpoint k-point identity mismatch')
    if len(predictor.input_shape) != 2 or predictor.input_shape[0] != predictor.input_shape[1]:
        raise ValueError('square 2D predictor inputs required')
    if not manifest['rows'] or any(row['label'] != 'anchor' for row in manifest['rows']):
        raise ValueError('anchor-only manifest required')
    output_path = Path(output)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.monotonic()
    rows = []
    for row in manifest['rows']:
        structure = FullCellStructureSpec.from_mapping(row['structure'])
        if structure.parameter_hash != row['parameter_hash']:
            raise ValueError('candidate parameter identity mismatch')
        predictions = []
        for protocol in (a,b):
            grids = periodic_perturbation_grids(structure,resolution=protocol['raster_resolution'],
                                                 offsets=tuple(protocol['offsets']))
            predictions.append(predict_perturbed_gap(grids,predictor,input_resolution=predictor.input_shape[0],
                lower_band=manifest['lower_band'],target_frequency=manifest['target_frequency']))
        if any(item['status'] != 'surrogate_only' for item in predictions):
            raise ValueError('invalid surrogate prediction')
        if a['raster_resolution'] == manifest.get('raster_resolution'):
            zero = a['offsets'].index(0.0)
            if predictions[0]['grid_sha256'][zero] != row['master_grid_sha256']:
                raise ValueError('manifest master raster identity mismatch')
        rows.append({'source_index':row['source_index'],'parameter_hash':structure.parameter_hash,
                     'prediction_a':predictions[0], 'prediction_b':predictions[1]})
    payload={'status':'exploratory_raster_proxy_only', 'manifest_sha256':digest,
             'protocol_a_sha256':a_digest,'protocol_b_sha256':b_digest,'checkpoint_sha256':checkpoint_sha256,
             'k_points_sha256':predictor.k_points_sha256,'anchor_count':len(rows),
             'inference_elapsed_s':time.monotonic()-started,'rows':rows}
    with output_path.open('x',encoding='utf8') as handle:
        json.dump(payload,handle,indent=2,allow_nan=False)
    return output_path


def main() -> None:
    parser=argparse.ArgumentParser(description='Compare surrogate input raster resolutions')
    for name in ('manifest-path','manifest-sha256','protocol-a-path','protocol-a-sha256',
                 'protocol-b-path','protocol-b-sha256','checkpoint','checkpoint-sha256','device','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    print(run(**vars(args)))


if __name__=='__main__':
    main()
