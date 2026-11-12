"""Frozen-checkpoint continuous-input gradient diagnostics on audit anchors.

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

from phc2d.data.full_cell_raster import block_mean_downsample
from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.candidate_gradient_audit import score_candidate_directions
from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
from phc2d.inverse.input_gradient_audit import audit_input_gradient


def run(manifest_path: str | Path, *, manifest_sha256: str, checkpoint: str | Path,
        checkpoint_sha256: str, device: str, output: str | Path,
        candidate_directions: bool = False) -> Path:
    """Audit one frozen model on all manifest anchors, without geometry gradients."""
    manifest, digest = read_checked_json(manifest_path, manifest_sha256)
    predictor = FrozenBandAdapter(checkpoint, sha256=checkpoint_sha256, device=device)
    points = np.ascontiguousarray(manifest['k_points'], dtype=np.float32)
    if hashlib.sha256(points.tobytes()).hexdigest() != predictor.k_points_sha256:
        raise ValueError('checkpoint k-point identity mismatch')
    if len(predictor.input_shape) != 2 or predictor.input_shape[0] != predictor.input_shape[1]:
        raise ValueError('expected square 2D checkpoint input')
    anchors = [r for r in manifest['rows'] if r['label'] == 'anchor']
    if not anchors or len({r['source_index'] for r in anchors}) != len(anchors):
        raise ValueError('missing or duplicate anchors')
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.monotonic()
    results = []
    for index, row in enumerate(anchors):
        structure = FullCellStructureSpec.from_mapping(row['structure'])
        if structure.parameter_hash != row['parameter_hash']:
            raise ValueError('parameter identity mismatch')
        grid = np.ascontiguousarray(structure.rasterize((manifest['raster_resolution'],) * 2), dtype=np.uint8)
        if hashlib.sha256(grid.tobytes()).hexdigest() != row['master_grid_sha256']:
            raise ValueError('master raster identity mismatch')
        image = block_mean_downsample(grid, predictor.input_shape[0])
        result = audit_input_gradient(predictor, image[None], lower_band=manifest['lower_band'],
                                      target_frequency=manifest['target_frequency'],
                                      steps=(.01, .03, .1), seed=20261030 + index)
        entry = {'source_index': row['source_index'], 'parameter_hash': structure.parameter_hash,
                 'master_grid_sha256': row['master_grid_sha256'], **result}
        if candidate_directions:
            neighbors = [item for item in manifest['rows']
                         if item['source_index'] == row['source_index'] and item['label'] != 'anchor']
            candidates = []
            for item in neighbors:
                other = FullCellStructureSpec.from_mapping(item['structure'])
                if other.parameter_hash != item['parameter_hash']:
                    raise ValueError('candidate parameter identity mismatch')
                other_grid = np.ascontiguousarray(other.rasterize((manifest['raster_resolution'],) * 2), dtype=np.uint8)
                if hashlib.sha256(other_grid.tobytes()).hexdigest() != item['master_grid_sha256']:
                    raise ValueError('candidate raster identity mismatch')
                candidate_image = block_mean_downsample(other_grid, predictor.input_shape[0])
                candidates.append({'label': item['label'], 'parameter_hash': item['parameter_hash'],
                                   'input': candidate_image[None]})
            entry['candidate_directions'] = score_candidate_directions(
                predictor, image[None], candidates, lower_band=manifest['lower_band'],
                target_frequency=manifest['target_frequency'])
        results.append(entry)
    payload = {'status': 'exploratory_continuous_input_diagnostic_only', 'manifest_sha256': digest,
               'checkpoint_sha256': checkpoint_sha256, 'k_points_sha256': predictor.k_points_sha256,
               'device': device, 'steps': [.01, .03, .1], 'anchor_count': len(results),
               'candidate_directions_included': candidate_directions,
               'elapsed_s': time.monotonic() - started, 'rows': results}
    with output_path.open('x', encoding='utf8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit gradients on frozen inverse anchors')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate-directions', action='store_true')
    args = parser.parse_args()
    print(run(args.manifest, manifest_sha256=args.manifest_sha256, checkpoint=args.checkpoint,
              checkpoint_sha256=args.checkpoint_sha256, device=args.device, output=args.output,
              candidate_directions=args.candidate_directions))


if __name__ == '__main__':
    main()
