"""Score frozen inverse candidates using a sampled surrogate common gap.

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
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids, predict_perturbed_gap


def validate_paired_proxy(proxy: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *,
                          manifest_sha256: str, protocol_sha256: str,
                          checkpoint_sha256: str, k_points_sha256: str) -> list[dict[str, Any]]:
    """Check the surrogate and physical runs use identical candidates and grids."""
    if (proxy.get('status') != 'exploratory_surrogate_common_gap_only'
            or proxy.get('manifest_sha256') != manifest_sha256 or proxy.get('protocol_sha256') != protocol_sha256
            or proxy.get('checkpoint_sha256') != checkpoint_sha256 or proxy.get('k_points_sha256') != k_points_sha256):
        raise ValueError('proxy provenance mismatch')
    def key(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return row['source_index'], row['label'], row['parameter_hash']
    indexed = {key(item): item for item in records}
    if len(indexed) != len(records) or len(proxy['rows']) != len(records):
        raise ValueError('incomplete or duplicate candidate records')
    paired = []
    seen = set()
    for item in proxy['rows']:
        identity = key(item)
        if identity in seen or identity not in indexed:
            raise ValueError('proxy candidate identity mismatch')
        seen.add(identity)
        record = indexed[identity]
        if record.get('manifest_sha256') != manifest_sha256 or record.get('protocol_sha256') != protocol_sha256:
            raise ValueError('physical provenance mismatch')
        observations = record['observations']
        if ([r['offset'] for r in observations] != list(proxy['offsets'])
                or [r['provenance']['evaluated_grid_sha256'] for r in observations] != item['grid_sha256']):
            raise ValueError('offset grid identity mismatch')
        if item.get('status') != 'surrogate_only' or item.get('predicted_margin') is None:
            raise ValueError('proxy candidate has invalid score')
        paired.append(dict(item))
    return paired


def run(manifest_path: str | Path, *, manifest_sha256: str, protocol_path: str | Path,
        protocol_sha256: str, checkpoint: str | Path, checkpoint_sha256: str,
        device: str, output: str | Path) -> Path:
    """Predict all candidate offsets; caller verifies checkpoint raster contract."""
    manifest, digest = read_checked_json(manifest_path, manifest_sha256)
    protocol, protocol_digest = read_checked_json(protocol_path, protocol_sha256)
    predictor = FrozenBandAdapter(checkpoint, sha256=checkpoint_sha256, device=device)
    points = np.ascontiguousarray(manifest['k_points'], dtype=np.float32)
    if hashlib.sha256(points.tobytes()).hexdigest() != predictor.k_points_sha256:
        raise ValueError('checkpoint k-point identity mismatch')
    if len(predictor.input_shape) != 2 or predictor.input_shape[0] != predictor.input_shape[1]:
        raise ValueError('square 2D predictor inputs required')
    offsets = tuple(protocol['offsets'])
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.monotonic()
    rows = []
    for source in manifest['rows']:
        structure = FullCellStructureSpec.from_mapping(source['structure'])
        if structure.parameter_hash != source['parameter_hash']:
            raise ValueError('parameter identity mismatch')
        grids = periodic_perturbation_grids(structure, resolution=protocol['raster_resolution'], offsets=offsets)
        result = predict_perturbed_gap(grids, predictor, input_resolution=predictor.input_shape[0],
                                       lower_band=manifest['lower_band'], target_frequency=manifest['target_frequency'])
        zero = offsets.index(0.0)
        if result['grid_sha256'][zero] != source['master_grid_sha256']:
            raise ValueError('master raster identity mismatch')
        row = {'source_index': source['source_index'], 'label': source['label'],
               'parameter_hash': structure.parameter_hash, 'grid_sha256': result['grid_sha256'],
               'status': result['status'], 'predicted_margin': None}
        if result['gap'] is not None:
            row['predicted_margin'] = result['gap']['target_margin']
        rows.append(row)
    payload = {'status': 'exploratory_surrogate_common_gap_only', 'manifest_sha256': digest,
               'protocol_sha256': protocol_digest, 'checkpoint_sha256': checkpoint_sha256,
               'k_points_sha256': predictor.k_points_sha256, 'offsets': offsets,
               'elapsed_s': time.monotonic() - started, 'rows': rows}
    with output_path.open('x', encoding='utf8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict frozen candidate common gaps')
    for name in ('manifest', 'manifest-sha256', 'protocol', 'protocol-sha256',
                 'checkpoint', 'checkpoint-sha256', 'device', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(run(args.manifest, manifest_sha256=args.manifest_sha256, protocol_path=args.protocol,
              protocol_sha256=args.protocol_sha256, checkpoint=args.checkpoint,
              checkpoint_sha256=args.checkpoint_sha256, device=args.device, output=args.output))


if __name__ == '__main__':
    main()
