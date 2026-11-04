"""Prepare MPB pixel-geometry requests from a locked pure-surrogate batch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec


def build_physical_request(lock: dict[str, Any], config: dict[str, Any], *, lock_sha256: str,
                           config_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return compatible MPB manifest and nominal zero-offset raster protocol."""
    if lock.get('config_sha256') != config_sha256 or len(lock.get('records', [])) != 12:
        raise ValueError('lock does not match the registered origin batch')
    points = np.ascontiguousarray(lock['k_points'], dtype=np.float32)
    if hashlib.sha256(points.tobytes()).hexdigest() != lock['k_points_sha256']:
        raise ValueError('lock k-point hash mismatch')
    protocol = config['physical_protocol']
    if protocol['offsets'] != [0.0] or protocol['raster_resolution'] != config['source_resolution']:
        raise ValueError('expected nominal master-grid protocol')
    rows, role_coverage, seen = [], [], {}
    for record in lock['records']:
        candidate = record['candidate']
        coverage = {'label': record['label'], 'case_index': None}
        if candidate is None or candidate['status'] not in ('surrogate_only', 'duplicate_geometry'):
            coverage['status'] = 'no_valid_candidate' if candidate is None else candidate['status']
            role_coverage.append(coverage)
            continue
        structure = FullCellStructureSpec.from_mapping(candidate['structure'])
        grid = np.ascontiguousarray(structure.rasterize((config['source_resolution'],)*2), dtype=np.uint8)
        digest = hashlib.sha256(grid.tobytes()).hexdigest()
        if digest != candidate['master_grid_sha256'] or structure.parameter_hash != candidate['parameter_hash']:
            raise ValueError('locked geometry or parameter hash mismatch')
        if digest not in seen:
            seen[digest] = len(rows)
            rows.append({'label': record['label'], 'source_index': 8,
                         'parameter_hash': structure.parameter_hash,
                         'master_grid_sha256': digest, 'structure': structure.to_mapping()})
        coverage.update(status='queued', case_index=seen[digest])
        role_coverage.append(coverage)
    protocol_bytes = json.dumps(protocol, indent=2, allow_nan=False).encode('utf8')
    manifest = {'scope': 'locked_pure_surrogate_origin_batch', 'selection_lock_sha256': lock_sha256,
                'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
                'k_points': lock['k_points'], 'lower_band': config['lower_band'],
                'target_frequency': config['target_frequency'], 'solver_config': config['solver_config'],
                'raster_resolution': config['source_resolution'], 'rows': rows, 'role_coverage': role_coverage}
    return manifest, protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    config_raw, lock_raw = Path(args.config).read_bytes(), Path(args.lock).read_bytes()
    manifest, protocol = build_physical_request(json.loads(lock_raw), json.loads(config_raw),
        lock_sha256=hashlib.sha256(lock_raw).hexdigest(), config_sha256=hashlib.sha256(config_raw).hexdigest())
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (('manifest.json', manifest), ('protocol.json', protocol)):
        (output/name).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    main()
