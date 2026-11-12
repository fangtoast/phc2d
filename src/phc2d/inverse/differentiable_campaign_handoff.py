"""Build deduplicated MPB requests from a locked paired surrogate campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from phc2d.data.full_cell_spec import FullCellStructureSpec


def build_campaign_request(lock: Mapping[str, Any], manifest: Mapping[str, Any],
                           base: Mapping[str, Any], *, lock_sha256: str,
                           manifest_sha256: str, base_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate locked structures and preserve all role-to-unique-case mappings."""
    if (lock.get('scope') != 'locked_paired_surrogate_campaign'
            or lock.get('manifest_sha256') != manifest_sha256
            or manifest.get('base_config_sha256') != base_sha256):
        raise ValueError('campaign lock or base identity mismatch')
    points = np.ascontiguousarray(lock['k_points'], dtype=np.float32)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or hashlib.sha256(points.tobytes()).hexdigest() != base['k_points_sha256']):
        raise ValueError('campaign k-point identity mismatch')
    by_trial = {item['trial_id']: item for item in manifest['trials']}
    if len(by_trial) != len(manifest['trials']):
        raise ValueError('duplicate trial identity')
    roles, seen_labels, rows, grids = [], set(), [], {}
    resolution = base['source_resolution']
    for record in lock['records']:
        trial = by_trial.get(record['trial_id'])
        label = f"{record['trial_id']}/{record['role']}"
        if trial is None or label in seen_labels:
            raise ValueError('invalid or duplicated locked role')
        seen_labels.add(label)
        candidate = record['candidate']
        coverage = {'label': label, 'trial_id': record['trial_id'], 'role': record['role'],
                    'case_index': None}
        if candidate is None or candidate['status'] not in ('surrogate_only', 'duplicate_geometry'):
            coverage['status'] = record.get('status', 'no_valid_candidate') if candidate is None else candidate['status']
            roles.append(coverage)
            continue
        structure = FullCellStructureSpec.from_mapping(candidate['structure'])
        raster = np.ascontiguousarray(structure.rasterize((resolution, resolution)), dtype=np.uint8)
        digest = hashlib.sha256(raster.tobytes()).hexdigest()
        if digest != candidate['master_grid_sha256'] or structure.parameter_hash != candidate['parameter_hash']:
            raise ValueError('locked campaign structure hash mismatch')
        if digest not in grids:
            grids[digest] = len(rows)
            rows.append({'label': label, 'source_index': trial['source_id'],
                         'parameter_hash': structure.parameter_hash, 'master_grid_sha256': digest,
                         'structure': structure.to_mapping()})
        coverage.update(status='queued', case_index=grids[digest])
        roles.append(coverage)
    protocol = manifest['physical_protocol'] if manifest.get('scope') == 'preregistered_robust_design' else base['physical_protocol']
    if manifest.get('scope') == 'preregistered_robust_design':
        offsets = manifest['offsets']
        expected_protocol = {**base['physical_protocol'], 'offsets': offsets}
        if (protocol != expected_protocol or protocol['offsets'] != offsets or len(offsets) != 3 or 0.0 not in offsets
                or not any(value < 0 for value in offsets) or not any(value > 0 for value in offsets)):
            raise ValueError('robust physical offsets differ from the registered campaign')
    elif protocol['offsets'] != [0.0]:
        raise ValueError('expected nominal physical protocol')
    if protocol['raster_resolution'] != resolution:
        raise ValueError('physical and master-grid resolutions differ')
    raw = json.dumps(protocol, indent=2, allow_nan=False).encode('utf8')
    physical = {'scope': 'locked_paired_surrogate_campaign', 'selection_lock_sha256': lock_sha256,
                'campaign_manifest_sha256': manifest_sha256, 'protocol_sha256': hashlib.sha256(raw).hexdigest(),
                'k_points': lock['k_points'], 'lower_band': base['lower_band'],
                'target_frequency': base['target_frequency'], 'solver_config': base['solver_config'],
                'raster_resolution': resolution, 'rows': rows, 'role_coverage': roles}
    return physical, protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lock', 'manifest', 'base', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw_lock, raw_manifest, raw_base = (Path(getattr(args, key)).read_bytes() for key in ('lock', 'manifest', 'base'))
    physical, protocol = build_campaign_request(json.loads(raw_lock), json.loads(raw_manifest), json.loads(raw_base),
        lock_sha256=hashlib.sha256(raw_lock).hexdigest(), manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        base_sha256=hashlib.sha256(raw_base).hexdigest())
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (('manifest.json', physical), ('protocol.json', protocol)):
        (output / name).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    main()
