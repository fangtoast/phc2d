"""Freeze paired feedback stencils before their initial MPB evaluations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def lock_stencils(seeds: Sequence[Mapping[str, Any]], *, design_manifest: Mapping[str, Any],
                  design_lock: Mapping[str, Any], base: Mapping[str, Any], campaign: Mapping[str, Any],
                  manifest_sha256: str, design_lock_sha256: str, campaign_sha256: str,
                  base_sha256: str) -> dict[str, Any]:
    """Bind all seed roles to the frozen pure-design initials and model identity."""
    if (design_lock.get('scope') != 'locked_paired_surrogate_campaign'
            or design_lock.get('manifest_sha256') != manifest_sha256
            or design_manifest.get('base_config_sha256') != base_sha256):
        raise ValueError('design lock and manifest identity mismatch')
    points = np.ascontiguousarray(design_lock['k_points'], dtype=np.float32)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or hashlib.sha256(points.tobytes()).hexdigest() != base['k_points_sha256']):
        raise ValueError('feedback seed k-point identity mismatch')
    source_map = {row['source_id']: row for row in design_manifest['sources']}
    expected = {(source_id, row['initial_index']) for source_id, source in source_map.items()
                for row in source['initials']}
    by_key = {(item['source_id'], item['initial_index']): item for item in seeds}
    if len(by_key) != len(seeds) or set(by_key) != expected:
        raise ValueError('feedback seed roster missing or duplicated')
    offsets = [np.asarray(x, dtype=float) for x in campaign['initial_offsets']]
    if len(offsets) != 5:
        raise ValueError('five registered feedback offsets required')
    commits, records = set(), []
    for source_id, initial_index in sorted(expected):
        seed = by_key[(source_id, initial_index)]
        initial_rows = [row for row in source_map[source_id]['initials'] if row['initial_index'] == initial_index]
        initial = initial_rows[0]
        if (seed.get('design_manifest_sha256') != manifest_sha256 or seed.get('design_lock_sha256') != design_lock_sha256
                or seed.get('campaign_sha256') != campaign_sha256 or seed.get('base_config_sha256') != base_sha256
                or seed.get('checkpoint_sha256') != base['checkpoint_sha256']
                or seed.get('k_points_sha256') != base['k_points_sha256'] or not seed.get('code_commit')
                or not seed.get('cluster_id')):
            raise ValueError('feedback seed identity mismatch')
        commits.add(seed['code_commit'])
        trial_id = f'source_{source_id:04d}_initial_{initial_index}_adam'
        if initial['status'] != 'ready':
            if seed.get('status') != initial['status']:
                raise ValueError('invalid initial status mismatch')
            for i in range(5):
                records.append({'label': f'{trial_id}/offset_{i}', 'trial_id': trial_id,
                                'role': f'offset_{i}', 'candidate': None, 'status': initial['status']})
            continue
        if (seed.get('status') != 'prepared' or seed.get('initial') != initial['initial']
                or seed.get('source_parameter_hash') != source_map[source_id]['source_parameter_hash']
                or len(seed.get('records', [])) != 5):
            raise ValueError('feedback seed initial or source mismatch')
        runtime = seed.get('runtime', {})
        if (runtime.get('contract') != base['runtime'] or runtime.get('matmul_tf32_enabled') is not False
                or runtime.get('cudnn_tf32_enabled') is not False):
            raise ValueError('feedback seed precision mismatch')
        center = np.asarray(initial['initial'], dtype=float)
        for i, (record, offset) in enumerate(zip(seed['records'], offsets)):
            if (record['label'] != f'offset_{i}' or offset.shape != center.shape
                    or not np.allclose(record['point'], center + offset, rtol=0, atol=1e-12)):
                raise ValueError('feedback offset mismatch')
            item = {'label': f'{trial_id}/offset_{i}', 'trial_id': trial_id,
                    'role': f'offset_{i}', 'point': record['point'], 'candidate': record['candidate']}
            if record['candidate'] is None:
                item['status'] = record['status']
            else:
                item['surrogate_edges'] = record.get('surrogate_edges')
                if record['candidate']['status'] in ('surrogate_only', 'duplicate_geometry'):
                    edges = item['surrogate_edges']
                    if not isinstance(edges, Mapping) or any(
                            np.asarray(edges.get(key), dtype=float).shape != (2,)
                            or not np.isfinite(np.asarray(edges.get(key), dtype=float)).all()
                            for key in ('hard_edges', 'soft_exact_edges', 'soft_smooth_edges')):
                        raise ValueError('feedback seed surrogate edge predictions missing')
            records.append(item)
    if len(commits) != 1:
        raise ValueError('feedback seed code identities differ')
    return {'scope': 'locked_paired_surrogate_campaign', 'purpose': 'feedback_initial_stencil',
            'manifest_sha256': manifest_sha256, 'design_lock_sha256': design_lock_sha256,
            'campaign_sha256': campaign_sha256, 'code_commit': next(iter(commits)),
            'k_points': design_lock['k_points'], 'records': records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('seed-directory', 'design-manifest', 'design-lock', 'base', 'campaign', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw_manifest, raw_lock, raw_base, raw_campaign = (Path(getattr(args, name)).read_bytes()
        for name in ('design_manifest', 'design_lock', 'base', 'campaign'))
    manifest, lock, base, campaign = (json.loads(raw) for raw in (raw_manifest, raw_lock, raw_base, raw_campaign))
    seed_paths = sorted(Path(args.seed_directory).glob('source_*_initial_*.json'))
    seeds = [json.loads(path.read_bytes()) for path in seed_paths]
    result = lock_stencils(seeds, design_manifest=manifest, design_lock=lock, base=base, campaign=campaign,
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(), design_lock_sha256=hashlib.sha256(raw_lock).hexdigest(),
        base_sha256=hashlib.sha256(raw_base).hexdigest(), campaign_sha256=hashlib.sha256(raw_campaign).hexdigest())
    result['seed_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in seed_paths}
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
