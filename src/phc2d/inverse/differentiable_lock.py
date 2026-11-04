"""Combine pure-surrogate MVP shards into an immutable candidate lock.

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


def combine_shards(shards: Mapping[str, dict[str, Any]], *, n56_takeover: bool = False) -> dict[str, Any]:
    """Assemble the nine fixed direction roles and three algorithm Top-1 roles."""
    roles = ('directions', 'adam', 'de', 'random')
    if set(shards) != set(roles) or any(shards[r].get('role') != r for r in roles):
        raise ValueError('exactly one shard for each required role is needed')
    platforms = {'directions': 'paracloud_n40', 'adam': 'paracloud_n40',
                 'de': 'paracloud_n56', 'random': 'paracloud_n56'}
    if n56_takeover:
        platforms.update(directions='paracloud_n56', adam='paracloud_n56')
    if any(shards[role].get('cluster_id') != platforms[role] for role in roles):
        raise ValueError('shard platform assignment mismatch')
    fields = ('config_sha256', 'source_sha256', 'source_parameter_hash', 'checkpoint_sha256',
              'k_points_sha256', 'code_commit', 'k_points', 'device_parity_tolerance', 'precision_contract')
    for field in fields:
        if not shards['directions'].get(field) or any(shards[r].get(field) != shards['directions'][field] for r in roles):
            raise ValueError(f'shard identity mismatch: {field}')
    direction = shards['directions']
    if direction.get('diagnostics', {}).get('fd_passed') is not True:
        raise ValueError('parameter gradient finite-difference check failed')
    reference = direction['selection']['candidates'][0]
    if reference.get('status') != 'surrogate_only' or 'bands' not in reference:
        raise ValueError('center reference prediction unavailable')
    center_bands = np.asarray(reference['bands'], dtype=np.float64)
    tolerance = direction['device_parity_tolerance']
    if not np.isfinite(tolerance) or tolerance < 0 or not np.isfinite(center_bands).all():
        raise ValueError('invalid center prediction or parity tolerance')
    parity = {}
    for role in roles[1:]:
        candidate = shards[role]['selection']['candidates'][0]
        if candidate.get('master_grid_sha256') != reference['master_grid_sha256'] or 'bands' not in candidate:
            raise ValueError('shard center geometry or prediction mismatch')
        bands = np.asarray(candidate['bands'], dtype=np.float64)
        if bands.shape != center_bands.shape or not np.isfinite(bands).all():
            raise ValueError('invalid shard center prediction')
        parity[role] = float(np.max(np.abs(bands - center_bands)))
        if parity[role] > tolerance:
            raise ValueError('center prediction device parity failed')
    expected_labels = ['center', 'gradient_0.05', 'negative_gradient_0.05', 'random_0.05',
                       'negative_random_0.05', 'gradient_0.1', 'negative_gradient_0.1',
                       'random_0.1', 'negative_random_0.1']
    if [row.get('label') for row in direction['history']] != expected_labels:
        raise ValueError('direction roster does not match the origin sub-batch')
    if len(direction['selection']['candidates']) != len(expected_labels):
        raise ValueError('direction candidate count mismatch')
    records = []
    for index, label in enumerate(expected_labels):
        records.append({'label': label, 'role': 'directions',
                        'candidate': direction['selection']['candidates'][index]})
    for role in roles[1:]:
        shard = shards[role]
        if len(shard['history']) != 20 or len(shard['selection']['candidates']) != 20:
            raise ValueError(f'{role} trajectory does not contain 20 evaluations')
        selected = shard['selection']['selected_index']
        candidates = shard['selection']['candidates']
        if selected is None:
            records.append({'label': f'{role}_top1', 'role': role, 'candidate': None,
                            'status': 'no_valid_candidate'})
        else:
            if not isinstance(selected, int) or not 0 <= selected < len(candidates):
                raise ValueError('invalid selected index')
            row = candidates[selected]
            if row['status'] != 'surrogate_only':
                raise ValueError('selected candidate is not eligible')
            eligible = [item for item in candidates if item['status'] == 'surrogate_only']
            maximum = max(float(item['predicted_gap']['target_margin']) for item in eligible)
            first = next(item['index'] for item in eligible if float(item['predicted_gap']['target_margin']) == maximum)
            if selected != first:
                raise ValueError('selected candidate does not match deterministic hard ranking')
            records.append({'label': f'{role}_top1', 'role': role, 'candidate': row})
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        candidate = record['candidate']
        if candidate is not None:
            digest = candidate['master_grid_sha256']
            if digest in seen:
                record['duplicate_of'] = seen[digest]
            else:
                seen[digest] = index
    return {'scope': 'surrogate_only_locked_origin_subbatch', 'records': records, 'device_parity_max_abs': parity,
            'platforms_by_role': {role: shards[role]['cluster_id'] for role in roles},
            **{field: shards['directions'][field] for field in fields}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('directions', 'adam', 'de', 'random'):
        parser.add_argument('--' + role, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n56-takeover', action='store_true')
    args = parser.parse_args()
    shards = {}
    hashes = {}
    for role in ('directions', 'adam', 'de', 'random'):
        raw = Path(getattr(args, role)).read_bytes()
        hashes[role] = hashlib.sha256(raw).hexdigest()
        shards[role] = json.loads(raw)
    lock = combine_shards(shards, n56_takeover=args.n56_takeover)
    lock['shard_sha256'] = hashes
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(lock, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
