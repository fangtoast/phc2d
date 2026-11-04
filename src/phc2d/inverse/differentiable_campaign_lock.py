"""Freeze paired-campaign trial selections and preregistered diagnostics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def lock_campaign(manifest: Mapping[str, Any], trials: Mapping[str, Mapping[str, Any]], *,
                  manifest_sha256: str, diagnostic_counts: list[int]) -> dict[str, Any]:
    """Validate all registered trajectories and retain frozen diagnostic roles."""
    expected = manifest['trials']
    ids = [row['trial_id'] for row in expected]
    if len(ids) != len(set(ids)) or set(ids) != set(trials):
        raise ValueError('trial roster incomplete or duplicated')
    if diagnostic_counts != [1, 20, 40, 60]:
        raise ValueError('diagnostic schedule differs from registration')
    records, commits, k_reference = [], set(), None
    sources = {row['source_id']: row for row in manifest['sources']}
    if len(sources) != len(manifest['sources']):
        raise ValueError('duplicate source identity')
    for spec in expected:
        item = trials[spec['trial_id']]
        if (item.get('manifest_sha256') != manifest_sha256 or item.get('trial_id') != spec['trial_id']
                or not item.get('code_commit') or not item.get('cluster_id')):
            raise ValueError('trial identity mismatch')
        commits.add(item['code_commit'])
        if spec['status'] != 'ready':
            if item.get('status') != spec['status']:
                raise ValueError('unavailable initial status mismatch')
            records.append({'label': f"{spec['trial_id']}/top1", 'trial_id': spec['trial_id'], 'role': 'top1', 'candidate': None,
                            'status': spec['status']})
            continue
        if (item.get('status') != 'completed' or item.get('role') != spec['method']
                or item.get('config') != spec['config'] or item.get('source_id') != spec['source_id']
                or item.get('initial_index') != spec['initial_index']
                or item.get('source_parameter_hash') != sources[spec['source_id']]['source_parameter_hash']
                or item.get('source_sha256') != spec['config']['source_sha256']
                or item.get('checkpoint_sha256') != spec['config']['checkpoint_sha256']
                or item.get('k_points_sha256') != spec['config']['k_points_sha256']):
            raise ValueError('trial configuration mismatch')
        points = np.ascontiguousarray(item['k_points'], dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                or hashlib.sha256(points.tobytes()).hexdigest() != spec['config']['k_points_sha256']):
            raise ValueError('trial k-point identity mismatch')
        if k_reference is not None and points.tolist() != k_reference:
            raise ValueError('trial k-point coordinates differ')
        k_reference = points.tolist()
        history, selection = item['history'], item['selection']
        candidates = selection['candidates']
        if (len(history) != 60 or len(candidates) != 60
                or history[0]['parameters'] != spec['config']['initial']
                or any(candidate['index'] != i for i, candidate in enumerate(candidates))):
            raise ValueError('trial trajectory, pairing or evaluation count mismatch')
        eligible = [row for row in candidates if row['status'] == 'surrogate_only']
        if any(not math.isfinite(float(row['predicted_gap']['target_margin'])) for row in eligible):
            raise ValueError('nonfinite hard score')
        selected = selection['selected_index']
        if eligible:
            winner = max(eligible, key=lambda row: (float(row['predicted_gap']['target_margin']), -row['index']))
            if selected != winner['index'] or selection['status'] != 'selected':
                raise ValueError('hard Top-1 ranking mismatch')
            top = candidates[selected]
        else:
            if selected is not None or selection['status'] != 'no_valid_candidates':
                raise ValueError('invalid empty selection')
            top = None
        def diagnostic(index: int) -> dict[str, Any]:
            candidate = dict(candidates[index])
            if candidate['status'] == 'duplicate_geometry' and 'bands' not in candidate:
                prior = candidate['duplicate_of']
                if not isinstance(prior, int) or not 0 <= prior < index:
                    raise ValueError('invalid duplicate reference')
                original = candidates[prior]
                if (original['status'] != 'surrogate_only' or 'bands' not in original
                        or original['master_grid_sha256'] != candidate['master_grid_sha256']):
                    raise ValueError('duplicate prediction identity mismatch')
                candidate['bands'] = original['bands']
                candidate['predicted_gap'] = original['predicted_gap']
            return candidate
        for count in diagnostic_counts:
            records.append({'label': f"{spec['trial_id']}/evaluation_{count}", 'trial_id': spec['trial_id'], 'role': f'evaluation_{count}',
                            'candidate': diagnostic(count - 1)})
        records.append({'label': f"{spec['trial_id']}/top1", 'trial_id': spec['trial_id'], 'role': 'top1', 'candidate': top,
                        'status': 'selected' if top is not None else 'no_valid_candidate'})
    if len(commits) != 1:
        raise ValueError('trial code identities differ')
    return {'scope': 'locked_paired_surrogate_campaign', 'manifest_sha256': manifest_sha256,
            'code_commit': next(iter(commits)), 'k_points': k_reference, 'records': records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'campaign', 'trial-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw = Path(args.manifest).read_bytes()
    manifest = json.loads(raw)
    campaign_raw = Path(args.campaign).read_bytes()
    if hashlib.sha256(campaign_raw).hexdigest() != manifest['campaign_sha256']:
        raise ValueError('campaign configuration identity mismatch')
    campaign = json.loads(campaign_raw)
    trials = {}
    for spec in manifest['trials']:
        path = Path(args.trial_directory) / f"{spec['trial_id']}.json"
        trials[spec['trial_id']] = json.loads(path.read_bytes())
    lock = lock_campaign(manifest, trials, manifest_sha256=hashlib.sha256(raw).hexdigest(),
                         diagnostic_counts=campaign['diagnostic_evaluation_counts'])
    lock['trial_sha256'] = {spec['trial_id']: hashlib.sha256((Path(args.trial_directory) /
                           f"{spec['trial_id']}.json").read_bytes()).hexdigest() for spec in manifest['trials']}
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(lock, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
