"""Freeze robust campaign selections before physical evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def lock_robust_campaign(manifest: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], *,
                         manifest_sha256: str) -> dict[str, Any]:
    """Validate paired trajectories and freeze the nominal or scenario-based Top-1."""
    if manifest.get('scope') != 'preregistered_robust_design' or not manifest_sha256:
        raise ValueError('invalid robust campaign identity')
    specs = manifest['trials']
    ids = [spec['trial_id'] for spec in specs]
    if len(ids) != len(set(ids)) or set(ids) != set(results):
        raise ValueError('robust trial roster incomplete or duplicated')
    sources = {source['source_id']: source for source in manifest['sources']}
    commits, paired, records, reference_k = set(), {}, [], None
    for spec in specs:
        item = results[spec['trial_id']]
        if (item.get('manifest_sha256') != manifest_sha256 or item.get('trial_id') != spec['trial_id']
                or not item.get('code_commit') or not item.get('cluster_id')):
            raise ValueError('robust trial identity mismatch')
        commits.add(item['code_commit'])
        prefix = {'trial_id': spec['trial_id'], 'label': f"{spec['trial_id']}/top1", 'role': 'top1'}
        if spec['status'] != 'ready':
            if item.get('status') != spec['status']:
                raise ValueError('robust initial status mismatch')
            records.append({**prefix, 'candidate': None, 'status': spec['status']})
            continue
        cfg = spec['config']
        runtime = item.get('runtime', {})
        if (item.get('status') != 'completed' or item.get('mode') != spec['mode']
                or item.get('config') != cfg or item.get('source_id') != spec['source_id']
                or item.get('initial_index') != spec['initial_index'] or spec['mode'] not in ('nominal', 'mean', 'worst')
                or item.get('source_parameter_hash') != sources[spec['source_id']]['source_parameter_hash']
                or item.get('source_sha256') != cfg['source_sha256']
                or item.get('checkpoint_sha256') != cfg['checkpoint_sha256']
                or item.get('k_points_sha256') != cfg['k_points_sha256']
                or runtime.get('contract') != cfg['runtime']
                or runtime.get('matmul_tf32_enabled') is not False or runtime.get('cudnn_tf32_enabled') is not False):
            raise ValueError('robust trial configuration mismatch')
        points = np.ascontiguousarray(item['k_points'], dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                or hashlib.sha256(points.tobytes()).hexdigest() != cfg['k_points_sha256']
                or (reference_k is not None and points.tolist() != reference_k)):
            raise ValueError('robust trial k-point identity mismatch')
        reference_k = points.tolist()
        history, selection = item['history'], item['selection']
        candidates = selection['candidates']
        if (len(history) != manifest['evaluations'] or len(candidates) != len(history)
                or history[0]['parameters'] != cfg['initial']
                or any(row['index'] != index for index, row in enumerate(candidates))):
            raise ValueError('robust trajectory or initial mismatch')
        center, pair = candidates[0], (spec['source_id'], spec['initial_index'])
        if pair in paired:
            prior = paired[pair]
            nominal_status = center.get('nominal_status', center['status'])
            prior_status = prior.get('nominal_status', prior['status'])
            if (center['master_grid_sha256'] != prior['master_grid_sha256']
                    or center['input_sha256'] != prior['input_sha256'] or nominal_status != prior_status):
                raise ValueError('paired robust initial geometry differs')
            if nominal_status == 'surrogate_only':
                bands, old = np.asarray(center['bands'], dtype=float), np.asarray(prior['bands'], dtype=float)
                if (bands.shape != old.shape or not np.isfinite(bands).all() or not np.isfinite(old).all()
                        or np.max(np.abs(bands - old)) > float(cfg['device_parity_tolerance'])):
                    raise ValueError('paired robust initial spectrum differs')
        else:
            paired[pair] = center
        eligible = [row for row in candidates if row['status'] == 'surrogate_only']
        def score(row: Mapping[str, Any]) -> float:
            if spec['mode'] == 'nominal':
                value = float(row['predicted_gap']['target_margin'])
            else:
                prediction = row['robust_prediction']
                bands = np.asarray(prediction['bands'], dtype=np.float64)
                pair_index, target = cfg['lower_band'], float(cfg['target_frequency'])
                if (prediction['status'] != 'surrogate_only' or bands.ndim != 3
                        or bands.shape[0] != len(manifest['offsets']) or not np.isfinite(bands).all()
                        or bands.shape[2] <= pair_index or not np.isfinite(target) or target <= 0):
                    raise ValueError('invalid robust candidate spectrum')
                margins = 2 * np.minimum(target - bands[:, :, pair_index - 1].max(axis=1),
                                          bands[:, :, pair_index].min(axis=1) - target) / target
                value = float(margins.mean() if spec['mode'] == 'mean' else margins.min())
                if not np.allclose(margins, prediction['scenario_target_margins'], rtol=0, atol=1e-10):
                    raise ValueError('robust per-scenario margin mismatch')
                if not math.isclose(value, float(prediction['selection_score']), rel_tol=0, abs_tol=1e-10):
                    raise ValueError('robust candidate score mismatch')
            if not math.isfinite(value):
                raise ValueError('nonfinite robust candidate score')
            return value
        if eligible:
            winner = max(eligible, key=lambda row: (score(row), -row['index']))
            if selection['status'] != 'selected' or selection['selected_index'] != winner['index']:
                raise ValueError('robust Top-1 selection mismatch')
            top = winner
        else:
            if selection['status'] != 'no_valid_candidates' or selection['selected_index'] is not None:
                raise ValueError('invalid empty robust selection')
            top = None
        records.append({'trial_id': spec['trial_id'], 'label': f"{spec['trial_id']}/initial",
                        'role': 'initial', 'candidate': center})
        records.append({**prefix, 'candidate': top, 'status': 'selected' if top else 'no_valid_candidate'})
    if len(commits) != 1 or reference_k is None:
        raise ValueError('robust code or k-point identity missing or inconsistent')
    return {'scope': 'locked_paired_surrogate_campaign', 'purpose': 'robust_design',
            'manifest_sha256': manifest_sha256, 'code_commit': next(iter(commits)),
            'k_points': reference_k, 'records': records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'settings', 'trial-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw_manifest, raw_settings = Path(args.manifest).read_bytes(), Path(args.settings).read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest['campaign_sha256'] != hashlib.sha256(raw_settings).hexdigest():
        raise ValueError('robust campaign settings differ')
    trials, hashes = {}, {}
    for spec in manifest['trials']:
        raw = (Path(args.trial_directory)/f"{spec['trial_id']}.json").read_bytes()
        trials[spec['trial_id']] = json.loads(raw)
        hashes[spec['trial_id']] = hashlib.sha256(raw).hexdigest()
    result = lock_robust_campaign(manifest, trials, manifest_sha256=hashlib.sha256(raw_manifest).hexdigest())
    result['trial_sha256'] = hashes
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
