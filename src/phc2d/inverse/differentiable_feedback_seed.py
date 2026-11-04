"""Prepare the registered local observation stencil for one feedback seed.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from phc2d.inverse.differentiable_selection import rank_hard_trajectory
from phc2d.inverse.differentiable_soft_audit import audit_locked_soft
from phc2d.inverse.differentiable_mvp import load_design_inputs


def prepare_feedback_stencil(source: Any, predictor: Any, *, base: Mapping[str, Any],
                             campaign: Mapping[str, Any], initial: list[float]) -> dict[str, Any]:
    """Evaluate fixed nominal stencil geometries without any MPB observation."""
    started = time.monotonic()
    keys = tuple((int(i), str(component)) for i, component in base['coefficient_keys'])
    reference = np.asarray([source.parameters['modes'][i][component] for i, component in keys]
                           + [source.parameters['threshold']], dtype=np.float64)
    center = np.asarray(initial, dtype=np.float64)
    scale, lo, hi = (np.asarray(base[key], dtype=np.float64) for key in ('scales', 'lower', 'upper'))
    if (center.shape != reference.shape or scale.shape != center.shape or lo.shape != center.shape
            or hi.shape != center.shape or not all(np.isfinite(x).all() for x in (center, scale, lo, hi))
            or np.any(scale <= 0) or np.any(center < lo) or np.any(center > hi)):
        raise ValueError('invalid feedback seed or parameter contract')
    offsets = [np.asarray(row, dtype=np.float64) for row in campaign['initial_offsets']]
    if (len(offsets) != 5 or any(row.shape != center.shape or not np.isfinite(row).all() for row in offsets)
            or len({tuple(row) for row in offsets}) != len(offsets)
            or not np.all(offsets[0] == 0)):
        raise ValueError('invalid preregistered feedback stencil')
    history, indices, records = [], [], [None] * len(offsets)
    for index, offset in enumerate(offsets):
        point = center + offset
        label = f'offset_{index}'
        if np.any(point < lo) or np.any(point > hi):
            records[index] = {'label': label, 'point': point.tolist(), 'candidate': None,
                              'status': 'outside_parameter_bounds'}
            continue
        indices.append(index)
        history.append({'parameters': point.tolist(), 'fourier_parameters': (reference + scale * point).tolist()})
    if history:
        screening = rank_hard_trajectory(source, predictor, history, coefficient_keys=keys,
            max_mode_radius=base['max_mode_radius'], raster_method='master_block_mean_v1',
            source_resolution=base['source_resolution'], input_resolution=base['input_resolution'],
            material_fraction_bounds=tuple(base['material_fraction_bounds']),
            lower_band=base['lower_band'], target_frequency=base['target_frequency'])
        for index, candidate, item in zip(indices, screening['candidates'], history):
            records[index] = {'label': f'offset_{index}', 'point': item['parameters'], 'candidate': candidate}
    soft = audit_locked_soft({'records': records}, predictor, base)
    beta, pair = float(base['beta']), int(base['lower_band'])
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('invalid feedback smoothing parameter')
    hard_predictions = {row['candidate']['master_grid_sha256']: row['candidate']['bands']
                        for row in records if row['candidate'] is not None and 'bands' in row['candidate']}
    for record, soft_row in zip(records, soft['roles']):
        if soft_row['status'] != 'computed':
            continue
        band = np.asarray(soft_row['soft_bands'], dtype=np.float64)
        hard = np.asarray(hard_predictions[record['candidate']['master_grid_sha256']], dtype=np.float64)
        low, upper = band[:, pair - 1], band[:, pair]
        low_max, upper_min = float(low.max()), float(upper.min())
        record['surrogate_edges'] = {
            'hard_edges': [float(hard[:, pair - 1].max()), float(hard[:, pair].min())],
            'soft_exact_edges': [low_max, upper_min],
            'soft_smooth_edges': [float(low_max + np.log(np.exp(beta * (low - low_max)).sum()) / beta),
                                  float(upper_min - np.log(np.exp(-beta * (upper - upper_min)).sum()) / beta)]}
    return {'scope': 'feedback_initial_stencil_surrogate_only', 'source_parameter_hash': source.parameter_hash,
            'initial': center.tolist(), 'records': records, 'soft_audit': soft,
            'elapsed_s': time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'base', 'design-manifest', 'design-lock', 'source', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--source-id', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    campaign_raw = Path(args.campaign).read_bytes()
    base_raw = Path(args.base).read_bytes()
    manifest_raw = Path(args.design_manifest).read_bytes()
    lock_raw = Path(args.design_lock).read_bytes()
    campaign, base, manifest, lock = (json.loads(raw) for raw in (campaign_raw, base_raw, manifest_raw, lock_raw))
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if (lock.get('scope') != 'locked_paired_surrogate_campaign' or lock.get('manifest_sha256') != manifest_sha
            or manifest.get('base_config_sha256') != hashlib.sha256(base_raw).hexdigest()):
        raise ValueError('feedback seed does not match the locked design campaign')
    source_rows = [row for row in manifest['sources'] if row['source_id'] == args.source_id]
    trials = [row for row in manifest['trials'] if row['source_id'] == args.source_id and row['status'] == 'ready']
    if len(source_rows) != 1 or not trials:
        raise ValueError('source absent from design manifest or has no valid initial')
    source, predictor, runtime = load_design_inputs(trials[0]['config'], args.source, args.checkpoint, device=args.device)
    if source.parameter_hash != source_rows[0]['source_parameter_hash']:
        raise ValueError('source parameter identity mismatch')
    commit, cluster = os.environ['PHC2D_DEPLOYED_GIT_COMMIT'], os.environ['PHC2D_CLUSTER_ID']
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    campaign_sha, lock_sha = hashlib.sha256(campaign_raw).hexdigest(), hashlib.sha256(lock_raw).hexdigest()
    for item in source_rows[0]['initials']:
        index = item['initial_index']
        path = root / f'source_{args.source_id:04d}_initial_{index}.json'
        identity = {'source_id': args.source_id, 'initial_index': index, 'design_lock_sha256': lock_sha,
                    'design_manifest_sha256': manifest_sha, 'campaign_sha256': campaign_sha,
                    'base_config_sha256': hashlib.sha256(base_raw).hexdigest(),
                    'k_points_sha256': predictor.k_points_sha256,
                    'checkpoint_sha256': predictor.checkpoint_sha256, 'code_commit': commit, 'cluster_id': cluster}
        if path.exists():
            prior = json.loads(path.read_bytes())
            if any(prior.get(key) != value for key, value in identity.items()):
                raise ValueError('existing feedback seed identity mismatch')
            continue
        if item['status'] == 'ready':
            result = prepare_feedback_stencil(source, predictor, base=base, campaign=campaign, initial=item['initial'])
            result['status'] = 'prepared'
            result['runtime'] = runtime
        else:
            result = {'status': item['status'], 'initial': None, 'records': []}
        result.update(identity)
        with path.open('x', encoding='utf8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'source_id': args.source_id, 'initial_count': len(source_rows[0]['initials'])}))


if __name__ == '__main__':
    main()
