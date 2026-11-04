"""Freeze physically corrected robust trials before their final MPB evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, allow_nan=False).encode('utf8')


def lock_corrected_robust_campaign(manifest: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], *,
                                   manifest_sha256: str, feedback_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate corrected soft selection and prepare matching physical role identities."""
    if manifest.get('scope') != 'preregistered_robust_design' or not manifest_sha256 or not feedback_sha256:
        raise ValueError('corrected robust campaign identity mismatch')
    specs = manifest['trials']
    names = [spec['trial_id'] + '_corrected' for spec in specs]
    if len(names) != len(set(names)) or set(names) != set(results):
        raise ValueError('corrected robust trial roster incomplete or duplicated')
    handoff = copy.deepcopy(dict(manifest))
    handoff['parent_robust_manifest_sha256'] = manifest_sha256
    for spec in handoff['trials']:
        spec['trial_id'] += '_corrected'
    handoff_sha = hashlib.sha256(_bytes(handoff)).hexdigest()
    sources = {source['source_id']: source for source in manifest['sources']}
    records, commits, reference_k = [], set(), None
    for spec in specs:
        trial_id = spec['trial_id'] + '_corrected'
        item = results[trial_id]
        if (item.get('trial_id') != trial_id or item.get('manifest_sha256') != manifest_sha256
                or item.get('feedback_sha256') != feedback_sha256 or not item.get('code_commit')
                or not item.get('cluster_id')):
            raise ValueError('corrected robust trial identity mismatch')
        commits.add(item['code_commit'])
        prefix = {'trial_id': trial_id, 'label': f'{trial_id}/top1', 'role': 'top1'}
        if item['status'] != 'completed':
            records.append({**prefix, 'candidate': None, 'status': item['status']})
            continue
        cfg, expected = item['config'], spec['config']
        used_offsets = [0.0] if spec['mode'] == 'nominal' else manifest['offsets']
        cfg_base = {key: value for key, value in cfg.items()
                    if key not in ('center', 'radius', 'ridge', 'surrogate_contract_sha256', 'scenario_reduction')}
        expected_base = {**expected, 'offsets': used_offsets}
        runtime = item['runtime']
        if (cfg_base != expected_base or cfg['center'] != expected['initial']
                or cfg['radius'] != manifest['feedback']['radius'] or cfg['ridge'] != manifest['feedback']['ridge']
                or cfg['scenario_reduction'] != ('mean' if spec['mode'] == 'mean' else 'worst')
                or item['mode'] != spec['mode'] or item['source_id'] != spec['source_id']
                or item['initial_index'] != spec['initial_index']
                or item['source_parameter_hash'] != sources[spec['source_id']]['source_parameter_hash']
                or item['checkpoint_sha256'] != expected['checkpoint_sha256']
                or item['source_sha256'] != expected['source_sha256']
                or item['k_points_sha256'] != expected['k_points_sha256']
                or runtime.get('contract') != expected['runtime']
                or runtime.get('matmul_tf32_enabled') is not False or runtime.get('cudnn_tf32_enabled') is not False):
            raise ValueError('corrected robust trial configuration mismatch')
        points = np.ascontiguousarray(item['k_points'], dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                or hashlib.sha256(points.tobytes()).hexdigest() != expected['k_points_sha256']
                or (reference_k is not None and points.tolist() != reference_k)):
            raise ValueError('corrected robust trial k-point mismatch')
        reference_k = points.tolist()
        history, screening = item['history'], item['hard_screening']
        candidates = screening['candidates']
        if (len(history) != manifest['evaluations'] or len(candidates) != len(history)
                or history[0]['parameters'] != expected['initial']
                or any(row['index'] != index for index, row in enumerate(candidates))):
            raise ValueError('corrected robust trajectory or initial mismatch')
        eligible = [row for row in candidates if row['status'] == 'surrogate_only']
        if not eligible or any(not math.isfinite(float(history[row['index']]['objective'])) for row in eligible):
            raise ValueError('completed corrected robust trial has no finite candidate')
        winner = max(eligible, key=lambda row: (float(history[row['index']]['objective']), -row['index']))
        if item['selected_index'] != winner['index'] or item['prefit_prediction']['point'] != history[winner['index']]['parameters']:
            raise ValueError('corrected robust selection or prefit prediction mismatch')
        diagnostic = item['full_scenarios_diagnostic']['candidates']
        if len(diagnostic) != len(candidates):
            raise ValueError('full-scenario diagnostic roster mismatch')
        def candidate(index: int) -> dict[str, Any]:
            row, scenario = candidates[index], diagnostic[index]
            if row['master_grid_sha256'] != scenario['master_grid_sha256']:
                raise ValueError('corrected scenario diagnostic geometry mismatch')
            if scenario['status'] == 'duplicate_geometry':
                scenario = diagnostic[scenario['duplicate_of']]
            return {**row, 'robust_prediction': scenario.get('robust_prediction'),
                    'scenario_prediction_status': scenario['status']}
        records.append({'trial_id': trial_id, 'label': f'{trial_id}/initial', 'role': 'initial',
                        'candidate': candidate(0)})
        records.append({**prefix, 'candidate': candidate(winner['index']), 'status': 'selected',
                        'prefit_prediction': item['prefit_prediction']})
    if len(commits) != 1 or reference_k is None:
        raise ValueError('corrected robust code or k-point identity inconsistent')
    lock = {'scope': 'locked_paired_surrogate_campaign', 'purpose': 'robust_design',
            'manifest_sha256': handoff_sha, 'parent_robust_manifest_sha256': manifest_sha256,
            'feedback_sha256': feedback_sha256, 'code_commit': next(iter(commits)),
            'k_points': reference_k, 'records': records}
    return handoff, lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'feedback', 'trial-directory', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    manifest_raw, feedback_raw = Path(args.manifest).read_bytes(), Path(args.feedback).read_bytes()
    manifest = json.loads(manifest_raw)
    results, hashes = {}, {}
    for spec in manifest['trials']:
        name = spec['trial_id'] + '_corrected'
        raw = (Path(args.trial_directory)/f'{name}.json').read_bytes()
        results[name], hashes[name] = json.loads(raw), hashlib.sha256(raw).hexdigest()
    handoff, lock = lock_corrected_robust_campaign(manifest, results,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), feedback_sha256=hashlib.sha256(feedback_raw).hexdigest())
    lock['trial_sha256'] = hashes
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output/'manifest.json').write_bytes(_bytes(handoff))
    (output/'selection.lock.json').write_bytes(_bytes(lock))


if __name__ == '__main__':
    main()
