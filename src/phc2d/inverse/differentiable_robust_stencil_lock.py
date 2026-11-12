"""Freeze scenario-specific feedback stencils before their physical evaluations.

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


def lock_robust_stencils(manifest: Mapping[str, Any], stencils: Mapping[str, Mapping[str, Any]],
                         nominal: Mapping[str, Mapping[str, Any]], *, manifest_sha256: str,
                         design_lock: Mapping[str, Any], nominal_sha256: Mapping[str, str]) -> dict[str, Any]:
    """Validate the full paired stencil roster and retain every registered role."""
    if (manifest.get('scope') != 'preregistered_robust_design' or not manifest_sha256
            or design_lock.get('manifest_sha256') != manifest['parent_manifest_sha256']):
        raise ValueError('robust stencil manifest identity mismatch')
    expected = {(source['source_id'], row['initial_index']): (source, row)
                for source in manifest['sources'] for row in source['initials']}
    names = {f'source_{source:04d}_initial_{index}' for source, index in expected}
    if not expected or set(stencils) != names or set(nominal) != names or set(nominal_sha256) != names:
        raise ValueError('robust stencil roster incomplete or duplicated')
    trials = {row['trial_id']: row for row in manifest['trials']}
    commits, records = set(), []
    for (source_id, initial_index), (source, initial) in sorted(expected.items()):
        name = f'source_{source_id:04d}_initial_{initial_index}'
        scenario, origin = stencils[name], nominal[name]
        trial_id = f'{name}_nominal'
        if (trial_id not in trials or scenario.get('manifest_sha256') != manifest_sha256
                or scenario.get('source_id') != source_id or scenario.get('initial_index') != initial_index
                or scenario.get('stencil_sha256') != nominal_sha256[name]
                or not scenario.get('code_commit') or not scenario.get('cluster_id')
                or origin.get('design_manifest_sha256') != manifest['parent_manifest_sha256']):
            raise ValueError('robust stencil identity mismatch')
        commits.add(scenario['code_commit'])
        if initial['status'] != 'ready':
            if scenario.get('status') != initial['status'] or origin.get('status') != initial['status']:
                raise ValueError('robust stencil initial status mismatch')
            for index in range(5):
                role = f'offset_{index}'
                records.append({'label': f'{trial_id}/{role}', 'trial_id': trial_id, 'role': role,
                                'candidate': None, 'status': initial['status']})
            continue
        if (scenario.get('checkpoint_sha256') != trials[trial_id]['config']['checkpoint_sha256']
                or scenario.get('k_points_sha256') != trials[trial_id]['config']['k_points_sha256']
                or origin.get('checkpoint_sha256') != scenario.get('checkpoint_sha256')
                or origin.get('k_points_sha256') != scenario.get('k_points_sha256')
                or scenario.get('offsets') != manifest['offsets']
                or origin.get('source_parameter_hash') != source['source_parameter_hash']
                or scenario.get('source_parameter_hash') != source['source_parameter_hash']
                or origin.get('initial') != initial['initial'] or scenario.get('initial') != initial['initial']):
            raise ValueError('robust stencil source or checkpoint mismatch')
        runtime = scenario['runtime']
        if (runtime.get('contract') != trials[trial_id]['config']['runtime']
                or runtime.get('matmul_tf32_enabled') is not False or runtime.get('cudnn_tf32_enabled') is not False):
            raise ValueError('robust stencil precision mismatch')
        if len(origin['records']) != 5 or len(scenario['records']) != 5:
            raise ValueError('robust stencil requires all registered roles')
        for index, (old, prediction) in enumerate(zip(origin['records'], scenario['records'])):
            role = f'offset_{index}'
            if old['label'] != role or prediction['label'] != role:
                raise ValueError('robust stencil role order mismatch')
            candidate = old['candidate']
            item = {'label': f'{trial_id}/{role}', 'trial_id': trial_id, 'role': role,
                    'point': old.get('point'), 'candidate': candidate, 'scenario_prediction': prediction}
            if prediction['status'] != 'computed':
                item.update(candidate=None, status=prediction['status'])
            elif (candidate is None or prediction['point'] != old['point']
                  or prediction['parameter_hash'] != candidate['parameter_hash']
                  or prediction['hard']['grid_sha256'][manifest['offsets'].index(0.0)] != candidate['master_grid_sha256']):
                raise ValueError('robust stencil candidate identity mismatch')
            records.append(item)
    if len(commits) != 1:
        raise ValueError('robust stencil code identities differ')
    return {'scope': 'locked_paired_surrogate_campaign', 'purpose': 'robust_feedback_stencil',
            'manifest_sha256': manifest_sha256, 'code_commit': next(iter(commits)),
            'k_points': design_lock['k_points'], 'records': records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'design-lock', 'stencil-directory', 'nominal-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw = Path(args.manifest).read_bytes()
    def load(directory: str) -> tuple[dict[str, Any], dict[str, str]]:
        paths = list(Path(directory).glob('source_*_initial_*.json'))
        contents = {path.stem: path.read_bytes() for path in paths}
        return ({name: json.loads(value) for name, value in contents.items()},
                {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()})
    stencils, _ = load(args.stencil_directory)
    nominal, hashes = load(args.nominal_directory)
    result = lock_robust_stencils(json.loads(raw), stencils, nominal, manifest_sha256=hashlib.sha256(raw).hexdigest(),
                                  design_lock=json.loads(Path(args.design_lock).read_bytes()), nominal_sha256=hashes)
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
