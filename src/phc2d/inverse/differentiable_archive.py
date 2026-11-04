"""Archive locked design shards with standard project run artifacts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping


def archive_design(*, config_raw: bytes, lock_raw: bytes, shard_raw: Mapping[str, bytes],
                   output: Path) -> None:
    """Write a design-only run after validating all immutable file identities."""
    import yaml

    config, lock = json.loads(config_raw), json.loads(lock_raw)
    roles = ('directions', 'adam', 'de', 'random')
    if set(shard_raw) != set(roles) or hashlib.sha256(config_raw).hexdigest() != lock['config_sha256']:
        raise ValueError('configuration or shard roster mismatch')
    for role in roles:
        if hashlib.sha256(shard_raw[role]).hexdigest() != lock['shard_sha256'][role]:
            raise ValueError(f'{role} shard hash mismatch')
    shards = {role: json.loads(shard_raw[role]) for role in roles}
    if any(shards[role]['role'] != role for role in roles):
        raise ValueError('shard role mismatch')
    from phc2d.inverse.differentiable_lock import combine_shards

    reconstructed = combine_shards(shards)
    if any(lock.get(key) != value for key, value in reconstructed.items()):
        raise ValueError('selection lock cannot be reconstructed from the shards')
    output.mkdir(parents=True, exist_ok=False)
    directory = output/'shards'
    directory.mkdir()
    for role in roles:
        (directory/f'{role}.json').write_bytes(shard_raw[role])
    (output/'selection.lock.json').write_bytes(lock_raw)
    (output/'config.resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf8')
    (output/'run.meta.json').write_text(json.dumps({'experiment_id': config['experiment_id'],
        'batch': config['batch'], 'scope': 'design_only', 'code_commit': lock['code_commit'],
        'config_sha256': lock['config_sha256'], 'checkpoint_sha256': lock['checkpoint_sha256'],
        'source_sha256': lock['source_sha256'], 'k_points_sha256': lock['k_points_sha256'],
        'lock_sha256': hashlib.sha256(lock_raw).hexdigest()}, indent=2), encoding='utf8')
    with (output/'history.csv').open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('role', 'step', 'parameters', 'fourier_parameters', 'objective', 'gradient'))
        writer.writeheader()
        for role in roles:
            for row in shards[role]['history']:
                writer.writerow({'role': role, 'step': row['step'],
                    'parameters': json.dumps(row['parameters']), 'fourier_parameters': json.dumps(row['fourier_parameters']),
                    'objective': row.get('objective'), 'gradient': json.dumps(row.get('gradient'))})
    result = {'status': 'design_locked', 'physical_validation': 'not_run',
              'roles': [{'label': row['label'], 'status': row['candidate']['status'] if row['candidate']
                         else row['status'], 'duplicate_of': row.get('duplicate_of')} for row in lock['records']]}
    counts = {}
    for role in roles:
        shard = shards[role]
        hard = sum(row['status'] in ('surrogate_only', 'band_order_violation',
                                     'negative_frequency', 'nonfinite_prediction')
                   for row in shard['selection']['candidates'])
        if role == 'directions':
            gradient = shard['diagnostics']['gradient']
            finite_difference = 2 * len(gradient) * len(shard['diagnostics']['finite_differences'])
            with_grad, backward, without_grad = 1, 1, finite_difference + hard
        elif role == 'adam':
            with_grad = backward = len(shard['history'])
            without_grad = hard
        else:
            with_grad = backward = 0
            without_grad = len(shard['history']) + hard
        counts[role] = {'forward_with_grad': with_grad, 'forward_without_grad': without_grad,
                        'backward': backward, 'hard_inference': hard}
    metrics = {'design_elapsed_s_by_shard': {role: shards[role]['elapsed_s'] for role in roles},
               'inference_counts_derived_from_completed_shards': counts,
               'physical_evaluations': 0, 'scope': 'surrogate_only'}
    for name, value in (('results.json', result), ('metrics.json', metrics)):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    (output/'run.log').write_text(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} archived locked design shards\n', encoding='utf8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'lock', 'directions', 'adam', 'de', 'random', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    roles = ('directions', 'adam', 'de', 'random')
    archive_design(config_raw=Path(args.config).read_bytes(), lock_raw=Path(args.lock).read_bytes(),
                   shard_raw={role: Path(getattr(args, role)).read_bytes() for role in roles},
                   output=Path(args.output))


if __name__ == '__main__':
    main()
