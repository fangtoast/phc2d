"""Advance matched feedback arms from a frozen round's physical observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from phc2d.inverse.differentiable_feedback_campaign import advance_feedback_arm
from phc2d.inverse.differentiable_feedback_round import save_round_completion


def advance_feedback_round(arms: Mapping[str, Any], observations: Mapping[str, Any], *,
                           lock: Mapping[str, Any], lock_sha256: str,
                           request_directory: Path) -> dict[str, Any]:
    """Resume successful arms and retain failures without inventing solver results."""
    if (lock.get('purpose') != 'feedback_round' or observations.get('scope') != 'feedback_round_observations'
            or observations.get('selection_lock_sha256') != lock_sha256):
        raise ValueError('feedback round lock or observation identity mismatch')
    rows = {row['trial_id']: row for row in observations['records']}
    locked = {row['trial_id']: row for row in lock['records']}
    if len(rows) != len(observations['records']) or set(rows) != set(locked):
        raise ValueError('feedback round observation roster mismatch')
    updated = copy.deepcopy(dict(arms))
    output = []
    for arm in arms['arms']:
        name = f"{arm['seed_id']}_{arm['arm']}"
        if name not in rows:
            raise ValueError('feedback arm missing from round')
        row, saved = rows[name], locked[name]
        if arm['status'] != 'ready':
            output.append(copy.deepcopy(arm))
            continue
        directory = request_directory/name
        request = json.loads((directory/'request.json').read_bytes())
        if request['frozen']['frozen_sha256'] != saved['frozen_sha256']:
            raise ValueError('saved round request identity mismatch')
        if row['status'] == 'cached_physical_observation':
            composite = request['frozen']['cached_observation']
            physical = arm['physical_observations'][composite['parameter_hash']]
        elif row['status'] == 'observed':
            composite, physical = row['composite'], row['physical']
        else:
            failed = copy.deepcopy(dict(arm))
            failed.update(status='round_physical_unavailable', round_failure=row['status'])
            output.append(failed)
            continue
        record = save_round_completion(directory, physical=physical, composite=composite)
        output.append(advance_feedback_arm(arm, record))
    updated['arms'] = output
    updated['previous_round_lock_sha256'] = lock_sha256
    updated['completed_rounds'] = lock['round_index'] + 1
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('arms', 'observations', 'lock', 'request-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw_arms, raw_observations, raw_lock = (Path(getattr(args, name)).read_bytes()
                                             for name in ('arms', 'observations', 'lock'))
    result = advance_feedback_round(json.loads(raw_arms), json.loads(raw_observations), lock=json.loads(raw_lock),
        lock_sha256=hashlib.sha256(raw_lock).hexdigest(), request_directory=Path(args.request_directory))
    result['parent_arms_sha256'] = hashlib.sha256(raw_arms).hexdigest()
    result['round_observations_sha256'] = hashlib.sha256(raw_observations).hexdigest()
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
