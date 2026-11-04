"""Prepare matched nominal and manufacturing-scenario design trials.

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


def build_robust_campaign(design: Mapping[str, Any], settings: Mapping[str, Any], *,
                          design_manifest_sha256: str) -> dict[str, Any]:
    """Pair all modes on the previously frozen source and initial identities."""
    if (design.get('scope') != 'preregistered_surrogate_design' or not design_manifest_sha256
            or settings.get('experiment_id') != 'EXP-INV-015'):
        raise ValueError('robust design campaign identity mismatch')
    modes = settings['modes']
    offsets = [float(value) for value in settings['offsets']]
    feedback = settings['feedback']
    if (len(modes) != len(set(modes)) or set(modes) != {'nominal', 'mean', 'worst'}
            or len(offsets) != 3 or len(set(offsets)) != 3 or 0.0 not in offsets
            or not all(math.isfinite(x) for x in offsets)
            or not any(x < 0 for x in offsets) or not any(x > 0 for x in offsets)
            or isinstance(settings['evaluations'], bool) or settings['evaluations'] <= 0
            or not isinstance(settings['evaluations'], int)
            or not all(math.isfinite(float(feedback[key])) and float(feedback[key]) > 0 for key in ('radius', 'ridge'))):
        raise ValueError('invalid robust design modes, offsets or evaluation budget')
    trials, seen, protocol = [], set(), None
    for source in design['sources']:
        source_id = source['source_id']
        for initial in source['initials']:
            key = (source_id, initial['initial_index'])
            if key in seen:
                raise ValueError('duplicate robust source and initial')
            seen.add(key)
            old = [row for row in design['trials'] if row['source_id'] == source_id
                   and row['initial_index'] == initial['initial_index'] and row['method'] == 'adam']
            if len(old) != 1 or old[0]['status'] != initial['status']:
                raise ValueError('paired source or initial differs from design')
            for mode in modes:
                config = None
                if initial['status'] == 'ready':
                    config = copy.deepcopy(old[0]['config'])
                    if config['initial'] != initial['initial'] or config['source_sha256'] != source['source_sha256']:
                        raise ValueError('source configuration does not match frozen initial')
                    config.update(experiment_id=settings['experiment_id'], evaluations=settings['evaluations'],
                                  offsets=offsets)
                    nominal_protocol = config['physical_protocol']
                    if nominal_protocol['offsets'] != [0.0]:
                        raise ValueError('parent physical protocol must be nominal')
                    proposed = {**copy.deepcopy(nominal_protocol), 'offsets': offsets}
                    if protocol is not None and protocol != proposed:
                        raise ValueError('paired physical protocols differ')
                    protocol = proposed
                trials.append({'trial_id': f"source_{source_id:04d}_initial_{initial['initial_index']}_{mode}",
                    'source_id': source_id, 'initial_index': initial['initial_index'], 'mode': mode,
                    'method': mode, 'status': initial['status'], 'config': config})
    if not trials or protocol is None:
        raise ValueError('empty robust campaign or no ready trial')
    return {'experiment_id': settings['experiment_id'], 'scope': 'preregistered_robust_design',
            'parent_manifest_sha256': design_manifest_sha256, 'base_config_sha256': design['base_config_sha256'],
            'sources': copy.deepcopy(design['sources']), 'offsets': offsets, 'physical_protocol': protocol, 'modes': list(modes),
            'evaluations': settings['evaluations'], 'feedback': copy.deepcopy(feedback), 'trials': trials}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('design-manifest', 'settings', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    design_raw, settings_raw = Path(args.design_manifest).read_bytes(), Path(args.settings).read_bytes()
    settings = json.loads(settings_raw)
    digest = hashlib.sha256(design_raw).hexdigest()
    if settings['parent_manifest_sha256'] != digest:
        raise ValueError('registered parent design manifest differs')
    result = build_robust_campaign(json.loads(design_raw), settings, design_manifest_sha256=digest)
    result['campaign_sha256'] = hashlib.sha256(settings_raw).hexdigest()
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
