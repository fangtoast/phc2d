"""Prepare registered higher-accuracy replays of frozen inverse endpoints.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _encode(value: Any) -> bytes:
    return json.dumps(value, indent=2, allow_nan=False).encode('utf8')


def build_high_accuracy_replay(lock: Mapping[str, Any], manifest: Mapping[str, Any], base: Mapping[str, Any],
                               registration: Mapping[str, Any], *, lock_sha256: str, manifest_sha256: str,
                               base_sha256: str, registration_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Change only the solver contract for preregistered, already locked endpoints."""
    if (lock.get('scope') != 'locked_paired_surrogate_campaign' or lock.get('manifest_sha256') != manifest_sha256
            or manifest.get('base_config_sha256') != base_sha256 or not lock_sha256 or not registration_sha256
            or registration.get('experiment_id') != manifest.get('experiment_id')
            or float(registration['target_frequency']) != float(base['target_frequency'])):
        raise ValueError('high-accuracy replay parent identity mismatch')
    resolution, verification = registration['resolution'], registration['verification_resolution']
    if (isinstance(resolution, bool) or not isinstance(resolution, int) or isinstance(verification, bool)
            or not isinstance(verification, int) or not base['solver_config']['resolution'] < resolution < verification
            or base['physical_protocol']['offsets'] != [0.0] or registration['offsets'] != [0.0]):
        raise ValueError('invalid high-accuracy resolution or physical offsets')
    fields = ('initial_spacing', 'edge_tolerance', 'resolution_tolerance')
    if any(not math.isfinite(float(registration[key])) or float(registration[key]) <= 0
           or float(registration[key]) >= float(base['physical_protocol']['refinement'][key]) for key in fields):
        raise ValueError('high-accuracy refinement must be tighter than the parent')
    roles, records = registration['roles'], lock['records']
    labels = [row['label'] for row in records]
    if not roles or len(roles) != len(set(roles)) or len(labels) != len(set(labels)) or not set(roles).issubset(labels):
        raise ValueError('high-accuracy role roster mismatch')
    selected = [copy.deepcopy(row) for row in records if row['label'] in roles]
    if any(row['role'] not in ('top1', 'final') for row in selected):
        raise ValueError('high-accuracy roles must be selected endpoints')
    updated_base = copy.deepcopy(dict(base))
    updated_base['solver_config']['resolution'] = resolution
    refinement = updated_base['physical_protocol']['refinement']
    refinement.update({key: float(registration[key]) for key in fields}, verification_resolution=verification)
    updated_manifest = copy.deepcopy(dict(manifest))
    updated_manifest.update(base_config_sha256=hashlib.sha256(_encode(updated_base)).hexdigest(),
        parent_nominal_manifest_sha256=manifest_sha256, high_accuracy_registration_sha256=registration_sha256)
    updated_lock = copy.deepcopy(dict(lock))
    updated_lock.update(manifest_sha256=hashlib.sha256(_encode(updated_manifest)).hexdigest(), records=selected,
        parent_selection_lock_sha256=lock_sha256, high_accuracy_registration_sha256=registration_sha256)
    return updated_base, updated_manifest, updated_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lock', 'manifest', 'base', 'registration', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    names = ('lock', 'manifest', 'base', 'registration')
    raw = {name: Path(getattr(args, name)).read_bytes() for name in names}
    updated = build_high_accuracy_replay(*(json.loads(raw[name]) for name in names),
        **{name+'_sha256': hashlib.sha256(raw[name]).hexdigest() for name in names})
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in zip(('base.json', 'manifest.json', 'selection.lock.json'), updated):
        (output/name).write_bytes(_encode(payload))


if __name__ == '__main__':
    main()
