"""Derive preregistered unseen-offset replays from frozen robust selections.

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


def build_unseen_replay(lock: Mapping[str, Any], manifest: Mapping[str, Any],
                        registration: Mapping[str, Any], *, lock_sha256: str,
                        manifest_sha256: str, registration_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep frozen selected geometries and register new physical offsets only."""
    if (manifest.get('scope') != 'preregistered_robust_design' or lock.get('purpose') != 'robust_design'
            or lock.get('manifest_sha256') != manifest_sha256 or not lock_sha256 or not registration_sha256
            or registration.get('experiment_id') != manifest.get('experiment_id')):
        raise ValueError('unseen replay identity mismatch')
    offsets = [float(value) for value in registration['offsets']]
    if (len(offsets) != 3 or len(set(offsets)) != 3 or 0.0 not in offsets
            or not all(math.isfinite(value) for value in offsets)
            or not any(value < 0 for value in offsets) or not any(value > 0 for value in offsets)
            or set(offsets).intersection(manifest['offsets']) != {0.0}):
        raise ValueError('unseen offsets must differ from the design scenarios')
    roles = registration['roles']
    if not roles or len(roles) != len(set(roles)):
        raise ValueError('invalid unseen role registration')
    records = lock['records']
    labels = [row['label'] for row in records]
    if len(labels) != len(set(labels)):
        raise ValueError('duplicate parent lock role')
    selected = [copy.deepcopy(row) for row in records if row['label'] in roles]
    if not selected or any(row['role'] != 'top1' for row in selected):
        raise ValueError('unseen replay requires registered top1 roles')
    for row in selected:
        if row['candidate'] is not None:
            row['candidate'].pop('robust_prediction', None)
    new_manifest = copy.deepcopy(dict(manifest))
    new_manifest['parent_robust_manifest_sha256'] = manifest_sha256
    new_manifest['unseen_registration_sha256'] = registration_sha256
    new_manifest['offsets'] = offsets
    new_manifest['physical_protocol']['offsets'] = offsets
    new_lock = copy.deepcopy(dict(lock))
    new_lock.update(manifest_sha256=hashlib.sha256(_encode(new_manifest)).hexdigest(), records=selected,
                    parent_selection_lock_sha256=lock_sha256, unseen_registration_sha256=registration_sha256)
    return new_manifest, new_lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lock', 'manifest', 'registration', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    lock_raw, manifest_raw, registration_raw = (Path(getattr(args, key)).read_bytes()
                                                for key in ('lock', 'manifest', 'registration'))
    manifest, lock = build_unseen_replay(json.loads(lock_raw), json.loads(manifest_raw), json.loads(registration_raw),
        lock_sha256=hashlib.sha256(lock_raw).hexdigest(), manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        registration_sha256=hashlib.sha256(registration_raw).hexdigest())
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output/'manifest.json').write_bytes(_encode(manifest))
    (output/'selection.lock.json').write_bytes(_encode(lock))


if __name__ == '__main__':
    main()
