"""Compare independent GPU preflights before starting inverse-design shards.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def compare_preflights(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Check matching source, checkpoint, code, center input, gradient gate, and spectrum."""
    if len(records) != 2 or any(row.get('role') != 'preflight' for row in records):
        raise ValueError('two preflight records are required')
    if {row.get('cluster_id') for row in records} != {'paracloud_n40', 'paracloud_n56'}:
        raise ValueError('one N40 and one N56 platform preflight are required')
    reference = records[0]
    fields = ('config_sha256', 'source_sha256', 'source_parameter_hash', 'checkpoint_sha256',
              'k_points_sha256', 'code_commit', 'device_parity_tolerance', 'precision_contract')
    if any(not reference.get(key) or any(row.get(key) != reference[key] for row in records[1:]) for key in fields):
        raise ValueError('preflight identity mismatch')
    center = reference['center']
    bands = np.asarray(center['bands'], dtype=np.float64)
    tolerance = float(reference['device_parity_tolerance'])
    if bands.ndim != 2 or not np.isfinite(bands).all() or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid reference bands or tolerance')
    maximum = []
    for row in records:
        other = row['center']
        if (row.get('precision_contract') != {'precision': 'fp32', 'amp_enabled': False, 'tf32_enabled': False}
                or row.get('runtime', {}).get('matmul_tf32_enabled') is not False
                or row.get('runtime', {}).get('cudnn_tf32_enabled') is not False):
            raise ValueError('preflight precision contract mismatch')
        if (other['status'] != 'surrogate_only' or not row['diagnostics'].get('fd_passed')
                or any(other[key] != center[key] for key in ('parameter_hash', 'master_grid_sha256', 'input_sha256'))):
            raise ValueError('preflight geometry or gradient gate mismatch')
        current = np.asarray(other['bands'], dtype=np.float64)
        if current.shape != bands.shape or not np.isfinite(current).all():
            raise ValueError('invalid preflight bands')
        difference = float(np.max(np.abs(current - bands)))
        if difference > tolerance:
            raise ValueError('preflight spectrum parity failed')
        maximum.append(difference)
    return {'status': 'passed', 'max_absolute_band_differences': maximum,
            'tolerance': tolerance, 'config_sha256': reference['config_sha256'],
            'checkpoint_sha256': reference['checkpoint_sha256'], 'code_commit': reference['code_commit']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare_preflights([json.loads(Path(path).read_bytes()) for path in args.input])
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
