"""Compare selected WSL solver replays against locked M9 cases.

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

import numpy as np


def compare_crosscheck(manifest: Mapping[str, Any], roles: list[str], *,
                       reference: Mapping[int, Mapping[str, Any]],
                       replay: Mapping[int, Mapping[str, Any]], tolerance: float,
                       manifest_sha256: str) -> dict[str, Any]:
    """Compare same-grid full spectra and report coverage for preregistered roles."""
    if not np.isfinite(tolerance) or tolerance <= 0 or len(set(roles)) != len(roles):
        raise ValueError('invalid tolerance or role roster')
    coverage = {row['label']: row['case_index'] for row in manifest['role_coverage']}
    if any(role not in coverage or coverage[role] is None for role in roles):
        raise ValueError('crosscheck role absent from locked manifest')
    points = np.asarray(manifest['k_points'], dtype=np.float64)
    rows = []
    for role in roles:
        idx = coverage[role]
        row = {'label': role, 'case_index': idx}
        if idx not in reference or idx not in replay:
            rows.append({**row, 'status': 'missing'})
            continue
        a, b = reference[idx], replay[idx]
        physical = manifest['rows'][idx]
        for result in (a, b):
            if (result['case_index'] != idx or result['parameter_hash'] != physical['parameter_hash']
                    or result['manifest_sha256'] != manifest_sha256
                    or result['protocol_sha256'] != manifest['protocol_sha256']
                    or len(result['observations']) != 1
                    or result['observations'][0]['provenance']['evaluated_grid_sha256'] != physical['master_grid_sha256']):
                raise ValueError('crosscheck geometry or case identity mismatch')
        if any(a[key] != b[key] for key in ('manifest_sha256', 'protocol_sha256')):
            raise ValueError('crosscheck manifest or protocol mismatch')
        aa, bb = a['observations'][0]['arrays'], b['observations'][0]['arrays']
        ka, kb = np.asarray(aa['k_points'], dtype=float), np.asarray(bb['k_points'], dtype=float)
        va, vb = np.asarray(aa['bands'], dtype=float), np.asarray(bb['bands'], dtype=float)
        n = len(points)
        if (ka.ndim != 2 or kb.ndim != 2 or len(ka) < n or len(kb) < n
                or not np.allclose(ka[:n], points, atol=1e-7, rtol=0)
                or not np.allclose(kb[:n], points, atol=1e-7, rtol=0)
                or va.ndim != 2 or vb.ndim != 2 or va[:n].shape != vb[:n].shape
                or len(va) < n or len(vb) < n or not np.isfinite(va[:n]).all()
                or not np.isfinite(vb[:n]).all()):
            raise ValueError('crosscheck k grid or spectrum mismatch')
        delta = float(np.max(np.abs(va[:n] - vb[:n])))
        stable = all(x['aggregate']['status'] == 'numerically_stable' for x in (a, b))
        rows.append({**row, 'status': 'passed' if stable and delta <= tolerance else 'failed',
                     'max_absolute_spectrum_difference': delta, 'both_numerically_stable': stable,
                     'm9_elapsed_s': a['elapsed_s'], 'wsl_elapsed_s': b['elapsed_s']})
    return {'scope': 'selected_same_k_cross_environment', 'frequency_absolute_tolerance': tolerance,
            'roles': rows, 'all_passed': all(row['status'] == 'passed' for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'resources', 'reference-root', 'replay-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    manifest_raw = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_raw)
    config = json.loads(Path(args.resources).read_bytes())['wsl_crosscheck']
    def load(root: str) -> dict[int, Any]:
        return {int(path.parent.name[5:]): json.loads(path.read_bytes())
                for path in Path(root).glob('case_*/results.json')}
    result = compare_crosscheck(manifest, config['roles'], reference=load(args.reference_root),
                                replay=load(args.replay_root), tolerance=config['max_absolute_spectrum_difference'],
                                manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
