"""Compare locked surrogate bands with pixel-geometry MPB replay results.

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


def compare_physical_results(lock: Mapping[str, Any], manifest: Mapping[str, Any],
                             results: Mapping[int, Mapping[str, Any]], *,
                             lock_sha256: str, manifest_sha256: str,
                             protocol_sha256: str) -> dict[str, Any]:
    """Compare same-k bands and refined edges, retaining missing and failed roles."""
    if (manifest['selection_lock_sha256'] != lock_sha256 or manifest['protocol_sha256'] != protocol_sha256
            or len(lock['records']) != len(manifest['role_coverage'])):
        raise ValueError('lock, manifest or protocol identity mismatch')
    points = np.asarray(manifest['k_points'], dtype=np.float64)
    pair = int(manifest['lower_band'])
    target = float(manifest['target_frequency'])
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or not np.isfinite(target) or target <= 0:
        raise ValueError('invalid physical comparison grid or target')
    predicted = {}
    for record in lock['records']:
        candidate = record['candidate']
        if candidate is not None and 'bands' in candidate:
            predicted[candidate['master_grid_sha256']] = candidate['bands']
    rows = []
    for record, coverage in zip(lock['records'], manifest['role_coverage']):
        if record['label'] != coverage['label']:
            raise ValueError('role order mismatch')
        entry = {'label': record['label'], 'case_index': coverage['case_index']}
        index = coverage['case_index']
        if index is None:
            rows.append({**entry, 'status': coverage['status']})
            continue
        if index not in results:
            rows.append({**entry, 'status': 'physical_result_missing'})
            continue
        result = results[index]
        physical = manifest['rows'][index]
        if (result['case_index'] != index or result['manifest_sha256'] != manifest_sha256
                or result['protocol_sha256'] != protocol_sha256
                or result['parameter_hash'] != physical['parameter_hash']):
            raise ValueError('physical result identity mismatch')
        traces = result['observations']
        if len(traces) != 1 or traces[0]['offset'] != 0.0:
            raise ValueError('nominal zero-offset physical result required')
        trace = traces[0]
        if trace['provenance']['evaluated_grid_sha256'] != physical['master_grid_sha256']:
            raise ValueError('physical master grid mismatch')
        arrays = trace['arrays']
        solver_points = np.asarray(arrays['k_points'], dtype=np.float64)
        all_bands = np.asarray(arrays['bands'], dtype=np.float64)
        estimate = np.asarray(predicted.get(physical['master_grid_sha256']), dtype=np.float64)
        n = len(points)
        if (solver_points.shape[0] < n or not np.allclose(solver_points[:n], points, atol=1e-7, rtol=0)
                or all_bands.ndim != 2 or all_bands.shape[0] < n or estimate.shape != all_bands[:n].shape
                or not np.isfinite(estimate).all() or not np.isfinite(all_bands[:n]).all()
                or not 1 <= pair < estimate.shape[1]):
            raise ValueError('same-k spectrum missing, invalid or mismatched')
        physical_bands = all_bands[:n]
        delta = estimate - physical_bands
        pred_edges = [float(np.max(estimate[:, pair-1])), float(np.min(estimate[:, pair]))]
        physical_edges = [float(np.max(physical_bands[:, pair-1])), float(np.min(physical_bands[:, pair]))]
        row = trace['row']
        refined = [float(row['refined_lower_edge_frequency']), float(row['refined_upper_edge_frequency'])]
        rows.append({**entry, 'status': result['aggregate']['status'],
            'same_k_spectrum_mae': float(np.mean(np.abs(delta))),
            'same_k_spectrum_max_abs': float(np.max(np.abs(delta))),
            'surrogate_edges': pred_edges, 'physical_same_k_edges': physical_edges,
            'physical_refined_edges': refined,
            'physical_refined_target_margin': float(2*min(target-refined[0], refined[1]-target)/target),
            'elapsed_s': result['elapsed_s']})
    return {'scope': 'locked_surrogate_vs_physical', 'roles': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lock', 'manifest', 'protocol', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--result', action='append', default=[])
    args = parser.parse_args()
    raw_lock, raw_manifest, raw_protocol = (Path(getattr(args, key)).read_bytes()
                                             for key in ('lock', 'manifest', 'protocol'))
    results = {}
    for filename in args.result:
        result = json.loads(Path(filename).read_bytes())
        index = result['case_index']
        if index in results:
            raise ValueError('duplicate physical case result')
        results[index] = result
    report = compare_physical_results(json.loads(raw_lock), json.loads(raw_manifest), results,
        lock_sha256=hashlib.sha256(raw_lock).hexdigest(),
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        protocol_sha256=hashlib.sha256(raw_protocol).hexdigest())
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
