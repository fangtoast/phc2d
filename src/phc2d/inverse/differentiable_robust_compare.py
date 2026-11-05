"""Compare locked hard-scenario surrogate spectra with refined MPB results.

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

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids


def compare_robust_results(lock: Mapping[str, Any], manifest: Mapping[str, Any],
                           protocol: Mapping[str, Any], results: Mapping[int, Mapping[str, Any]], *,
                           lock_sha256: str, manifest_sha256: str, protocol_sha256: str,
                           failures: Mapping[int, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Report per-offset and common margins without changing the locked selection."""
    if (lock.get('purpose') != 'robust_design' or manifest['selection_lock_sha256'] != lock_sha256
            or manifest['protocol_sha256'] != protocol_sha256
            or len(lock['records']) != len(manifest['role_coverage'])):
        raise ValueError('robust lock, physical manifest or protocol mismatch')
    offsets = [float(value) for value in protocol['offsets']]
    points = np.asarray(manifest['k_points'], dtype=float)
    pair, target = int(manifest['lower_band']), float(manifest['target_frequency'])
    if (len(offsets) != 3 or len(set(offsets)) != 3 or 0.0 not in offsets
            or not np.isfinite(offsets).all() or points.ndim != 2 or points.shape[1] != 2
            or not np.isfinite(points).all() or not np.isfinite(target) or target <= 0 or pair < 1
            or not any(value < 0 for value in offsets) or not any(value > 0 for value in offsets)):
        raise ValueError('invalid robust comparison protocol')
    failures = {} if failures is None else failures
    if set(results).intersection(failures):
        raise ValueError('physical case is both successful and failed')
    rows = []
    for record, coverage in zip(lock['records'], manifest['role_coverage']):
        if record['label'] != coverage['label']:
            raise ValueError('robust role identity mismatch')
        index = coverage['case_index']
        entry = {'label': record['label'], 'case_index': index}
        if index is None:
            rows.append({**entry, 'status': coverage['status']})
            continue
        if index in failures:
            failure = failures[index]
            if (failure['case_index'] != index or failure['manifest_sha256'] != manifest_sha256
                    or failure['protocol_sha256'] != protocol_sha256):
                raise ValueError('robust physical failure identity mismatch')
            rows.append({**entry, 'status': 'physical_solver_failed', 'error_type': failure['error_type']})
            continue
        if index not in results:
            rows.append({**entry, 'status': 'physical_result_missing'})
            continue
        result, physical, candidate = results[index], manifest['rows'][index], record['candidate']
        if (result['case_index'] != index or result['manifest_sha256'] != manifest_sha256
                or result['protocol_sha256'] != protocol_sha256 or result['parameter_hash'] != physical['parameter_hash']
                or candidate is None or candidate['master_grid_sha256'] != physical['master_grid_sha256']):
            raise ValueError('robust physical result identity mismatch')
        structure = FullCellStructureSpec.from_mapping(candidate['structure'])
        grids = periodic_perturbation_grids(structure, resolution=protocol['raster_resolution'], offsets=tuple(offsets))
        hashes = [hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest() for grid in grids]
        traces = result['observations']
        if len(traces) != len(offsets) or [float(trace['offset']) for trace in traces] != offsets:
            raise ValueError('robust physical scenario order mismatch')
        prediction = candidate.get('robust_prediction')
        predicted = None
        if prediction is not None and prediction.get('status') == 'surrogate_only':
            if prediction['grid_sha256'] != hashes:
                raise ValueError('robust surrogate grid identity mismatch')
            predicted = np.asarray(prediction['bands'], dtype=np.float64)
            if predicted.ndim != 3 or predicted.shape[:2] != (len(offsets), len(points)) or not np.isfinite(predicted).all():
                raise ValueError('robust surrogate spectrum mismatch')
        nominal_prediction = None
        if predicted is None and candidate.get('bands') is not None:
            nominal_prediction = np.asarray(candidate['bands'], dtype=np.float64)
            if (nominal_prediction.ndim != 2 or nominal_prediction.shape[0] != len(points)
                    or nominal_prediction.shape[1] <= pair or not np.isfinite(nominal_prediction).all()):
                raise ValueError('nominal surrogate spectrum mismatch')
        scenario_rows, refined = [], []
        stable = result['aggregate']['status'] == 'numerically_stable'
        gates = ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged')
        for i, trace in enumerate(traces):
            if trace['provenance']['evaluated_grid_sha256'] != hashes[i]:
                raise ValueError('robust physical scenario grid mismatch')
            point_array = np.asarray(trace['arrays']['k_points'], dtype=np.float64)
            bands = np.asarray(trace['arrays']['bands'], dtype=np.float64)
            if (point_array.ndim != 2 or point_array.shape[0] < len(points)
                    or not np.allclose(point_array[:len(points)], points, rtol=0, atol=1e-7)
                    or bands.ndim != 2 or bands.shape[0] < len(points) or bands.shape[1] <= pair
                    or not np.isfinite(bands[:len(points)]).all()):
                raise ValueError('robust physical spectrum or k-points differ')
            row = trace['row']
            edges = [float(row['refined_lower_edge_frequency']), float(row['refined_upper_edge_frequency'])]
            if not np.isfinite(edges).all() or (stable and not all(row.get(key) is True for key in gates)):
                raise ValueError('robust refined edges or numerical status invalid')
            refined.append(edges)
            item = {'offset': offsets[i], 'physical_refined_edges': edges,
                    'physical_refined_target_margin': float(2 * min(target-edges[0], edges[1]-target) / target)}
            estimated = predicted[i] if predicted is not None else nominal_prediction if offsets[i] == 0.0 else None
            if estimated is not None:
                if estimated.shape != bands[:len(points)].shape:
                    raise ValueError('robust spectrum shape mismatch')
                delta = estimated - bands[:len(points)]
                item['same_k_spectrum_mae'] = float(np.abs(delta).mean())
                item['same_k_spectrum_rmse'] = float(np.sqrt(np.square(delta).mean()))
                item['same_k_spectrum_mre_pct'] = float(100*np.mean(np.abs(delta)/np.maximum(np.abs(bands[:len(points)]), 1e-8)))
                item['surrogate_target_margin'] = float(2 * min(target-estimated[:, pair-1].max(),
                                                                estimated[:, pair].min()-target) / target)
            scenario_rows.append(item)
        lower, upper = max(edge[0] for edge in refined), min(edge[1] for edge in refined)
        common = float(2 * min(target-lower, upper-target) / target)
        aggregate = result['aggregate']['row']
        if (not np.allclose([lower, upper], [aggregate['refined_lower_edge_frequency'],
                                            aggregate['refined_upper_edge_frequency']], rtol=0, atol=1e-10)
                or stable and not all(aggregate.get(key) is True for key in gates)):
            raise ValueError('robust aggregate does not match per-scenario results')
        row = {**entry, 'status': result['aggregate']['status'], 'scenarios': scenario_rows,
               'physical_common_edges': [lower, upper], 'physical_common_target_margin': common,
               'physical_contains_target': (common > 0) if stable else None,
               'physical_nominal_target_margin': scenario_rows[offsets.index(0.0)]['physical_refined_target_margin'],
               'elapsed_s': result['elapsed_s']}
        if predicted is not None:
            row['surrogate_common_target_margin'] = min(item['surrogate_target_margin'] for item in scenario_rows)
        rows.append(row)
    cases = {row['case_index'] for row in rows if row['case_index'] is not None}
    return {'scope': 'locked_robust_surrogate_vs_physical', 'roles': rows,
            'unique_cases_registered': len(cases), 'unique_cases_with_results': len(cases.intersection(results)),
            'unique_cases_failed': len(cases.intersection(failures)),
            'unique_cases_missing': len(cases - set(results) - set(failures))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lock', 'manifest', 'protocol', 'result-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    lock_raw, manifest_raw, protocol_raw = (Path(getattr(args, key)).read_bytes()
                                            for key in ('lock', 'manifest', 'protocol'))
    results, failures = {}, {}
    for filename, output in (('results.json', results), ('failure.json', failures)):
        for path in Path(args.result_directory).glob(f'case_*/{filename}'):
            item = json.loads(path.read_bytes())
            if item['case_index'] in output:
                raise ValueError('duplicate robust physical case')
            output[item['case_index']] = item
    report = compare_robust_results(json.loads(lock_raw), json.loads(manifest_raw), json.loads(protocol_raw), results,
        lock_sha256=hashlib.sha256(lock_raw).hexdigest(), manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        protocol_sha256=hashlib.sha256(protocol_raw).hexdigest(), failures=failures)
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
