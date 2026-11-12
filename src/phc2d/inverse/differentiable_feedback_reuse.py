"""Plan physical seed reuse by immutable grid and solver contracts.

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


def plan_seed_reuse(target: Mapping[str, Any], reference: Mapping[str, Any], *,
                    target_sha256: str, reference_sha256: str,
                    allow_target_change: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map equal hard grids to prior cases and create a manifest for new cases."""
    if not isinstance(allow_target_change, bool):
        raise ValueError('target-change reuse must be explicit')
    fields = ('protocol_sha256', 'k_points', 'lower_band', 'solver_config', 'raster_resolution')
    if not allow_target_change:
        fields += ('target_frequency',)
    if any(target.get(key) != reference.get(key) for key in fields):
        raise ValueError('physical reuse contracts differ')
    if allow_target_change:
        import math
        if any(isinstance(item.get('target_frequency'), bool) or not isinstance(item.get('target_frequency'), (int, float))
               or not math.isfinite(item['target_frequency']) or item['target_frequency'] <= 0 for item in (target, reference)):
            raise ValueError('positive finite target frequencies required for reuse')
    prior = {}
    for index, row in enumerate(reference['rows']):
        digest = row['master_grid_sha256']
        if not digest or digest in prior:
            raise ValueError('reference physical grids must be unique')
        prior[digest] = index
    reused, fresh, rows = [], [], []
    seen = set()
    for index, row in enumerate(target['rows']):
        digest = row['master_grid_sha256']
        if not digest or digest in seen:
            raise ValueError('target physical grids must be unique')
        seen.add(digest)
        if digest in prior:
            reused.append({'target_case_index': index, 'reference_case_index': prior[digest], 'master_grid_sha256': digest})
        else:
            fresh.append({'target_case_index': index, 'fresh_case_index': len(rows), 'master_grid_sha256': digest})
            rows.append({**copy.deepcopy(row), 'target_case_index': index})
    fresh_manifest = {**copy.deepcopy(dict(target)), 'scope': 'feedback_seed_fresh_cases',
                      'parent_manifest_sha256': target_sha256, 'rows': rows,
                      'role_coverage': [{'label': row['label'], 'case_index': i, 'status': 'queued'}
                                        for i, row in enumerate(rows)]}
    plan = {'scope': 'feedback_seed_physical_reuse_plan', 'target_manifest_sha256': target_sha256,
            'reference_manifest_sha256': reference_sha256, 'reused': reused, 'fresh': fresh}
    if allow_target_change:
        plan['allow_target_change'] = True
        plan['target_frequencies'] = {'target': target['target_frequency'], 'reference': reference['target_frequency']}
    return plan, fresh_manifest


def rebind_seed_results(target: Mapping[str, Any], reference: Mapping[str, Any],
                        fresh: Mapping[str, Any], plan: Mapping[str, Any], *,
                        target_sha256: str, reference_sha256: str, fresh_sha256: str,
                        reference_results: Mapping[int, tuple[Mapping[str, Any], str]],
                        fresh_results: Mapping[int, tuple[Mapping[str, Any], str]]) -> dict[int, dict[str, Any]]:
    """Create explicitly derived target records without changing any raw solver output."""
    expected, expected_fresh = plan_seed_reuse(target, reference, target_sha256=target_sha256,
        reference_sha256=reference_sha256, allow_target_change=plan.get('allow_target_change', False))
    if (any(plan.get(key) != value for key, value in expected.items())
            or plan.get('fresh_manifest_sha256') != fresh_sha256 or fresh != expected_fresh):
        raise ValueError('reuse plan or fresh manifest differs from frozen mapping')
    output = {}
    for kind, entries, source_manifest, digest, available, key in (
            ('reference', plan['reused'], reference, reference_sha256, reference_results, 'reference_case_index'),
            ('fresh', plan['fresh'], fresh, fresh_sha256, fresh_results, 'fresh_case_index')):
        for item in entries:
            target_index, source_index = item['target_case_index'], item[key]
            if source_index not in available:
                continue
            source, source_result_sha = available[source_index]
            source_row, target_row = source_manifest['rows'][source_index], target['rows'][target_index]
            traces = source['observations']
            if (source.get('case_index') != source_index or source.get('manifest_sha256') != digest
                    or source.get('protocol_sha256') != target['protocol_sha256']
                    or source.get('parameter_hash') != source_row['parameter_hash']
                    or source_row['master_grid_sha256'] != target_row['master_grid_sha256']
                    or item['master_grid_sha256'] != target_row['master_grid_sha256']
                    or not traces or len({trace['offset'] for trace in traces}) != len(traces)
                    or sum(trace['offset'] == 0.0 for trace in traces) != 1
                    or next(trace for trace in traces if trace['offset'] == 0.0)['provenance']['evaluated_grid_sha256'] != item['master_grid_sha256']
                    or source['aggregate']['provenance']['parameter_hash'] != source_row['parameter_hash']
                    or not source_result_sha):
                raise ValueError('reused solver evidence identity mismatch')
            if len(traces) > 1:
                import numpy as np
                from phc2d.data.full_cell_spec import FullCellStructureSpec
                from phc2d.inverse.perturbation_screen import periodic_perturbation_grids
                structure = FullCellStructureSpec.from_mapping(target_row['structure'])
                offsets = tuple(trace['offset'] for trace in traces)
                grids = periodic_perturbation_grids(structure, resolution=target['raster_resolution'], offsets=offsets)
                hashes = [hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest() for grid in grids]
                if hashes != [trace['provenance']['evaluated_grid_sha256'] for trace in traces]:
                    raise ValueError('reused solver scenario grids differ')
            derived = copy.deepcopy(dict(source))
            derived.update(case_index=target_index, source_index=target_row['source_index'], label=target_row['label'],
                           parameter_hash=target_row['parameter_hash'], manifest_sha256=target_sha256,
                           evidence_scope='derived_same_grid_solver_reuse',
                           source_evidence={'kind': kind, 'manifest_sha256': digest, 'case_index': source_index,
                                            'parameter_hash': source_row['parameter_hash'],
                                            'result_sha256': source_result_sha})
            derived['aggregate']['provenance']['parameter_hash'] = target_row['parameter_hash']
            for trace in derived['observations']:
                if 'parameter_hash' in trace['provenance']:
                    trace['provenance']['parameter_hash'] = target_row['parameter_hash']
            output[target_index] = derived
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('target-manifest', 'reference-manifest', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--allow-target-change', action='store_true')
    args = parser.parse_args()
    target_raw, reference_raw = Path(args.target_manifest).read_bytes(), Path(args.reference_manifest).read_bytes()
    plan, fresh = plan_seed_reuse(json.loads(target_raw), json.loads(reference_raw),
        target_sha256=hashlib.sha256(target_raw).hexdigest(), reference_sha256=hashlib.sha256(reference_raw).hexdigest(),
        allow_target_change=args.allow_target_change)
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    fresh_raw = json.dumps(fresh, indent=2, allow_nan=False).encode('utf8')
    plan['fresh_manifest_sha256'] = hashlib.sha256(fresh_raw).hexdigest()
    (output/'fresh_manifest.json').write_bytes(fresh_raw)
    (output/'reuse_plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    main()
