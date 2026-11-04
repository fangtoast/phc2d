"""Bind frozen feedback seed predictions to independently solved MPB observations.

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

import numpy as np

from phc2d.inverse.composite_feedback import composite_edge_observation


def bind_seed_observations(seed_lock: Mapping[str, Any], manifest: Mapping[str, Any],
                           results: Mapping[int, Mapping[str, Any]], *, base: Mapping[str, Any],
                           lock_sha256: str, manifest_sha256: str) -> dict[str, Any]:
    """Authenticate physical results and create fit-compatible per-role observations."""
    if (seed_lock.get('purpose') not in ('feedback_initial_stencil', 'feedback_round')
            or manifest.get('selection_lock_sha256') != lock_sha256
            or len(seed_lock['records']) != len(manifest['role_coverage'])):
        raise ValueError('feedback seed lock and physical manifest mismatch')
    contract = {'checkpoint_sha256': base['checkpoint_sha256'], 'k_points_sha256': base['k_points_sha256'],
                'input_resolution': base['input_resolution'], 'source_resolution': base['source_resolution'],
                'temperature': base['temperature'], 'beta': base['beta'], 'lower_band': base['lower_band'],
                'coefficient_keys': base['coefficient_keys'], 'scales': base['scales']}
    surrogate_sha = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                                allow_nan=False).encode()).hexdigest()
    points = np.asarray(manifest['k_points'], dtype=np.float64)
    rows = []
    for record, coverage in zip(seed_lock['records'], manifest['role_coverage']):
        if record['label'] != coverage['label']:
            raise ValueError('feedback role order mismatch')
        index = coverage['case_index']
        row = {'label': record['label'], 'trial_id': record['trial_id'], 'role': record['role'],
               'point': record.get('point'), 'case_index': index}
        if index is None:
            rows.append({**row, 'status': coverage['status']})
            continue
        if index not in results:
            rows.append({**row, 'status': 'physical_result_missing'})
            continue
        result = results[index]
        expected = manifest['rows'][index]
        candidate = record['candidate']
        if (result['case_index'] != index or result['manifest_sha256'] != manifest_sha256
                or result['protocol_sha256'] != manifest['protocol_sha256']
                or result['parameter_hash'] != expected['parameter_hash']
                or candidate['master_grid_sha256'] != expected['master_grid_sha256']
                or len(result['observations']) != 1):
            raise ValueError('feedback physical result identity mismatch')
        trace = result['observations'][0]
        if (trace['offset'] != 0.0
                or trace['provenance']['evaluated_grid_sha256'] != expected['master_grid_sha256']):
            raise ValueError('feedback physical raster or offset mismatch')
        solver_points = np.asarray(trace['arrays']['k_points'], dtype=float)
        bands = np.asarray(trace['arrays']['bands'], dtype=float)
        if (solver_points.ndim != 2 or len(solver_points) < len(points) or bands.ndim != 2
                or len(bands) < len(points) or not np.allclose(solver_points[:len(points)], points, rtol=0, atol=1e-7)
                or not np.isfinite(bands[:len(points)]).all()):
            raise ValueError('feedback physical spectrum or k-points mismatch')
        physical = copy.deepcopy(result['aggregate'])
        gates = ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged')
        stable = physical.get('status') == 'numerically_stable' and all(physical['row'].get(k) is True for k in gates)
        if not stable:
            rows.append({**row, 'status': 'physical_numerical_gate_failed'})
            continue
        original_hash = physical['provenance']['parameter_hash']
        if original_hash != expected['parameter_hash']:
            raise ValueError('feedback aggregate identity mismatch')
        physical['provenance']['parameter_hash'] = candidate['parameter_hash']
        reused_hash = result.get('source_evidence', {}).get('parameter_hash', original_hash)
        physical['provenance']['reused_from_parameter_hash'] = reused_hash
        edges = record['surrogate_edges']
        physical_edges = [physical['row']['refined_lower_edge_frequency'], physical['row']['refined_upper_edge_frequency']]
        composite = composite_edge_observation(parameter_hash=candidate['parameter_hash'], parameters=record['point'],
            physical_edges=physical_edges, hard_edges=edges['hard_edges'], soft_exact_edges=edges['soft_exact_edges'],
            soft_smooth_edges=edges['soft_smooth_edges'],
            comparison_contract_sha256=physical['provenance']['comparison_contract_sha256'],
            surrogate_contract_sha256=surrogate_sha, numerical_gate_passed=True)
        rows.append({**row, 'status': 'observed', 'physical': physical, 'composite': composite,
                     'physical_reused_from_parameter_hash': reused_hash,
                     'source_evidence': result.get('source_evidence')})
    scope = 'feedback_initial_observations' if seed_lock['purpose'] == 'feedback_initial_stencil' else 'feedback_round_observations'
    return {'scope': scope, 'selection_lock_sha256': lock_sha256,
            'physical_manifest_sha256': manifest_sha256, 'surrogate_contract_sha256': surrogate_sha,
            'design_manifest_sha256': seed_lock['manifest_sha256'],
            'records': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('seed-lock', 'manifest', 'base', 'result-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw_lock, raw_manifest, raw_base = (Path(getattr(args, key)).read_bytes()
                                        for key in ('seed_lock', 'manifest', 'base'))
    results = {}
    for path in Path(args.result_directory).glob('case_*/results.json'):
        result = json.loads(path.read_bytes())
        if result['case_index'] in results:
            raise ValueError('duplicate physical result')
        results[result['case_index']] = result
    output = bind_seed_observations(json.loads(raw_lock), json.loads(raw_manifest), results, base=json.loads(raw_base),
        lock_sha256=hashlib.sha256(raw_lock).hexdigest(), manifest_sha256=hashlib.sha256(raw_manifest).hexdigest())
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
