"""Compare frozen robust solver cases across execution environments.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from phc2d.inverse.feedback_handoff import validate_robust_observation


def compare_locked_robust_environments(
    *, batch_bytes: bytes, batch_sha256: str, manifest_bytes: bytes, manifest_sha256: str,
    protocol_bytes: bytes, protocol_sha256: str, reference_results: Mapping[int, bytes],
    replay_results: Mapping[int, bytes], replay_exit_codes: Mapping[int, int],
    expected_case_count: int, expected_solver_identity: Mapping[str, Any], frequency_tolerance: float,
) -> dict[str, Any]:
    """Compare each predeclared offset, k point, band and numerical gate.

    The caller authenticates source deployments, lock timing, exit receipts and
    raw archive identities. This finite-grid comparison is not a P2 or BZ proof.
    """
    if (hashlib.sha256(batch_bytes).hexdigest() != batch_sha256
            or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256
            or hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha256):
        raise ValueError('batch, manifest or protocol digest mismatch')
    if (isinstance(expected_case_count, bool) or not isinstance(expected_case_count, int) or expected_case_count < 1
            or isinstance(frequency_tolerance, bool) or not np.isfinite(frequency_tolerance) or frequency_tolerance < 0):
        raise ValueError('invalid cross-environment contract')
    batch, manifest, protocol = map(json.loads, (batch_bytes, manifest_bytes, protocol_bytes))
    if (batch.get('status') != 'ready_for_physical_evaluation' or batch.get('manifest') != manifest
            or manifest.get('protocol_sha256') != protocol_sha256):
        raise ValueError('frozen batch mismatch')
    roles = batch['cross_environment_roles']
    ids = [role['case_index'] for role in roles]
    if (len(ids) != expected_case_count or len(set(ids)) != expected_case_count
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(manifest['rows']) for i in ids)
            or any(role not in batch['role_map'] for role in roles)
            or set(reference_results) != set(ids) or set(replay_exit_codes) != set(ids)
            or any(isinstance(code, bool) or not isinstance(code, int) for code in replay_exit_codes.values())
            or set(replay_results) != {i for i in ids if replay_exit_codes[i] == 0}):
        raise ValueError('cross-environment case or exit roster mismatch')
    cases = []
    for index in ids:
        left_raw = reference_results[index]
        if not isinstance(left_raw, bytes):
            raise ValueError('reference result must be raw bytes')
        left = json.loads(left_raw)
        validated = validate_robust_observation(manifest_bytes=manifest_bytes, expected_manifest_sha256=manifest_sha256,
            protocol_bytes=protocol_bytes, protocol_sha256=protocol_sha256, case_index=index, result=left)
        if validated['provenance']['solver_identity'] != dict(expected_solver_identity):
            raise ValueError('reference solver identity mismatch')
        record: dict[str, Any] = {'case_index':index, 'reference_sha256':hashlib.sha256(left_raw).hexdigest(),
                                 'replay_exit_code':replay_exit_codes[index]}
        if replay_exit_codes[index] != 0:
            record.update(passed=False, reason='replay_process_failed', replay_sha256=None, offsets=[])
            cases.append(record)
            continue
        right_raw = replay_results[index]
        if not isinstance(right_raw, bytes):
            raise ValueError('replay result must be raw bytes')
        right = json.loads(right_raw)
        replay = validate_robust_observation(manifest_bytes=manifest_bytes, expected_manifest_sha256=manifest_sha256,
            protocol_bytes=protocol_bytes, protocol_sha256=protocol_sha256, case_index=index, result=right)
        if replay['provenance']['solver_identity'] != dict(expected_solver_identity):
            raise ValueError('replay solver identity mismatch')
        comparisons = []
        for offset, a, b in zip(protocol['offsets'],left['observations'],right['observations'],strict=True):
            same_grid = a['provenance']['evaluated_grid_sha256'] == b['provenance']['evaluated_grid_sha256']
            same_k = all(np.array_equal(a['arrays'][key],b['arrays'][key]) for key in ('k_points','verification_k_points'))
            gate_keys = ('last_refinement_stable','resolution_stable','adaptive_refinement_converged')
            same_gates = all(a['row'][key] == b['row'][key] for key in gate_keys)
            difference = None
            if same_k:
                pairs = [(np.asarray(a['arrays'][key]),np.asarray(b['arrays'][key])) for key in ('bands','verification_bands')]
                if all(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all() for x,y in pairs):
                    difference = max(float(np.max(np.abs(x-y))) for x,y in pairs)
            passed = bool(a['offset'] == offset == b['offset'] and same_grid and same_k and same_gates
                          and difference is not None and difference <= frequency_tolerance)
            comparisons.append({'offset':offset, 'grid_match':same_grid, 'k_points_match':same_k,
                                'gates_match':same_gates, 'max_frequency_difference':difference, 'passed':passed})
        record.update(passed=all(x['passed'] for x in comparisons), replay_sha256=hashlib.sha256(right_raw).hexdigest(),
                      offsets=comparisons)
        cases.append(record)
    return {'scope':'finite_grid_cross_environment_not_continuous_bz', 'batch_sha256':batch_sha256,
            'manifest_sha256':manifest_sha256, 'protocol_sha256':protocol_sha256,
            'frequency_absolute_tolerance':frequency_tolerance, 'cases':cases,
            'all_passed':all(case['passed'] for case in cases), 'p2_gate':'not_evaluated'}
