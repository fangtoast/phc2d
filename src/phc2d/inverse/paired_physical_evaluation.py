"""Evaluate prelocked selected and random candidates on solver observations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.inverse.audit_ranking import summarize_seed_averaged_stratified_gains
from phc2d.inverse.feedback_handoff import validate_robust_observation


def evaluate_locked_paired_physics(
    *, batch_bytes: bytes, batch_sha256: str, manifest_bytes: bytes, manifest_sha256: str,
    protocol_bytes: bytes, protocol_sha256: str, results_by_case: Mapping[int, bytes],
    source_strata: Mapping[int, str], seeds: Sequence[int], expected_counts: Mapping[str, int],
    stratum_weights: Mapping[str, float], bootstrap_seed: int, bootstrap_samples: int,
    expected_solver_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare frozen choices after validating every required solver case.

    The caller authenticates the lock's pre-physical timing, source and trace
    identities, all raw artifacts, resource accounting and cross-environment
    evidence. This operation reports descriptive statistics and cannot pass P2.
    """
    if (hashlib.sha256(batch_bytes).hexdigest() != batch_sha256
            or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256
            or hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha256):
        raise ValueError('batch, manifest or protocol digest mismatch')
    batch, manifest = json.loads(batch_bytes), json.loads(manifest_bytes)
    if (batch.get('status') != 'ready_for_physical_evaluation' or batch.get('physical_validation') != 'not_run'
            or batch.get('manifest') != manifest or manifest.get('protocol_sha256') != protocol_sha256):
        raise ValueError('frozen physical batch mismatch')
    cases = manifest['rows']
    if (not cases or set(results_by_case) != set(range(len(cases)))
            or any(isinstance(i, bool) or not isinstance(i, int) for i in results_by_case)):
        raise ValueError('complete solver case roster required')
    roles: dict[tuple[int, int | None, str], Mapping[str, Any]] = {}
    for role in batch['role_map']:
        sid, name = role['source_index'], role['role']
        seed = None if name == 'anchor' else role.get('model_seed')
        key = (sid, seed, name)
        case = role['case_index']
        if (sid not in source_strata or key in roles or isinstance(case, bool) or not isinstance(case, int)
                or not 0 <= case < len(cases) or name not in ('anchor', 'surrogate_choice', 'random_choice')
                or (name != 'anchor' and seed not in seeds)
                or role['nominal_grid_sha256'] != cases[case]['master_grid_sha256']):
            raise ValueError('invalid frozen role mapping')
        roles[key] = role
    expected_roles = {(sid, None, 'anchor') for sid in source_strata}
    expected_roles.update((sid, seed, role) for sid in source_strata for seed in seeds
                          for role in ('surrogate_choice', 'random_choice'))
    if set(roles) != expected_roles:
        raise ValueError('incomplete frozen role roster')
    observed, failed, case_hashes = {}, [], {}
    for case_index in range(len(cases)):
        raw = results_by_case[case_index]
        if not isinstance(raw, bytes):
            raise ValueError('raw solver results must be bytes')
        result = json.loads(raw)
        observation = validate_robust_observation(manifest_bytes=manifest_bytes,
            expected_manifest_sha256=manifest_sha256, protocol_bytes=protocol_bytes,
            protocol_sha256=protocol_sha256, case_index=case_index, result=result)
        if observation['provenance']['solver_identity'] != dict(expected_solver_identity):
            raise ValueError('solver identity mismatch')
        row = observation['row']
        case_hashes[case_index] = hashlib.sha256(raw).hexdigest()
        if (result['aggregate']['status'] != 'numerically_stable'
                or any(row.get(key) is not True for key in
                       ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged'))):
            failed.append(case_index)
            continue
        lower, upper = float(row['refined_lower_edge_frequency']), float(row['refined_upper_edge_frequency'])
        target = float(manifest['target_frequency'])
        if not np.isfinite([lower, upper, target]).all() or target <= 0:
            raise ValueError('invalid physical target or band edges')
        observed[case_index] = 2*min(target-lower, upper-target)/target
    if failed:
        return {'status':'unscoreable_numerical_gate_failures', 'physical_validation':'incomplete',
                'failed_case_indices':failed, 'statistics':None, 'p2_gate':'not_evaluated',
                'batch_sha256':batch_sha256, 'manifest_sha256':manifest_sha256, 'protocol_sha256':protocol_sha256,
                'case_result_sha256':case_hashes}
    groups = []
    for sid in sorted(source_strata):
        relevant = sorted({roles[key]['case_index'] for key in expected_roles if key[0] == sid})
        group_hash = hashlib.sha256(json.dumps([{'case_index':i,'result_sha256':case_hashes[i]}
            for i in relevant], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        anchor = observed[roles[(sid, None, 'anchor')]['case_index']]
        for seed in seeds:
            selected = observed[roles[(sid, seed, 'surrogate_choice')]['case_index']]
            random = observed[roles[(sid, seed, 'random_choice')]['case_index']]
            groups.append({'source_index':sid, 'seed':seed, 'physical_group_sha256':group_hash,
                           'selected_gain':selected-anchor, 'random_expected_gain':random-anchor})
    summary = summarize_seed_averaged_stratified_gains(groups, source_strata=source_strata, seeds=seeds,
        expected_counts=expected_counts, stratum_weights=stratum_weights,
        bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    return {'status':'paired_statistics_available', 'physical_validation':'solver_cases_numerically_checked',
            'failed_case_indices':[], 'statistics':summary, 'groups':groups,
            'batch_sha256':batch_sha256, 'manifest_sha256':manifest_sha256, 'protocol_sha256':protocol_sha256,
            'p2_gate':'not_evaluated', 'case_result_sha256':case_hashes}
