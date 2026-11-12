"""Bind finite-grid robust solver evidence to a single held-out design.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from phc2d.inverse.feedback_handoff import validate_robust_observation


@dataclass(frozen=True)
class RobustSolverArtifact:
    """Immutable inputs and one result for the existing robust solver validator."""

    manifest_bytes: bytes
    manifest_sha256: str
    protocol_bytes: bytes
    protocol_sha256: str
    result_bytes: bytes
    result_sha256: str
    case_index: int


def _checked_artifact(artifact: RobustSolverArtifact, *, expected_solver_identity: Mapping[str, Any],
                      expected_frequency_normalization: str) -> tuple[dict, dict, dict, dict]:
    if not isinstance(artifact, RobustSolverArtifact):
        raise ValueError('robust solver artifact is required')
    if hashlib.sha256(artifact.result_bytes).hexdigest() != artifact.result_sha256:
        raise ValueError('solver result digest mismatch')
    manifest, protocol = json.loads(artifact.manifest_bytes), json.loads(artifact.protocol_bytes)
    result = json.loads(artifact.result_bytes)
    observation = validate_robust_observation(manifest_bytes=artifact.manifest_bytes,
        expected_manifest_sha256=artifact.manifest_sha256, protocol_bytes=artifact.protocol_bytes,
        protocol_sha256=artifact.protocol_sha256, case_index=artifact.case_index, result=result)
    if observation['provenance']['solver_identity'] != dict(expected_solver_identity):
        raise ValueError('unexpected solver identity')
    for trace in result['observations']:
        for stage in ('initial_solver', 'verification_solver'):
            if trace['provenance'][stage].get('frequency_normalization') != expected_frequency_normalization:
                raise ValueError('frequency normalization differs')
    return manifest, protocol, result, observation


def bind_joint_robust_evidence(nominal: RobustSolverArtifact, unseen: RobustSolverArtifact, *,
                               expected_solver_identity: Mapping[str, Any],
                               expected_frequency_normalization: str,
                               optimization_protocol_bytes: bytes | None = None,
                               optimization_protocol_sha256: str | None = None) -> dict[str, Any]:
    """Validate two records, keeping held-out offsets and grids disjoint.

    The zero-offset anchor can occur in both protocols; every nonzero held-out
    offset and raster must be absent from the supplied nominal artifact. The
    An optional independently identified optimization protocol can establish
    that its offsets are all included in the nominal artifact. The caller
    remains responsible for proving that this is the actual frozen protocol.
    This function checks finite numerical results, not continuous BZ or a real
    manufacturing distribution. The caller must establish the provenance of
    solver records and whether this evaluation set was frozen before design.
    Mock solver records support tests but are not physical acceptance.
    """
    if (not isinstance(expected_solver_identity, Mapping) or not expected_solver_identity
            or not isinstance(expected_frequency_normalization, str) or not expected_frequency_normalization.strip()):
        raise ValueError('expected solver and normalization identities are required')
    nominal_manifest, nominal_protocol, nominal_result, nominal_observation = _checked_artifact(
        nominal, expected_solver_identity=expected_solver_identity,
        expected_frequency_normalization=expected_frequency_normalization)
    unseen_manifest, unseen_protocol, unseen_result, unseen_observation = _checked_artifact(
        unseen, expected_solver_identity=expected_solver_identity,
        expected_frequency_normalization=expected_frequency_normalization)
    design = nominal_observation['provenance']['parameter_hash']
    if design != unseen_observation['provenance']['parameter_hash']:
        raise ValueError('solver artifacts refer to different base designs')
    if (nominal_manifest['lower_band'] != unseen_manifest['lower_band']
            or nominal_manifest['target_frequency'] != unseen_manifest['target_frequency']
            or nominal_protocol['raster_resolution'] != unseen_protocol['raster_resolution']):
        raise ValueError('band, target or base raster contract differs')
    for key in ('polarization', 'epsilon_air', 'epsilon_material'):
        if nominal_manifest['solver_config'][key] != unseen_manifest['solver_config'][key]:
            raise ValueError(f'physical solver {key} differs')
    seen_offsets = [float(x) for x in nominal_protocol['offsets']]
    coverage_verified = False
    if (optimization_protocol_bytes is None) != (optimization_protocol_sha256 is None):
        raise ValueError('optimization protocol bytes and digest must be supplied together')
    if optimization_protocol_bytes is not None:
        if (not isinstance(optimization_protocol_bytes, bytes)
                or not isinstance(optimization_protocol_sha256, str)
                or hashlib.sha256(optimization_protocol_bytes).hexdigest() != optimization_protocol_sha256):
            raise ValueError('optimization protocol digest mismatch')
        optimization = json.loads(optimization_protocol_bytes)
        offsets = optimization.get('offsets') if isinstance(optimization, dict) else None
        if (not isinstance(offsets, list) or not offsets
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                       for x in offsets)):
            raise ValueError('optimization protocol offsets are invalid')
        if any(not any(math.isclose(float(x), y, rel_tol=0, abs_tol=1e-12) for y in seen_offsets)
               for x in offsets):
            raise ValueError('nominal artifact does not cover optimization offsets')
        coverage_verified = True
    unseen_offsets = [float(x) for x in unseen_protocol['offsets'] if float(x) != 0.0]
    if not unseen_offsets or any(math.isclose(x, y, rel_tol=0, abs_tol=1e-12)
                                 for x in unseen_offsets for y in seen_offsets):
        raise ValueError('held-out offsets overlap optimization offsets or are empty')
    seen_hashes = {trace['provenance']['evaluated_grid_sha256'] for trace in nominal_result['observations']}
    heldout_hashes = [trace['provenance']['evaluated_grid_sha256'] for trace in unseen_result['observations']
                      if float(trace['offset']) != 0.0]
    if len(set(heldout_hashes)) != len(heldout_hashes) or seen_hashes.intersection(heldout_hashes):
        raise ValueError('held-out grids duplicate optimization or held-out grids')

    def evidence(observation: Mapping[str, Any], artifact: RobustSolverArtifact) -> dict[str, Any]:
        row = observation['row']
        keys = ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged')
        if not all(row.get(key) is True for key in keys):
            return {'status': 'unresolved', 'reason': 'numerical_gate_failed',
                    'artifact_sha256': artifact.result_sha256, 'design_sha256': design}
        return {'status': 'verified', 'numerical_gate_passed': True, 'artifact_sha256': artifact.result_sha256,
                'design_sha256': design, 'lower_edge': row['refined_lower_edge_frequency'],
                'upper_edge': row['refined_upper_edge_frequency']}

    return {'scope': 'validated_finite_solver_records_not_formal_acceptance',
            'target_frequency': nominal_manifest['target_frequency'], 'lower_band': nominal_manifest['lower_band'],
            'nominal': evidence(nominal_observation, nominal), 'unseen': evidence(unseen_observation, unseen),
            'solver_identity': dict(expected_solver_identity),
            'frequency_normalization': expected_frequency_normalization,
            'optimization_coverage_verified_against_supplied_protocol': coverage_verified,
            'optimization_protocol_sha256': optimization_protocol_sha256}
