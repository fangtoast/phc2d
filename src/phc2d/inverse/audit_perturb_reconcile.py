"""Verify complete periodic-offset refinement audit results.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids
from phc2d.inverse.refinement_feedback import aggregate_refined_perturbations


def reconcile_perturbations(manifest_path: str | Path, records: Sequence[Mapping[str, Any]], *,
                            manifest_sha256: str, protocol_path: str | Path,
                            protocol_sha256: str) -> dict[str, Any]:
    """Rebuild offset-grid identities and common gap, retaining all failed gates."""
    manifest, digest = read_checked_json(manifest_path, manifest_sha256)
    protocol, protocol_digest = read_checked_json(protocol_path, protocol_sha256)
    expected = manifest['rows']
    indexed = {r['case_index']: r for r in records}
    if len(records) != len(expected) or len(indexed) != len(expected) or set(indexed) != set(range(len(expected))):
        raise ValueError('incomplete or duplicate perturbation matrix')
    target = float(manifest['target_frequency'])
    if not math.isfinite(target) or target <= 0:
        raise ValueError('invalid target frequency')
    offsets = tuple(protocol['offsets'])
    signatures = set()
    results = []
    for index, source in enumerate(expected):
        item = indexed[index]
        structure = FullCellStructureSpec.from_mapping(source['structure'])
        if (structure.parameter_hash != source['parameter_hash'] or item['parameter_hash'] != source['parameter_hash']
                or item['source_index'] != source['source_index'] or item['label'] != source['label']
                or item['manifest_sha256'] != digest or item['protocol_sha256'] != protocol_digest):
            raise ValueError('candidate or input identity mismatch')
        grids = periodic_perturbation_grids(structure, resolution=protocol['raster_resolution'], offsets=offsets)
        observations = item['observations']
        by_offset = {float(record['offset']): record for record in observations}
        if len(by_offset) != len(observations) or set(by_offset) != set(offsets):
            raise ValueError('offset records differ from protocol')
        for offset, grid in zip(offsets, grids):
            obs = by_offset[offset]
            provenance = obs['provenance']
            grid_hash = hashlib.sha256(np.ascontiguousarray(grid, dtype=np.uint8).tobytes()).hexdigest()
            if (provenance.get('evaluated_grid_sha256') != grid_hash
                    or provenance.get('initial_solver', {}).get('grid_sha256') != grid_hash
                    or provenance.get('verification_solver', {}).get('grid_sha256') != grid_hash):
                raise ValueError('offset grid identity mismatch')
            if obs['row']['lower_band_one_based'] != manifest['lower_band']:
                raise ValueError('band-pair identity mismatch')
        aggregate = aggregate_refined_perturbations(observations, expected_offsets=offsets)
        if aggregate != item['aggregate']:
            raise ValueError('stored and recomputed aggregate differ')
        signatures.add(aggregate['provenance']['comparison_contract_sha256'])
        row = aggregate['row']
        lower, upper = row['refined_lower_edge_frequency'], row['refined_upper_edge_frequency']
        results.append({'case_index': index, 'source_index': source['source_index'], 'label': source['label'],
                        'parameter_hash': structure.parameter_hash, 'lower_edge': lower, 'upper_edge': upper,
                        'target_margin': 2 * min(target - lower, upper - target) / target,
                        'numerical_gate_passed': aggregate['status'] == 'numerically_stable'})
    if len(signatures) != 1:
        raise ValueError('perturbation comparison contracts differ')
    return {'status': 'exploratory_finite_offsets_and_local_refinement', 'manifest_sha256': digest,
            'protocol_sha256': protocol_digest, 'comparison_contract_sha256': next(iter(signatures)),
            'candidate_count': len(results), 'numerical_gate_passed_count': sum(r['numerical_gate_passed'] for r in results),
            'rows': results}


def fingerprint_validated_perturbation_groups(
    *, manifest_bytes: bytes, manifest_sha256: str, protocol_bytes: bytes, protocol_sha256: str,
    record_bytes: Sequence[bytes], expected_solver_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate each complete result and hash its ordered source group.

    The result binds each group fingerprint to the manifest, protocol and
    exact raw result bytes. It retains failed gates and does not authenticate
    scheduler origin or upgrade finite-k results to physical acceptance.
    """
    from phc2d.inverse.feedback_handoff import validate_robust_observation

    if (hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256
            or hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha256):
        raise ValueError('manifest or protocol digest mismatch')
    if not isinstance(expected_solver_identity, Mapping) or not expected_solver_identity:
        raise ValueError('expected solver identity is required')
    manifest = json.loads(manifest_bytes)
    expected = manifest['rows']
    if not isinstance(record_bytes, Sequence) or isinstance(record_bytes, (str, bytes)):
        raise ValueError('raw result sequence is required')
    indexed = {}
    for raw in record_bytes:
        if not isinstance(raw, bytes):
            raise ValueError('each solver result must be raw bytes')
        record = json.loads(raw)
        index = record.get('case_index')
        if isinstance(index, bool) or not isinstance(index, int) or index in indexed:
            raise ValueError('duplicate or invalid case index')
        indexed[index] = (raw, record)
    if not expected or set(indexed) != set(range(len(expected))):
        raise ValueError('incomplete or unexpected physical matrix')
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, entry in enumerate(expected):
        raw, result = indexed[index]
        observation = validate_robust_observation(manifest_bytes=manifest_bytes,
            expected_manifest_sha256=manifest_sha256, protocol_bytes=protocol_bytes,
            protocol_sha256=protocol_sha256, case_index=index, result=result)
        if observation['provenance']['solver_identity'] != dict(expected_solver_identity):
            raise ValueError('unexpected solver identity')
        source = entry['source_index']
        if isinstance(source, bool) or not isinstance(source, int) or source < 0:
            raise ValueError('invalid source identity')
        row = observation['row']
        keys = ('last_refinement_stable', 'resolution_stable', 'adaptive_refinement_converged')
        grouped.setdefault(source, []).append({'case_index': index, 'label': entry['label'],
            'parameter_hash': entry['parameter_hash'], 'result_sha256': hashlib.sha256(raw).hexdigest(),
            'numerical_gate_passed': all(row.get(key) is True for key in keys)})
    groups = []
    for source, members in sorted(grouped.items()):
        identity = {'manifest_sha256': manifest_sha256, 'protocol_sha256': protocol_sha256,
                    'source_index': source, 'members': members}
        raw = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
        groups.append({'source_index': source, 'physical_group_sha256': hashlib.sha256(raw).hexdigest(),
                       'candidate_count': len(members), 'numerical_gate_passed_count':
                           sum(member['numerical_gate_passed'] for member in members),
                       'all_numerical_gates_passed': all(member['numerical_gate_passed'] for member in members),
                       'members': members})
    return {'scope': 'validated_finite_solver_records_not_formal_acceptance', 'groups': groups}
