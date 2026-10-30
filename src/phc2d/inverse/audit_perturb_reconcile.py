"""Verify complete periodic-offset refinement audit results.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
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
