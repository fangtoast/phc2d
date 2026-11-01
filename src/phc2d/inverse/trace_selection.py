"""Freeze paired surrogate and same-trace random candidate selections.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.surrogate_design import audit_surrogate_design


def lock_paired_trace_selection(
    output_root: Path, *, config_bytes: bytes, source_bytes: bytes,
    checkpoint_sha256: str, source_index: int, model_seed: int,
    search_seed: int, random_choice_seed: int,
) -> dict[str, Any]:
    """Select audited nonanchor candidates without consulting physical results.

    Candidate evaluations with the same nominal physical raster count once;
    the earliest evaluation represents that raster. Callers must separately
    establish actual execution, frozen source/checkpoint identity, and
    cross-trace physical deduplication before using this lock.
    """
    for value in (source_index, model_seed, search_seed, random_choice_seed):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('source and seeds must be nonnegative integers')
    config = json.loads(config_bytes)
    options = config['search_options']
    offsets = options.get('perturbation_offsets')
    if (options.get('projection_contract_sha256') is None or not isinstance(offsets, (list, tuple))
            or offsets.count(0.0) != 1):
        raise ValueError('projected perturbation contract with nominal offset required')
    audit = audit_surrogate_design(Path(output_root), config_bytes=config_bytes,
        source_bytes=source_bytes, checkpoint_sha256=checkpoint_sha256)
    if search_seed not in config['seeds']:
        raise ValueError('search seed not present in audited run')
    source = FullCellStructureSpec.from_mapping(json.loads(source_bytes))
    anchor_grid = np.ascontiguousarray(source.rasterize(
        (options['source_resolution'], options['source_resolution'])), dtype=np.uint8)
    anchor_hash = hashlib.sha256(anchor_grid.tobytes()).hexdigest()
    trace_raw = (Path(output_root)/f'seed_{search_seed}'/'search.json').read_bytes()
    trace = json.loads(trace_raw)
    summary_raw = (Path(output_root)/'summary.json').read_bytes()
    summary = json.loads(summary_raw)
    zero_index = list(offsets).index(0.0)
    eligible = []
    duplicate_indices = []
    seen_grids = {anchor_hash}
    for row in trace['evaluations']:
        if row['status'] != 'projected_surrogate_only':
            continue
        nominal_hash = row['perturbation_grid_sha256'][zero_index]
        if nominal_hash in seen_grids:
            duplicate_indices.append(row['candidate_index'])
            continue
        seen_grids.add(nominal_hash)
        eligible.append(row)
    # The audited trace has consecutive indices; explicit sorting makes the
    # random mapping and tie break independent of incidental container order.
    eligible.sort(key=lambda row: row['candidate_index'])
    result: dict[str, Any] = {
        'status': 'unscoreable_insufficient_eligible', 'physical_validation': 'not_run',
        'source_index': source_index, 'model_seed': model_seed, 'search_seed': search_seed,
        'random_choice_seed': random_choice_seed, 'source_parameter_hash': source.parameter_hash,
        'source_grid_sha256': anchor_hash, 'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_k_points_sha256': summary['provenance'].get('k_points_sha256'),
        'source_resolution': options['source_resolution'], 'perturbation_offsets': list(offsets),
        'lower_band': options['lower_band'], 'target_frequency': options['target_frequency'],
        'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'trace_sha256': hashlib.sha256(trace_raw).hexdigest(),
        'summary_sha256': hashlib.sha256(summary_raw).hexdigest(),
        'eligible_indices': [row['candidate_index'] for row in eligible],
        'excluded_duplicate_indices': duplicate_indices,
        'baseline_unit': 'unique_nominal_grid_earliest_evaluation', 'audit_status': audit['status'],
        'surrogate_choice': None, 'random_choice': None,
    }
    if len(eligible) < 2:
        return result
    best = max(eligible, key=lambda row: row['selection_score'])
    rng = np.random.default_rng(np.random.SeedSequence([random_choice_seed, source_index, model_seed]))
    baseline = eligible[int(rng.integers(len(eligible)))]

    def selection(row: dict[str, Any]) -> dict[str, Any]:
        return {'candidate_index': row['candidate_index'], 'parameter_hash': row['parameter_hash'],
                'nominal_grid_sha256': row['perturbation_grid_sha256'][zero_index],
                'selection_score': row['selection_score'], 'structure': row['structure']}

    result.update(status='locked_surrogate_only', surrogate_choice=selection(best),
                  random_choice=selection(baseline))
    return result
