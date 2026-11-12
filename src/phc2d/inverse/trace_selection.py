"""Freeze paired surrogate and same-trace random candidate selections.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.surrogate_design import audit_surrogate_design


def lock_static_nominal_candidates(
    source: FullCellStructureSpec, traces: list[list[dict[str, Any]]], *,
    raster_resolution: int, candidates: int, evaluations_per_trace: int,
) -> dict[str, Any]:
    """Rank distinct nominal grids across independent static-search traces.

    Call only after independently authenticating the model, source, seeds,
    search bounds, and original traces. This function performs no physics and
    does not establish that the traces were generated without feedback.
    """
    if (source.material_kind != 'binary' or not traces or not isinstance(raster_resolution, int) or isinstance(raster_resolution, bool)
            or raster_resolution <= 0 or not isinstance(candidates, int) or isinstance(candidates, bool)
            or candidates <= 0 or not isinstance(evaluations_per_trace, int)
            or isinstance(evaluations_per_trace, bool) or evaluations_per_trace <= 0):
        raise ValueError('invalid candidate, trace or raster budget')
    def grid_hash(structure: FullCellStructureSpec) -> str:
        grid = np.ascontiguousarray(structure.rasterize((raster_resolution, raster_resolution)), dtype=np.uint8)
        return hashlib.sha256(grid.tobytes()).hexdigest()

    anchor_grid = grid_hash(source)
    seen_grids = {anchor_grid}
    eligible = []
    duplicates = []
    for trace_index, trace in enumerate(traces):
        if not isinstance(trace, list) or len(trace) > evaluations_per_trace:
            raise ValueError('trace exceeds evaluation budget')
        for index, row in enumerate(trace):
            if row.get('candidate_index') != index:
                raise ValueError('trace indices must be consecutive')
            if row.get('status') != 'surrogate_only':
                continue
            if row.get('perturbation_offsets') not in (None, [0.0], (0.0,)):
                raise ValueError('eligible trace is not nominal')
            score = row.get('selection_score')
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError('eligible selection score must be finite')
            structure = FullCellStructureSpec.from_mapping(row['structure'])
            if row.get('parameter_hash') != structure.parameter_hash:
                raise ValueError('candidate parameter identity mismatch')
            digest = grid_hash(structure)
            identity = {'trace_index': trace_index, 'candidate_index': index}
            if digest in seen_grids:
                duplicates.append(identity)
                continue
            seen_grids.add(digest)
            eligible.append({**identity, 'parameter_hash': structure.parameter_hash,
                'nominal_grid_sha256': digest, 'selection_score': float(score),
                'structure': structure.to_mapping()})
    eligible.sort(key=lambda row: (-row['selection_score'], row['trace_index'], row['candidate_index']))
    return {'status': 'locked_surrogate_only' if len(eligible) >= candidates else 'unscoreable_insufficient_eligible',
            'physical_validation': 'not_run', 'source_parameter_hash': source.parameter_hash,
            'source_grid_sha256': anchor_grid, 'eligible_count': len(eligible),
            'excluded_duplicates': duplicates, 'selected': eligible[:candidates] if len(eligible) >= candidates else []}


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
