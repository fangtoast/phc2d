"""Extend frozen inverse-development exclusion identities from audited manifests.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Mapping, Sequence

import numpy as np


def _addition_identity(row: Mapping[str, Any], *, low_resolution: int,
                       audit_resolution: int, physical_resolution: int) -> dict[str, str]:
    from phc2d.data.full_cell_hashing import hash_raster_content
    from phc2d.data.full_cell_raster import block_mean_downsample
    from phc2d.data.full_cell_spec import FullCellStructureSpec

    structure = FullCellStructureSpec.from_mapping(row['structure'])
    if structure.material_kind != 'binary' or structure.parameter_hash != row['parameter_hash']:
        raise ValueError('addition structure identity or material mismatch')
    audit = np.ascontiguousarray(structure.rasterize((audit_resolution, audit_resolution)), dtype=np.uint8)
    physical = np.ascontiguousarray(structure.rasterize((physical_resolution, physical_resolution)), dtype=np.uint8)
    if not np.isin(audit, (0, 1)).all() or not np.isin(physical, (0, 1)).all():
        raise ValueError('addition raster is not binary')
    low = block_mean_downsample(audit, low_resolution)
    result = {'parameter_hash': structure.parameter_hash,
              f'content_hash_r{low_resolution}': hash_raster_content(low, material_kind='area_fraction'),
              f'content_hash_r{audit_resolution}': hash_raster_content(audit, material_kind='binary'),
              'physical_grid_sha256': hashlib.sha256(physical.tobytes()).hexdigest()}
    for key in ('parameter_hash', f'content_hash_r{low_resolution}', f'content_hash_r{audit_resolution}'):
        if key in row and row[key] != result[key]:
            raise ValueError(f'addition recorded {key} mismatch')
    if row.get('master_grid_resolution') == physical_resolution and row.get('master_grid_sha256') != result['physical_grid_sha256']:
        raise ValueError('addition physical raster identity mismatch')
    return result


def extend_source_exclusions(
    previous_bytes: bytes, *, previous_sha256: str, config_bytes: bytes,
    additions: Sequence[Mapping[str, Any]], low_resolution: int,
    audit_resolution: int, physical_resolution: int, workers: int,
    cpu_limit: int, memory_budget_mib: int, worker_memory_mib: int, setup_reserve_mib: int,
) -> dict[str, Any]:
    """Reconstruct added structures and preserve prior exclusions and lineage.

    Each addition contains a stable name, exact bytes and expected SHA-256 for
    a JSON manifest with ``rows`` holding structures. The caller must inventory
    and authenticate all required datasets and past search traces; this helper
    cannot establish that the provided inventory is exhaustive.
    """
    if hashlib.sha256(previous_bytes).hexdigest() != previous_sha256:
        raise ValueError('prior exclusions hash mismatch')
    nums = (low_resolution, audit_resolution, physical_resolution, workers, cpu_limit,
            memory_budget_mib, worker_memory_mib, setup_reserve_mib)
    if (any(isinstance(v, bool) or not isinstance(v, int) for v in nums)
            or low_resolution <= 0 or audit_resolution <= low_resolution or audit_resolution % low_resolution
            or physical_resolution < audit_resolution or workers < 1 or workers > cpu_limit
            or min(memory_budget_mib, worker_memory_mib) < 1 or setup_reserve_mib < 0
            or workers*worker_memory_mib+setup_reserve_mib > memory_budget_mib):
        raise ValueError('invalid resolution or resource budget')
    if 'SLURM_CPUS_PER_TASK' in os.environ and cpu_limit > int(os.environ['SLURM_CPUS_PER_TASK']):
        raise ValueError('CPU limit exceeds scheduler allocation')
    if 'SLURM_MEM_PER_NODE' in os.environ and memory_budget_mib > int(os.environ['SLURM_MEM_PER_NODE']):
        raise ValueError('memory budget exceeds scheduler allocation')
    previous = json.loads(previous_bytes)
    keys = ('parameter_hash', f'content_hash_r{low_resolution}', f'content_hash_r{audit_resolution}')
    identity_sets: dict[str, set[str]] = {}
    for key in keys:
        values = previous['identity_sets'].get(key)
        if not isinstance(values, list) or len(set(values)) != len(values) or not all(isinstance(v, str) and v for v in values):
            raise ValueError('invalid previous identity set')
        identity_sets[key] = set(values)
    old_physical = previous.get('prior_candidate_master_grid_sha256')
    if (not isinstance(old_physical, list) or len(set(old_physical)) != len(old_physical)
            or not all(isinstance(v, str) and v for v in old_physical)):
        raise ValueError('invalid previous physical grid set')
    physical = set(old_physical)
    inventory, rows = [], []
    names = set()
    for item in additions:
        name, raw, expected = item['name'], item['bytes'], item['sha256']
        if not isinstance(name, str) or not name or name in names or not isinstance(raw, bytes):
            raise ValueError('invalid addition inventory')
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('addition hash mismatch')
        names.add(name)
        entries = json.loads(raw).get('rows')
        if not isinstance(entries, list) or not entries:
            raise ValueError('empty or malformed addition manifest')
        inventory.append({'name': name, 'sha256': expected, 'rows': len(entries)})
        rows.extend(entries)
    if not rows:
        raise ValueError('at least one addition required')
    work = partial(_addition_identity, low_resolution=low_resolution,
                   audit_resolution=audit_resolution, physical_resolution=physical_resolution)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for identity in executor.map(work, rows, chunksize=1):
            for key in keys:
                identity_sets[key].add(identity[key])
            physical.add(identity['physical_grid_sha256'])
    return {'scope': 'extended_source_exclusions_not_exhaustiveness_certification',
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'previous_exclusions_sha256': previous_sha256, 'additions': inventory,
            'identity_sets': {key: sorted(value) for key, value in identity_sets.items()},
            'prior_candidate_master_grid_sha256': sorted(physical)}
