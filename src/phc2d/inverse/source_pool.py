"""Generate a frozen, collision-checked Fourier development source pool.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


def _source(index: int, *, prefix: str, seed: int, generator_config: dict[str, Any],
            low_resolution: int, master_resolution: int) -> dict[str, Any]:
    from phc2d.data.full_cell_hashing import hash_raster_content
    from phc2d.data.full_cell_raster import block_mean_downsample
    from phc2d.generators.full_cell_latent import generate_unified_fourier_binary

    structure = generate_unified_fourier_binary(f'{prefix}{index:04d}', master_seed=seed,
        material_fraction_bin=0, config=generator_config)
    grid = np.ascontiguousarray(structure.rasterize((master_resolution, master_resolution)), dtype=np.uint8)
    low = block_mean_downsample(grid, low_resolution)
    return {'source_index': index, 'label': 'anchor', 'parameter_hash': structure.parameter_hash,
            f'content_hash_r{low_resolution}': hash_raster_content(low, material_kind='area_fraction'),
            f'content_hash_r{master_resolution}': hash_raster_content(grid, material_kind='binary'),
            'geometry_hash': structure.geometry_hash, 'master_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest(),
            'structure': structure.to_mapping()}


def generate_source_pool(*, config: Path, scope: Path, scope_sha256: str, exclusions: Path,
                         reference_manifest: Path, output: Path, workers: int, cpu_limit: int,
                         memory_budget_mib: int, worker_memory_mib: int, setup_reserve_mib: int) -> dict[str, Any]:
    """Generate deterministic records without replacing collisions or partial output.

    Exclusions and the fixed-k reference must be frozen before invocation. The
    output is only an input for sampled MPB screening, never a physical result.
    """
    import yaml

    nums = (workers, cpu_limit, memory_budget_mib, worker_memory_mib, setup_reserve_mib)
    if (any(isinstance(v, bool) or not isinstance(v, int) for v in nums) or workers < 1
            or workers > cpu_limit or min(memory_budget_mib, worker_memory_mib) < 1
            or setup_reserve_mib < 0 or workers * worker_memory_mib + setup_reserve_mib > memory_budget_mib):
        raise ValueError('invalid worker CPU or memory budget')
    if 'SLURM_CPUS_PER_TASK' in os.environ and cpu_limit > int(os.environ['SLURM_CPUS_PER_TASK']):
        raise ValueError('CPU limit exceeds scheduler allocation')
    if 'SLURM_MEM_PER_NODE' in os.environ and memory_budget_mib > int(os.environ['SLURM_MEM_PER_NODE']):
        raise ValueError('memory budget exceeds scheduler allocation')
    scope_raw, config_raw = scope.read_bytes(), config.read_bytes()
    if hashlib.sha256(scope_raw).hexdigest() != scope_sha256:
        raise ValueError('source generation scope hash mismatch')
    contract, params = json.loads(scope_raw), yaml.safe_load(config_raw)
    if (contract.get('status') != 'frozen_before_generation'
            or contract.get('experiment_id') != params.get('experiment_id')
            or contract.get('config_sha256') != hashlib.sha256(config_raw).hexdigest()):
        raise ValueError('scope and experiment configuration differ')
    exclusions_raw, reference_raw = exclusions.read_bytes(), reference_manifest.read_bytes()
    if (hashlib.sha256(exclusions_raw).hexdigest() != contract.get('exclusions_sha256')
            or hashlib.sha256(reference_raw).hexdigest() != contract.get('reference_manifest_sha256')):
        raise ValueError('frozen exclusion or reference hash mismatch')
    excluded, reference = json.loads(exclusions_raw), json.loads(reference_raw)
    if excluded.get('config_sha256') != contract['config_sha256']:
        raise ValueError('exclusions belong to another configuration')
    pool, geometry = params['source_pool'], params['geometry']
    if geometry.get('family') != 'unified_fourier_binary':
        raise ValueError('unsupported source geometry family')
    count, seed = pool['count'], pool['generator_seed']
    low, master = geometry['model_input_resolution'], geometry['generation_audit_resolution']
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or not isinstance(low, int) or not isinstance(master, int)
            or low <= 0 or master <= low or master % low):
        raise ValueError('invalid source count, seed or raster resolution')
    prefix = contract.get('sample_id_prefix')
    if not isinstance(prefix, str) or not prefix:
        raise ValueError('frozen sample identity prefix required')
    coarse = pool['coarse_solver']
    solver = reference['solver_config']
    for key, value in coarse.items():
        if solver.get(key, 'frequency_ascending' if key == 'band_order_policy' else None) != value:
            raise ValueError('reference coarse solver differs from configured contract')
    if (reference['lower_band'] != params['target']['lower_band_one_based']
            or reference['target_frequency'] != params['target']['frequency_a_over_lambda']
            or solver.get('polarization') != params['target']['polarization']
            or solver.get('epsilon_air') != geometry['epsilon_air']
            or solver.get('epsilon_material') != geometry['epsilon_material']):
        raise ValueError('reference target differs from configured contract')
    points = np.ascontiguousarray(reference['k_points'], dtype=np.float32)
    if hashlib.sha256(points.tobytes()).hexdigest() != params['target']['checkpoint_k_points_sha256']:
        raise ValueError('reference k-point identity mismatch')
    keys = ('parameter_hash', f'content_hash_r{low}', f'content_hash_r{master}')
    known = {}
    for key in keys:
        values = excluded['identity_sets'].get(key)
        if not isinstance(values, list) or len(set(values)) != len(values) or not all(isinstance(v, str) and v for v in values):
            raise ValueError('invalid exclusion identity set')
        known[key] = set(values)
    generator_config = {'fourier_cutoff': geometry['cutoff'], 'audit_resolution': master,
        'epsilon_air': geometry['epsilon_air'], 'epsilon_material': geometry['epsilon_material'],
        'binary_target_material_fraction_range': geometry['target_material_fraction_range']}
    if output.exists():
        raise FileExistsError('source output exists; reconcile before retry')
    from functools import partial
    work = partial(_source, prefix=prefix, seed=seed, generator_config=generator_config,
                   low_resolution=low, master_resolution=master)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(work, range(count), chunksize=1):
            if any(row[key] in known[key] for key in keys):
                raise ValueError(f"source identity collision at index {row['source_index']}")
            for key in keys:
                known[key].add(row[key])
            rows.append(row)
    manifest = {'experiment_id': params['experiment_id'], 'scope': 'new_source_development_not_physical',
        'scope_sha256': scope_sha256, 'exclusions_sha256': contract['exclusions_sha256'],
        'reference_manifest_sha256': contract['reference_manifest_sha256'],
        'generator_seed': seed, 'sample_count': count, 'sample_id_prefix': prefix,
        'selection_seed': pool['selection_seed'], 'stratification_threshold': pool['coarse_margin_threshold'],
        'lower_band': reference['lower_band'], 'target_frequency': reference['target_frequency'],
        'k_points': reference['k_points'], 'solver_config': solver, 'rows': rows}
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(raw)
    return {'status': 'source_manifest_created_not_physically_screened', 'count': len(rows),
            'manifest_sha256': hashlib.sha256(raw).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a frozen collision-checked source pool')
    for name in ('config', 'scope', 'scope-sha256', 'exclusions', 'reference-manifest', 'output'):
        parser.add_argument('--'+name, required=True)
    for name in ('workers', 'cpu-limit', 'memory-budget-mib', 'worker-memory-mib', 'setup-reserve-mib'):
        parser.add_argument('--'+name, type=int, required=True)
    a = parser.parse_args()
    result = generate_source_pool(config=Path(a.config), scope=Path(a.scope), scope_sha256=a.scope_sha256,
        exclusions=Path(a.exclusions), reference_manifest=Path(a.reference_manifest), output=Path(a.output),
        workers=a.workers, cpu_limit=a.cpu_limit, memory_budget_mib=a.memory_budget_mib,
        worker_memory_mib=a.worker_memory_mib, setup_reserve_mib=a.setup_reserve_mib)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
