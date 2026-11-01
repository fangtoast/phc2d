"""Build frozen target-driven search requests from an audited source selection.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec


def build_paired_search_requests(
    *, config_bytes: bytes, source_manifest_bytes: bytes, source_audit_bytes: bytes,
    selection_bytes: bytes, expected_source_audit_sha256: str, expected_selection_sha256: str,
    checkpoint_paths: Mapping[int, str], source_directory: str,
) -> dict[str, Any]:
    """Build one search config per selected source and frozen model seed.

    Callers must authenticate the source audit, selection chronology, physical
    coarse-screen evidence and checkpoint files independently. This function
    creates contracts only; it does not execute a model or verify a design.
    Serialize each returned source using sorted JSON keys and compact separators
    to reproduce its recorded SHA-256 before writing the referenced file.
    """
    import yaml

    config_sha = hashlib.sha256(config_bytes).hexdigest()
    manifest_sha = hashlib.sha256(source_manifest_bytes).hexdigest()
    audit_sha = hashlib.sha256(source_audit_bytes).hexdigest()
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    if audit_sha != expected_source_audit_sha256 or selection_sha != expected_selection_sha256:
        raise ValueError('source audit or selection hash mismatch')
    config = yaml.safe_load(config_bytes)
    manifest, audit, selection = map(json.loads, (source_manifest_bytes, source_audit_bytes, selection_bytes))
    pool, geometry, search, surrogate, target, physics = (config[k] for k in
        ('source_pool', 'geometry', 'search', 'surrogate', 'target', 'physics'))
    if (config.get('experiment_id') != manifest.get('experiment_id')
            or audit.get('status') != 'pass' or audit.get('manifest_sha256') != manifest_sha
            or audit.get('config_sha256') != config_sha or audit.get('reconstructed_count') != len(manifest['rows'])
            or audit.get('seed_regenerated_count') != len(manifest['rows'])
            or audit.get('scope') != 'source_identity_only_no_physical_validation'
            or audit.get('verified_resolutions') != [geometry['model_input_resolution'],
                geometry['generation_audit_resolution'], geometry['physical_master_raster_resolution']]
            or pool.get('check_physical_grid_collisions') is not True
            or manifest.get('scope') != 'new_source_development_not_physical'
            or manifest.get('sample_count') != len(manifest['rows'])
            or selection.get('source_manifest_sha256') != manifest_sha or selection.get('status') != 'ready'
            or selection.get('seed') != pool['selection_seed'] or selection.get('per_stratum') != pool['per_stratum']
            or selection.get('lower_band') != target['lower_band_one_based']
            or selection.get('target_frequency') != target['frequency_a_over_lambda']
            or selection.get('margin_threshold') != pool['coarse_margin_threshold']):
        raise ValueError('source selection or audit contract mismatch')
    offsets = physics.get('offsets_a')
    if (geometry.get('family') != 'unified_fourier_binary'
            or search.get('algorithm') != 'differential_evolution'
            or surrogate.get('band_order_method') != 'ascending_projection_preserve_raw'
            or not isinstance(offsets, list) or not offsets or offsets.count(0.0) != 1
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in offsets)
            or len(set(offsets)) != len(offsets)):
        raise ValueError('unsupported search, geometry or projection contract')
    seeds = sorted(surrogate['checkpoint_sha256_by_seed'])
    seed_values = search['search_seeds']
    if (not seeds or set(seeds) != set(checkpoint_paths) or len(seeds) != len(seed_values)
            or len(set(seed_values)) != len(seed_values) or not source_directory
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
            or any(not isinstance(checkpoint_paths[seed], str) or not checkpoint_paths[seed] for seed in seeds)):
        raise ValueError('invalid checkpoint or search seed roster')
    source_by_index = {}
    for row in manifest['rows']:
        index = row['source_index']
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in source_by_index:
            raise ValueError('invalid source index')
        source_by_index[index] = row
    strata = pool['stratum_order']
    selected = selection['selected']
    if (len(set(strata)) != len(strata) or len(selected) != len(strata)*pool['per_stratum']
            or any(sum(x['stratum'] == s for x in selected) != pool['per_stratum'] for s in strata)
            or len({x['index'] for x in selected}) != len(selected)
            or any(x['index'] not in source_by_index or x['stratum'] not in strata for x in selected)):
        raise ValueError('invalid selected source roster')
    keys = [tuple(key) for key in search['coefficient_keys']]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('invalid coefficient key roster')
    requests = []
    for item in selected:
        index = item['index']
        source = FullCellStructureSpec.from_mapping(source_by_index[index]['structure'])
        if (source.material_kind != 'binary' or source.parameter_hash != source_by_index[index]['parameter_hash']
                or source.epsilon_air != geometry['epsilon_air'] or source.epsilon_material != geometry['epsilon_material']):
            raise ValueError('selected source identity mismatch')
        source_raw = json.dumps(source.to_mapping(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        bounds = []
        for mode_index, component in keys:
            if (isinstance(mode_index, bool) or not isinstance(mode_index, int)
                    or not 0 <= mode_index < len(source.parameters['modes']) or component not in ('cos', 'sin')):
                raise ValueError('invalid coefficient key')
            mode = source.parameters['modes'][mode_index]
            if np.hypot(mode['mx'], mode['my']) > search['max_mode_radius']:
                raise ValueError('search mode exceeds radius')
            center = float(mode[component])
            radius = float(search['coefficient_radius'])
            bounds.append([center-radius, center+radius])
        center = float(source.parameters['threshold'])
        radius = float(search['threshold_radius'])
        bounds.append([center-radius, center+radius])
        if not np.isfinite(bounds).all() or any(lo >= hi for lo, hi in bounds):
            raise ValueError('invalid search bounds')
        for position, model_seed in enumerate(seeds):
            checkpoint_hash = surrogate['checkpoint_sha256_by_seed'][model_seed]
            if (not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64
                    or any(ch not in '0123456789abcdef' for ch in checkpoint_hash)):
                raise ValueError('invalid checkpoint digest')
            options = {'coefficient_keys': [list(k) for k in keys], 'bounds': bounds,
                'max_mode_radius': search['max_mode_radius'], 'population_size': search['population_size'],
                'evaluation_budget': search['evaluation_budget_per_source_seed'],
                'raster_method': geometry['model_raster_method'],
                'source_resolution': geometry['physical_master_raster_resolution'],
                'input_resolution': geometry['model_input_resolution'],
                'material_fraction_bounds': geometry['material_fraction_bounds'],
                'lower_band': target['lower_band_one_based'], 'target_frequency': target['frequency_a_over_lambda'],
                'perturbation_offsets': physics['offsets_a'], 'projection_contract_sha256': config_sha}
            run_config = {'source': f"{source_directory.rstrip('/')}/source_{index:04d}.json",
                'checkpoint': checkpoint_paths[model_seed], 'checkpoint_sha256': checkpoint_hash,
                'seeds': [seed_values[position]], 'device_parity_absolute_tolerance': surrogate['device_parity_absolute_tolerance'],
                'search_options': options}
            requests.append({'source_index': index, 'stratum': item['stratum'], 'model_seed': model_seed,
                'search_seed': seed_values[position], 'source_sha256': hashlib.sha256(source_raw).hexdigest(),
                'source': source.to_mapping(), 'config': run_config})
    return {'scope': 'frozen_search_requests_not_executed', 'config_sha256': config_sha,
            'source_manifest_sha256': manifest_sha, 'source_audit_sha256': audit_sha,
            'selection_sha256': selection_sha, 'requests': requests}
