"""Assemble audited paired search choices into a deduplicated solver batch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


def build_nominal_static_physical_request(*, campaign_bytes: bytes, campaign_sha256: str,
                                          selections: Mapping[int, bytes], pilot_config_bytes: bytes,
                                          prior_manifest_bytes: bytes, prior_protocol_bytes: bytes) -> dict[str, Any]:
    """Build all frozen static candidate requests without submitting or solving.

    The caller authenticates the campaign's provenance, scheduler chronology,
    and execution lock. Identical grids across sources are retained as separate
    registered roles rather than silently changing the attempt contract.
    """
    import yaml

    if hashlib.sha256(campaign_bytes).hexdigest() != campaign_sha256:
        raise ValueError('static campaign identity mismatch')
    campaign, pilot = json.loads(campaign_bytes), yaml.safe_load(pilot_config_bytes)
    prior, previous_protocol = json.loads(prior_manifest_bytes), json.loads(prior_protocol_bytes)
    if (campaign.get('scope') != 'static_campaign_selection_not_physical_execution_lock'
            or campaign.get('status') != 'locked_surrogate_only'
            or campaign.get('pilot_config_sha256') != hashlib.sha256(pilot_config_bytes).hexdigest()
            or hashlib.sha256(prior_manifest_bytes).hexdigest() != pilot['physical_manifest_sha256']
            or hashlib.sha256(prior_protocol_bytes).hexdigest() != pilot['physics']['prior_protocol_sha256']
            or prior.get('protocol_sha256') != pilot['physics']['prior_protocol_sha256']):
        raise ValueError('static selection or original physical protocol mismatch')
    ids = list(pilot['sources'].values())
    records = campaign['sources']
    if (not ids or len(set(ids)) != len(ids) or set(selections) != set(ids)
            or [row['source_index'] for row in records] != ids):
        raise ValueError('static source roster incomplete')
    resolution = pilot['physics']['master_raster_resolution']
    if (previous_protocol.get('offsets', []).count(0.0) != 1
            or previous_protocol['raster_resolution'] != resolution or prior['raster_resolution'] != resolution
            or prior['target_frequency'] != pilot['target']['frequency_a_over_lambda']
            or prior['lower_band'] != pilot['target']['lower_band_one_based']):
        raise ValueError('nominal geometry or target differs from prior contract')
    config = FullCellMpbBandSolverConfig(**prior['solver_config'])
    if config.polarization != pilot['target']['polarization'] or config.resolution != pilot['physics']['primary_resolution']:
        raise ValueError('nominal solver differs from pilot contract')
    protocol = {**previous_protocol, 'offsets': [0.0]}
    if protocol['refinement']['verification_resolution'] != pilot['physics']['verification_resolution']:
        raise ValueError('verification resolution differs from pilot contract')
    protocol_raw = json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    rows = []
    for source_index, record in zip(ids, records):
        raw = selections[source_index]
        if hashlib.sha256(raw).hexdigest() != record['selection_sha256'] or record['status'] != 'locked_surrogate_only':
            raise ValueError('static source selection identity mismatch')
        selection = json.loads(raw)
        candidates = selection['selection']['selected']
        if (selection.get('source_index') != source_index or selection['selection']['status'] != 'locked_surrogate_only'
                or len(candidates) != pilot['search']['candidate_physical_attempts_per_method']):
            raise ValueError('static candidate roster incomplete')
        seen = set()
        for rank, item in enumerate(candidates):
            structure = FullCellStructureSpec.from_mapping(item['structure'])
            grid = np.ascontiguousarray(structure.rasterize((resolution, resolution)), dtype=np.uint8)
            digest = hashlib.sha256(grid.tobytes()).hexdigest()
            if (structure.material_kind != 'binary' or not np.isin(grid, (0, 1)).all()
                    or structure.parameter_hash != item['parameter_hash'] or digest != item['nominal_grid_sha256']
                    or digest in seen or digest == selection['selection']['source_grid_sha256']
                    or structure.epsilon_air != config.epsilon_air or structure.epsilon_material != config.epsilon_material):
                raise ValueError('static candidate geometry or identity mismatch')
            seen.add(digest)
            rows.append({'source_index': source_index, 'label': 'candidate', 'method': 'static', 'rank': rank,
                         'parameter_hash': structure.parameter_hash, 'nominal_grid_sha256': digest,
                         'structure': structure.to_mapping()})
    if len(rows) > pilot['resources']['new_physical_attempts_max']:
        raise ValueError('static requests exceed total physical attempts')
    manifest = {'scope': 'nominal_static_solver_request_not_execution_lock', 'status': 'surrogate_replay_request',
                'campaign_sha256': campaign_sha256, 'k_points': prior['k_points'], 'lower_band': prior['lower_band'],
                'target_frequency': prior['target_frequency'], 'solver_config': prior['solver_config'],
                'raster_resolution': resolution, 'protocol_sha256': hashlib.sha256(protocol_raw).hexdigest(), 'rows': rows}
    return {'scope': 'static_request_constructed_not_submitted', 'manifest': manifest,
            'protocol': protocol, 'protocol_sha256': hashlib.sha256(protocol_raw).hexdigest()}


def materialize_nominal_static_physical_request(request: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Save a constructed nominal request once; this does not authorize execution."""
    if request.get('scope') != 'static_request_constructed_not_submitted':
        raise ValueError('constructed nominal static request required')
    manifest, protocol = request['manifest'], request['protocol']
    protocol_raw = json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    protocol_hash = hashlib.sha256(protocol_raw).hexdigest()
    if (protocol_hash != request.get('protocol_sha256') or manifest.get('protocol_sha256') != protocol_hash
            or manifest.get('scope') != 'nominal_static_solver_request_not_execution_lock'
            or protocol.get('offsets') != [0.0]):
        raise ValueError('constructed manifest or protocol mismatch')
    manifest_raw = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    with (root/'protocol.json').open('xb') as stream:
        stream.write(protocol_raw)
    with (root/'manifest.json').open('xb') as stream:
        stream.write(manifest_raw)
    return {'scope': 'materialized_static_request_not_execution_lock',
            'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(), 'protocol_sha256': protocol_hash,
            'case_count': len(manifest['rows'])}


def build_paired_physical_batch(
    sources: Sequence[Mapping[str, Any]], locks: Sequence[Mapping[str, Any]], *,
    expected_source_counts: Mapping[str, int], checkpoint_sha256_by_seed: Mapping[int, str],
    search_seed_by_model_seed: Mapping[int, int], random_choice_seed: int,
    k_points: Any, k_points_sha256: str, config: FullCellMpbBandSolverConfig,
    lower_band: int, target_frequency: float, protocol_bytes: bytes, protocol_sha256: str,
) -> dict[str, Any]:
    """Create a physical request from independently authenticated trace locks.

    The caller must recompute each lock from audited artifacts and authenticate
    source selection, checkpoint execution, chronology and experiment config.
    This operation neither performs a physical solve nor establishes a gate.
    """
    if hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha256:
        raise ValueError('protocol hash mismatch')
    protocol = json.loads(protocol_bytes)
    resolution = protocol['raster_resolution']
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError('invalid raster resolution')
    offsets = protocol.get('offsets')
    if (not isinstance(offsets, list) or not offsets or offsets.count(0.0) != 1
            or not np.isfinite(offsets).all() or len(set(offsets)) != len(offsets)
            or protocol.get('refinement', {}).get('verification_resolution', 0) <= config.resolution):
        raise ValueError('invalid perturbation or refinement protocol')
    points = np.asarray(k_points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
            or hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest() != k_points_sha256):
        raise ValueError('k-point identity mismatch')
    if (not isinstance(lower_band, int) or isinstance(lower_band, bool) or not 1 <= lower_band < config.num_bands
            or isinstance(target_frequency, bool) or not np.isfinite(target_frequency) or target_frequency <= 0):
        raise ValueError('invalid target')
    seeds = sorted(checkpoint_sha256_by_seed)
    if (not seeds or set(seeds) != set(search_seed_by_model_seed) or 0 not in seeds
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)):
        raise ValueError('invalid model seed roster')
    source_by_id: dict[int, Mapping[str, Any]] = {}
    for source in sources:
        sid = source['source_index']
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 0 or sid in source_by_id:
            raise ValueError('invalid or duplicate source')
        source_by_id[sid] = source
    counts = {name: sum(s['stratum'] == name for s in sources) for name in expected_source_counts}
    if (not expected_source_counts or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                                          for v in expected_source_counts.values())
            or counts != dict(expected_source_counts) or sum(counts.values()) != len(sources)):
        raise ValueError('source strata incomplete')
    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for lock in locks:
        key = (lock['source_index'], lock['model_seed'])
        if (any(isinstance(x, bool) or not isinstance(x, int) for x in key)
                or key in by_key or key[0] not in source_by_id or key[1] not in seeds):
            raise ValueError('duplicate or unexpected trace lock')
        by_key[key] = lock
    if set(by_key) != {(sid, seed) for sid in source_by_id for seed in seeds}:
        raise ValueError('trace lock roster incomplete')
    unscoreable = []
    rows: list[dict[str, Any]] = []
    role_map: list[dict[str, Any]] = []
    by_grid: dict[str, int] = {}

    def add_structure(structure: FullCellStructureSpec, sid: int, label: str, expected_grid: str) -> int:
        if (structure.material_kind != 'binary' or structure.epsilon_air != config.epsilon_air
                or structure.epsilon_material != config.epsilon_material):
            raise ValueError('structure material contract mismatch')
        grid = np.ascontiguousarray(structure.rasterize((resolution, resolution)), dtype=np.uint8)
        if not np.isin(grid, (0, 1)).all():
            raise ValueError('nonbinary raster')
        actual = hashlib.sha256(grid.tobytes()).hexdigest()
        if actual != expected_grid:
            raise ValueError('structure grid identity mismatch')
        if actual not in by_grid:
            by_grid[actual] = len(rows)
            rows.append({'source_index': sid, 'label': label, 'parameter_hash': structure.parameter_hash,
                         'structure': structure.to_mapping(), 'master_grid_sha256': actual})
        return by_grid[actual]

    for sid in sorted(source_by_id):
        source = source_by_id[sid]
        anchor = FullCellStructureSpec.from_mapping(source['structure'])
        if anchor.parameter_hash != source['parameter_hash']:
            raise ValueError('source parameter identity mismatch')
        anchor_grid = hashlib.sha256(np.ascontiguousarray(anchor.rasterize((resolution, resolution)), dtype=np.uint8).tobytes()).hexdigest()
        if any(r['role'] == 'anchor' and r['nominal_grid_sha256'] == anchor_grid for r in role_map):
            raise ValueError('duplicate source raster')
        anchor_case = add_structure(anchor, sid, 'anchor', anchor_grid)
        role_map.append({'source_index': sid, 'role': 'anchor', 'case_index': anchor_case,
                         'parameter_hash': anchor.parameter_hash, 'nominal_grid_sha256': anchor_grid})
        for seed in seeds:
            lock = by_key[(sid, seed)]
            if (lock.get('physical_validation') != 'not_run' or lock.get('audit_status') != 'artifact_consistent'
                    or lock.get('checkpoint_sha256') != checkpoint_sha256_by_seed[seed]
                    or lock.get('search_seed') != search_seed_by_model_seed[seed]
                    or lock.get('random_choice_seed') != random_choice_seed
                    or lock.get('checkpoint_k_points_sha256') != k_points_sha256
                    or lock.get('source_resolution') != resolution
                    or lock.get('perturbation_offsets') != offsets
                    or lock.get('lower_band') != lower_band
                    or lock.get('target_frequency') != target_frequency
                    or lock.get('source_parameter_hash') != anchor.parameter_hash
                    or lock.get('source_grid_sha256') != anchor_grid):
                raise ValueError('trace lock provenance mismatch')
            if lock.get('status') != 'locked_surrogate_only':
                if lock.get('status') != 'unscoreable_insufficient_eligible':
                    raise ValueError('unexpected trace status')
                unscoreable.append({'source_index': sid, 'model_seed': seed, 'status': lock['status']})
                continue
            for role in ('surrogate_choice', 'random_choice'):
                choice = lock[role]
                structure = FullCellStructureSpec.from_mapping(choice['structure'])
                if (structure.parameter_hash != choice['parameter_hash']
                        or choice['candidate_index'] not in lock['eligible_indices']
                        or choice['nominal_grid_sha256'] == anchor_grid):
                    raise ValueError('choice identity mismatch')
                case_index = add_structure(structure, sid, f'{role}_seed_{seed}', choice['nominal_grid_sha256'])
                lock_sha = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(',', ':'),
                                                     allow_nan=False).encode()).hexdigest()
                role_map.append({'source_index': sid, 'model_seed': seed, 'role': role,
                                 'candidate_index': choice['candidate_index'], 'case_index': case_index,
                                 'parameter_hash': structure.parameter_hash,
                                 'nominal_grid_sha256': choice['nominal_grid_sha256'], 'trace_lock_sha256': lock_sha})
    if unscoreable:
        return {'status': 'unscoreable_trace_roster', 'physical_validation': 'not_run',
                'unscoreable': unscoreable, 'manifest': None, 'role_map': role_map,
                'cross_environment_roles': []}
    cross_roles = []
    for stratum in expected_source_counts:
        sid = min(sid for sid, source in source_by_id.items() if source['stratum'] == stratum)
        selected = [r for r in role_map if r['source_index'] == sid and
                    (r['role'] == 'anchor' or (r['role'] == 'surrogate_choice' and r.get('model_seed') == 0))]
        if len(selected) != 2:
            raise ValueError('cross-environment role missing')
        cross_roles.extend(selected)
    if len({r['case_index'] for r in cross_roles}) != len(cross_roles):
        return {'status': 'cross_environment_roles_not_unique', 'physical_validation': 'not_run',
                'manifest': None, 'role_map': role_map, 'cross_environment_roles': cross_roles,
                'unscoreable': []}
    manifest = {'scope': 'paired_surrogate_replay_request_not_physical_validation',
                'k_points': points.tolist(), 'lower_band': lower_band, 'target_frequency': float(target_frequency),
                'raster_resolution': resolution, 'protocol_sha256': protocol_sha256,
                'solver_config': asdict(config), 'rows': rows}
    return {'status': 'ready_for_physical_evaluation', 'physical_validation': 'not_run',
            'manifest': manifest, 'role_map': role_map, 'cross_environment_roles': cross_roles,
            'unscoreable': []}
