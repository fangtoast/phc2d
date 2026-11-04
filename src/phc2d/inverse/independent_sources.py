"""Freeze held-out inverse sources without reading their band labels.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def select_heldout_sources(dataset: Path, *, dataset_sha256: str, method_lock_sha256: str,
                           seed: int, count: int, source_id_offset: int, master_resolution: int,
                           excluded_parameter_hashes: Sequence[str], excluded_grid_hashes: Sequence[str],
                           epsilon_air: float, epsilon_material: float) -> dict[str, Any]:
    """Select deterministic forward-test structures, excluding known inverse development identities."""
    import h5py
    from phc2d.data.full_cell_spec import FullCellStructureSpec

    if (not dataset_sha256 or not method_lock_sha256 or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or isinstance(count, bool) or not isinstance(count, int) or count < 1
            or isinstance(source_id_offset, bool) or not isinstance(source_id_offset, int) or source_id_offset < 0
            or isinstance(master_resolution, bool) or not isinstance(master_resolution, int) or master_resolution < 1
            or not np.isfinite([epsilon_air, epsilon_material]).all() or epsilon_air <= 0 or epsilon_material <= 0):
        raise ValueError('invalid independent-source selection contract')
    digest = hashlib.sha256()
    with dataset.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != dataset_sha256:
        raise ValueError('independent-source dataset identity mismatch')
    excluded_params, excluded_grids = set(excluded_parameter_hashes), set(excluded_grid_hashes)
    if not excluded_params or not excluded_grids or any(not isinstance(x, str) or not x for x in excluded_params | excluded_grids):
        raise ValueError('known inverse development exclusion identities required')
    rows, rejected = [], []
    with h5py.File(dataset, 'r') as data:
        test = np.asarray(data['idx_test'][:], dtype=np.int64)
        train, validation = set(map(int, data['idx_train'][:])), set(map(int, data['idx_val'][:]))
        sample_count = len(data['structure_json'])
        if (test.ndim != 1 or len(test) != len(set(map(int, test))) or np.any(test < 0) or np.any(test >= sample_count)
                or set(map(int, test)) & (train | validation) or train & validation):
            raise ValueError('forward dataset splits overlap or are malformed')
        for index in np.random.default_rng(seed).permutation(test):
            index = int(index)
            raw = data['structure_json'][index]
            structure = FullCellStructureSpec.from_mapping(json.loads(raw))
            if (structure.material_kind != 'binary' or structure.epsilon_air != epsilon_air
                    or structure.epsilon_material != epsilon_material):
                raise ValueError('held-out structure material contract mismatch')
            original_hash = data['parameter_hash'][index]
            if isinstance(original_hash, bytes):
                original_hash = original_hash.decode('utf8')
            grid = np.ascontiguousarray(structure.rasterize((master_resolution, master_resolution)), dtype=np.uint8)
            if not np.isin(grid, (0, 1)).all():
                raise ValueError('held-out structure did not rasterize to binary material')
            grid_hash = hashlib.sha256(grid.tobytes()).hexdigest()
            if (structure.parameter_hash in excluded_params or original_hash in excluded_params or grid_hash in excluded_grids):
                rejected.append({'dataset_index': index, 'reason': 'known_inverse_development_identity'})
                continue
            source_id = source_id_offset + index
            rows.append({'source_id': source_id, 'dataset_index': index, 'split': 'test',
                         'parameter_hash': structure.parameter_hash, 'dataset_parameter_hash': original_hash,
                         'master_grid_sha256': grid_hash, 'structure': structure.to_mapping()})
            excluded_params.update((structure.parameter_hash, original_hash))
            excluded_grids.add(grid_hash)
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError('insufficient distinct held-out source structures')
    return {'scope': 'heldout_inverse_source_selection_no_band_labels', 'dataset_sha256': dataset_sha256,
            'method_lock_sha256': method_lock_sha256, 'selection_seed': seed, 'source_id_offset': source_id_offset,
            'master_resolution': master_resolution, 'rows': rows, 'rejected': rejected}


def write_heldout_sources(selection: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Write source JSON files and a source roster for the existing design campaign."""
    if selection.get('scope') != 'heldout_inverse_source_selection_no_band_labels':
        raise ValueError('invalid independent-source selection')
    output.mkdir(parents=True, exist_ok=False)
    sources = []
    for row in selection['rows']:
        raw = json.dumps(row['structure'], indent=2, allow_nan=False).encode('utf8')
        (output / f"source_{row['source_id']:04d}.json").write_bytes(raw)
        sources.append({'id': row['source_id'], 'sha256': hashlib.sha256(raw).hexdigest(),
                        'dataset_index': row['dataset_index'], 'split': row['split'],
                        'parameter_hash': row['parameter_hash'], 'master_grid_sha256': row['master_grid_sha256']})
    return {'scope': selection['scope'], 'dataset_sha256': selection['dataset_sha256'],
            'method_lock_sha256': selection['method_lock_sha256'], 'selection_seed': selection['selection_seed'],
            'master_resolution': selection['master_resolution'], 'sources': sources,
            'rejected': selection['rejected']}


def collect_inverse_exclusions(inputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collect recorded candidate identities from authenticated prior inverse artifacts."""
    parameters, grids, inventory = set(), set(), []
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            parameter, grid = value.get('parameter_hash'), value.get('master_grid_sha256')
            if isinstance(parameter, str) and parameter and isinstance(grid, str) and grid:
                parameters.add(parameter)
                grids.add(grid)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
    for entry in inputs:
        raw, expected = entry['bytes'], entry['sha256']
        if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('inverse exclusion artifact identity mismatch')
        visit(json.loads(raw))
        inventory.append({'name': entry['name'], 'sha256': expected})
    if not parameters or not grids:
        raise ValueError('inverse exclusion artifacts contain no candidate identities')
    return {'scope': 'recorded_inverse_candidates_not_exhaustive_history', 'inventory': inventory,
            'parameter_hashes': sorted(parameters), 'master_grid_sha256': sorted(grids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('settings', 'dataset', 'method-lock', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--exclusion', action='append', required=True)
    parser.add_argument('--exclusion-sha256', action='append', required=True)
    args = parser.parse_args()
    settings_raw, method_raw = Path(args.settings).read_bytes(), Path(args.method_lock).read_bytes()
    settings = json.loads(settings_raw)
    method_sha = hashlib.sha256(method_raw).hexdigest()
    if (settings.get('experiment_id') != 'EXP-INV-016' or settings.get('method_lock_sha256') != method_sha
            or len(args.exclusion) != len(args.exclusion_sha256)):
        raise ValueError('independent test method or exclusion inventory mismatch')
    inputs = [{'name': Path(name).name, 'bytes': Path(name).read_bytes(), 'sha256': digest}
              for name, digest in zip(args.exclusion, args.exclusion_sha256)]
    exclusions = collect_inverse_exclusions(inputs)
    selection = select_heldout_sources(Path(args.dataset), dataset_sha256=settings['dataset_sha256'],
        method_lock_sha256=method_sha, seed=settings['source_selection_seed'], count=settings['source_count'],
        source_id_offset=settings['source_id_offset'], master_resolution=settings['master_resolution'],
        excluded_parameter_hashes=exclusions['parameter_hashes'], excluded_grid_hashes=exclusions['master_grid_sha256'],
        epsilon_air=settings['epsilon_air'], epsilon_material=settings['epsilon_material'])
    output = Path(args.output_directory)
    roster = write_heldout_sources(selection, output)
    roster.update(settings_sha256=hashlib.sha256(settings_raw).hexdigest(), exclusion_inventory=exclusions['inventory'],
                  exclusion_scope=exclusions['scope'])
    (output/'manifest.json').write_text(json.dumps(roster, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    main()
