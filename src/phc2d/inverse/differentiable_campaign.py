"""Prepare paired initial geometries for surrogate inverse campaigns.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.fourier_design import update_fourier_binary
from phc2d.inverse.fourier_screen import screen_fourier_geometry


def prepare_paired_initials(source: Any, *, base: Mapping[str, Any],
                            campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Choose each first eligible registered initial without consulting a surrogate or solver."""
    import numpy as np
    keys = tuple((int(i), str(component)) for i, component in base['coefficient_keys'])
    reference = [float(source.parameters['modes'][i][component]) for i, component in keys]
    reference.append(float(source.parameters['threshold']))
    scale, lo, hi = base['scales'], base['lower'], base['upper']
    if any(len(x) != len(reference) for x in (scale, lo, hi)):
        raise ValueError('parameter dimensions differ')
    records = []
    for index, options in enumerate(campaign['initial_options']):
        attempts = []
        selected = None
        for option in options:
            if len(option) != len(reference) or any(not lo[i] <= z <= hi[i] for i, z in enumerate(option)):
                raise ValueError('initial point outside parameter bounds')
            values = [reference[i] + scale[i] * z for i, z in enumerate(option)]
            structure = update_fourier_binary(source, sample_id=f'{source.sample_id}-initial-{index}',
                coefficients={key: float(values[i]) for i, key in enumerate(keys)},
                threshold=float(values[-1]), max_mode_radius=base['max_mode_radius'])
            geometry = screen_fourier_geometry(structure, raster_method='master_block_mean_v1',
                source_resolution=base['source_resolution'], input_resolution=base['input_resolution'],
                material_fraction_bounds=tuple(base['material_fraction_bounds']))
            grid = np.ascontiguousarray(structure.rasterize((base['source_resolution'],)*2), dtype=np.uint8)
            attempts.append({'point': list(option), 'status': geometry.status,
                             'parameter_hash': structure.parameter_hash,
                             'master_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest(),
                             'material_fraction': geometry.material_fraction})
            if geometry.status == 'eligible':
                selected = list(option)
                break
        records.append({'initial_index': index, 'status': 'ready' if selected is not None else 'no_valid_initial',
                        'initial': selected, 'attempts': attempts})
    return records


def build_campaign(campaign: Mapping[str, Any], base: Mapping[str, Any],
                   sources: Mapping[int, Any]) -> dict[str, Any]:
    """Freeze paired initial attempts and per-trial configs before solver observations."""
    roster = campaign['sources']
    ids = [row['id'] for row in roster]
    if len(ids) != len(set(ids)) or set(ids) != set(sources):
        raise ValueError('campaign source roster missing or duplicated')
    methods = campaign['methods']
    if len(methods) != len(set(methods)) or set(methods) != {'adam', 'de', 'random'}:
        raise ValueError('campaign methods must be adam, de and random')
    if campaign['evaluations'] != 60:
        raise ValueError('campaign requires sixty evaluations')
    prepared, trials = [], []
    for source_entry in roster:
        source_id = source_entry['id']
        source = sources[source_id]
        initials = prepare_paired_initials(source, base=base, campaign=campaign)
        prepared.append({'source_id': source_id, 'source_sha256': source_entry['sha256'],
                         'source_parameter_hash': source.parameter_hash, 'initials': initials})
        for item in initials:
            for method in methods:
                cfg = dict(base, experiment_id=campaign['experiment_id'], source_sha256=source_entry['sha256'],
                           evaluations=campaign['evaluations'], seed=campaign['seed'], initial=item['initial'])
                trials.append({'trial_id': f"source_{source_id:04d}_initial_{item['initial_index']}_{method}",
                               'source_id': source_id, 'initial_index': item['initial_index'],
                               'method': method, 'status': item['status'],
                               'config': cfg if item['status'] == 'ready' else None})
    return {'experiment_id': campaign['experiment_id'], 'scope': 'preregistered_surrogate_design',
            'sources': prepared, 'trials': trials}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'base', 'source-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw = Path(args.campaign).read_bytes()
    campaign = json.loads(raw)
    base, digest = read_checked_json(args.base, campaign['base_config_sha256'])
    sources = {}
    for entry in campaign['sources']:
        path = Path(args.source_directory) / f"source_{entry['id']:04d}.json"
        source, _ = read_checked_json(path, entry['sha256'])
        sources[entry['id']] = FullCellStructureSpec.from_mapping(source)
    manifest = build_campaign(campaign, base, sources)
    manifest.update(campaign_sha256=hashlib.sha256(raw).hexdigest(), base_config_sha256=digest)
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
