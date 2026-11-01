"""Produce Fourier structures for a target using a frozen forward surrogate.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.fourier_search import search_fourier


def run_surrogate_design(source: FullCellStructureSpec, predictor: Any, *, seeds: Sequence[int],
                         search_options: Mapping[str, Any], output_root: Path,
                         provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Run independent searches and save predicted structures and full traces.

    This is a surrogate-only exploratory workflow; neither the fixed k-grid
    objective nor its returned geometries constitute MPB-verified designs.
    The caller is responsible for checkpoint/raster and data-split provenance.
    """
    if (not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('unique nonnegative integer seeds required')
    options = dict(search_options)
    if 'score_adjuster' in options or 'seed' in options:
        raise ValueError('score adjustment and seeds must be controlled by the workflow')
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError('explicit provenance is required')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in seeds:
        result = search_fourier(source, predictor, seed=seed, **options)
        folder = root/f'seed_{seed}'
        folder.mkdir()
        result_dict = asdict(result)
        (folder/'search.json').write_text(json.dumps(result_dict, indent=2, allow_nan=False), encoding='utf8')
        structure_hash = None
        if result.best_structure is not None:
            structure = FullCellStructureSpec.from_mapping(result.best_structure)
            structure_hash = structure.parameter_hash
            raw = json.dumps(structure.to_mapping(), indent=2, allow_nan=False).encode('utf8')
            (folder/'structure.json').write_bytes(raw)
            raster = structure.rasterize((options['source_resolution'], options['source_resolution']))
            np.save(folder/'structure.npy', raster, allow_pickle=False)
        records.append({'seed': seed, 'status': result.status, 'structure_parameter_hash': structure_hash,
                        'predicted_target_margin': result.best_margin, 'evaluation_count': len(result.evaluations)})
    overall_status = 'surrogate_only' if any(item['structure_parameter_hash'] is not None for item in records) else 'no_valid_candidates'
    output = {'status': overall_status, 'physical_validation': 'not_run',
              'source_parameter_hash': source.parameter_hash, 'provenance': dict(provenance),
              'search_options': options, 'trajectories': records}
    (root/'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf8')
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description='Search a target band gap using a frozen forward checkpoint')
    parser.add_argument('--config', required=True, help='JSON with checkpoint, source, seeds and search_options')
    parser.add_argument('--output-root', required=True, help='new output directory')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    source_bytes = Path(config['source']).read_bytes()
    source = FullCellStructureSpec.from_mapping(json.loads(source_bytes))
    from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
    predictor = FrozenBandAdapter(config['checkpoint'], sha256=config['checkpoint_sha256'], device=args.device)
    options = dict(config['search_options'])
    options['coefficient_keys'] = tuple(tuple(key) for key in options['coefficient_keys'])
    options['bounds'] = tuple(tuple(bound) for bound in options['bounds'])
    options['material_fraction_bounds'] = tuple(options['material_fraction_bounds'])
    if options.get('perturbation_offsets') is not None:
        options['perturbation_offsets'] = tuple(options['perturbation_offsets'])
    result = run_surrogate_design(source, predictor, seeds=config['seeds'], search_options=options,
        output_root=Path(args.output_root), provenance={'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(), 'checkpoint_sha256': predictor.checkpoint_sha256,
        'model_name': predictor.model_name, 'device': args.device})
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
