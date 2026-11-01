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
from phc2d.inverse.fourier_screen import screen_fourier_geometry
from phc2d.inverse.perturbation_screen import periodic_perturbation_grids, predict_perturbed_gap
from phc2d.inverse.robust_gap import common_gap


def _replay_selected_prediction(structure: FullCellStructureSpec, predictor: Any,
                                options: Mapping[str, Any]) -> dict[str, Any]:
    """Re-predict the selected design; this is not the original search evaluation."""
    offsets = options.get('perturbation_offsets')
    if offsets is not None:
        grids = periodic_perturbation_grids(structure, resolution=options['source_resolution'],
                                            offsets=tuple(offsets))
        result = predict_perturbed_gap(grids, predictor, input_resolution=options['input_resolution'],
            lower_band=options['lower_band'], target_frequency=options['target_frequency'])
        return {'offsets': list(offsets), **result}
    import torch
    geometry = screen_fourier_geometry(structure, raster_method=options['raster_method'],
        source_resolution=options['source_resolution'], input_resolution=options['input_resolution'],
        material_fraction_bounds=options['material_fraction_bounds'],
        geometry_constraints=options.get('geometry_constraints'))
    if geometry.status != 'eligible':
        return {'status': geometry.status, 'offsets': [0.0], 'gap': None}
    with torch.no_grad():
        bands = predictor.predict_tensor(torch.as_tensor(geometry.raster[None, ...], dtype=torch.float32)).detach().cpu().numpy()
    if bands.ndim != 3 or bands.shape[0] != 1 or any(size == 0 for size in bands.shape):
        raise ValueError('predictor must return [1,k,band] frequencies')
    if not np.isfinite(bands).all():
        return {'status': 'nonfinite_prediction', 'offsets': [0.0], 'gap': None}
    if np.any(np.diff(bands, axis=2) < 0):
        return {'status': 'band_order_violation', 'offsets': [0.0], 'gap': None, 'bands': bands.tolist()}
    lower = options['lower_band']
    if not 1 <= lower < bands.shape[2]:
        raise ValueError('invalid target band')
    pair = bands[:, :, lower-1:lower+1]
    if np.any(pair < 0):
        return {'status': 'negative_frequency', 'offsets': [0.0], 'gap': None, 'bands': bands.tolist()}
    return {'status': 'surrogate_only', 'offsets': [0.0], 'bands': bands.tolist(),
            'gap': asdict(common_gap(pair, lower_band=1, target_frequency=options['target_frequency']))}


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
        response_status = 'not_available'
        if result.best_structure is not None:
            structure = FullCellStructureSpec.from_mapping(result.best_structure)
            structure_hash = structure.parameter_hash
            raw = json.dumps(structure.to_mapping(), indent=2, allow_nan=False).encode('utf8')
            (folder/'structure.json').write_bytes(raw)
            raster = structure.rasterize((options['source_resolution'], options['source_resolution']))
            np.save(folder/'structure.npy', raster, allow_pickle=False)
            response = _replay_selected_prediction(structure, predictor, options)
            response_status = response['status']
            response['prediction_stage'] = 'post_search_reprediction'
            response['structure_parameter_hash'] = structure_hash
            response['search_best_margin'] = result.best_margin
            response['target_margin_difference'] = (None if response['gap'] is None else
                response['gap']['target_margin'] - result.best_margin)
            (folder/'predicted_response.json').write_text(json.dumps(response, indent=2, allow_nan=False), encoding='utf8')
        records.append({'seed': seed, 'status': result.status, 'structure_parameter_hash': structure_hash,
                        'predicted_target_margin': result.best_margin, 'evaluation_count': len(result.evaluations),
                        'response_reprediction_status': response_status})
    overall_status = 'surrogate_only' if any(item['structure_parameter_hash'] is not None for item in records) else 'no_valid_candidates'
    output = {'status': overall_status, 'physical_validation': 'not_run',
              'source_parameter_hash': source.parameter_hash, 'provenance': dict(provenance),
              'search_options': options, 'trajectories': records}
    (root/'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf8')
    return output


def build_surrogate_replay_request(*, summary_bytes: bytes, summary_sha256: str,
                                   structure_bytes: bytes, seed: int, k_points: Any,
                                   k_points_sha256: str, solver_config: Any,
                                   source_index: int, protocol_bytes: bytes,
                                   protocol_sha256: str) -> dict[str, Any]:
    """Bind a selected surrogate structure to a real-MPB request, without solving it.

    The k-grid digest is computed from contiguous float32 coordinates as in the
    frozen forward checkpoint. Offset amplitudes and raster resolution must
    match the frozen physical protocol; solver fidelity remains caller-owned.
    """
    from phc2d.inverse.feedback_handoff import build_structure_solver_manifest
    if hashlib.sha256(summary_bytes).hexdigest() != summary_sha256:
        raise ValueError('search summary hash mismatch')
    summary = json.loads(summary_bytes)
    if summary.get('status') != 'surrogate_only' or summary.get('physical_validation') != 'not_run':
        raise ValueError('expected an unverified surrogate search')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('integer seed required')
    matches = [item for item in summary['trajectories'] if item['seed'] == seed]
    if len(matches) != 1 or matches[0]['status'] != 'surrogate_only':
        raise ValueError('seed has no unique candidate')
    structure = FullCellStructureSpec.from_mapping(json.loads(structure_bytes))
    if structure.parameter_hash != matches[0]['structure_parameter_hash']:
        raise ValueError('structure identity differs from search')
    points = np.asarray(k_points, dtype=np.float64)
    canonical_points = np.ascontiguousarray(points, dtype=np.float32)
    if hashlib.sha256(canonical_points.tobytes()).hexdigest() != k_points_sha256:
        raise ValueError('k-grid hash mismatch')
    checkpoint_k = summary['provenance'].get('k_points_sha256')
    if checkpoint_k != k_points_sha256:
        raise ValueError('k grid differs from frozen checkpoint')
    options = summary['search_options']
    if hashlib.sha256(protocol_bytes).hexdigest() != protocol_sha256:
        raise ValueError('physical protocol hash mismatch')
    protocol = json.loads(protocol_bytes)
    offsets = options.get('perturbation_offsets')
    if (offsets is None or list(offsets) != protocol.get('offsets')
            or options['source_resolution'] != protocol.get('raster_resolution')):
        raise ValueError('physical protocol differs from search perturbations or raster')
    request = build_structure_solver_manifest(structure, k_points=points, config=solver_config,
        lower_band=options['lower_band'], source_index=source_index,
        target_frequency=options['target_frequency'], surrogate_contract_sha256=summary_sha256,
        status='surrogate_replay_request')
    request['search_seed'] = seed
    request['checkpoint_sha256'] = summary['provenance']['checkpoint_sha256']
    request['k_points_sha256'] = k_points_sha256
    request['protocol_sha256'] = protocol_sha256
    request['search_perturbation_offsets'] = list(offsets)
    return request


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
        'model_name': predictor.model_name, 'k_points_sha256': predictor.k_points_sha256,
        'device': args.device})
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
