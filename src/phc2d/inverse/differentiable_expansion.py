"""Prepare one of four preregistered local inverse-direction diagnostics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.differentiable_objective import fourier_surrogate_margin
from phc2d.inverse.differentiable_selection import rank_hard_trajectory
from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter


def combine_centers(shards: list[Mapping[str, Any]], extension: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the complete four-center roster without consuming physical results."""
    if len(shards) != 4 or len(extension['centers']) != 4:
        raise ValueError('the registered extension requires four center shards')
    by_index = {item['center_index']: item for item in shards}
    if set(by_index) != set(range(4)):
        raise ValueError('center indices are missing or duplicated')
    first = by_index[0]
    identities = ('base_config_sha256', 'extension_config_sha256', 'source_sha256', 'source_parameter_hash',
                  'checkpoint_sha256', 'k_points_sha256', 'k_points', 'code_commit')
    if (first.get('base_config_sha256') != extension['base_config_sha256']
            or any(not first.get(key) or any(by_index[i].get(key) != first[key] for i in range(4))
                   for key in identities)):
        raise ValueError('center shards differ in configuration, source, model, or code')
    labels = ['center', 'gradient_0.05', 'negative_gradient_0.05', 'random_0.05',
              'negative_random_0.05', 'gradient_0.1', 'negative_gradient_0.1', 'random_0.1', 'negative_random_0.1']
    records = []
    seen = {}
    for index in range(4):
        shard = by_index[index]
        if shard['center'] != extension['centers'][index] or [r['label'] for r in shard['records']] != labels:
            raise ValueError('center or direction roster mismatch')
        for record in shard['records']:
            item = {**record, 'label': f'center_{index}/{record["label"]}', 'center_index': index}
            candidate = item['candidate']
            if candidate is not None:
                digest = candidate['master_grid_sha256']
                if digest in seen:
                    item['duplicate_of'] = seen[digest]
                else:
                    seen[digest] = len(records)
            records.append(item)
    return {'scope': 'surrogate_only_four_center_extension_locked',
            **{key: first[key] for key in identities}, 'records': records,
            'center_fd_passed': [bool(by_index[i]['fd_passed']) for i in range(4)]}


def evaluate_center(source: Any, predictor: Any, *, base_config: Mapping[str, Any],
                    extension: Mapping[str, Any], center_index: int) -> dict[str, Any]:
    """Evaluate one center and its eight fixed local direction roles using GL only."""
    import torch

    centers = extension['centers']
    if isinstance(center_index, bool) or not isinstance(center_index, int) or not 0 <= center_index < len(centers):
        raise ValueError('invalid preregistered center index')
    keys = tuple((int(i), str(component)) for i, component in base_config['coefficient_keys'])
    base = torch.tensor([source.parameters['modes'][i][component] for i, component in keys]
                        + [source.parameters['threshold']], dtype=torch.float64, device=predictor.device)
    scale = torch.as_tensor(base_config['scales'], dtype=base.dtype, device=base.device)
    center = np.asarray(centers[center_index], dtype=np.float64)
    low, high = np.asarray(base_config['lower']), np.asarray(base_config['upper'])
    if (center.shape != tuple(base.shape) or scale.shape != base.shape or low.shape != center.shape
            or high.shape != center.shape or not np.isfinite(center).all()
            or not torch.isfinite(scale).all() or (scale <= 0).any()
            or np.any(center < low) or np.any(center > high)):
        raise ValueError('invalid normalized center, parameter scales, or bounds')
    started = time.monotonic()
    def objective(z: Any) -> Any:
        return fourier_surrogate_margin(source, predictor, base + scale*z, coefficient_keys=keys,
            master_resolution=base_config['source_resolution'], input_resolution=base_config['input_resolution'],
            temperature=base_config['temperature'], lower_band=base_config['lower_band'],
            target_frequency=base_config['target_frequency'], beta=base_config['beta'])
    z = torch.tensor(center, dtype=base.dtype, device=base.device, requires_grad=True)
    score = objective(z)
    score_value = float(score.detach().cpu())
    score_finite = bool(np.isfinite(score_value))
    gradient = torch.autograd.grad(score, z, allow_unused=True)[0] if score.requires_grad and score_finite else None
    gradient_values = np.zeros(len(center)) if gradient is None else gradient.detach().cpu().numpy()
    norm = float(np.linalg.norm(gradient_values))
    gradient_valid = bool(np.isfinite(gradient_values).all() and np.isfinite(norm) and norm > 0)
    fd = []
    if gradient_valid:
        with torch.no_grad():
            for step in base_config['finite_difference_steps']:
                values = []
                for i in range(len(z)):
                    delta = torch.zeros_like(z); delta[i] = step
                    estimate = float((objective(z + delta) - objective(z - delta)) / (2*step))
                    values.append(estimate if np.isfinite(estimate) else None)
                fd.append({'step': step, 'gradient': values})
    rng = np.random.default_rng(int(base_config['seed']) + int(extension['seed_offset']) + center_index)
    random = rng.normal(size=len(center)); random /= np.linalg.norm(random)
    direction = gradient_values/norm if gradient_valid else None
    directions = [('gradient', direction), ('negative_gradient', None if direction is None else -direction),
                  ('random', random), ('negative_random', -random)]
    proposals = [{'label': 'center', 'point': center, 'clipped': False}]
    for radius in base_config['direction_radii']:
        for label, vector in directions:
            raw = None if vector is None else center + radius*vector
            clipped = None if raw is None else np.clip(raw, low, high)
            proposals.append({'label': f'{label}_{radius:g}', 'point': clipped,
                              'clipped': None if raw is None else bool(np.any(clipped != raw))})
    active = [item for item in proposals if item['point'] is not None]
    history = [{'fourier_parameters': (base.detach().cpu().numpy() + scale.detach().cpu().numpy()*item['point']).tolist()}
               for item in active]
    ranked = rank_hard_trajectory(source, predictor, history, coefficient_keys=keys,
        max_mode_radius=base_config['max_mode_radius'], raster_method='master_block_mean_v1',
        source_resolution=base_config['source_resolution'], input_resolution=base_config['input_resolution'],
        material_fraction_bounds=tuple(base_config['material_fraction_bounds']),
        lower_band=base_config['lower_band'], target_frequency=base_config['target_frequency'])
    iterator = iter(ranked['candidates'])
    records = []
    for item in proposals:
        candidate = next(iterator) if item['point'] is not None else None
        records.append({'label': item['label'], 'normalized_point': None if item['point'] is None else item['point'].tolist(),
                        'clipped': item['clipped'], 'candidate': candidate,
                        'status': 'direction_unavailable' if candidate is None else candidate['status']})
    absolute = base_config['gradient_fd_absolute_tolerance']
    relative = base_config['gradient_fd_relative_tolerance']
    valid_steps = sum(all(b is not None and np.isfinite(b) and abs(a-b) <= absolute + relative*max(abs(a), abs(b))
                          for a, b in zip(gradient_values, row['gradient'])) for row in fd)
    return {'scope': 'surrogate_only_four_center_extension', 'center_index': center_index,
            'center': center.tolist(), 'center_objective': score_value if score_finite else None,
            'gradient': gradient_values.tolist() if gradient_valid else None,
            'gradient_status': 'available' if gradient_valid else 'zero_or_nonfinite',
            'finite_differences': fd, 'fd_valid_steps': valid_steps,
            'fd_passed': valid_steps >= base_config['gradient_fd_minimum_steps'],
            'random_unit_direction': random.tolist(), 'records': records, 'elapsed_s': time.monotonic()-started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-config', 'extension-config', 'source', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--center-index', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    base_raw, extension_raw = Path(args.base_config).read_bytes(), Path(args.extension_config).read_bytes()
    config, extension = json.loads(base_raw), json.loads(extension_raw)
    base_hash = hashlib.sha256(base_raw).hexdigest()
    if base_hash != extension['base_config_sha256'] or len(extension['centers']) != 4:
        raise ValueError('extension is not bound to the registered base configuration')
    source_mapping, _ = read_checked_json(args.source, config['source_sha256'])
    source = FullCellStructureSpec.from_mapping(source_mapping)
    predictor = FrozenBandAdapter(args.checkpoint, sha256=config['checkpoint_sha256'], device=args.device)
    if (predictor.model_name != 'global_local_band_operator_v1' or predictor.k_points_sha256 != config['k_points_sha256']
            or predictor.input_shape != (config['input_resolution'],)*2 or predictor.output_shape != tuple(config['output_shape'])):
        raise ValueError('checkpoint contract mismatch')
    result = evaluate_center(source, predictor, base_config=config, extension=extension, center_index=args.center_index)
    result.update(base_config_sha256=base_hash, extension_config_sha256=hashlib.sha256(extension_raw).hexdigest(),
                  source_sha256=config['source_sha256'], source_parameter_hash=source.parameter_hash,
                  checkpoint_sha256=predictor.checkpoint_sha256, k_points_sha256=predictor.k_points_sha256,
                  k_points=predictor.k_points.detach().cpu().tolist(),
                  code_commit=os.environ.get('PHC2D_DEPLOYED_GIT_COMMIT'),
                  cluster_id=os.environ.get('PHC2D_CLUSTER_ID'), device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
