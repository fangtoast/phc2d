"""Run a shard of the frozen-source differentiable inverse design MVP.

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
from typing import Any

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.inverse.audit_refine import read_checked_json
from phc2d.inverse.differentiable_objective import (fourier_surrogate_margin,
    optimize_fourier_adam, optimize_fourier_black_box)
from phc2d.inverse.differentiable_selection import rank_hard_trajectory
from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
from phc2d.inverse.differentiable_runtime import configure_design_runtime


def run_shard(config: dict[str, Any], source: FullCellStructureSpec, predictor: Any,
              *, role: str) -> dict[str, Any]:
    """Compute one pure-surrogate shard; no physical results enter this function."""
    import numpy as np
    import torch

    if role not in ('adam', 'de', 'random', 'directions'):
        raise ValueError('invalid shard role')
    keys = tuple((int(i), str(component)) for i, component in config['coefficient_keys'])
    shared = dict(coefficient_keys=keys, scales=config['scales'], lower=config['lower'],
        upper=config['upper'], master_resolution=config['source_resolution'],
        input_resolution=config['input_resolution'], temperature=config['temperature'],
        lower_band=config['lower_band'], target_frequency=config['target_frequency'], beta=config['beta'])
    start = time.monotonic()
    diagnostics = None
    if role == 'adam':
        history = optimize_fourier_adam(source, predictor, **shared,
            learning_rate=config['learning_rate'], steps=config['evaluations'] - 1,
            initial=config.get('initial'))
    elif role in ('de', 'random'):
        history = optimize_fourier_black_box(source, predictor, **shared,
            evaluations=config['evaluations'], seed=config['seed'], method=role,
            initial=config.get('initial'))
    else:
        base = torch.tensor([source.parameters['modes'][i][component] for i, component in keys]
                            + [source.parameters['threshold']], dtype=torch.float64, device=predictor.device)
        scales = torch.as_tensor(config['scales'], dtype=base.dtype, device=base.device)
        def objective(z: Any) -> Any:
            return fourier_surrogate_margin(source, predictor, base + scales*z,
                coefficient_keys=keys, master_resolution=config['source_resolution'],
                input_resolution=config['input_resolution'], temperature=config['temperature'],
                lower_band=config['lower_band'], target_frequency=config['target_frequency'], beta=config['beta'])
        z = torch.zeros_like(base, requires_grad=True)
        score = objective(z)
        gradient, = torch.autograd.grad(score, z)
        if not torch.isfinite(gradient).all() or float(torch.linalg.vector_norm(gradient)) == 0:
            raise ValueError('nonfinite or zero center gradient')
        fd = []
        with torch.no_grad():
            for step in config['finite_difference_steps']:
                estimates = []
                for i in range(len(z)):
                    delta = torch.zeros_like(z); delta[i] = step
                    estimates.append(float((objective(z + delta) - objective(z - delta)) / (2*step)))
                fd.append({'step': step, 'gradient': estimates})
        grad = gradient.detach().cpu().numpy()
        rng = np.random.default_rng(config['seed'])
        random = rng.normal(size=len(grad)); random /= np.linalg.norm(random)
        unit = grad / np.linalg.norm(grad)
        directions = [('gradient', unit), ('negative_gradient', -unit),
                      ('random', random), ('negative_random', -random)]
        history = [{'step': 0, 'label': 'center', 'parameters': [0.0]*len(grad),
                    'fourier_parameters': base.detach().cpu().tolist()}]
        for radius in config['direction_radii']:
            for label, direction in directions:
                point = np.clip(radius * direction, config['lower'], config['upper'])
                params = base.detach().cpu().numpy() + np.asarray(config['scales']) * point
                history.append({'step': len(history), 'label': f'{label}_{radius:g}',
                                'parameters': point.tolist(), 'fourier_parameters': params.tolist()})
        absolute = config['gradient_fd_absolute_tolerance']
        relative = config['gradient_fd_relative_tolerance']
        valid_steps = sum(all(abs(a-b) <= absolute + relative*max(abs(a), abs(b))
                              for a, b in zip(grad, row['gradient'])) for row in fd)
        diagnostics = {'objective': float(score.detach().cpu()), 'gradient': grad.tolist(),
                       'finite_differences': fd, 'random_unit_direction': random.tolist(),
                       'fd_valid_steps': valid_steps,
                       'fd_passed': valid_steps >= config['gradient_fd_minimum_steps']}
    selection = rank_hard_trajectory(source, predictor, history, coefficient_keys=keys,
        max_mode_radius=config['max_mode_radius'], raster_method='master_block_mean_v1',
        source_resolution=config['source_resolution'], input_resolution=config['input_resolution'],
        material_fraction_bounds=tuple(config['material_fraction_bounds']),
        lower_band=config['lower_band'], target_frequency=config['target_frequency'])
    return {'role': role, 'history': history, 'diagnostics': diagnostics, 'selection': selection,
            'elapsed_s': time.monotonic() - start, 'scope': 'surrogate_only'}


def load_design_inputs(config: dict[str, Any], source_path: str, checkpoint_path: str,
                       *, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load and verify the frozen surrogate and a source under the design contract."""
    import torch
    runtime_info = configure_design_runtime(config['runtime'], device=device)
    source_mapping, _ = read_checked_json(source_path, config['source_sha256'])
    source = FullCellStructureSpec.from_mapping(source_mapping)
    predictor = FrozenBandAdapter(checkpoint_path, sha256=config['checkpoint_sha256'], device=device)
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (predictor.model_name != 'global_local_band_operator_v1'
            or predictor.k_points_sha256 != config['k_points_sha256']
            or predictor.input_shape != (config['input_resolution'],)*2
            or predictor.output_shape != tuple(config['output_shape'])
            or payload['model_params'].get('local_kernel') != config['local_kernel']
            or payload['resolved_config']['experiment']['domain'] != config['dataset_domain']
            or payload['resolved_config']['experiment']['seed'] != config['model_seed']
            or payload['model_identity']['architecture_fingerprint'] != config['architecture_fingerprint']
            or any(payload['resolved_config']['runtime'].get(key) != value
                   for key, value in config['runtime'].items())
            or source.epsilon_air != config['solver_config']['epsilon_air']
            or source.epsilon_material != config['solver_config']['epsilon_material']):
        raise ValueError('checkpoint identity or input contract mismatch')
    return source, predictor, runtime_info


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source', 'checkpoint', 'role', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists() or path.with_suffix(path.suffix + '.failure.json').exists():
        raise FileExistsError('output or prior failure exists; reconcile before retry')
    raw = Path(args.config).read_bytes()
    config = json.loads(raw)
    source, predictor, runtime_info = load_design_inputs(config, args.source, args.checkpoint, device=args.device)
    result = None
    try:
        if args.role == 'preflight':
            probe = run_shard(config, source, predictor, role='directions')
            center = probe['selection']['candidates'][0]
            result = {'role': 'preflight', 'scope': 'engineering_preflight_only',
                      'elapsed_s': probe['elapsed_s'], 'diagnostics': probe['diagnostics'],
                      'precision_contract': runtime_info['contract'], 'runtime': runtime_info,
                      'center': {key: center.get(key) for key in
                                 ('status', 'parameter_hash', 'master_grid_sha256', 'input_sha256', 'bands')},
                      'cuda_peak_memory_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None}
            if result['center']['status'] != 'surrogate_only' or not result['diagnostics']['fd_passed']:
                raise ValueError('preflight center prediction or gradient gate failed')
        else:
            result = run_shard(config, source, predictor, role=args.role)
    except Exception as exc:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix + '.failure.json').open('x', encoding='utf8') as stream:
            json.dump({'role': args.role, 'error_type': type(exc).__name__, 'message': str(exc),
                       'config_sha256': hashlib.sha256(raw).hexdigest(), 'partial_result': result},
                      stream, indent=2, allow_nan=False)
        raise
    result.update(config_sha256=hashlib.sha256(raw).hexdigest(), source_sha256=config['source_sha256'],
                  source_parameter_hash=source.parameter_hash, checkpoint_sha256=predictor.checkpoint_sha256,
                  k_points_sha256=predictor.k_points_sha256,
                  k_points=predictor.k_points.detach().cpu().tolist(),
                  device_parity_tolerance=config['device_parity_tolerance'],
                  precision_contract=runtime_info['contract'], runtime=runtime_info,
                  code_commit=os.environ.get('PHC2D_DEPLOYED_GIT_COMMIT'),
                  cluster_id=os.environ.get('PHC2D_CLUSTER_ID'), device=args.device)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
