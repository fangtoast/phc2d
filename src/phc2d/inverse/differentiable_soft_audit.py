"""Record soft/hard surrogate predictions for preselected inverse candidates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.data.differentiable_fourier import fourier_material_weight
from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
from phc2d.inverse.differentiable_runtime import configure_design_runtime


def audit_locked_soft(lock: Mapping[str, Any], predictor: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Predict fixed soft geometries without changing the locked hard selection."""
    import torch

    keys = tuple((int(i), str(component)) for i, component in config['coefficient_keys'])
    started = time.monotonic()
    lower_band = int(config['lower_band'])
    target = float(config['target_frequency'])
    hard_predictions: dict[str, Any] = {}
    for record in lock['records']:
        candidate = record.get('candidate')
        if candidate is not None and 'bands' in candidate:
            hard_predictions.setdefault(candidate['master_grid_sha256'], candidate['bands'])
    rows = []
    for record in lock['records']:
        candidate = record.get('candidate')
        row = {'label': record['label']}
        if candidate is None or candidate['status'] not in ('surrogate_only', 'duplicate_geometry'):
            rows.append({**row, 'status': 'not_eligible'})
            continue
        structure = FullCellStructureSpec.from_mapping(candidate['structure'])
        if structure.parameter_hash != candidate['parameter_hash']:
            raise ValueError('locked parameter identity mismatch')
        master = np.ascontiguousarray(structure.rasterize((config['source_resolution'],)*2), dtype=np.uint8)
        if hashlib.sha256(master.tobytes()).hexdigest() != candidate['master_grid_sha256']:
            raise ValueError('locked master geometry mismatch')
        params = torch.tensor([structure.parameters['modes'][i][component] for i, component in keys]
                              + [structure.parameters['threshold']], dtype=torch.float64, device=predictor.device)
        soft = fourier_material_weight(structure, params, coefficient_keys=keys,
            master_resolution=config['source_resolution'], output_resolution=config['input_resolution'],
            temperature=config['temperature'])
        with torch.no_grad():
            bands = predictor.predict_tensor(soft.unsqueeze(0)).detach().cpu().numpy()[0]
        hard = np.asarray(candidate.get('bands', hard_predictions.get(candidate['master_grid_sha256'])), dtype=np.float64)
        bands = np.asarray(bands, dtype=np.float64)
        if (bands.ndim != 2 or hard.shape != bands.shape or not np.isfinite(bands).all()
                or not np.isfinite(hard).all() or not 1 <= lower_band < bands.shape[1]
                or not np.isfinite(target) or target <= 0):
            raise ValueError('invalid soft or locked hard band prediction')
        def margin(values: np.ndarray) -> float:
            return float(2 * min(target - values[:, lower_band-1].max(),
                                 values[:, lower_band].min() - target) / target)
        row.update(status='computed', master_grid_sha256=candidate['master_grid_sha256'],
                   parameter_hash=structure.parameter_hash, soft_bands=bands.tolist(),
                   soft_band_order_violations=int(np.count_nonzero(np.diff(bands, axis=1) < 0)),
                   soft_exact_margin=margin(bands), hard_exact_margin=margin(hard),
                   soft_minus_hard_margin=margin(bands)-margin(hard),
                   soft_hard_spectrum_mae=float(np.mean(np.abs(bands-hard))))
        rows.append(row)
    return {'scope': 'locked_surrogate_soft_hard_diagnostic', 'temperature': config['temperature'],
            'forward_inferences': sum(row['status'] == 'computed' for row in rows),
            'elapsed_s': time.monotonic() - started, 'roles': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'lock', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--campaign-manifest')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    config_raw, lock_raw = Path(args.config).read_bytes(), Path(args.lock).read_bytes()
    config, lock = json.loads(config_raw), json.loads(lock_raw)
    runtime_info = configure_design_runtime(config['runtime'], device=args.device)
    config_sha = hashlib.sha256(config_raw).hexdigest()
    if args.campaign_manifest:
        campaign_raw = Path(args.campaign_manifest).read_bytes()
        campaign = json.loads(campaign_raw)
        points = np.ascontiguousarray(lock['k_points'], dtype=np.float32)
        if (hashlib.sha256(campaign_raw).hexdigest() != lock.get('manifest_sha256')
                or campaign.get('base_config_sha256') != config_sha
                or hashlib.sha256(points.tobytes()).hexdigest() != config['k_points_sha256']):
            raise ValueError('campaign lock and configuration differ')
        k_sha, checkpoint_sha = config['k_points_sha256'], config['checkpoint_sha256']
    else:
        if lock['config_sha256'] != config_sha or lock['precision_contract'] != runtime_info['contract']:
            raise ValueError('lock and configuration differ')
        k_sha, checkpoint_sha = lock['k_points_sha256'], lock['checkpoint_sha256']
    predictor = FrozenBandAdapter(args.checkpoint, sha256=config['checkpoint_sha256'], device=args.device)
    if predictor.k_points_sha256 != k_sha or predictor.checkpoint_sha256 != checkpoint_sha:
        raise ValueError('locked model or k-point identity mismatch')
    report = audit_locked_soft(lock, predictor, config)
    report.update(lock_sha256=hashlib.sha256(lock_raw).hexdigest(), config_sha256=config_sha,
                  checkpoint_sha256=checkpoint_sha, k_points_sha256=k_sha,
                  runtime=runtime_info)
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
